"""Per-turn state tokens in ember's interpreter (paperkiln IR_SPEC_DRAFT.md sections 1.1-1.4), CPU, on
the synthetic model make_state_gguf.py writes (chat_sftfinal_v3 + an untrained state_embd.weight,
k = 4, paperkiln.state_tokens.k = 4). No trained state-token model exists yet, so every check is about
the engine's arithmetic and bookkeeping:
  (a) with no state ids, turn_state (eviction on or off) equals plain causal attention bit for bit;
  (b) prefill equals token-by-token decode bit for bit on multi-turn sequences with state blocks; the
      mask against an independent numpy reference of section 1.3 (ROUNDING), and against the causal
      mask (the mask is read);
  (c) eviction on and off bit-equal over a multi-turn session with generation, a regenerated reply and
      rewinds into evicted turns (state_tokens_test run), with fewer kv rows held when evicting;
  (d) the template rule (state_tokens_test template) and the /chat server end to end: prompt ids with
      <|endoftext|> + k ids after every reply (also one cut by max_new_tokens), the generated reply's
      block prefilled, prefix reuse after an edited earlier message resuming at that turn's start,
      replies equal across eviction on, eviction off and no prefix reuse;
  plus load-time and run-time refusals.
Writes state_tokens_gate_cpu.json beside this file. Usage: python3 state_tokens_gate.py [DIR]"""
import copy
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PK = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch"
EMBER = os.path.dirname(os.path.dirname(HERE))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ember_wt/state_tokens")
B = os.path.expanduser("~/ember_wt/build_cpu")
GGUF = os.path.join(D, "chat_state_k4.gguf")
ORIG = "/mnt/c/ml_artifacts/chat_base/sftfinal/chat_sftfinal_v3.gguf"
SPEC = os.path.join(PK, "spec", "tools")  # the IR tools (keramos): $KERAMOS_TOOLS, spec/tools, tools/
sys.path.insert(0, os.path.join(PK, "tools"))
sys.path.insert(0, os.path.expanduser(os.environ.get("KERAMOS_TOOLS") or (
    SPEC if os.path.isfile(os.path.join(SPEC, "ir_numpy.py")) else os.path.join(PK, "tools"))))
import ir_numpy  # noqa: E402

V, K = 8192, 4
EOT, USER, ASST = 1, 2, 3
S = [V + j for j in range(K)]
rng = np.random.default_rng(7)
IR = {n: json.load(open(os.path.join(D, n + ".ir.json"))) for n in ("ts", "ts_noevict", "ts_causal", "plain")}


def text(n):
    return [int(x) for x in rng.integers(4, V, n)]


def env_with(**kw):
    e = {k: v for k, v in os.environ.items() if k not in ("EMBER_IR", "EMBER_EVICT")}
    e.update({k: v for k, v in kw.items() if v is not None})
    return e


def ir_path(name, doc=None):
    p = os.path.join(D, name + ".ir.json")
    if doc is not None:
        json.dump(doc, open(p, "w"))
    return p


def ir_logits(gguf, seqs, ir=None, **kw):
    r = subprocess.run([os.path.join(B, "ir_logits"), gguf], input=json.dumps(seqs), capture_output=True, text=True,
                       env=env_with(EMBER_IR=ir, **kw))
    return r


def logits(gguf, seqs, ir=None, **kw):
    r = ir_logits(gguf, seqs, ir, **kw)
    if r.returncode:
        raise SystemExit("ir_logits failed: " + r.stderr[-1500:])
    return [{k: np.array(v[k], dtype=np.float32) for k in ("prefill", "decode")} for v in json.loads(r.stdout)]


def exchange(u, a, closed=True):
    return [USER] + text(u) + [ASST] + text(a) + ([EOT] + S if closed else [])


report = {"gguf": GGUF, "k": K, "vocab": V}

# ---- (a) no state ids: turn_state == causal, bit for bit ----
plain_seqs = [[USER] + text(20) + [ASST] + text(30) + [EOT] + [USER] + text(15) + [ASST] + text(10),
              text(64), [USER] + text(40) + [ASST]]
ref = logits(ORIG, plain_seqs)  # the original file, its own causal IR
a = {}
for name in ("ts", "ts_noevict"):
    got = logits(GGUF, plain_seqs, ir_path(name))
    a[name] = {"prefill_bit_equal": all(np.array_equal(g["prefill"], r["prefill"]) for g, r in zip(got, ref)),
               "decode_bit_equal": all(np.array_equal(g["decode"], r["decode"]) for g, r in zip(got, ref))}
a["pass"] = all(v["prefill_bit_equal"] and v["decode_bit_equal"] for v in a.values())
report["a_no_state_ids_equals_causal"] = a

# ---- (b) prefill == decode with state blocks; the mask against numpy ----
state_seqs = [exchange(8, 12) + exchange(6, 9) + exchange(10, 5) + [USER] + text(7) + [ASST] + text(4),
              exchange(5, 3) + exchange(4, 0) + [USER] + text(3) + [ASST],  # an empty reply
              exchange(12, 20) + [USER] + text(6) + [ASST] + text(5) + [EOT] + S[:2]]  # ends in an open block
W = ir_numpy.load_weights(GGUF)


def segments(ids):
    s, c, done, run = [], [], 0, 0
    for t in ids:
        s.append(done)
        c.append(t >= V)
        if t >= V:
            run += 1
            if run == K:
                done, run = done + 1, 0
    return np.array(s), np.array(c)


def visible(ids, kind):
    T = len(ids)
    p, j = np.arange(T)[:, None], np.arange(T)[None, :]
    if kind == "causal":
        return j <= p
    s, c = segments(ids)
    return (j <= p) & ((s[None, :] == s[:, None]) | (c[None, :] & (s[None, :] < s[:, None])))


def numpy_forward(doc, ids, kind):
    env = {"tokens": np.array(ids), "pos": np.arange(len(ids))}
    table = np.concatenate([W["token_embd.weight"], W["state_embd.weight"]])
    for nd in doc["embed"]["nodes"]:
        env[nd["out"]] = table[np.array(ids)]
    vis = visible(ids, kind)
    for i in range(doc["block"]["repeat"]):
        for nd in doc["block"]["nodes"]:
            if nd["op"] != "attention":
                ir_numpy.run_nodes([nd], env, W, i)
                continue
            q, k, v = (env[n] for n in nd["in"])
            H, hd, KV = nd["heads"], nd["head_dim"], nd["kv_heads"]
            outs = []
            for h in range(H):
                kh = h // (H // KV)
                sc = (q[:, h * hd:(h + 1) * hd] @ k[:, kh * hd:(kh + 1) * hd].T) * np.float32(nd["scale"])
                sc[~vis] = -np.inf
                sc = np.exp(sc - sc.max(axis=1, keepdims=True))
                sc /= sc.sum(axis=1, keepdims=True)
                outs.append(sc @ v[:, kh * hd:(kh + 1) * hd])
            env[nd["out"]] = np.concatenate(outs, axis=1).astype(np.float32)
    for nd in doc["head"]["nodes"]:
        ir_numpy.run_nodes([nd], env, W)
    return env["logits"]


b = {}
for name in ("ts", "ts_noevict"):
    got = logits(GGUF, state_seqs, ir_path(name))
    rows = []
    for ids, g in zip(state_seqs, got):
        refl = numpy_forward(IR["ts"], ids, "turn_state")
        rng_ = refl.max(axis=-1) - refl.min(axis=-1)
        rows.append({"length": len(ids), "decode_eq_prefill": bool(np.array_equal(g["decode"], g["prefill"])),
                     "vs_numpy_rel": float((np.abs(g["prefill"] - refl).max(axis=-1) / rng_).max()),
                     "top1_vs_numpy": f"{int((g['prefill'].argmax(1) == refl.argmax(1)).sum())}/{len(ids)}"})
    b[name] = rows
got_ts = logits(GGUF, state_seqs, ir_path("ts"))
got_c = logits(GGUF, state_seqs, ir_path("ts_causal"))
b["ts_noevict_eq_ts"] = all(np.array_equal(x["prefill"], y["prefill"]) and np.array_equal(x["decode"], y["decode"])
                            for x, y in zip(logits(GGUF, state_seqs, ir_path("ts_noevict")), got_ts))
caus = []
for ids, g, c in zip(state_seqs, got_ts, got_c):
    refc = numpy_forward(IR["ts"], ids, "causal")
    rng_ = refc.max(axis=-1) - refc.min(axis=-1)
    first_masked = int(np.argmax(np.array(segments(ids)[0]) > 0))  # the first position of segment 1
    caus.append({"causal_vs_numpy_rel": float((np.abs(c["prefill"] - refc).max(axis=-1) / rng_).max()),
                 "segment0_equal_to_causal": bool(np.array_equal(g["prefill"][:first_masked],
                                                                 c["prefill"][:first_masked])),
                 "later_max_change_vs_causal": float(np.abs(g["prefill"][first_masked:] -
                                                            c["prefill"][first_masked:]).max())})
b["mask_is_read"] = caus
b["pass"] = (all(r["decode_eq_prefill"] and r["vs_numpy_rel"] <= 1e-5 for n in ("ts", "ts_noevict") for r in b[n])
             and b["ts_noevict_eq_ts"] and all(c["segment0_equal_to_causal"] and c["later_max_change_vs_causal"] > 0
                                                and c["causal_vs_numpy_rel"] <= 1e-5 for c in caus))
report["b_prefill_eq_decode_and_mask"] = b

# ---- refusals ----
ref_cases = {}


def refused(name, gguf, seqs, doc=None, irname=None, want=""):
    p = ir_path("refuse_" + name, doc) if doc is not None else (ir_path(irname) if irname else None)
    r = ir_logits(gguf, seqs, p)
    msg = [ln for ln in r.stderr.splitlines() if ln.strip()]
    ok = r.returncode != 0 and want in r.stderr
    ref_cases[name] = {"refused": r.returncode != 0, "message_has": want, "pass": ok,
                       "message": (msg[-1][-240:] if msg else "")}


one = [[USER, 5, ASST, 6]]
d = copy.deepcopy(IR["ts_noevict"])
[n.update(window=8) for n in d["block"]["nodes"] if n["op"] == "attention"]
d["block"]["state"][0]["window"] = 8
refused("window", GGUF, one, d, want="turn_state")
d = copy.deepcopy(IR["ts_noevict"])
[n.update(sinks=2) for n in d["block"]["nodes"] if n["op"] == "attention"]
refused("sinks", GGUF, one, d, want="sinks")
d = copy.deepcopy(IR["ts_causal"])
d["block"]["state"][0]["evict"] = "turn_state"
refused("evict_without_mask", GGUF, one, d, want="evict")
d = copy.deepcopy(IR["ts"])
d["block"]["state"][0]["evict"] = "lru"
refused("evict_unknown", GGUF, one, d, want="evict")
d = copy.deepcopy(IR["ts"])
[n.update(mask="bidirectional") for n in d["block"]["nodes"] if n["op"] == "attention"]
refused("mask_unknown", GGUF, one, d, want="mask")
refused("metadata_k_without_state_weight", GGUF, one, irname="plain", want="state_tokens.k")
d = copy.deepcopy(IR["ts"])
[n.update(state_weight="blk.0.ffn_gate.weight") for n in d["embed"]["nodes"]]
refused("state_weight_rows_differ_from_k", GGUF, one, d, want="state_tokens.k")
d = copy.deepcopy(IR["ts"])
[n.update(state_weight="blk.0.ffn_down.weight") for n in d["embed"]["nodes"]]
refused("state_weight_wrong_width", GGUF, one, d, want="not [k, d]")
refused("block_out_of_order", GGUF, [[USER, 5, ASST, 6, EOT, S[0], S[2], S[1], S[3]]], irname="ts", want="expected S_1")
refused("text_inside_block", GGUF, [[USER, 5, ASST, 6, EOT, S[0], S[1], USER]], irname="ts", want="incomplete state block")
refused("block_not_from_S0", GGUF, [[USER, 5, ASST, 6, EOT, S[1]]], irname="ts", want="S_0")
refused("id_past_k", GGUF, [[USER, 5, ASST, 6, EOT, V + K]], irname="ts", want="past the")
refused("state_id_without_state_weight", ORIG, [[USER, 5, ASST, 6, EOT, V]], want="out of range")
report["refusals"] = ref_cases
report["refusals_pass"] = all(v["pass"] for v in ref_cases.values())

# ---- (c) eviction on vs off: a session with generation, regeneration and rewinds ----
t = [exchange(9, 11), exchange(7, 13), exchange(10, 8), exchange(6, 12), exchange(8, 7)]
conv = sum(t, [])
cut = lambda n: sum(t[:n], [])  # noqa: E731
calls = []
# turn 0: the prompt, then the reply token by token, <|endoftext|> and the block (the generation loop)
p0 = cut(0) + t[0][:10 + 1]  # <|user|> U <|assistant|>
calls.append({"seq": p0, "from": 0})
calls.append({"seq": cut(1), "from": len(p0), "step": True})
for n in (1, 2):  # the next prompts reuse the prefix; replies decoded
    pn = cut(n) + t[n][:t[n].index(ASST) + 1]
    calls.append({"seq": pn, "from": len(cut(n))})
    calls.append({"seq": cut(n + 1), "from": len(pn), "step": True})
# regenerate the latest reply (turn 2): back to its <|assistant|>, a different reply, not evicted
regen = cut(2) + t[2][:t[2].index(ASST) + 1] + text(6) + [EOT] + S
calls.append({"seq": regen, "from": len(cut(2)) + t[2].index(ASST) + 1, "step": True})
# an edited earlier message: turn 1's user text changes at its 4th token -> rewind into evicted turn 1
edited1 = copy.deepcopy(t[1])
edited1[4] = (edited1[4] + 1) % (V - 4) + 4
seq_e = cut(1) + edited1 + t[2] + [USER] + text(5) + [ASST]
calls.append({"seq": seq_e, "from": len(cut(1)) + 4})
calls.append({"seq": seq_e + text(5) + [EOT] + S, "from": len(seq_e), "step": True})
# the whole five-turn conversation in one prefill from 0, then a rewind into evicted turn 0
calls.append({"seq": conv + [USER] + text(4) + [ASST], "from": 0})
seq0 = t[0][:3] + text(5) + [ASST] + text(9) + [EOT] + S + t[1] + [USER] + text(3) + [ASST]
calls.append({"seq": seq0, "from": 3})
calls.append({"seq": seq0 + text(4), "from": len(seq0), "step": True})
script = os.path.join(D, "evict_script.json")
json.dump({"calls": calls}, open(script, "w"))
r = subprocess.run([os.path.join(B, "state_tokens_test"), "run", GGUF], stdin=open(script), capture_output=True,
                   text=True, env=env_with(EMBER_IR=ir_path("ts")))
if r.returncode not in (0, 1):
    raise SystemExit("state_tokens_test run: " + r.stderr[-1500:])
c = json.loads(r.stdout)
c["evicting_holds_fewer_rows"] = any(x["kv_rows_held"]["on"] < x["kv_rows_held"]["off"] for x in c["calls"])
c["rewinds_resumed_at_turn_start"] = [x["resumed_at"]["on"] for x in c["calls"] if "direct_rewind_refused" in x]
c["pass"] = c["pass"] and c["evicting_holds_fewer_rows"] and len(c["rewinds_resumed_at_turn_start"]) >= 2
report["c_evict_on_eq_off"] = c

# ---- (d) the template rule and the /chat server ----
r = subprocess.run([os.path.join(B, "state_tokens_test"), "template"], capture_output=True, text=True)
report["d_template_rule"] = json.loads(r.stdout)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


U1, U2, U3, U4 = ("Hi! How are you today?", "What is your favourite colour and why?",
                  "Tell me a short story about a cat.", "Thanks. Can you say that again in one line?")


def session(extra_env):
    port = free_port()
    log = open(os.path.join(D, "server_" + "_".join(f"{k}{v}" for k, v in extra_env.items()) + ".log"), "w")
    srv = subprocess.Popen([os.path.join(B, "tinyllama_server"), GGUF, str(port), "127.0.0.1",
                            os.path.join(EMBER, "www"), "0"], stdout=log, stderr=subprocess.STDOUT,
                           env=env_with(**extra_env))
    try:
        for _ in range(300):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        g = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0, "return_ids": True}
        out = []
        r1 = post(port, {"user_input": U1, "history": [], "max_new_tokens": 6, **g})  # cut by max_new_tokens
        out.append(r1)
        h = [{"user": U1, "assistant": r1["reply"]}]
        r2 = post(port, {"user_input": U2, "history": h, "max_new_tokens": 48, **g})
        out.append(r2)
        h2 = h + [{"user": U2, "assistant": r2["reply"]}]
        out.append(post(port, {"user_input": U3, "history": h2, "max_new_tokens": 32, **g}))
        # regenerate the latest reply (same request again)
        out.append(post(port, {"user_input": U3, "history": h2, "max_new_tokens": 32, **g}))
        # an edited earlier message (turn 1's user text): prefix reuse ends inside an evicted turn
        h_ed = h + [{"user": U2.replace("colour", "food"), "assistant": r2["reply"]}]
        out.append(post(port, {"user_input": U4, "history": h_ed, "max_new_tokens": 32, **g}))
        return out
    finally:
        srv.terminate()
        srv.wait()


def blocks_ok(ids):
    """Every id >= V sits in a complete ordered block right after an <|endoftext|>, and every <|user|>
    after the first is preceded by such a block."""
    i, n_blocks = 0, 0
    while i < len(ids):
        if ids[i] >= V:
            if ids[i - 1] != EOT or ids[i:i + K] != S:
                return False, n_blocks
            n_blocks += 1
            if i + K < len(ids) and ids[i + K] != USER:
                return False, n_blocks
            i += K
            continue
        if ids[i] == USER and i > 0 and ids[i - 1] != S[-1]:
            return False, n_blocks
        i += 1
    return True, n_blocks


runs = {"evict_on": session({}), "evict_off": session({"EMBER_EVICT": "0"}),
        "no_prefix_reuse": session({"EMBER_PREFIX_REUSE": "0"})}
srv = {"requests": []}
names = ["r1_cut_by_max_new_tokens", "r2", "r3", "r3_regenerated", "r4_edited_turn1"]
for i, nm in enumerate(names):
    row = {"name": nm}
    for run_name, rr in runs.items():
        x = rr[i]
        ok, nb = blocks_ok(x["prompt_ids"])
        row[run_name] = {"stop_reason": x["stop_reason"], "tokens": x["tokens"], "prompt_tokens": x["prompt_tokens"],
                         "prefix_reused_tokens": x["ttft"]["prefix_reused_tokens"], "state_blocks": nb,
                         "blocks_well_placed": ok}
    row["ids_equal"] = len({json.dumps(rr[i]["ids"]) for rr in runs.values()}) == 1
    row["prompt_ids_equal"] = len({json.dumps(rr[i]["prompt_ids"]) for rr in runs.values()}) == 1
    srv["requests"].append(row)
on = runs["evict_on"]
p4 = on[4]["prompt_ids"]
seg1_start = next(i for i in range(len(p4)) if p4[i:i + K] == S) + K  # after turn 0's block
first_diff = next(i for i, (x, y) in enumerate(zip(p4, on[3]["prompt_ids"])) if x != y)
srv["r1_reply_cut"] = on[0]["stop_reason"] == "length"
srv["r2_prompt_has_eot_and_block_after_cut_reply"] = (
    p4[:seg1_start] == on[1]["prompt_ids"][:seg1_start] and on[1]["prompt_ids"][seg1_start - K - 1] == EOT)
srv["r4_edit_position"] = first_diff
srv["r4_turn1_start"] = seg1_start
srv["r4_resumed_at_turn_start"] = on[4]["ttft"]["prefix_reused_tokens"] == seg1_start
srv["r4_without_eviction_reuses_to_edit"] = runs["evict_off"][4]["ttft"]["prefix_reused_tokens"] == first_diff
gen_closed = on[1]["stop_reason"] == "eos"
srv["r2_ended_eos"] = gen_closed
r3_prefix = len(on[1]["prompt_ids"]) + on[1]["tokens"] + 1 + K
srv["r3_reused_through_generated_block"] = (on[2]["ttft"]["prefix_reused_tokens"] >= r3_prefix) if gen_closed else None
srv["pass"] = (all(r["ids_equal"] and r["prompt_ids_equal"] and all(r[k]["blocks_well_placed"] for k in runs)
                   for r in srv["requests"]) and srv["r1_reply_cut"]
               and srv["r2_prompt_has_eot_and_block_after_cut_reply"] and srv["r4_resumed_at_turn_start"]
               and srv["r4_without_eviction_reuses_to_edit"])
report["d_server"] = srv

report["pass"] = all(report[k]["pass"] for k in ("a_no_state_ids_equals_causal", "b_prefill_eq_decode_and_mask",
                                                  "c_evict_on_eq_off", "d_template_rule", "d_server")) \
    and report["refusals_pass"]
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(HERE, "state_tokens_gate_cpu.json"), "w"), indent=1)
sys.exit(0 if report["pass"] else 1)
