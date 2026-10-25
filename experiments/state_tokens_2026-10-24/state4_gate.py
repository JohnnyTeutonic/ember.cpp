"""IR_SPEC_DRAFT.md section 5 gate 2 (CPU) on paperkiln's first trained state-token model,
/mnt/c/ml_artifacts/state_gate/state4 (llama_state: d 64, 2 layers, vocab 400, k 4, mask turn_state, kv
evict turn_state; 40 multi-turn prompts, prompt 7 with a reply cut after one word):
  - sha256 of every file against manifest.json;
  - ember's interpreter (ir_logits, the IR embedded in state4.gguf) with eviction on (the IR's own
    setting) and off (EMBER_EVICT=0): per-token log-probs of numpy's greedy continuations against
    mtstudio score (score_out.jsonl), ROUNDING (1e-5 of max(1, max |ref|)); ember's argmax at every
    scored position equal to the continuation, 40/40; decode equal to prefill and eviction on equal
    to off, bit for bit;
  - kv rows held after the context with and without eviction against mtstudio's (manifest kv_rows_evict,
    kv_rows_keep), and on = off = a fresh prefill bit for bit (state_tokens_test run);
  - the template rule against paperkiln's serialiser (tools/state_tokens.py insert / error) on 3000
    random id sequences (state_tokens_test insert);
  - /chat end to end (tinyllama_server, greedy): history and message parsed from prompts_text.jsonl, the
    prompt ids equal score_in's (prompt 7's cut reply closed with <|endoftext|> and the block), the reply
    plus its stop id equal the continuation, 40/40, identical with eviction on, off and no prefix reuse;
    every prompt equal to the serialiser's insert of its own text ids;
  - state4_q8_0.gguf: loads and runs, numbers reported.
Writes state4_gate_cpu.json beside this file."""
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

D = "/mnt/c/ml_artifacts/state_gate/state4"
HERE = os.path.dirname(os.path.abspath(__file__))
EMBER = os.path.dirname(os.path.dirname(HERE))
B = os.path.expanduser("~/ember_wt/build_cpu")
PK = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch"
SPEC = os.path.join(PK, "spec", "tools")  # the IR tools (keramos): $KERAMOS_TOOLS, spec/tools, tools/
sys.path.insert(0, os.path.join(PK, "tools"))
sys.path.insert(0, os.path.expanduser(os.environ.get("KERAMOS_TOOLS") or (
    SPEC if os.path.isfile(os.path.join(SPEC, "state_tokens.py")) else os.path.join(PK, "tools"))))
import state_tokens as pk_state  # noqa: E402

GGUF = os.path.join(D, "state4.gguf")
V, K, EOT, USER, ASST = 400, 4, 1, 2, 3


def rows(name):
    return [json.loads(x) for x in open(os.path.join(D, name)) if x.strip()]


def env_with(**kw):
    e = {k: v for k, v in os.environ.items() if k not in ("EMBER_IR", "EMBER_EVICT", "EMBER_PREFIX_REUSE")}
    e.update(kw)
    return e


man = json.load(open(os.path.join(D, "manifest.json")))
report = {"dir": D, "sha256": {}}
for f, h in man["files"].items():
    got = hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest()
    report["sha256"][f] = got == h
if not all(report["sha256"].values()):
    print(json.dumps(report, indent=1))
    raise SystemExit("sha256 mismatch: not gating")

ins, outs = rows("score_in.jsonl"), rows("score_out.jsonl")
seqs = [r["prompt_ids"] + r["continuation_ids"] for r in ins]


def ir_logits(path, **kw):
    r = subprocess.run([os.path.join(B, "ir_logits"), path], input=json.dumps(seqs), capture_output=True, text=True,
                       env=env_with(**kw))
    if r.returncode:
        raise SystemExit(path + ": " + r.stderr[-1500:])
    return r.stdout, json.loads(r.stdout)


def score(em, ref_rows):
    worst, greedy, dec_eq = 0.0, 0, True
    for k, (a, b) in enumerate(zip(ins, ref_rows)):
        assert b["continuation_ids"] == a["continuation_ids"]
        pre = np.array(em[k]["prefill"], dtype=np.float64)
        dec_eq = dec_eq and bool(np.array_equal(np.array(em[k]["decode"]), np.array(em[k]["prefill"])))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
    return {"logprob_worst_rel": worst, "greedy": f"{greedy}/{len(ins)}", "decode_eq_prefill": dec_eq}


t_on, em_on = ir_logits(GGUF)
t_off, em_off = ir_logits(GGUF, EMBER_EVICT="0")
ok_n = f"{len(ins)}/{len(ins)}"
interp = {"evict_on": score(em_on, outs), "evict_off": score(em_off, outs),
          "evict_on_vs_mtstudio_evict": score(em_on, rows("score_out_evict.jsonl"))["logprob_worst_rel"],
          "evict_on_eq_off_bit": t_on == t_off}
interp["pass"] = (interp["evict_on_eq_off_bit"] and
                  all(interp[m]["logprob_worst_rel"] <= 1e-5 and interp[m]["greedy"] == ok_n and
                      interp[m]["decode_eq_prefill"] for m in ("evict_on", "evict_off")))
report["interpreter"] = interp

# kv rows after the context (prompt + continuation but its last token), against mtstudio's.
calls = [{"seq": r["prompt_ids"] + r["continuation_ids"][:-1], "from": 0} for r in ins]
script = os.path.expanduser("~/ember_wt/state4_rows.json")
json.dump({"calls": calls}, open(script, "w"))
r = subprocess.run([os.path.join(B, "state_tokens_test"), "run", GGUF], stdin=open(script), capture_output=True,
                   text=True, env=env_with())
os.remove(script)
if r.returncode not in (0, 1):
    raise SystemExit("state_tokens_test run: " + r.stderr[-1500:])
c = json.loads(r.stdout)
rows_on = [x["kv_rows_held"]["on"] for x in c["calls"]]
rows_off = [x["kv_rows_held"]["off"] for x in c["calls"]]
report["kv_rows"] = {"evict_equal_mtstudio": rows_on == man["numpy_vs_mtstudio"]["kv_rows_evict"],
                     "keep_equal_mtstudio": rows_off == man["numpy_vs_mtstudio"]["kv_rows_keep"],
                     "on_off_fresh_bit_equal": c["pass"], "evict_mean": float(np.mean(rows_on)),
                     "keep_mean": float(np.mean(rows_off))}
report["kv_rows"]["pass"] = all(v for k, v in report["kv_rows"].items() if not k.endswith("mean"))

# The template rule against paperkiln's serialiser, on random sequences (vocab 10, k 3).
rng = np.random.default_rng(24)
fz = []
for _ in range(3000):
    n = int(rng.integers(0, 24))
    alphabet = [0, 1, 2, 3, 1, 2, 3, 4, 5, 6] + ([10, 11, 12] if rng.random() < 0.4 else [])
    s = [int(rng.choice(alphabet)) for _ in range(n)]
    if rng.random() < 0.3:  # well-formed blocks after some <|endoftext|>s
        t = []
        for x in s:
            t.append(x)
            if x == 1 and rng.random() < 0.5:
                t += [10, 11, 12]
        s = t
    fz.append(s)
r = subprocess.run([os.path.join(B, "state_tokens_test"), "insert"], capture_output=True, text=True,
                   input=json.dumps({"vocab": 10, "k": 3, "eot": 1, "user": 2, "assistant": 3, "seqs": fz}))
mine = json.loads(r.stdout)
agree, accepted, with_state = 0, 0, 0
for s, m in zip(fz, mine):
    err = pk_state.error(s, 10, 3, 1, 2, 3)
    want = pk_state.insert([x for x in s if x < 10], 10, 3, 1, 2, 3)[0] if not err else None
    with_state += any(x >= 10 for x in s)
    if err:
        agree += "refused" in m
    else:
        accepted += 1
        agree += m.get("out") == want
report["template_vs_serialiser"] = {"sequences": len(fz), "with_state_ids": with_state, "accepted": accepted,
                                    "agree": agree, "pass": agree == len(fz)}

# /chat end to end.
prompts = rows("prompts_text.jsonl")


def parse(text):
    parts = [p for p in re.split(r"(<\|user\|>|<\|assistant\|>|<\|endoftext\|>)", text) if p != ""]
    hist, user, reply, cur = [], None, None, None
    for p in parts:
        if p == "<|user|>":
            if user is not None and reply is not None:
                hist.append({"user": user, "assistant": reply})
            user, reply, cur = "", None, "u"
        elif p == "<|assistant|>":
            reply, cur = "", "a"
        elif p == "<|endoftext|>":
            cur = None
        elif cur == "u":
            user += p
        elif cur == "a":
            reply += p
    assert reply == "", text  # the prompt ends at <|assistant|>
    return hist, user


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve(extra):
    port = free_port()
    log = open(os.path.expanduser("~/ember_wt/state4_server.log"), "w")
    srv = subprocess.Popen([os.path.join(B, "tinyllama_server"), GGUF, str(port), "127.0.0.1", os.path.join(EMBER, "www"), "0"],
                           stdout=log, stderr=subprocess.STDOUT, env=env_with(**extra))
    try:
        for _ in range(300):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.3)
        res = []
        for p, a in zip(prompts, ins):
            hist, user = parse(p["prompt"])
            mx = min(len(a["continuation_ids"]) + 2, 128 - len(a["prompt_ids"]))
            body = {"user_input": user, "history": hist, "max_new_tokens": mx, "temperature": 1.0, "top_k": 1,
                    "top_p": 1.0, "seed": 0, "return_ids": True}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            res.append(json.load(urllib.request.urlopen(req, timeout=300)))
        return res
    finally:
        srv.terminate()
        srv.wait()


chat = {}
runs = {"evict_on": serve({}), "evict_off": serve({"EMBER_EVICT": "0"}), "no_prefix_reuse": serve({"EMBER_PREFIX_REUSE": "0"})}
for name, res in runs.items():
    pid = sum(x["prompt_ids"] == a["prompt_ids"] for x, a in zip(res, ins))
    rep = sum(x["ids"] + ([x["stop_id"]] if x["stop_id"] >= 0 else []) == a["continuation_ids"] for x, a in zip(res, ins))
    ser = sum(pk_state.insert([t for t in x["prompt_ids"] if t < V], V, K, EOT, USER, ASST)[0] == x["prompt_ids"]
              for x in res)
    chat[name] = {"prompt_ids_equal_score_in": f"{pid}/{len(ins)}", "reply_and_stop_equal_continuation": f"{rep}/{len(ins)}",
                  "prompt_equal_serialiser_insert": f"{ser}/{len(ins)}",
                  "prefix_reused_tokens": [x["ttft"]["prefix_reused_tokens"] for x in res]}
chat["truncated_prompt_7_ids_equal"] = runs["evict_on"][7]["prompt_ids"] == ins[7]["prompt_ids"]
chat["identical_across_runs"] = all(json.dumps([x["ids"] for x in runs["evict_on"]]) == json.dumps([x["ids"] for x in res])
                                    for res in runs.values())
chat["pass"] = (chat["identical_across_runs"] and chat["truncated_prompt_7_ids_equal"] and
                all(chat[n][k] == ok_n for n in runs for k in ("prompt_ids_equal_score_in",
                                                               "reply_and_stop_equal_continuation",
                                                               "prompt_equal_serialiser_insert")))
report["chat"] = chat

_, em_q = ir_logits(os.path.join(D, "state4_q8_0.gguf"))
report["q8_0"] = score(em_q, outs)

report["pass"] = all(report[k]["pass"] for k in ("interpreter", "kv_rows", "template_vs_serialiser", "chat"))
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(HERE, "state4_gate_cpu.json"), "w"), indent=1)
sys.exit(0 if report["pass"] else 1)
