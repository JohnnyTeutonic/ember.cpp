#!/usr/bin/env python3
"""Self-test of ember's IR conformance adapter (tools/ir_adapter.cpp; paperkiln spec/IR_SPEC.md section 11.3).

Drives ~/ember_wt/build_cpu/ir_adapter over JSON Lines, as tools/ir_conformance.py would, on paperkiln's gate
models copied to ext4 (copy_models.sh: ~/ember_wt/adapter_models). Checks:
  prefill    one forward of each sequence against token-by-token decode and against random chunks: logits,
             kv rows, recurrent tensors and depths bit-equal;
  truncate   forward, truncate to a cut, forward a continuation: equal to a fresh run fed the same chunks (bits)
             and to a one-call prefill (ROUNDING, reported); kv rows and recurrent state after the truncate equal
             to a fresh run over the prefix; a recurrent truncate past the 64 kept snapshots re-prefills;
  snapshot   snapshot, forward, restore (also after a truncate below the snapshot), forward again: the reports
             at restore equal those at the snapshot, and the second forward equals the first, bit for bit;
  evict      state4 turn by turn with eviction as declared and with evict false: logits bit-equal, rows held
             fewer under eviction, rows held equal to ember's (lazy) rule and, after every call and every decoded
             token, one of section 11.3's allowed position sets (eager, or lazy while no text position follows
             the last complete block); token-by-token decode under eviction; rewinds into evicted turns 0 and 1 re-prefill and
             equal a fresh run, rows held as section 6.9 says; evict true accepted on state4, an error on ar3;
  mor        depths equal mtstudio's trace (score_out.jsonl) in prefill and decode; kv rows per application
             #{t : step_i = -1 or depth_t > step_i} at those tokens' positions;
  refuse     check accepts every gate model and refuses broken documents (accepted false); depth on a model
             without depth_route and run_node skip; keys outside the schema's closed sets (IR_SPEC 10.1 rules
             1, 4) refused, naming the key; "_" keys and tensor_index under weight_index accepted.
Writes adapter_selftest_cpu.json beside this file; exits 1 on any failure."""
import copy
import json
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# IR_ADAPTER_BIN tests another build (a pinned one); its report then goes to the scratch directory, not here.
BIN = os.environ.get("IR_ADAPTER_BIN", os.path.expanduser("~/ember_wt/build_cpu/ir_adapter"))
M = os.path.expanduser("~/ember_wt/adapter_models")
SCRATCH = os.path.expanduser("~/ember_wt/adapter_scratch")
os.makedirs(SCRATCH, exist_ok=True)


class Adapter:
    def __init__(self):
        self.log = open(os.path.join(SCRATCH, "adapter_stderr.log"), "w")
        self.p = subprocess.Popen([BIN, "--conformance"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, bufsize=1)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        assert r["id"] == self.n, r
        return r

    def ok(self, method, **params):
        r = self.call(method, **params)
        if "result" not in r:
            raise RuntimeError(f"{method} {params.get('session', '')}: {r}")
        return r["result"]

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def f32(t):
    b = bytes.fromhex(t["f32"])
    a = np.frombuffer(b, dtype=">u4").astype(np.uint32).view(np.float32)
    return a.reshape(t["shape"])


def rec_arrays(rep):
    return {s: [{k: f32(v) for k, v in app.items()} for app in apps] for s, apps in rep.items()}


def rec_equal(a, b):
    return all(np.array_equal(x[k].view(np.uint32), y[k].view(np.uint32))
               for s in a for x, y in zip(a[s], b[s]) for k in x) and a.keys() == b.keys()


def bits_equal(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def rel(a, b):
    if a.size == 0:
        return 0.0
    return float(np.abs(a.astype(np.float64) - b).max() / max(1.0, np.abs(b).max()))


def rows(path, n):
    out = []
    for x in open(path):
        if x.strip():
            r = json.loads(x)
            out.append(r["prompt_ids"] + r["continuation_ids"][:-1])
        if len(out) == n:
            break
    return out


rng = random.Random(25)
kimi_long = [rng.randrange(5, 8000) for _ in range(100)]
short = [[5, 17, 300, 42, 9, 77, 128, 3], [11, 2, 250, 99, 1, 7, 4000, 12]]
MODELS = {
    "state4": (f"{M}/state4/state4.gguf", rows(f"{M}/state4/score_in.jsonl", 3)),
    "ar3": (f"{M}/ar3/ar3.gguf", rows(f"{M}/ar3/score_in.jsonl", 3)),
    "b2x4": (f"{M}/b2x4/r.gguf", rows(f"{M}/b2x4/score_in.jsonl", 3)),
    "mor3": (f"{M}/mor3/mor3.gguf", rows(f"{M}/mor3/score_in.jsonl", 3)),
    "mor2p0": (f"{M}/mor2p0/mor2p0.gguf", rows(f"{M}/mor2p0/score_in.jsonl", 3)),
    "kimi_tiny": (f"{M}/kimi_tiny/kimi_tiny.gguf", short + [kimi_long]),
    "mamba_tiny": (f"{M}/mamba_tiny/mamba_tiny.gguf", short),
    "swa_tiny": (f"{M}/swa_tiny/swa_tiny.gguf", short + [[rng.randrange(5, 8000) for _ in range(60)]]),
}
MOR = {"mor3", "mor2p0"}
REC = {"kimi_tiny", "mamba_tiny"}

ad = Adapter()
report = {"bin": BIN, "hello": ad.ok("hello"), "checks": {}}
failures = []


def record(name, ok, **info):
    report["checks"][name] = dict(info)
    report["checks"][name]["pass"] = bool(ok)
    if not ok:
        failures.append(name)


def reps(model):
    r = ["logits", "kv_rows", "kv_positions"]
    if model in MOR:
        r.append("depth")
    if model in REC:
        r.append("recurrent")
    return r


def run(gguf, chunks, model, evict=None):
    s = ad.ok("open", gguf=gguf, ir=None, evict=evict)["session"]
    outs = [ad.ok("forward", session=s, ids=c, report=reps(model)) for c in chunks]
    return s, outs


def stack(outs):
    return np.concatenate([f32(o["logits"]) for o in outs])


def depths(outs):
    return [d for o in outs for d in o.get("depth", [])]


# ---- prefill vs decode vs chunks
for model, (gguf, seqs) in MODELS.items():
    for k, seq in enumerate(seqs):
        s1, pre = run(gguf, [seq], model)
        s2, dec = run(gguf, [[t] for t in seq], model)
        cuts, i = [], 0
        while i < len(seq):
            j = min(len(seq), i + rng.randint(1, 5))
            cuts.append(seq[i:j])
            i = j
        s3, chk = run(gguf, cuts, model)
        a, b, c = stack(pre), stack(dec), stack(chk)
        last = [pre[-1], dec[-1], chk[-1]]
        same_rows = last[0]["kv_positions"] == last[1]["kv_positions"] == last[2]["kv_positions"]
        same_rec = model not in REC or (rec_equal(rec_arrays(last[0]["recurrent"]), rec_arrays(last[1]["recurrent"])) and
                                        rec_equal(rec_arrays(last[0]["recurrent"]), rec_arrays(last[2]["recurrent"])))
        same_depth = model not in MOR or depths(pre) == depths(dec) == depths(chk)
        record(f"prefill/{model}/{k}", bits_equal(a, b) and bits_equal(a, c) and same_rows and same_rec and same_depth,
               positions=len(seq), decode_bits=bits_equal(a, b), chunks_bits=bits_equal(a, c),
               decode_rel=rel(b, a), chunks_rel=rel(c, a), kv_rows=last[0]["kv_rows"], kv_rows_equal=same_rows,
               recurrent_bits=same_rec, depth_equal=same_depth)
        for s in (s1, s2, s3):
            ad.ok("close", session=s)

# ---- truncate then re-forward
for model, (gguf, seqs) in MODELS.items():
    for k, seq in enumerate(seqs):
        if len(seq) < 6:
            continue
        cut = len(seq) // 2
        if model == "state4":
            alt = seq[cut:]  # the block structure stays valid
        else:
            alt = list(reversed(seq[cut:]))
        s, _ = run(gguf, [seq], model)
        tr = ad.ok("truncate", session=s, n=cut, report=[r for r in reps(model) if r not in ("logits", "depth")])
        after = ad.ok("forward", session=s, ids=alt, report=reps(model))
        f, fo = run(gguf, [seq[:cut], alt], model)
        one, oo = run(gguf, [seq[:cut] + alt], model)
        fresh_cut = fo[0]
        a, b = f32(after["logits"]), f32(fo[1]["logits"])
        c = f32(oo[0]["logits"])[cut:]
        rows_ok = tr["kv_positions"] == fresh_cut["kv_positions"] and after["kv_positions"] == fo[1]["kv_positions"]
        rec_ok = model not in REC or (rec_equal(rec_arrays(tr["recurrent"]), rec_arrays(fresh_cut["recurrent"])) and
                                      rec_equal(rec_arrays(after["recurrent"]), rec_arrays(fo[1]["recurrent"])))
        dep_ok = model not in MOR or after["depth"] == fo[1]["depth"] == oo[0]["depth"][cut:]
        record(f"truncate/{model}/{k}", tr["len"] == cut and bits_equal(a, b) and rows_ok and rec_ok and dep_ok,
               cut=cut, len=len(seq), fresh_bits=bits_equal(a, b), one_call_rel=rel(a, c),
               kv_rows_after_truncate=tr["kv_rows"], kv_rows_equal_fresh=rows_ok, recurrent_bits=rec_ok,
               depth_equal=dep_ok)
        for x in (s, f, one):
            ad.ok("close", session=x)

# a recurrent truncate past the kept snapshots (at most 64): the adapter re-prefills from resume_point
gguf = MODELS["kimi_tiny"][0]
s, _ = run(gguf, [kimi_long], "kimi_tiny")
tr = ad.ok("truncate", session=s, n=10, report=["recurrent"])
f, fo = run(gguf, [kimi_long[:10]], "kimi_tiny")
after = ad.ok("forward", session=s, ids=kimi_long[10:20], report=["logits", "recurrent"])
f2 = ad.ok("forward", session=f, ids=kimi_long[10:20], report=["logits", "recurrent"])
record("truncate/kimi_tiny/past_snapshots",
       rec_equal(rec_arrays(tr["recurrent"]), rec_arrays(fo[0]["recurrent"])) and
       bits_equal(f32(after["logits"]), f32(f2["logits"])) and
       rec_equal(rec_arrays(after["recurrent"]), rec_arrays(f2["recurrent"])), cut=10, len=len(kimi_long))
ad.ok("close", session=s)
ad.ok("close", session=f)

# ---- snapshot / restore
for model, (gguf, seqs) in MODELS.items():
    seq = max(seqs, key=len)
    a = len(seq) // 3
    rq = [r for r in reps(model) if r not in ("logits", "depth")]
    s, first = run(gguf, [seq[:a]], model)
    snap = ad.ok("snapshot", session=s, tag="t")
    x1 = ad.ok("forward", session=s, ids=seq[a:], report=reps(model))
    back = ad.ok("restore", session=s, tag="t", report=rq)
    x2 = ad.ok("forward", session=s, ids=seq[a:], report=reps(model))
    ad.ok("truncate", session=s, n=1, report=[])
    back2 = ad.ok("restore", session=s, tag="t", report=rq)
    x3 = ad.ok("forward", session=s, ids=seq[a:], report=reps(model))
    same_back = all(back[r] == first[0][r] for r in rq if r != "recurrent") and \
        all(back2[r] == first[0][r] for r in rq if r != "recurrent")
    if model in REC:
        same_back = same_back and rec_equal(rec_arrays(back["recurrent"]), rec_arrays(first[0]["recurrent"])) and \
            rec_equal(rec_arrays(back2["recurrent"]), rec_arrays(first[0]["recurrent"]))
    same_fwd = bits_equal(f32(x1["logits"]), f32(x2["logits"])) and bits_equal(f32(x1["logits"]), f32(x3["logits"])) and \
        x1["kv_positions"] == x2["kv_positions"] == x3["kv_positions"]
    record(f"snapshot/{model}", snap["len"] == a and back["len"] == a and back2["len"] == a and same_back and same_fwd,
           at=a, len=len(seq), restore_reports_equal=same_back, forward_bits=same_fwd)
    ad.ok("close", session=s)

# ---- state4 eviction
VOCAB, K = 400, 4


def codes(ids):
    """c(p), s(p) of section 6.8.2, and m after each prefix (complete blocks in 0 .. L-1)."""
    c, s, m, seg, run_ = [], [], [], 0, 0
    for i in ids:
        if i >= VOCAB:
            c.append(1)
            s.append(seg)
            run_ = run_ + 1 if i - VOCAB == run_ else (1 if i == VOCAB else 0)
            if run_ == K:
                seg, run_ = seg + 1, 0
        else:
            c.append(0)
            s.append(seg)
            run_ = 0
        m.append(seg)
    return c, s, m


def allowed_positions(ids):
    """Section 11.3 / 6.8.3: the eager position set (state rows + segment m's text) and, while no text position
    follows q (the last position of block m - 1), the set that still holds segment m - 1's text."""
    c, s, m = codes(ids)
    mm = m[-1]
    eager = [p for p in range(len(ids)) if c[p] == 1 or s[p] == mm]
    out = [eager]
    if mm > 0:
        q = max(p for p in range(len(ids)) if c[p] == 1 and s[p] == mm - 1)
        if not any(c[p] == 0 for p in range(q + 1, len(ids))):
            out.append([p for p in range(len(ids)) if c[p] == 1 or s[p] >= mm - 1])
    return out


def lazy_rows(ids):
    c, s, _ = codes(ids)
    last = s[-1]
    return sum(c) + sum(1 for p in range(len(ids)) if c[p] == 0 and s[p] == last)


def turns(seq):
    starts = [i for i, t in enumerate(seq) if t == 2] + [len(seq)]
    return [seq[a:b] for a, b in zip(starts, starts[1:]) if b > a]


gguf = MODELS["state4"][0]
for k, seq in enumerate(MODELS["state4"][1]):
    chunks = turns(seq)
    se, ev = run(gguf, chunks, "state4", None)
    sn, ne = run(gguf, chunks, "state4", False)
    so, oe = run(gguf, [seq], "state4", None)
    sd, de = run(gguf, [[t] for t in seq], "state4", None)
    logits_ok = all(bits_equal(f32(a["logits"]), f32(b["logits"])) for a, b in zip(ev, ne)) and \
        bits_equal(stack(ev), stack(oe)) and bits_equal(stack(ev), stack(de))
    held_e = [o["kv_rows"]["kv"][0] for o in ev]
    held_n = [o["kv_rows"]["kv"][0] for o in ne]
    ends = np.cumsum([len(c) for c in chunks])
    lazy = [lazy_rows(seq[:L]) for L in ends]
    allowed = [allowed_positions(seq[:L]) for L in ends]
    legal = [o["kv_positions"]["kv"][0] in a and o["kv_positions"]["kv"][1] in a for o, a in zip(ev, allowed)]
    rows_ok = held_e == lazy and all(legal) and held_n == list(map(int, ends)) and held_e[-1] < held_n[-1]
    record(f"evict/state4/{k}/turns", logits_ok and rows_ok, calls=len(chunks), held_evict=held_e, held_keep=held_n,
           ember_lazy_rule=lazy, allowed_11_3=[[len(x) for x in a] for a in allowed], legal_11_3=legal,
           eager_count_calls=sum(h == len(a[0]) for h, a in zip(held_e, allowed)),
           decode_rows_evict=de[-1]["kv_rows"]["kv"][0], prefill_rows_evict=oe[-1]["kv_rows"]["kv"][0])
    # decode token by token: after every token, the rows held are a legal set of section 11.3
    sd2 = ad.ok("open", gguf=gguf, ir=None, evict=None)["session"]
    dec_legal = 0
    for i, t in enumerate(seq):
        o = ad.ok("forward", session=sd2, ids=[t], report=["kv_positions"])
        dec_legal += o["kv_positions"]["kv"][0] in allowed_positions(seq[:i + 1])
    record(f"evict/state4/{k}/decode_legal_rows", dec_legal == len(seq), positions=len(seq), legal=dec_legal)
    ad.ok("close", session=sd2)
    # rewinds into evicted turns: n inside turn 0's text, and inside turn 1's text
    _, s_, _ = codes(seq)
    t1 = len(chunks[0])  # start(1): the position after block 0
    for name, n in (("turn0", 3), ("turn1", t1 + 3)):
        tr_e = ad.ok("truncate", session=se, n=n, report=["kv_rows", "kv_positions"])
        tr_n = ad.ok("truncate", session=sn, n=n, report=["kv_rows", "kv_positions"])
        fe = ad.ok("forward", session=se, ids=seq[n:], report=["logits", "kv_rows", "kv_positions"])
        fn = ad.ok("forward", session=sn, ids=seq[n:], report=["logits", "kv_rows", "kv_positions"])
        sf, fo = run(gguf, [seq[:n], seq[n:]], "state4", None)
        c, s, m = codes(seq[:n])
        mm = m[-1]
        # 6.9: the rows a never-evicting run holds (0 .. n-1) less the dropped text rows of segments < m
        want = [p for p in range(n) if not (c[p] == 0 and s[p] < mm)]
        ok = tr_e["kv_positions"]["kv"][0] == want and tr_n["kv_positions"]["kv"][0] == list(range(n)) and \
            bits_equal(f32(fe["logits"]), f32(fn["logits"])) and bits_equal(f32(fe["logits"]), f32(fo[1]["logits"])) and \
            fe["kv_positions"] == fo[1]["kv_positions"]
        record(f"evict/state4/{k}/rewind_{name}", ok, n=n, segment=mm, rows_after_truncate=tr_e["kv_rows"]["kv"][0],
               rows_expected=len(want), keep_rows_after_truncate=tr_n["kv_rows"]["kv"][0])
        ad.ok("close", session=sf)
        # back to the full run for the next rewind
        ad.ok("truncate", session=se, n=0, report=[])
        ad.ok("truncate", session=sn, n=0, report=[])
        for ch in chunks:
            ad.ok("forward", session=se, ids=ch, report=[])
            ad.ok("forward", session=sn, ids=ch, report=[])
    for x in (se, sn, so, sd):
        ad.ok("close", session=x)

r = ad.call("open", gguf=gguf, ir=None, evict=True)
record("evict/true_on_declared", "result" in r, reply=r.get("result"))
if "result" in r:
    ad.ok("close", session=r["result"]["session"])
r = ad.call("open", gguf=MODELS["ar3"][0], ir=None, evict=True)
record("evict/true_on_undeclared_is_error", "error" in r, reply=r)

# ---- MoR depths and rows per application against mtstudio's trace
for model in sorted(MOR):
    gguf, _ = MODELS[model]
    ir = json.load(open(f"{M}/{model}/{model}.ir.json"))
    step = ir["block"]["nodes"][0]["step"]["per_layer"]
    ins = [json.loads(x) for x in open(f"{M}/{model}/score_in.jsonl") if x.strip()][:10]
    outs = [json.loads(x) for x in open(f"{M}/{model}/score_out.jsonl") if x.strip()][:10]
    good = 0
    for a, b in zip(ins, outs):
        seq = a["prompt_ids"] + a["continuation_ids"][:-1]
        s1, pre = run(gguf, [seq], model)
        s2, dec = run(gguf, [[t] for t in seq], model)
        d = pre[0]["depth"]
        want_rows = [[p for p in range(len(seq)) if st == -1 or d[p] > st] for st in step]
        good += d == b["depth"] and depths(dec) == b["depth"] and pre[0]["kv_positions"]["kv"] == want_rows and \
            dec[-1]["kv_positions"]["kv"] == want_rows
        ad.ok("close", session=s1)
        ad.ok("close", session=s2)
    record(f"mor/{model}", good == len(ins), prompts=len(ins), equal=good, step=step)

# ---- check: acceptance and refusals
for model, (gguf, _) in MODELS.items():
    r = ad.ok("check", gguf=gguf, ir=None)
    record(f"check/accept/{model}", r["accepted"], reply=r)


def variant(base, name, edit):
    d = copy.deepcopy(json.load(open(base)))
    edit(d)
    p = os.path.join(SCRATCH, f"adapter_{name}.ir.json")
    json.dump(d, open(p, "w"))
    return p


def kv(d):
    return [s for s in d["block"]["state"] if s["kind"] == "kv"][0]


def set_(obj, key, value):
    obj[key] = value


S4 = (MODELS["state4"][0], f"{M}/state4/state4.ir.json")
MR = (MODELS["mor3"][0], f"{M}/mor3/mor3.ir.json")
AR = (MODELS["ar3"][0], f"{M}/ar3/ar3.ir.json")
refusals = {
    "ir_version_2": (S4, lambda d: set_(d, "ir_version", 2)),
    "evict_lru": (S4, lambda d: set_(kv(d), "evict", "lru")),
    "kv_from": (MR, lambda d: set_(kv(d), "kv_from", {"per_layer": list(range(8))})),
    "route_expert_choice": (MR, lambda d: set_(d["block"]["nodes"][0], "route", "expert_choice")),
    "depth_route_without_kv_rows": (MR, lambda d: kv(d).pop("kv_rows")),
    "mask_bidirectional": (S4, lambda d: [set_(n, "mask", "bidirectional") for n in d["block"]["nodes"]
                                          if n["op"] == "attention"]),
}
for name, ((gguf, base), edit) in refusals.items():
    r = ad.ok("check", gguf=gguf, ir=variant(base, name, edit))
    record(f"check/refuse/{name}", not r["accepted"], message=r.get("message"))
for name, (gguf, base) in (("state4_side_file", S4), ("mor3_side_file", MR)):
    r = ad.ok("check", gguf=gguf, ir=base)
    record(f"check/accept/{name}", r["accepted"], reply=r)
r = ad.ok("check", gguf="/nonexistent/model.gguf", ir=None)
record("check/refuse/missing_gguf", not r["accepted"], message=r.get("message"))
# Closed sets (IR_SPEC 10.1 rules 1 and 4) refused at load, the message naming the key; tensor_index outside a
# weight_index block refused (rule 19), and accepted in one (section 7.2).
B2 = (MODELS["b2x4"][0], f"{M}/b2x4/r.ir.json")


def attn(d):
    return [n for n in d["block"]["nodes"] if n["op"] == "attention"][0]


def tensor_index_valid(d):
    # A node with a tensor name without {i} under weight_index: the document stays valid (every name binds).
    n = d["block"]["nodes"][0]
    n["weight"] = n["weight"].replace("{i}", "0")
    n["tensor_index"] = "application"


closed = {
    "unknown_attribute": (S4, lambda d: set_(attn(d), "flavour", "vanilla"), "'flavour'"),
    "unknown_top_level_key": (S4, lambda d: set_(d, "extras", {}), "'extras'"),
    "unknown_model_key": (S4, lambda d: set_(d["model"], "hidden", 64), "'hidden'"),
    "unknown_slot_key": (S4, lambda d: set_(kv(d), "pages", 4), "'pages'"),
    "unknown_block_key": (S4, lambda d: set_(d["block"], "loop", 2), "'loop'"),
    "attribute_of_another_op": (S4, lambda d: set_(d["block"]["nodes"][0], "theta", 10000.0), "'theta'"),
    "required_attribute_missing": (S4, lambda d: [n.pop("theta") for n in d["block"]["nodes"] if n["op"] == "rope"],
                                   "'theta'"),
    "tensor_index_without_weight_index": (AR, lambda d: set_(d["block"]["nodes"][0], "tensor_index", "application"),
                                          "tensor_index"),
}
# The load-time refusals of IR_SPEC 10.1 that ember lacked before (each previously loaded, or failed only at the
# first forward); the container rules on edited GGUFs are in ../ir_spec_2026-10-25/agreement.py.
closed.update({
    "rule3_n_layers_ne_repeat": (S4, lambda d: set_(d["model"], "n_layers", 3), "IR_SPEC 3.3"),
    "rule11_slot_read_by_no_node": (AR, lambda d: d["block"]["state"].append(dict(kv(d), name="kv2")), "IR_SPEC 6.1"),
    "rule11_attention_names_no_slot": (AR, lambda d: set_(attn(d), "state", "nope"), "IR_SPEC 6.1"),
    "rule12_window_on_unwindowed_slot": (AR, lambda d: set_(attn(d), "window", 4), "IR_SPEC 5.15"),
    "rule12_kv_heads_differs_at_load": (AR, lambda d: set_(attn(d), "kv_heads", 1), "IR_SPEC 5.15"),
    "rule15_turn_state_without_state_weight": (AR, lambda d: set_(attn(d), "mask", "turn_state"), "IR_SPEC 6.8.2"),
    "rule17a_two_attention_nodes_one_slot": (AR, lambda d: d["block"]["nodes"].insert(
        d["block"]["nodes"].index(attn(d)) + 1, dict(attn(d), out="a2")), "IR_SPEC 6.1"),
    "rule17a_reads_tokens": (S4, lambda d: set_([n for n in d["block"]["nodes"] if n["op"] == "silu"][0], "in",
                                                ["tokens"]), "IR_SPEC 3.5"),
    "rule17a_reads_depth_route_out": (MR, lambda d: d["block"]["nodes"].append(
        {"op": "silu", "in": [d["block"]["nodes"][0]["out"]], "out": "junk"}), "IR_SPEC 7.4"),
    "rule23_depth_route_with_lists": (MR, lambda d: d["block"]["nodes"].append({"op": "list", "in": ["x"], "out": "b"}),
                                      "IR_SPEC 7.4"),
    "rule26_router_ne_at_load": (MR, lambda d: set_(d["block"]["nodes"][0], "weight", "blk.0.attn_norm.weight"),
                                 "IR_SPEC 7.4"),
    "rule28_specials_name_other_ids": (S4, lambda d: set_(d["tokenizer"]["specials"], "eot", 5), "IR_SPEC 8.2"),
    "two_kv_slots_capability": (AR, lambda d: (d["block"]["state"].append(dict(kv(d), name="kv2")),
                                               d["block"]["nodes"].insert(d["block"]["nodes"].index(attn(d)) + 1,
                                                                          dict(attn(d), state="kv2", out="a2"))),
                                "more than one kv slot"),
})
for name, ((gguf, base), edit, want) in closed.items():
    r = ad.ok("check", gguf=gguf, ir=variant(base, name, edit))
    record(f"check/refuse/{name}", not r["accepted"] and want in r.get("message", ""), message=r.get("message"))
# depth_attend's S is structural (IR_SPEC 5.18): ar3 without its S = 1 pseudo-query (blk.0.attnres_attn_q, never
# read) loads; without an S = 2 one (blk.1.attnres_attn_q) it is refused at load, not at the first forward.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ir_spec_2026-10-25"))
from agreement import copy_gguf  # noqa: E402
for name, drop, accept in (("s1_pseudo_query_absent", "blk.0.attnres_attn_q.weight", True),
                           ("s2_pseudo_query_absent", "blk.1.attnres_attn_q.weight", False)):
    g = copy_gguf(AR[0], os.path.join(SCRATCH, f"adapter_{name}.gguf"), drop_tensors=(drop,))
    r = ad.ok("check", gguf=g, ir=None)
    ok = r["accepted"] if accept else (not r["accepted"] and "5.18" in r.get("message", ""))
    record(f"check/{'accept' if accept else 'refuse'}/{name}", ok, reply=r)
    if accept:  # and it runs: the logits equal ar3's own
        s1, o1 = run(g, [MODELS["ar3"][1][0]], "ar3")
        s2, o2 = run(AR[0], [MODELS["ar3"][1][0]], "ar3")
        record(f"forward/{name}", bits_equal(stack(o1), stack(o2)))
        ad.ok("close", session=s1)
        ad.ok("close", session=s2)
r = ad.ok("check", gguf=S4[0], ir=variant(S4[1], "prose_keys", lambda d: [set_(attn(d), "_why", "prose"),
                                                                      set_(d["model"], "_note", "prose")]))
record("check/accept/underscore_keys", r["accepted"], reply=r)
r = ad.ok("check", gguf=B2[0], ir=variant(B2[1], "tensor_index_valid", tensor_index_valid))
record("check/accept/tensor_index_under_weight_index", r["accepted"], reply=r)

s = ad.ok("open", gguf=MODELS["ar3"][0], ir=None, evict=None)["session"]
r = ad.call("forward", session=s, ids=[1, 2], report=["depth"])
r2 = ad.ok("forward", session=s, ids=[1, 2], report=["kv_rows"])
record("skip/depth_without_depth_route", "skip" in r and r2["len"] == 2, reply=r)
r = ad.call("truncate", session=s, n=1, report=["logits"])
record("error/logits_on_truncate", "error" in r, reply=r)
r = ad.call("run_node", node={}, inputs={}, weights={}, state=None)
record("skip/run_node", "skip" in r, reply=r)
ad.ok("close", session=s)
# Token ids out of range are refused at run time (IR_SPEC 5.1, 6.8.1): >= vocab without a state_weight, >= vocab + k
# with one, negative always.
for model in ("ar3", "state4"):
    o = ad.ok("open", gguf=MODELS[model][0], ir=None, evict=None)
    bad = [o["vocab"] + o["state_k"], -1]
    errs = [ad.call("forward", session=o["session"], ids=[5, b], report=[]) for b in bad]
    record(f"error/token_id_out_of_range/{model}", all("error" in e for e in errs),
           ids=bad, replies=[e.get("error") for e in errs])
    ad.ok("close", session=o["session"])
ad.close()

report["requests"] = ad.n
report["failures"] = failures
report["pass"] = not failures
n = len(report["checks"])
print(f"{n - len(failures)}/{n} checks pass, {ad.n} requests" + ("" if not failures else ": FAIL " + ", ".join(failures)))
out_dir = SCRATCH if "IR_ADAPTER_BIN" in os.environ else HERE
json.dump(report, open(os.path.join(out_dir, "adapter_selftest_cpu.json"), "w"), indent=1)
sys.exit(0 if not failures else 1)
