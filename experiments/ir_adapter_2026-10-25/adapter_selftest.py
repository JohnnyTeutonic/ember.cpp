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
             without depth_route is [] (revision 1.2; revision 1.1's adapter skipped) and run_node skips; keys outside the schema's closed sets (IR_SPEC 10.1 rules
             1, 4) refused, naming the key; "_" keys and tensor_index under weight_index accepted.
  rev11      spec revision 1.1 (appended after the checks above, which it leaves unchanged): the suite's
             recurrence models (pk_recurrence, and pk_kimi, pk_srd, pk_mamba, tp_mamba1 with their documents
             rewritten as recurrences) prefill = token-by-token decode = random chunks, bits (logits and recurrent
             tensors), truncate = a fresh run, snapshot / restore; each rewrite against the file's hand-written
             ops (ROUNDING, bits reported); an init body (s = B - B equals +0 bit for bit, s = B changes the
             logits, decodes = prefill and is re-applied by truncate 0); rule 30 (paperkiln.ir_sha256 equal to the
             embedded document's digest, refused when wrong or not a string, also under a substituted document;
             a file without the key loads); the digest vectors and dequantisation blocks; run_node on a
             recurrence (rule 34 on a wrong width), run_node_grad and other ops skip.
  rev12      spec revision 1.2 (keramos v1.2.0; IR12_SUITE, the suite at the tag): rule 35 (a member name twice,
             at any depth and in prose, in a document read in place of the file's, in the file's own (which loads
             with a valid document in its place) and in a document checked alone), rule 36 (the suite's four refusals and two edits; a head reading a block
             value is accepted without depth_route), rule 20's run of steps, rule 37 when the rows arrive (the error
             names it and leaves the session as a fresh open's), depth_attend without eps = eps 1e-5 bit for bit,
             check with gguf null on the suite's documents alone, section 11.3's edge cases, render_chat on the
             suite's 32 chat cases, run_node on the suite's expr cases (bits), depth 204 and two shapes refused.
  rev13      spec revision 1.3 (keramos v1.3.0; IR13_SUITE): rule 38 (kimi and swa labelled "gpt2" refused, also with
             their own document read in place, their relabelled twins load and match the reference; pk_moe,
             pk_llama, pk_gpt2 and tp_gemma3 are their architecture's graph; metadata that disagrees and an
             architecture without a graph refused), the windowed rewind (a cut into what a ring buffer drops keeps
             the rows below the cut and equals a fresh run, bits), position_bias in a whole model (pk_llama with a
             bias table, causal and windowed with a sink: the reference within 1e-5, prefill = decode bits, the bias
             moves the logits) and attention in run_node (the suite's six cases).
IR_SUITE is the suite the rev11 checks read (revision 1.1's receipt: a keramos v1.1.0 archive; since revision 1.3 an
archive of v1.3.0, whose relabelled pk_kimi and pk_recurrence load under rule 38). IR12_SUITE and IR13_SUITE likewise.
Writes adapter_selftest_cpu.json beside this file (or to ADAPTER_SELFTEST_OUT); exits 1 on any failure.
ADAPTER_SCRATCH moves the scratch directory (default ~/ember_wt/adapter_scratch)."""
import copy
import json
import os
import random
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# IR_ADAPTER_BIN tests another build (a pinned one); its report then goes to the scratch directory, not here.
BIN = os.environ.get("IR_ADAPTER_BIN", os.path.expanduser("~/ember_wt/build_cpu/ir_adapter"))
# IR_ADAPTER_MODELS: the gate models' directory (default copy_models.sh's); since keramos v1.3.0 (rule 38) kimi_tiny and
# swa_tiny, which carry general.architecture "gpt2", load only relabelled (../ir_rev13_2026-10-27/relabel_models.sh).
M = os.path.expanduser(os.environ.get("IR_ADAPTER_MODELS", "~/ember_wt/adapter_models"))
SCRATCH = os.environ.get("ADAPTER_SCRATCH", os.path.expanduser("~/ember_wt/adapter_scratch"))
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
# Revision 1.2 (IR_SPEC 11.3): depth is [] for a document without depth_route (revision 1.1's adapter skipped).
record("depth/without_depth_route_empty", r.get("result", {}).get("depth") == [] and r2["len"] == 4, reply=r)
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

from agreement import embedded  # noqa: E402
# ---- spec revision 1.1 (paperkiln 7df7b10b): the recurrence op, the digest (rule 30), dequantisation, run_node.
# Appended after every earlier check, so their requests, random draws and results are unchanged.
SUITE = os.environ.get("IR_SUITE", "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/spec/conformance")
rng11 = random.Random(1101)
REC11 = {  # model: (gguf, document run in place of the file's or None, the original document's gguf for comparison)
    "pk_recurrence": (f"{SUITE}/models/pk_recurrence/pk_recurrence.gguf", None),
    "pk_kimi_rec": (f"{SUITE}/models/pk_kimi/pk_kimi.gguf", f"{SUITE}/recurrence/pk_kimi.ir.json"),
    "pk_srd_rec": (f"{SUITE}/models/pk_srd/pk_srd.gguf", f"{SUITE}/recurrence/pk_srd.ir.json"),
    "pk_mamba_rec": (f"{SUITE}/models/pk_mamba/pk_mamba.gguf", f"{SUITE}/recurrence/pk_mamba.ir.json"),
    "tp_mamba1_rec": (f"{SUITE}/models/tp_mamba1/tp_mamba1.gguf", f"{SUITE}/recurrence/tp_mamba1.ir.json"),
}


def run11(gguf, ir, chunks, rep=("logits", "kv_rows", "kv_positions", "recurrent")):
    s = ad.ok("open", gguf=gguf, ir=ir, evict=None)
    outs = [ad.ok("forward", session=s["session"], ids=c, report=list(rep)) for c in chunks]
    return s, outs


for model, (gguf, ir) in REC11.items():
    o = ad.ok("open", gguf=gguf, ir=ir, evict=None)
    V = o["vocab"]
    ad.ok("close", session=o["session"])
    seqs = [[rng11.randrange(1, V) for _ in range(n)] for n in (5, 9, 12)]
    for k, seq in enumerate(seqs):
        s1, pre = run11(gguf, ir, [seq])
        s2, dec = run11(gguf, ir, [[t] for t in seq])
        cuts, i = [], 0
        while i < len(seq):
            j = min(len(seq), i + rng11.randint(1, 4))
            cuts.append(seq[i:j])
            i = j
        s3, chk = run11(gguf, ir, cuts)
        a, b, c = stack(pre), stack(dec), stack(chk)
        ra, rb, rc = (rec_arrays(x[-1]["recurrent"]) for x in (pre, dec, chk))
        ok = bits_equal(a, b) and bits_equal(a, c) and rec_equal(ra, rb) and rec_equal(ra, rc)
        record(f"rev11/prefill/{model}/{k}", ok, positions=len(seq), decode_bits=bits_equal(a, b),
               chunks_bits=bits_equal(a, c), recurrent_bits=rec_equal(ra, rb) and rec_equal(ra, rc))
        for x in (s1, s2, s3):
            ad.ok("close", session=x["session"])
    # truncate and re-forward against a fresh run; snapshot / restore
    seq = seqs[-1]
    cut = len(seq) // 2
    alt = list(reversed(seq[cut:]))
    s, _ = run11(gguf, ir, [seq])
    tr = ad.ok("truncate", session=s["session"], n=cut, report=["recurrent"])
    after = ad.ok("forward", session=s["session"], ids=alt, report=["logits", "recurrent"])
    f, fo = run11(gguf, ir, [seq[:cut], alt])
    ok = rec_equal(rec_arrays(tr["recurrent"]), rec_arrays(fo[0]["recurrent"])) and \
        bits_equal(f32(after["logits"]), f32(fo[1]["logits"])) and \
        rec_equal(rec_arrays(after["recurrent"]), rec_arrays(fo[1]["recurrent"]))
    record(f"rev11/truncate/{model}", ok, cut=cut, len=len(seq))
    snap = ad.ok("snapshot", session=f["session"], tag="t")
    x1 = ad.ok("forward", session=f["session"], ids=seq[:3], report=["logits", "recurrent"])
    back = ad.ok("restore", session=f["session"], tag="t", report=["recurrent"])
    x2 = ad.ok("forward", session=f["session"], ids=seq[:3], report=["logits", "recurrent"])
    ok = back["len"] == snap["len"] and rec_equal(rec_arrays(back["recurrent"]), rec_arrays(fo[1]["recurrent"])) and \
        bits_equal(f32(x1["logits"]), f32(x2["logits"])) and rec_equal(rec_arrays(x1["recurrent"]), rec_arrays(x2["recurrent"]))
    record(f"rev11/snapshot/{model}", ok, at=snap["len"])
    for x in (s, f):
        ad.ok("close", session=x["session"])
    # the rewrite against the file's own document (hand-written ops): ROUNDING, bits reported
    if ir is not None:
        seq = seqs[1]
        s1, a1 = run11(gguf, ir, [seq], rep=("logits", "recurrent"))
        s2, a2 = run11(gguf, None, [seq], rep=("logits", "recurrent"))
        la, lb = stack(a1), stack(a2)
        ra, rb = rec_arrays(a1[0]["recurrent"]), rec_arrays(a2[0]["recurrent"])
        worst_rec = max(rel(ra[sl][i][t], rb[sl][i][t]) for sl in ra for i in range(len(ra[sl])) for t in ra[sl][i])
        record(f"rev11/rewrite_vs_handwritten/{model}", rel(la, lb) <= 1e-5 and worst_rec <= 1e-5,
               logits_bits=bits_equal(la, lb), logits_rel=rel(la, lb), recurrent_bits=rec_equal(ra, rb),
               recurrent_rel=worst_rec)
        for x in (s1, s2):
            ad.ok("close", session=x["session"])

# init bodies (IR_SPEC 6.10.1): pk_mamba's rewrite with s's initial state a body over its weights.
PM = REC11["pk_mamba_rec"]


def with_init(body):
    def edit(d):
        n = [x for x in d["block"]["nodes"] if x["op"] == "recurrence"][0]
        n["init"] = [body]
        d["block"]["state"][0]["tensors"][0]["init"] = "body"
    return edit


zero_init = variant(PM[1], "rev11_init_zero", with_init("s[r] = B[r] - B[r]"))
b_init = variant(PM[1], "rev11_init_b", with_init("s[r] = B[r]"))
seq = [3, 9, 4, 1, 7, 2]
s0, o0 = run11(PM[0], PM[1], [seq], rep=("logits", "recurrent"))
sz, oz = run11(PM[0], zero_init, [seq], rep=("logits", "recurrent"))
sb, ob = run11(PM[0], b_init, [seq], rep=("logits", "recurrent"))
sd, od = run11(PM[0], b_init, [[t] for t in seq], rep=("logits", "recurrent"))
ad.ok("truncate", session=sb["session"], n=0, report=[])
again = ad.ok("forward", session=sb["session"], ids=seq, report=["logits", "recurrent"])
ok = bits_equal(stack(o0), stack(oz)) and not bits_equal(stack(o0), stack(ob)) and bits_equal(stack(ob), stack(od)) and \
    bits_equal(stack(ob), f32(again["logits"])) and rec_equal(rec_arrays(ob[0]["recurrent"]), rec_arrays(again["recurrent"]))
record("rev11/init_body/pk_mamba", ok, zero_body_equals_plus_zero=bits_equal(stack(o0), stack(oz)),
       b_body_changes_logits=not bits_equal(stack(o0), stack(ob)), b_body_decode_bits=bits_equal(stack(ob), stack(od)),
       b_body_after_truncate_0=bits_equal(stack(ob), f32(again["logits"])))
for x in (s0, sz, sb, sd):
    ad.ok("close", session=x["session"])

# rule 30 (IR_SPEC 3.10): paperkiln.ir_sha256 checked against the embedded document, even under a substituted one.
PR = REC11["pk_recurrence"][0]
emb = embedded(PR)
emb_path = os.path.join(SCRATCH, "adapter_rev11_embedded.ir.json")
open(emb_path, "w", encoding="utf-8").write(json.dumps(emb))
want = None
import gguf as _gguf  # noqa: E402
_r = _gguf.GGUFReader(PR)
want = bytes(_r.fields["paperkiln.ir_sha256"].parts[-1]).decode()
bad = ("0" if want[0] != "0" else "1") + want[1:]
g_bad = copy_gguf(PR, os.path.join(SCRATCH, "adapter_rev11_bad_sha.gguf"),
                  set_keys={"paperkiln.ir_sha256": ("STRING", bad)})
g_none = copy_gguf(PR, os.path.join(SCRATCH, "adapter_rev11_no_sha.gguf"), drop_keys=("paperkiln.ir_sha256",))
g_int = copy_gguf(PR, os.path.join(SCRATCH, "adapter_rev11_int_sha.gguf"),
                  set_keys={"paperkiln.ir_sha256": ("UINT32", 7)})
r_ok = ad.ok("check", gguf=PR, ir=None)
r_bad = ad.ok("check", gguf=g_bad, ir=None)
r_bad_ir = ad.ok("check", gguf=g_bad, ir=emb_path)
r_none = ad.ok("check", gguf=g_none, ir=None)
r_int = ad.ok("check", gguf=g_int, ir=None)
dg = ad.ok("digest", text=json.dumps(emb), prose=False)
record("rev11/rule30", r_ok["accepted"] and not r_bad["accepted"] and "3.10" in r_bad.get("message", "") and
       not r_bad_ir["accepted"] and "3.10" in r_bad_ir.get("message", "") and r_none["accepted"] and
       not r_int["accepted"] and dg.get("sha256") == want,
       key=want, embedded_digest=dg.get("sha256"), wrong_key=r_bad.get("message"),
       wrong_key_with_ir=r_bad_ir.get("message"), no_key=r_none["accepted"], int_key=r_int.get("message"))

# digest and dequantize over the suite's vectors and blocks (the runner checks them too; here per reply shape)
vec = json.load(open(f"{SUITE}/digest/vectors.json", encoding="utf-8"))["vectors"]
good = 0
for v in vec:
    r = ad.ok("digest", text=v["text"], prose=v.get("prose", False))
    good += ("refused" in r) if "refused" in v else (r.get("sha256") == v["sha256"] and r.get("canonical") == v["canonical"])
record("rev11/digest_vectors", good == len(vec), vectors=len(vec), equal=good)
blocks = json.load(open(f"{SUITE}/quant/blocks.json", encoding="utf-8"))["cases"]
per_type = {}
for c in blocks:
    r = ad.ok("dequantize", type=c["type"], n=c["rows"] * c["n"], bytes=c["bytes"])
    per_type[c["type"]] = bits_equal(f32(r["values"]).reshape(c["rows"], c["n"]), f32(c["expect"]))
record("rev11/dequantize_blocks", all(per_type.values()), bits_equal=per_type)

# run_node: recurrence cases; a width that is not the declaration's (rule 34); other ops and gradients skip
ops = json.load(open(f"{SUITE}/ops/recurrence.json", encoding="utf-8"))["cases"]
c = ops[1]
r = ad.ok("run_node", node=c["node"], inputs=c["inputs"], weights=c["weights"], state=None)
record("rev11/run_node/ssm_scan_body_bits", bits_equal(f32(r["outputs"]["y"]), f32(c["expect"]["y"])))
bad_in = dict(c["inputs"])
u = f32(c["inputs"]["u"])
bad_in["u"] = {"shape": [u.shape[0], u.shape[1] - 1], "f32": "".join(c["inputs"]["u"]["f32"][i * 32:i * 32 + 24] for i in range(u.shape[0]))}
r = ad.call("run_node", node=c["node"], inputs=bad_in, weights=c["weights"], state=None)
record("rev11/run_node/width_rule34", "error" in r and "rule 34" in r["error"], reply=r)
r = ad.call("run_node_grad", node=c["node"], inputs=c["inputs"], weights=c["weights"], out_grad=c["expect"]["y"])
record("rev11/skip/run_node_grad", "skip" in r, reply=r)
r = ad.call("run_node", node={"op": "relu", "in": ["x"], "out": "y"}, inputs={}, weights={}, state=None)
record("rev11/skip/run_node_other_op", "skip" in r, reply=r)

# ---- spec revision 1.2 (keramos v1.2.0, d4b880ae): rules 35-37 and 20, check of a document alone, render_chat,
# the adapter's edge cases (section 11.3), expr in run_node, depth_attend's default eps. Appended after every
# earlier check. IR12_SUITE is the suite at v1.2.0 (an archive of the tag).
S12 = os.environ.get("IR12_SUITE", "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos/conformance")


def scratch_text(name, text):
    p = os.path.join(SCRATCH, f"adapter_rev12_{name}.ir.json")
    open(p, "wb").write(text.encode("utf-8"))
    return p


def scratch_doc(name, d):
    return scratch_text(name, json.dumps(d))


LL = f"{S12}/models/pk_llama/pk_llama.gguf"
ll = embedded(LL)
txt = json.dumps(ll)
n0 = json.dumps(ll["block"]["nodes"][0])
assert txt.count(n0) == 1
dups = {  # rule 35: a member name twice in one object, at any depth, "_" members included
    "top": txt[:-1] + ', "family_hint": "llama"}',
    "node": txt.replace(n0, n0[:-1] + ', "eps": ' + json.dumps(ll["block"]["nodes"][0]["eps"]) + "}"),
    "prose": '{"_a": 1, "_a": 1, ' + txt[1:],
    "nested_prose": '{"_note": {"x": [1, {"k": 1, "k": 2}]}, ' + txt[1:],
}
got = {}
for name, t in dups.items():
    r = ad.ok("check", gguf=LL, ir=scratch_text("dup_" + name, t))
    got[name] = r.get("message")
    record(f"rev12/rule35/{name}", not r["accepted"] and "(rule 35)" in r.get("message", ""), message=r.get("message"))
r = ad.ok("check", gguf=LL, ir=scratch_text("no_dup", txt))
record("rev12/rule35/same_text_without_the_duplicate", r["accepted"], reply=r)
DE = f"{S12}/refuse/duplicate_member_embedded.gguf"
r = ad.ok("check", gguf=DE, ir=None)
de_doc = embedded(DE)  # Python's json keeps the last of the two members
r2 = ad.ok("check", gguf=DE, ir=scratch_doc("embedded_deduplicated", de_doc))
# The rule governs the document read: with a valid one in its place the file loads (as the reference accepts it).
record("rev12/rule35/embedded", not r["accepted"] and "(rule 35)" in r.get("message", "") and r2["accepted"],
       message=r.get("message"), with_ir_in_place=r2)
r = ad.ok("check", gguf=None, ir=scratch_text("dup_alone", dups["node"]))
record("rev12/rule35/document_alone", not r["accepted"] and "(rule 35)" in r.get("message", ""), message=r.get("message"))

# rule 36 (and rule 20's run of steps): the suite's refusals, and one edit each of pk_llama and pk_mor
for name in ("reads_unwritten_value", "embed_writes_no_x", "head_writes_no_logits", "mor_reads_outside_value",
             "mor_step_gap"):
    c = json.load(open(f"{S12}/refuse/{name}.json", encoding="utf-8"))
    r = ad.ok("check", gguf=f"{S12}/{c['gguf']}", ir=scratch_doc(name, c["doc"]))
    # refused; a rule the message cites is the case's (ember's own load checks, which run first, cite none)
    cited = re.findall(r"[(]rule ([0-9]+)[)]", r.get("message", ""))
    record(f"rev12/rule{c['rule']}/{name}", not r["accepted"] and all(int(x) == c["rule"] for x in cited),
           message=r.get("message"))
MO = f"{S12}/models/pk_mor/pk_mor.gguf"
mo = embedded(MO)
edits36 = {
    "reads_a_later_write": (LL, ll, lambda d: d["block"]["nodes"][1].__setitem__("in", ["gu"])),
    "head_reads_block_value_under_mor": (MO, mo, lambda d: d["head"]["nodes"][1].__setitem__("in", ["h2"])),
}
for name, (g, base, edit) in edits36.items():
    d = copy.deepcopy(base)
    edit(d)
    r = ad.ok("check", gguf=g, ir=scratch_doc(name, d))
    record(f"rev12/rule36/{name}", not r["accepted"] and "(rule 36)" in r.get("message", ""), message=r.get("message"))
d = copy.deepcopy(mo)  # without depth_route the head reads the last application's values (section 3.5)
d["head"]["nodes"][1]["in"] = ["h2"]
d["block"]["nodes"] = d["block"]["nodes"][1:]
d["block"]["state"][0].pop("kv_rows")
r = ad.ok("check", gguf=None, ir=scratch_doc("head_reads_block_value", d))
record("rev12/rule36/head_reads_block_value_without_mor", r["accepted"], reply=r)

# rule 37 when the rows arrive: an ffn rmsnorm group that does not divide d (after layer 0's attention ran), and a
# residual add of the ffn width after a recurrence updated its state: the error names the rule, and the session is
# as before the call (its kv rows and recurrent tensors those of a fresh open).
d = copy.deepcopy(ll)
d["block"]["nodes"][8]["group"] = 3
g37 = scratch_doc("rule37_group", d)
PRG = f"{S12}/models/pk_recurrence/pk_recurrence.gguf"
pr = embedded(PRG)
d = copy.deepcopy(pr)
d["block"]["nodes"][-1]["in"] = ["x", "u"]
r37 = scratch_doc("rule37_add", d)
for name, (g, ir, rep) in {"rmsnorm_group": (LL, g37, "kv_rows"), "add_shapes": (PRG, r37, "recurrent")}.items():
    o = ad.ok("open", gguf=g, ir=ir, evict=None)
    fresh = ad.ok("truncate", session=o["session"], n=0, report=[rep])
    e = ad.call("forward", session=o["session"], ids=[1, 2, 3], report=["logits"])
    after = ad.ok("truncate", session=o["session"], n=0, report=[rep])
    record(f"rev12/rule37/{name}", "error" in e and "(rule 37)" in e["error"] and after == fresh,
           error=e.get("error"), session_unchanged=after == fresh)
    ad.ok("close", session=o["session"])

# depth_attend's eps optional, default 1e-5 (section 5.18): pk_attnres without it gives the same logits, bits
AT = f"{S12}/models/pk_attnres/pk_attnres.gguf"
at = embedded(AT)
d = copy.deepcopy(at)
n_eps = 0
for sect in ("embed", "block", "head"):
    for n in d[sect]["nodes"]:
        if n["op"] == "depth_attend" and "eps" in n:
            assert n["eps"] == 1e-5
            n.pop("eps")
            n_eps += 1
ids = [3, 1, 4, 1, 5, 9, 2, 6]
outs = []
for ir in (None, scratch_doc("attnres_no_eps", d)):
    o = ad.ok("open", gguf=AT, ir=ir, evict=None)
    outs.append(f32(ad.ok("forward", session=o["session"], ids=ids, report=["logits"])["logits"]))
    ad.ok("close", session=o["session"])
record("rev12/depth_attend_eps_default", n_eps > 0 and bits_equal(outs[0], outs[1]), nodes=n_eps,
       logits_bits=bits_equal(outs[0], outs[1]))

# check with gguf null: the document rules alone (the suite's acceptance documents with an "ir" and no "gguf")
acc = json.load(open(f"{S12}/accept/documents.json", encoding="utf-8"))["cases"]
alone = [c for c in acc if not c.get("gguf")]
res = {c["name"]: ad.ok("check", gguf=None, ir=os.path.normpath(f"{S12}/{c['ir']}")) for c in alone}
record("rev12/check_document_alone/accept", len(alone) > 0 and all(r["accepted"] for r in res.values()),
       documents=len(alone), refused={k: r["message"] for k, r in res.items() if not r["accepted"]})
r = ad.ok("check", gguf=None, ir=scratch_doc("alone_unknown_op", {**ll, "block": {**ll["block"], "nodes": [
    {**ll["block"]["nodes"][0], "op": "warp"}] + ll["block"]["nodes"][1:]}}))
record("rev12/check_document_alone/refuse", not r["accepted"] and "(rule 4)" in r.get("message", ""), message=r.get("message"))

# the adapter's edge cases (section 11.3)
o = ad.ok("open", gguf=LL, ir=None, evict=None)
s = o["session"]
e_empty = ad.call("forward", session=s, ids=[], report=["logits"])
a = ad.ok("forward", session=s, ids=[1, 2, 3, 4, 5], report=["logits", "depth", "recurrent", "kv_rows"])
e_past = ad.call("truncate", session=s, n=6, report=[])
e_neg = ad.call("truncate", session=s, n=-1, report=[])
same = ad.ok("truncate", session=s, n=5, report=["kv_rows", "kv_positions"])
ad.ok("snapshot", session=s, tag="t")
ad.ok("forward", session=s, ids=[6], report=[])
ad.ok("snapshot", session=s, tag="t")  # replaces the snapshot at 5
back = ad.ok("restore", session=s, tag="t", report=["kv_rows"])
e_tag = ad.call("restore", session=s, tag="never", report=[])
e_vocab = ad.call("forward", session=s, ids=[2, o["vocab"] + 3], report=["logits"])
nxt = ad.ok("forward", session=s, ids=[7, 8], report=["logits"])
f = ad.ok("open", gguf=LL, ir=None, evict=None)
fr = ad.ok("forward", session=f["session"], ids=[1, 2, 3, 4, 5, 6], report=[])
fn = ad.ok("forward", session=f["session"], ids=[7, 8], report=["logits"])
for x in (s, f["session"]):
    ad.ok("close", session=x)
ok = "error" in e_empty and a["len"] == 5 and a["depth"] == [] and a["recurrent"] == {} and "error" in e_past and \
    "error" in e_neg and same["len"] == 5 and same["kv_rows"] == a["kv_rows"] and back["len"] == 6 and \
    back["kv_rows"] == {"kv": [6, 6]} and "error" in e_tag and "error" in e_vocab and nxt["len"] == 8 and \
    bits_equal(f32(nxt["logits"]), f32(fn["logits"]))
record("rev12/edge_cases", ok, empty_forward=e_empty.get("error"), depth=a["depth"], recurrent=a["recurrent"],
       truncate_past=e_past.get("error"), truncate_negative=e_neg.get("error"), truncate_to_len=same["len"],
       snapshot_tag_reused_len=back["len"], restore_unknown=e_tag.get("error"), bad_id=e_vocab.get("error"),
       after_error_equals_fresh=bits_equal(f32(nxt["logits"]), f32(fn["logits"])))
o = ad.ok("open", gguf=f"{S12}/models/pk_recurrence/pk_recurrence.gguf", ir=None, evict=None)
a = ad.ok("forward", session=o["session"], ids=[1, 2, 3], report=["kv_rows", "kv_positions", "depth"])
record("rev12/edge_cases/no_kv_slot", a["kv_rows"] == {} and a["kv_positions"] == {} and a["depth"] == [], reply=a)
ad.ok("close", session=o["session"])

# render_chat over the suite's chat cases (section 8.1.3)
chat = json.load(open(f"{S12}/chat/render.json", encoding="utf-8"))
good, bad_cases = 0, []
for c in chat["cases"]:
    rec = chat["records"][c["record"]] if isinstance(c["record"], str) else c["record"]
    r = ad.ok("render_chat", record=rec, messages=c["messages"], generation_prompt=c.get("generation_prompt", False),
              thinking=c.get("thinking"))
    hit = ("refused" in r) if c.get("refused") else r.get("text") == c["expect"]
    good += hit
    if not hit:
        bad_cases.append(c["name"])
record("rev12/render_chat", good == len(chat["cases"]), cases=len(chat["cases"]), equal=good, failed=bad_cases)

# run_node on expr: the suite's expr cases, EXACT ones bit for bit, ROUNDING within the case's tolerance (rowsum of a constant is w additions; depth 200 parses);
# depth 204 (40 parentheses around x) refused; inputs of two shapes refused naming rule 37
ex = json.load(open(f"{S12}/ops/expr.json", encoding="utf-8"))["cases"]
per = {}
for c in ex:
    r = ad.ok("run_node", node=c["node"], inputs=c["inputs"], weights=c.get("weights", {}), state=None)
    per[c["name"]] = all(bits_equal(f32(r["outputs"][k]), f32(v)) if c["class"] == "EXACT" else
                         rel(f32(r["outputs"][k]), f32(v)) <= c["tol"] for k, v in c["expect"].items())
x0 = ex[0]["inputs"][ex[0]["node"]["in"][0]]
deep = ad.call("run_node", node={"op": "expr", "in": ["x"], "out": "y", "expr": "(" * 40 + "x" + ")" * 40},
               inputs={"x": x0}, weights={}, state=None)
two = ad.call("run_node", node={"op": "expr", "in": ["x", "y"], "out": "z", "expr": "x + a"},
              inputs={"x": x0, "y": {"shape": [x0["shape"][0], x0["shape"][1] - 1],
                                     "f32": x0["f32"][:8 * x0["shape"][0] * (x0["shape"][1] - 1)]}}, weights={}, state=None)
record("rev12/run_node/expr", all(per.values()) and "rowsum_constant" in per and "depth_200" in per and "error" in deep and
       "error" in two and "(rule 37)" in two["error"], equal_at_class=per, depth_204=deep.get("error"), two_shapes=two.get("error"))

# ---- spec revision 1.3 (keramos v1.3.0, da9ead61): rule 38 (a file's architecture), the builder's MoE llama, T5's
# position_bias (rule 39), the windowed rewind (6.9), attention in run_node. IR13_SUITE: the suite at the tag.
S13 = os.environ.get("IR13_SUITE", "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos/conformance")
sys.path.insert(0, os.path.join(os.path.dirname(S13), "tools"))
import ir_numpy  # noqa: E402  (the reference interpreter, a black-box oracle)
from ir_conformance import load_model  # noqa: E402
from pathlib import Path  # noqa: E402

# rule 38: an IR-only model labelled "gpt2" is refused at load (never served as plain GPT-2), its relabelled twin
# (the same document, digest and tensors) loads and runs its document; so does the twin under the file's own
# document read in place (the rule concerns the embedded document, which stays the old label's).
twins = {"kimi": ("refuse/architecture_kimi_as_gpt2.gguf", "models/pk_kimi/pk_kimi.gguf"),
         "swa": ("refuse/architecture_swa_as_gpt2.gguf", "models/pk_gpt2_swa/pk_gpt2_swa.gguf")}
for name, (old, new) in twins.items():
    go, gn = f"{S13}/{old}", f"{S13}/{new}"
    eo, en = embedded(go), embedded(gn)
    r_old = ad.ok("check", gguf=go, ir=None)
    r_old_ir = ad.ok("check", gguf=go, ir=scratch_doc(f"rule38_{name}_own", eo))
    o = ad.ok("open", gguf=gn, ir=None, evict=None)
    ids = [1, 4, 2, 7, 3, 9, 5]
    lg = f32(ad.ok("forward", session=o["session"], ids=ids, report=["logits"])["logits"])
    ad.ok("close", session=o["session"])
    doc_n, W_n, _, _ = load_model(Path(gn).parent)
    ref = ir_numpy.forward(doc_n, W_n, ids)
    same_doc = ad.ok("digest", text=json.dumps(eo), prose=False) == ad.ok("digest", text=json.dumps(en), prose=False)
    record(f"rev13/rule38/{name}_old_label_and_twin", not r_old["accepted"] and "(rule 38)" in r_old.get("message", "") and
           not r_old_ir["accepted"] and "(rule 38)" in r_old_ir.get("message", "") and same_doc and
           rel(lg, ref) <= 1e-5, old_label=r_old.get("message"), old_label_with_its_document_in_place=r_old_ir.get("message"),
           same_document=same_doc, twin_vs_reference_rel=rel(lg, ref))
# the third-party graphs a llama label names: pk_moe (Mixtral's layout) and pk_llama are their builder's graph
res = {m: ad.ok("check", gguf=f"{S13}/models/{m}/{m}.gguf", ir=None) for m in ("pk_moe", "pk_llama", "pk_gpt2", "tp_gemma3")}
record("rev13/rule38/architecture_graphs_accepted", all(r["accepted"] for r in res.values()), replies=res)
for name in ("architecture_metadata_disagrees", "architecture_without_graph"):
    r = ad.ok("check", gguf=f"{S13}/refuse/{name}.gguf", ir=None)
    record(f"rev13/rule38/{name}", not r["accepted"] and "(rule 38)" in r.get("message", ""), message=r.get("message"))

# 6.9's windowed rewind: ember keeps every row of a windowed slot (its cache holds every position up to max_len), so
# a cut into the region a ring-buffer cache would have dropped (llama.cpp's finding 4) keeps the rows of positions
# < n, and the next tokens' logits equal a fresh run's, bit for bit.
for name, model, n_ids, cut, more in (("pk_gpt2_swa", "pk_gpt2_swa", 14, 7, 3), ("tp_gemma3", "tp_gemma3", 48, 10, 4)):
    g = f"{S13}/models/{model}/{model}.gguf"
    o = ad.ok("open", gguf=g, ir=None, evict=None)
    V13 = o["vocab"]
    rng13 = random.Random(1300 + n_ids)
    seq = [rng13.randrange(1, V13) for _ in range(n_ids)]
    cont = [rng13.randrange(1, V13) for _ in range(more)]
    ad.ok("forward", session=o["session"], ids=seq, report=[])
    tr = ad.ok("truncate", session=o["session"], n=cut, report=["kv_positions"])
    after = f32(ad.ok("forward", session=o["session"], ids=cont, report=["logits"])["logits"])
    f = ad.ok("open", gguf=g, ir=None, evict=None)
    ad.ok("forward", session=f["session"], ids=seq[:cut], report=[])
    fresh = f32(ad.ok("forward", session=f["session"], ids=cont, report=["logits"])["logits"])
    for x in (o, f):
        ad.ok("close", session=x["session"])
    held = all(p == list(range(cut)) for p in tr["kv_positions"].get("kv", []))
    record(f"rev13/rewind_window/{name}", held and bits_equal(after, fresh), positions=n_ids, cut=cut,
           rows_kept_are_positions_below_the_cut=held, logits_bits_vs_fresh=bits_equal(after, fresh))

# position_bias in a whole model: pk_llama with a bias table [B 5, heads 2] added (ne {2, 5}) and its document's
# attention given position_bias, with and without a window (sinks 1): against the reference, and prefill = decode
LLP = os.path.join(SCRATCH, "adapter_rev13_pb")
os.makedirs(LLP, exist_ok=True)
import gguf as _g13  # noqa: E402
_r = _g13.GGUFReader(LL)
_w = _g13.GGUFWriter(os.path.join(LLP, "adapter_rev13_pb.gguf"), "llama")
for fname, fld in _r.fields.items():
    if fname.startswith("GGUF.") or fname in ("general.architecture", "general.alignment"):
        continue
    vt = fld.types[0]
    if vt == _g13.GGUFValueType.ARRAY:
        sub = fld.types[-1]
        val = [bytes(fld.parts[i]) for i in fld.data] if sub == _g13.GGUFValueType.STRING else \
            [fld.parts[i].tolist()[0] for i in fld.data]
        _w.add_key_value(fname, val, vt, sub_type=sub)
    elif vt == _g13.GGUFValueType.STRING:
        _w.add_key_value(fname, bytes(fld.parts[-1]), vt)
    else:
        _w.add_key_value(fname, fld.parts[-1].tolist()[0], vt)
for t in _r.tensors:
    _w.add_tensor(t.name, np.asarray(t.data).reshape([int(x) for x in reversed(t.shape.tolist())]).astype(np.float32))
_w.add_tensor("rel_bias.weight", np.random.default_rng(13).normal(0, 0.5, (5, 2)).astype(np.float32))
_w.write_header_to_file()
_w.write_kv_data_to_file()
_w.write_tensors_to_file()
_w.close()
PBG = os.path.join(LLP, "adapter_rev13_pb.gguf")
doc_pb, W_pb, _, _ = load_model(Path(LLP))
pb_ok = {}
for variant_name, window, sinks in (("causal", 0, 0), ("window_sinks", 3, 1)):
    d = copy.deepcopy(ll)
    for n in d["block"]["nodes"]:
        if n["op"] == "attention":
            n["position_bias"] = {"weight": "rel_bias.weight", "buckets": [0, 1, 2, 3, 3, 4]}
            n["window"], n["sinks"] = window, sinks
    d["block"]["state"][0]["window"], d["block"]["state"][0]["sinks"] = window, sinks
    p = scratch_doc(f"pb_{variant_name}", d)
    ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    o = ad.ok("open", gguf=PBG, ir=p, evict=None)
    pre = f32(ad.ok("forward", session=o["session"], ids=ids, report=["logits"])["logits"])
    ad.ok("close", session=o["session"])
    o = ad.ok("open", gguf=PBG, ir=p, evict=None)
    dec = np.concatenate([f32(ad.ok("forward", session=o["session"], ids=[t], report=["logits"])["logits"]) for t in ids])
    ad.ok("close", session=o["session"])
    ref = ir_numpy.forward(d, W_pb, ids)
    nob = copy.deepcopy(d)
    for n in nob["block"]["nodes"]:
        n.pop("position_bias", None)
    o = ad.ok("open", gguf=PBG, ir=scratch_doc(f"pb_{variant_name}_without", nob), evict=None)
    without = f32(ad.ok("forward", session=o["session"], ids=ids, report=["logits"])["logits"])
    ad.ok("close", session=o["session"])
    pb_ok[variant_name] = {"vs_reference_rel": rel(pre, ref), "decode_bits": bits_equal(pre, dec),
                           "bias_changes_logits": not bits_equal(pre, without)}
record("rev13/position_bias/whole_model", all(v["vs_reference_rel"] <= 1e-5 and v["decode_bits"] and v["bias_changes_logits"]
                                              for v in pb_ok.values()), variants=pb_ok)
# attention in run_node: the suite's six attention cases (three with position_bias) within the case tolerance
at_cases = json.load(open(f"{S13}/ops/attention.json", encoding="utf-8"))["cases"]
worst = {}
for c in at_cases:
    r = ad.ok("run_node", node=c["node"], inputs=c["inputs"], weights=c.get("weights", {}), state=None)
    worst[c["name"]] = rel(f32(r["outputs"]["a"]), f32(c["expect"]["a"]))
record("rev13/run_node/attention", all(v <= 1e-5 for v in worst.values()) and len(worst) == 6, worst_rel=worst)

ad.close()

report["requests"] = ad.n
report["failures"] = failures
report["pass"] = not failures
n = len(report["checks"])
print(f"{n - len(failures)}/{n} checks pass, {ad.n} requests" + ("" if not failures else ": FAIL " + ", ".join(failures)))
out_dir = SCRATCH if "IR_ADAPTER_BIN" in os.environ else HERE
out_path = os.environ.get("ADAPTER_SELFTEST_OUT", os.path.join(out_dir, "adapter_selftest_cpu.json"))
json.dump(report, open(out_path, "w"), indent=1)
sys.exit(0 if not failures else 1)
