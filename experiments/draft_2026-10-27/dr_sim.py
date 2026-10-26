"""Replay of ember's speculative loop over recorded greedy decodes (draft_2026-10-27).

    python3 dr_sim.py check RECORDS.jsonl CUDA_E2E.jsonl K           # PLD counts against a recorded CUDA run
    python3 dr_sim.py sweep RECORDS.jsonl[,...] TABLE.bin [--k 2,4,8] [--theta ...] [--mode ...] [--model pick|pk]
    python3 dr_sim.py vocab GGUF_A GGUF_B

Greedy speculation is exact: the verify pass's row i picks the token plain decode would generate there, so a run's
passes, drafts and acceptances follow from the plain decode's prompt ids and ids alone. This replays api.cpp's loop
(the first token from prefill; then, while tokens remain, room = min(k, steps - generated - 1); a non-empty draft
makes a pass that yields the matching prefix plus one token, else a plain step) for each drafter:
  pld:        api.cpp's prompt_lookup_draft (the last 3, then 2, tokens' most recent earlier occurrence);
  table:      the corpus table (draft_table.h): from the history plus the draft so far, the longest order whose
              context is in the table gives the next token; the draft stops when the product of the entries'
              shares falls below theta, or at room;
  pld+table:  pld when it has a draft, else table (EMBER_DRAFT=table's rule).
Estimated decode speed-up over plain decode uses the L4 verify costs of verify_2026-10-27 (decode-step units,
P 32 rows of vf_bench): pass(n) interpolated over n = 3, 5, 9 (pick 0.984 / 1.005 / 1.065, pk 0.886 / 0.913 /
0.973; n = 2 taken as n = 3), a plain step 1. It ignores the drafter's own time (measured on the GPU runs)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dr_table  # noqa: E402

COST = {"pick": {3: 0.984, 5: 1.005, 9: 1.065}, "pk": {3: 0.886, 5: 0.913, 9: 0.973}}


def pass_cost(model, n):
    c = COST[model]
    xs = sorted(c)
    n = max(n, xs[0])
    if n >= xs[-1]:
        # extrapolate linearly from the last two points
        a, b = xs[-2], xs[-1]
        return c[b] + (c[b] - c[a]) / (b - a) * (n - b)
    for a, b in zip(xs, xs[1:]):
        if a <= n <= b:
            return c[a] + (c[b] - c[a]) * (n - a) / (b - a)


def pld(h, k):
    n = len(h)
    for g in range(min(3, n - 1), 1, -1):
        tail = h[n - g:]
        for s in range(n - g - 1, -1, -1):
            if h[s:s + g] == tail:
                return h[s + g:min(n, s + g + k)]
    return []


class Table:
    def __init__(self, path):
        self.vocab, self.t = dr_table.read(path)
        self.orders = len(self.t)

    def lookup(self, ctx):
        for o in range(min(self.orders, len(ctx)), 0, -1):
            t = self.t[o - 1]
            if t["keys"].size == 0:
                continue
            key = np.uint64(dr_table.key_of(ctx[-o:]))
            i = int(np.searchsorted(t["keys"], key))
            if i < t["keys"].size and t["keys"][i] == key:
                return int(t["next"][i]), int(t["conf"][i]) / 255.0, o
        return None

    def draft(self, h, k, theta, min_order=1):
        d, p = [], 1.0
        ctx = list(h[-self.orders:])
        while len(d) < k:
            r = self.lookup(ctx)
            if r is None or r[2] < min_order:
                break
            t, c, _ = r
            p *= c
            if p < theta:
                break
            d.append(t)
            ctx = (ctx + [t])[-self.orders:]
        return d


def replay(rec, k, drafter):
    seq = list(rec["ids"]) + ([rec["stop_id"]] if rec.get("stop_id") is not None and rec["stop_id"] >= 0 else [])
    steps = rec["max_new"]
    hist = list(rec["prompt_ids"])
    if not seq:
        return {"tokens": 0, "plain": 0, "passes": 0, "drafted": 0, "accepted": 0, "rows": []}
    hist.append(seq[0])
    g = 1
    out = {"tokens": len(seq), "plain": 0, "passes": 0, "drafted": 0, "accepted": 0, "rows": []}
    while g < len(seq):
        room = min(k, steps - g - 1)
        d = drafter(hist, room) if room > 0 else []
        if d:
            acc = 0
            for i in range(len(d) + 1):
                t = seq[g]
                hist.append(t)
                g += 1
                if g >= len(seq):
                    break
                if i < len(d) and t == d[i]:
                    acc += 1
                    continue
                break
            out["passes"] += 1
            out["drafted"] += len(d)
            out["accepted"] += acc
            out["rows"].append(len(d) + 1)
        else:
            hist.append(seq[g])
            g += 1
            out["plain"] += 1
    return out


def load(paths):
    recs = []
    for p in paths.split(","):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            recs.append(r)
    return recs


def attach_max_new(recs, prompts_dir):
    for r in recs:
        r["max_new"] = {"chat40": 60, "chatdev": 60, "copy20": 80, "carry40": 24, "carrydev": 24}[r["set"]]
    return recs


def summarise(recs, k, drafter, model):
    tot = {"tokens": 0, "plain": 0, "passes": 0, "drafted": 0, "accepted": 0}
    cost = 0.0
    dec = 0
    for r in recs:
        o = replay(r, k, drafter)
        for f in tot:
            tot[f] += o[f]
        cost += o["plain"] + sum(pass_cost(model, n) for n in o["rows"])
        dec += max(0, o["tokens"] - 1)
    tot["tok_per_pass"] = (tot["accepted"] + tot["passes"]) / tot["passes"] if tot["passes"] else None
    tot["acceptance"] = tot["accepted"] / tot["drafted"] if tot["drafted"] else None
    tot["est_speedup"] = dec / cost if cost else None
    tot["decode_tokens"] = dec
    tot["steps"] = tot["plain"] + tot["passes"]
    return tot


def cmd_check(a):
    recs = attach_max_new(load(a[0]), None)
    cuda = [json.loads(l) for l in open(a[1], encoding="utf-8")]
    k = int(a[2])
    bad = 0
    for r, c in zip(recs, cuda):
        assert r["ids"] == c["ids"], ("ids differ", r["i"])
        o = replay(r, k, pld)
        if (o["drafted"], o["accepted"], o["passes"]) != (c["drafted"], c["accepted"], c["passes"]):
            bad += 1
    print(json.dumps({"records": len(recs), "k": k, "counts_differ": bad}))


def cmd_sweep(a):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("table")
    ap.add_argument("--k", default="4,8")
    ap.add_argument("--theta", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--model", default="pick")
    ap.add_argument("--modes", default="pld,table,pld+table")
    ap.add_argument("--min-order", type=int, default=1)
    x = ap.parse_args(a)
    recs = attach_max_new(load(x.records), None)
    tab = Table(x.table) if x.table != "-" else None
    res = []
    for k in [int(v) for v in x.k.split(",")]:
        for mode in x.modes.split(","):
            thetas = [None] if mode == "pld" else [float(v) for v in x.theta.split(",")]
            for th in thetas:
                if mode == "pld":
                    dr = pld
                elif mode == "table":
                    dr = lambda h, kk, th=th: tab.draft(h, kk, th, x.min_order)
                else:
                    dr = lambda h, kk, th=th: pld(h, kk) or tab.draft(h, kk, th, x.min_order)
                s = summarise(recs, k, dr, x.model)
                s.update({"k": k, "mode": mode, "theta": th})
                res.append(s)
                print(json.dumps(s), flush=True)
    return res


def cmd_vocab(a):
    sys.path.insert(0, "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
    from llama3_bpe import read_gguf_kv
    ks = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.model", "tokenizer.ggml.pre"]
    va, vb = read_gguf_kv(a[0], ks), read_gguf_kv(a[1], ks)
    print(json.dumps({k: va.get(k) == vb.get(k) for k in ks} | {"tokens": len(va["tokenizer.ggml.tokens"])}))


if __name__ == "__main__":
    {"check": cmd_check, "sweep": cmd_sweep, "vocab": cmd_vocab}[sys.argv[1]](sys.argv[2:])
