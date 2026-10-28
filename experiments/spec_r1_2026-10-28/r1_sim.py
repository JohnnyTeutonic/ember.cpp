"""Draft-quality candidates for ember's speculative loop, replayed over recorded greedy decodes (spec_r1_2026-10-28).

    python3 r1_sim.py build OUT.npz --corpus A.u16[,B.u16] | --pool POOL.jsonl[.gz] [--orders 6] [--min-count 8]
                     [--min-conf 0.10]
    python3 r1_sim.py merge OUT.bin TABLE[,TABLE...] [--orders 4]   (EMBERDT1; earlier tables first)
    python3 r1_sim.py sweep MODEL RECORDS.jsonl[,...] CONFIGS.json [--costs COSTS.json] [--out OUT.jsonl]

Greedy speculation is exact, so a run's passes, drafts and acceptances follow from plain decode's prompt ids and ids
alone; draft_2026-10-27's dr_sim.py replay (api.cpp's loop) is reused unchanged, and this file adds drafters.

Tables (`build`): for every order o = 1..orders and every context (the last o tokens) seen at least --min-count
times, the most frequent next token and its share, kept when the share is at least --min-conf. Keys are
draft_table.h's for o <= 4 (o <= 3 packed, o = 4 mix64 of the packing >> 16) and, for o >= 5, the order-4 hash
continued: h = mix64(pack(t0..t3)), then h = mix64(h ^ (t_j + 1) * 0x9E3779B97F4A7C15) for j = 4..o-1, key h >> 16
(t0 the most recent token). --corpus counts every position of the token streams; --pool counts only positions
whose next token is a model output in a dr_e2e.py record file (its prompt's last tokens as context, then its ids and
stop id), so a self-output table holds what the model itself generated.

Drafters (CONFIGS.json: a list of {"name", ...}):
  {"pld": true|false, "tables": [[path, max_order], ...], "theta": t, "mode": "fill"|"first"|"extend"|"conf",
   "conf_min": c, "adapt": null|{"c": x}}
  fill    prompt lookup first; where it has nothing, the tables (draft_table.h's rule, ember's EMBER_DRAFT=table);
  first   the tables first; prompt lookup where they have nothing;
  extend  prompt lookup's draft continued by the tables up to k (or the tables' draft where prompt lookup has none);
  conf    the tables' draft when its first entry's share is at least conf_min, else prompt lookup, else the tables.
  Tables are tried in list order, each by its longest order (the first table with a hit gives the token; with
  "chain": "order", the longest order over all tables, the earlier table first at equal order); the draft
  stops at k tokens or when the product of the shares falls below theta. adapt caps the draft length at
  max(1, ceil(c * a / (1 - a))) for the request's running acceptance a (passes so far; 0.5 before any).
Speed: COSTS.json {"step_ms", "pass_ms": {"1": .., ..., "16": ..}} (r1_summary.py costs, an L4 bench): a plain step
costs step_ms (decode step + host sampler), a pass of n rows pass_ms[n]; estimate = decode tokens' plain time over
the run's time. Without --costs, draft_2026-10-27's L4 pass costs in decode steps."""
import argparse
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "draft_2026-10-27"))
import dr_sim  # noqa: E402

G = np.uint64(0x9E3779B97F4A7C15)


def mix64(x):
    x = x.copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def keys_at(ids, o, ends):
    """Keys of the contexts of order o ending at positions `ends` (ids[e] the most recent token)."""
    ids = ids.astype(np.uint64)
    if o <= 4:
        k = np.zeros(ends.size, dtype=np.uint64)
        for j in range(o):
            k |= ids[ends - j] << np.uint64(16 * j)
        return mix64(k) >> np.uint64(16) if o == 4 else k
    h = np.zeros(ends.size, dtype=np.uint64)
    for j in range(4):
        h |= ids[ends - j] << np.uint64(16 * j)
    h = mix64(h)
    with np.errstate(over="ignore"):
        for j in range(4, o):
            h = mix64(h ^ ((ids[ends - j] + np.uint64(1)) * G))
    return h >> np.uint64(16)


def key_of(ctx):
    """One context's key (a list, most recent last)."""
    a = np.array(ctx, dtype=np.int64)
    return int(keys_at(a, len(ctx), np.array([len(ctx) - 1]))[0])


def build_table(ids, valid_next, orders, min_count, min_conf):
    """ids: one token stream (sentinel 65535 between segments); valid_next[i]: position i may be a predicted next."""
    sent = ids == 65535
    out = []
    for o in range(1, orders + 1):
        nxt_pos = np.arange(o, ids.size)
        ends = nxt_pos - 1
        ok = valid_next[nxt_pos]
        # no sentinel inside the context or as the next token
        bad = sent[nxt_pos].copy()
        for j in range(o):
            bad |= sent[ends - j]
        ok &= ~bad
        ends, nxt_pos = ends[ok], nxt_pos[ok]
        if ends.size == 0:
            out.append({"keys": np.zeros(0, np.uint64), "next": np.zeros(0, np.uint16), "conf": np.zeros(0, np.float64)})
            continue
        ck = keys_at(ids, o, ends)
        comb = (ck << np.uint64(16)) | ids[nxt_pos].astype(np.uint64)
        del ck
        vals, counts = np.unique(comb, return_counts=True)
        ctx = vals >> np.uint64(16)
        tokn = (vals & np.uint64(0xFFFF)).astype(np.uint16)
        starts = np.flatnonzero(np.concatenate(([True], ctx[1:] != ctx[:-1])))
        tot = np.add.reduceat(counts, starts)
        order = np.lexsort((-counts.astype(np.int64), ctx))
        first = order[starts]
        p = counts[first] / tot
        keep = (tot >= min_count) & (p >= min_conf)
        out.append({"keys": ctx[starts][keep], "next": tokn[first][keep],
                    "conf": np.round(p[keep] * 255).astype(np.uint8).astype(np.float64) / 255.0})
        print(json.dumps({"order": o, "contexts": int(starts.size), "kept": int(keep.sum())}), flush=True)
    return out


def cmd_build(a):
    if a.corpus:
        ids = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in a.corpus.split(",")])
        valid = np.ones(ids.size, dtype=bool)
    else:
        segs, val = [], []
        op = gzip.open if a.pool.endswith(".gz") else open
        for line in op(a.pool, "rt", encoding="utf-8"):
            r = json.loads(line)
            gen = list(r["ids"] or []) + ([r["stop_id"]] if r.get("stop_id") is not None and r["stop_id"] >= 0 else [])
            ctx = list(r["prompt_ids"][-a.orders:])
            segs.append(ctx + gen + [65535])
            val.append([False] * len(ctx) + [True] * len(gen) + [False])
        ids = np.array([t for s in segs for t in s], dtype=np.uint16)
        valid = np.array([v for s in val for v in s], dtype=bool)
    t = build_table(ids, valid, a.orders, a.min_count, a.min_conf)
    np.savez(a.out, **{f"{f}{o}": t[o][f] for o in range(len(t)) for f in ("keys", "next", "conf")}, orders=len(t))
    print(json.dumps({"out": a.out, "tokens": int(ids.size), "entries": [int(x["keys"].size) for x in t]}))


def cmd_merge(a):
    """Tables (npz or EMBERDT1, in priority order) into one EMBERDT1 file of orders 1..--orders (at most 8; orders 5-8 need ember at
    spec_r1_2026-10-28's draft_table.h): at each order the
    union of the contexts, the earlier table's entry where several hold one (r1_sim's "chain": "order")."""
    tabs = [Table(p) for p in a.tables.split(",")]
    orders = min(a.orders, 8, max(t.orders for t in tabs))
    out = []
    for o in range(orders):
        keys, nxt, conf = [], [], []
        for pri, t in enumerate(tabs):
            if o < t.orders:
                keys.append(t.t[o]["keys"].astype(np.uint64))
                nxt.append(t.t[o]["next"].astype(np.uint16))
                conf.append(np.round(t.t[o]["conf"] * 255).astype(np.uint8))
        k = np.concatenate(keys)
        n = np.concatenate(nxt)
        c = np.concatenate(conf)
        pri = np.concatenate([np.full(x.size, i, np.int64) for i, x in enumerate(keys)])
        idx = np.lexsort((pri, k))  # key ascending, the earlier table first
        k, n, c = k[idx], n[idx], c[idx]
        first = np.concatenate(([True], k[1:] != k[:-1]))
        out.append({"keys": k[first], "next": n[first], "conf": c[first]})
    dr_sim.dr_table.write(a.out, a.vocab_size, out)
    dr_sim.dr_table.info_path(a.out)


class Table:
    def __init__(self, path, max_order=None):
        if path.endswith(".npz"):
            z = np.load(path)
            n = int(z["orders"])
            self.t = [{"keys": z[f"keys{o}"], "next": z[f"next{o}"], "conf": z[f"conf{o}"]} for o in range(n)]
        else:  # an EMBERDT1 file (draft_2026-10-27)
            _, tabs = dr_sim.dr_table.read(path)
            self.t = [{"keys": t["keys"], "next": t["next"], "conf": t["conf"].astype(np.float64) / 255.0} for t in tabs]
        self.orders = min(len(self.t), max_order or len(self.t))

    def lookup_order(self, ctx, o):
        t = self.t[o - 1]
        if t["keys"].size == 0 or len(ctx) < o:
            return None
        k = np.uint64(key_of(ctx[-o:]))
        i = int(np.searchsorted(t["keys"], k))
        if i < t["keys"].size and t["keys"][i] == k:
            return int(t["next"][i]), float(t["conf"][i]), o
        return None

    def lookup(self, ctx):
        for o in range(min(self.orders, len(ctx)), 0, -1):
            r = self.lookup_order(ctx, o)
            if r is not None:
                return r
        return None


_cache = {}


def tables_of(spec):
    out = []
    for path, mo in spec:
        key = (path, mo)
        if key not in _cache:
            _cache[key] = Table(path, mo)
        out.append(_cache[key])
    return out


def chain_lookup(tabs, ctx, by_order=False):
    """by_order False: the first table with any hit (its longest order); True: the longest order over all tables,
    the earlier table first at equal order (one merged table)."""
    if by_order:
        for o in range(min(max(t.orders for t in tabs), len(ctx)), 0, -1):
            for t in tabs:
                if o > t.orders:
                    continue
                r = t.lookup_order(ctx, o)
                if r is not None:
                    return r
        return None
    for t in tabs:
        r = t.lookup(ctx)
        if r is not None:
            return r
    return None


def table_draft(tabs, h, k, theta, start=None, by_order=False):
    d, p = [], 1.0
    no = max(t.orders for t in tabs)
    ctx = list(h[-no:]) + list(start or [])
    while len(d) < k:
        r = chain_lookup(tabs, ctx[-no:], by_order)
        if r is None:
            break
        p *= r[1]
        if p < theta:
            break
        d.append(r[0])
        ctx.append(r[0])
    return d


def make_drafter(cfg):
    tabs = tables_of(cfg.get("tables", []))
    theta = cfg.get("theta", 0.02)
    mode = cfg.get("mode", "fill")
    use_pld = cfg.get("pld", True)
    bo = cfg.get("chain") == "order"

    def drafter(h, k):
        p = dr_sim.pld(h, k) if use_pld else []
        if not tabs:
            return p
        if mode == "fill":
            return p or table_draft(tabs, h, k, theta, by_order=bo)
        if mode == "first":
            return table_draft(tabs, h, k, theta, by_order=bo) or p
        if mode == "extend":
            if p and len(p) < k:
                return p + table_draft(tabs, list(h) + p, k - len(p), theta, by_order=bo)
            return p or table_draft(tabs, h, k, theta, by_order=bo)
        if mode == "conf":
            no = max(t.orders for t in tabs)
            r = chain_lookup(tabs, list(h[-no:]), bo)
            if r is not None and r[1] >= cfg.get("conf_min", 0.5):
                return table_draft(tabs, h, k, theta, by_order=bo)
            return p or table_draft(tabs, h, k, theta, by_order=bo)
        raise ValueError(mode)
    return drafter


def replay_adaptive(rec, k, drafter, c):
    """dr_sim.replay with the draft length capped by the request's running acceptance."""
    seq = list(rec["ids"]) + ([rec["stop_id"]] if rec.get("stop_id") is not None and rec["stop_id"] >= 0 else [])
    steps = rec["max_new"]
    hist = list(rec["prompt_ids"])
    out = {"tokens": len(seq), "plain": 0, "passes": 0, "drafted": 0, "accepted": 0, "rows": []}
    if not seq:
        return out
    hist.append(seq[0])
    g = 1
    while g < len(seq):
        room = min(k, steps - g - 1)
        if c is not None and room > 0:
            a = out["accepted"] / out["drafted"] if out["drafted"] else 0.5
            a = min(a, 0.95)
            room = min(room, max(1, math.ceil(c * a / (1 - a))))
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


def cmd_sweep(a):
    recs = dr_sim.attach_max_new(dr_sim.load(a.records), None)
    cfgs = json.load(open(a.configs))
    costs = json.load(open(a.costs)) if a.costs else None
    out = open(a.out, "a") if a.out else None
    for cfg in cfgs:
        dr = make_drafter(cfg)
        for k in cfg.get("k", [8]):
            tot = {"tokens": 0, "plain": 0, "passes": 0, "drafted": 0, "accepted": 0}
            t_run = 0.0
            dec = 0
            rows_hist = {}
            for r in recs:
                o = replay_adaptive(r, k, dr, (cfg.get("adapt") or {}).get("c"))
                for f in tot:
                    tot[f] += o[f]
                for n in o["rows"]:
                    rows_hist[n] = rows_hist.get(n, 0) + 1
                if costs:
                    t_run += o["plain"] * costs["step_ms"] + sum(costs["pass_ms"][str(n)] for n in o["rows"])
                else:
                    t_run += o["plain"] + sum(dr_sim.pass_cost(a.model, n) for n in o["rows"])
                dec += max(0, o["tokens"] - 1)
            plain_t = dec * (costs["step_ms"] if costs else 1.0)
            res = {"name": cfg["name"], "k": k, "model": a.model, "records": a.records, **tot,
                   "acceptance": tot["accepted"] / tot["drafted"] if tot["drafted"] else None,
                   "tok_per_pass": (tot["accepted"] + tot["passes"]) / tot["passes"] if tot["passes"] else None,
                   "positions": tot["plain"] + tot["passes"], "decode_tokens": dec,
                   "est_speedup": plain_t / t_run if t_run else None, "costs": a.costs,
                   "rows_hist": dict(sorted(rows_hist.items()))}
            print(json.dumps({k2: res[k2] for k2 in ("name", "k", "acceptance", "tok_per_pass", "positions",
                                                     "decode_tokens", "est_speedup")}), flush=True)
            if out:
                out.write(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--corpus")
    b.add_argument("--pool")
    b.add_argument("--orders", type=int, default=6)
    b.add_argument("--min-count", type=int, default=8)
    b.add_argument("--min-conf", type=float, default=0.10)
    mg = sub.add_parser("merge")
    mg.add_argument("out")
    mg.add_argument("tables")
    mg.add_argument("--vocab-size", type=int, default=16384)
    mg.add_argument("--orders", type=int, default=4)
    s = sub.add_parser("sweep")
    s.add_argument("model")
    s.add_argument("records")
    s.add_argument("configs")
    s.add_argument("--costs")
    s.add_argument("--out")
    a = ap.parse_args()
    {"build": cmd_build, "merge": cmd_merge, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
