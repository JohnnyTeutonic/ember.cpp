"""Summaries of verify_2026-10-27's VM outputs (bench_*.json from vf_bench, e2e/*.jsonl from vf_e2e.py).

    python3 vf_summary.py bench OUT_DIR      (text)
    python3 vf_summary.py e2e OUT_DIR        (text)
    python3 vf_summary.py json OUT_DIR       (both, as one JSON object on stdout)

bench: per model, mode and cached prefix: the decode step (p50 ms), and per n = 1 + k rows the legacy pass and
each ladder option in decode steps (p50 / decode p50); the per-stage times (synchronised, graph off) at n = 5, 9;
how many rows of the n = 9 pass equal the decode step's logits bit for bit, per option.
e2e: per model, request set and configuration: decode tokens per second, against plain decode in the same mode
(default against `off`, EMBER_DETERMINISTIC=1 runs against `det_off`), draft acceptance, tokens per pass, and
replies and ids identical to plain decode."""
import glob
import json
import os
import sys


def bench(out):
    res = {}
    for f in sorted(glob.glob(os.path.join(out, "bench_*_*.json"))):
        tag = os.path.basename(f)[len("bench_"):-len(".json")]
        d = json.load(open(f))
        r = res[tag] = {"gpu": d["gpu"], "prefixes": {}}
        for p in d["prefixes"]:
            dec = p["decode_ms"]["p50"]
            e = {"decode_ms": dec, "host_sample_ms": p["host_sample_ms"]["p50"],
                 "legacy_steps": {n: round(v["p50"] / dec, 3) for n, v in p["legacy_ms"].items()},
                 "legacy_ms": {n: round(v["p50"], 4) for n, v in p["legacy_ms"].items()},
                 "ladder_steps": {o: {n: round(v["p50"] / dec, 3) for n, v in lv.items()} for o, lv in p["ladder_ms"].items()},
                 "ladder_ms": {o: {n: round(v["p50"], 4) for n, v in lv.items()} for o, lv in p["ladder_ms"].items()},
                 "graph_smallm_plus_rows_to_host_steps": {n: round(v["p50"] / dec, 3) for n, v in
                                                          p["graph_smallm_plus_rows_to_host_ms"].items()},
                 "stages_ms": p["stages_ms"],
                 "bits_n9": {k: {"rows_bit_identical": sum(x["bit_equal"] == x["of"] for x in v), "rows": len(v),
                                 "max_abs": max(x["max_abs"] for x in v),
                                 "argmax_equal": sum(x["argmax_equal"] for x in v),
                                 **({"pick_ok": sum(x["pick_first_ok"] for x in v)} if "pick_first_ok" in v[0] else {})}
                             for k, v in p["bits_n9"].items()}}
            r["prefixes"][str(p["prefix"])] = e
    return res


def load(f):
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


def totals(recs):
    dec_tok = sum(max(0, (r["tokens"] or 0) - 1) for r in recs)
    dec_ms = sum(max(0.0, r["wall_ms"] - (r["first_token_ms"] or 0.0)) for r in recs)
    dr, ac, ps = (sum(r[k] for r in recs) for k in ("drafted", "accepted", "passes"))
    return {"tokens": sum(r["tokens"] or 0 for r in recs), "decode_tok_per_s": dec_tok / dec_ms * 1000.0 if dec_ms else 0.0,
            "drafted": dr, "accepted": ac, "passes": ps, "acceptance": ac / dr if dr else None,
            "tokens_per_pass": (ac + ps) / ps if ps else None}


def e2e(out):
    res = {}
    files = sorted(glob.glob(os.path.join(out, "e2e", "*.jsonl")))
    runs = {}
    for f in files:
        name = os.path.basename(f)[:-len(".jsonl")]
        model, cfg = name.split("_", 1)
        runs.setdefault(model, {})[cfg] = load(f)
    for model, cfgs in runs.items():
        res[model] = {}
        for cfg, recs in cfgs.items():
            base_name = "det_off" if cfg.startswith("det_") else "off"
            base = cfgs.get(base_name)
            for s in sorted(set(r["set"] for r in recs)) + ["all"]:
                rs = [r for r in recs if s == "all" or r["set"] == s]
                t = totals(rs)
                if base is not None:
                    bs = [r for r in base if s == "all" or r["set"] == s]
                    tb = totals(bs)
                    t["vs"] = base_name
                    t["speedup"] = t["decode_tok_per_s"] / tb["decode_tok_per_s"] if tb["decode_tok_per_s"] else None
                    t["replies_identical"] = sum(a["reply"] == b["reply"] for a, b in zip(rs, bs))
                    t["ids_identical"] = sum(a["ids"] == b["ids"] and a["stop_id"] == b["stop_id"] for a, b in zip(rs, bs))
                    t["n"] = len(rs)
                t["warm_pass_same"] = sum(r["warm_pass_same"] for r in rs)
                res[model].setdefault(cfg, {})[s] = t
    return res


def show_bench(b):
    for tag, r in b.items():
        print(f"== {tag} ({r['gpu']})")
        for P, e in r["prefixes"].items():
            print(f" prefix {P}: decode {e['decode_ms']:.3f} ms, host sampler {e['host_sample_ms']:.3f} ms")
            print("   legacy steps   " + " ".join(f"n{n}:{v}" for n, v in e["legacy_steps"].items()))
            for o, v in e["ladder_steps"].items():
                print(f"   {o:24s} " + " ".join(f"n{n}:{x}" for n, x in v.items()))
            print("   +rows to host   " + " ".join(f"n{n}:{v}" for n, v in e["graph_smallm_plus_rows_to_host_steps"].items()))
            for n, per in e["stages_ms"].items():
                for lin, st in per.items():
                    print(f"   stages n{n} {lin:6s} " + " ".join(f"{k}:{v:.3f}" for k, v in st.items()))
            for k, v in e["bits_n9"].items():
                print(f"   bits {k:22s} {v}")


def show_e2e(e):
    for model, cfgs in e.items():
        print(f"== {model}")
        for cfg, sets in cfgs.items():
            for s, t in sets.items():
                acc = f"{t['acceptance']:.2f}" if t["acceptance"] is not None else "-"
                tpp = f"{t['tokens_per_pass']:.2f}" if t["tokens_per_pass"] is not None else "-"
                sp = f"{t['speedup']:.3f}" if t.get("speedup") else "-"
                print(f"  {cfg:12s} {s:7s} {t['decode_tok_per_s']:8.1f} tok/s  x{sp} vs {t.get('vs', '-')}  "
                      f"acc {acc} tok/pass {tpp}  ids {t.get('ids_identical', '-')}/{t.get('n', '-')} "
                      f"replies {t.get('replies_identical', '-')}  warm {t['warm_pass_same']}")


if __name__ == "__main__":
    what, out = sys.argv[1], sys.argv[2]
    if what == "bench":
        show_bench(bench(out))
    elif what == "e2e":
        show_e2e(e2e(out))
    else:
        print(json.dumps({"bench": bench(out), "e2e": e2e(out)}, indent=1))
