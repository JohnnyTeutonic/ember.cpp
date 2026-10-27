"""Tables of decode_occupancy_2026-10-27 from the VM's output (and the CPU legs' references).

    python3 occ_summary.py bench OUT          the synthetic bench: per shape and length, every variant
    python3 occ_summary.py step OUT           the decode step in situ: split against tiled
    python3 occ_summary.py gates OUT [CPU]    x5 / verification / e2e / dumps comparisons (CPU: cpu_legs.sh's work dir)
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict


def bench(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "bench", "occ_bench.jsonl")) if l.strip()]
    hw = rows[0]
    print(f"{hw['gpu']}: {hw['sms']} SMs, L2 {hw['l2_bytes'] >> 20} MB, measured copy bandwidth {hw['copy_gbs']} GB/s")
    kern = defaultdict(dict)  # (range) -> {"split": avg_ns, "merge": avg_ns}
    p = os.path.join(out, "bench", "nsys_bench_nvtx_kern_sum.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            rng = r["NVTX Range"].lstrip(":")
            name = r["Kernel Name"]
            if "flush" in name or not rng:
                continue
            part = "merge" if "merge" in name else "split"
            kern[rng][part] = float(r["Avg (ns)"]) / 1000.0
    by = defaultdict(list)
    for r in rows[1:]:
        by[(r["shape"], r["L"], r["cap"])].append(r)
    for (shape, L, cap), rs in by.items():
        print(f"\n{shape} (H {rs[0]['H']}, head_dim {rs[0]['D']}), L {L} keys, cache {cap}: K+V {rs[0]['kv_bytes'] / 1e6:.3f} MB, "
              f"floor {rs[0]['floor_us_spec']:.2f} us at 300 GB/s, {rs[0]['floor_us_copy']:.2f} us at the measured copy rate")
        print(f"  {'variant':12s} {'chunk':>5s} {'blocks':>11s} {'graph us':>9s} {'hot split+merge':>16s} {'cold split+merge':>17s} "
              f"{'x floor':>7s} {'max err':>8s} bits")
        for r in sorted(rs, key=lambda r: r["us_per_layer_graph"]):
            tag = f"{r['variant']}|{shape}|L{L}|cap{cap}"
            h, c = kern.get(tag + "|hot", {}), kern.get(tag + "|cold", {})
            hs = f"{h.get('split', 0):.2f}+{h.get('merge', 0):.2f}" if h else "-"
            cs = f"{c.get('split', 0):.2f}+{c.get('merge', 0):.2f}" if c else "-"
            print(f"  {r['variant']:12s} {r['chunk']:5d} {r['active_blocks']:5d}/{r['grid_blocks']:5d} {r['us_per_layer_graph']:9.2f} "
                  f"{hs:>16s} {cs:>17s} {r['us_per_layer_graph'] / r['floor_us_spec']:7.1f} {r['max_err']:8.1e} "
                  f"{'yes' if r['run_to_run_bit_identical'] else 'NO'}")


def step(out):
    res = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(out, "step", "*.json"))):
        name = os.path.basename(f)[:-5]
        m, tag = name.split("_", 1)
        d = json.load(open(f))
        res[m][tag] = d
    for m, tags in res.items():
        any_ = next(iter(tags.values()))
        print(f"\n{m}: {os.path.basename(any_['model'])}, d {any_['hidden']}, {any_['layers']} layers x {any_['heads']} heads, "
              f"cache {any_['max_seq']}, {any_['gpu']}")
        ps = [p["prefix"] for p in any_["prefixes"]]
        cols = [t for t in ("split", "split2", "tiled", "tiled2", "t16", "t64", "det_split", "det_tiled") if t in tags]
        print("  keys " + " ".join(f"{c:>10s}" for c in cols) + "   tiled/split  tok/s split -> tiled  argmax same")
        for i, p in enumerate(ps):
            v = {c: tags[c]["prefixes"][i]["decode_ms"]["p50"] for c in cols}
            am = {c: tags[c]["prefixes"][i]["argmax"] for c in cols}
            sp = min(v.get("split", 1e9), v.get("split2", 1e9))
            ti = min(v.get("tiled", 1e9), v.get("tiled2", 1e9))
            cmp = f"   {sp / ti:10.3f}x  {1000 / sp:7.1f} -> {1000 / ti:7.1f}" if sp < 1e9 and ti < 1e9 else " " * 37
            print(f"  {p + 1:4d} " + " ".join(f"{v[c]:10.4f}" for c in cols) + f"{cmp}  {len(set(am.values())) == 1}")


def gates(out, cpu=None):
    print("x5_plain_collect comparisons (360 requests: tinychat 40 + carry5 320)")
    for f in sorted(glob.glob(os.path.join(out, "plain", "cmp_*.json"))):
        d = json.load(open(f))
        s = {k: (v["replies_identical"], v["ids_identical"], v.get("every_position", {}).get("bit_equal_rows"),
                 v.get("every_position", {}).get("rows"), v.get("every_position", {}).get("worst_abs"),
                 v.get("every_position", {}).get("worst_rel_to_range")) for k, v in d["sets"].items()}
        print(f"  {os.path.basename(f)[4:-5]:32s} bit_identical={d['bit_identical']} " +
              "; ".join(f"{k}: replies {a}/{d['sets'][k]['n']}, ids {b}, rows bit-equal {c}/{e}, worst |d| {w}, rel {r}"
                        for k, (a, b, c, e, w, r) in s.items()))
    print("\nspeculative verification (vd_compare, all60)")
    for f in sorted(glob.glob(os.path.join(out, "vd", "cmp_*.json"))):
        try:
            d = json.load(open(f))
        except ValueError:
            print(f"  {os.path.basename(f)}: unreadable")
            continue
        print(f"  {os.path.basename(f)[4:-5]:28s} " + json.dumps({k: d.get(k) for k in (
            "bit_identical", "requests", "ids_identical", "replies_identical", "rows_compared", "rows_bit_identical",
            "max_abs_diff")}))
    print("\nend to end (vf_e2e.py, all140: all60 + c12_40 + long40), decode tokens per second by set")
    runs = {}
    for f in sorted(glob.glob(os.path.join(out, "e2e", "*.jsonl"))):
        runs[os.path.basename(f)[:-6]] = [json.loads(l) for l in open(f) if l.strip()]
    cpu_runs = {}
    if cpu:
        for f in glob.glob(os.path.join(cpu, "e2e", "*_cpu.jsonl")):
            cpu_runs[os.path.basename(f)[:-10]] = [json.loads(l) for l in open(f) if l.strip()]

    def tps(recs, sets):
        r = [x for x in recs if x["set"] in sets]
        tok = sum(max(0, (x["tokens"] or 0) - 1) for x in r)
        ms = sum(max(0.0, x["wall_ms"] - (x["first_token_ms"] or 0)) for x in r)
        return tok / ms * 1000 if ms else 0

    groups = {"chat40": {"chat40"}, "copy20": {"copy20"}, "c12_40": {"c12_40"}, "long40": {"long40"}}
    for name, recs in runs.items():
        m = name.split("_")[0]
        line = f"  {name:16s} " + " ".join(f"{g} {tps(recs, s):7.1f}" for g, s in groups.items())
        base = runs.get(m + "_split") if "det" not in name else runs.get(m + "_det_split")
        if base and base is not recs:
            same = sum(x["ids"] == y["ids"] and x["stop_id"] == y["stop_id"] for x, y in zip(base, recs))
            line += f"  ids = split's {same}/{len(recs)}"
        if m in cpu_runs:
            c = cpu_runs[m]
            same = sum(x["ids"] == y["ids"] and x["stop_id"] == y["stop_id"] for x, y in zip(c, recs))
            per = {g: sum(x["ids"] == y["ids"] and x["stop_id"] == y["stop_id"] for x, y in zip(c, recs) if y["set"] in s)
                   for g, s in groups.items()}
            line += f"  ids = CPU's {same}/{len(recs)} {per}"
        print(line)
    print("\ninterpreter dumps (HEAD build vs this build)")
    for f in sorted(glob.glob(os.path.join(out, "dumps", "*.json"))):
        try:
            d = json.load(open(f))
        except ValueError:
            print(f"  {os.path.basename(f)}: {open(f).read()[:200]}")
            continue
        print(f"  {os.path.basename(f)[:-5]:24s} prefill bit-identical {d['prefill_bit_identical']}, decode rows bit-identical "
              f"{d['decode_rows_bit_identical']}/{d['decode_rows']}, head vs new {d['decode_head_vs_new_worst_abs']:.2e}, "
              f"argmax {d['decode_argmax_equal']}/{d['decode_rows']}, greedy {d['greedy_ids_identical']}/{d['prompts']}; "
              f"decode vs prefill: head {d['head_self']['decode_vs_prefill_worst_abs']:.2e} "
              f"({d['head_self']['decode_vs_prefill_worst_rounding_ratio']:.2e}), new "
              f"{d['new_self']['decode_vs_prefill_worst_abs']:.2e} ({d['new_self']['decode_vs_prefill_worst_rounding_ratio']:.2e})")


if __name__ == "__main__":
    {"bench": bench, "step": step, "gates": gates}[sys.argv[1]](*sys.argv[2:])
