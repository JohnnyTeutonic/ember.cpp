"""Summaries of spec_r1_2026-10-28's VM outputs.

    python3 r1_summary.py bench OUT          the decode step and the pass by linear kernel, in ms and in steps; bits
    python3 r1_summary.py nsys OUT           kernel time inside the graphs per call: linear, attention, the rest
    python3 r1_summary.py e2e OUT            decode tokens per second and ratios; ids; acceptance
    python3 r1_summary.py bits OUT           the dump comparisons
    python3 r1_summary.py rows OUT           every run's rows against the CPU (and main's deterministic run)
    python3 r1_summary.py costs OUT MODEL MODE OUT.json [P]
                                             r1_sim.py's cost file from the bench (MODE default|det; P default 384:
                                             step_ms = decode step + host sampler, pass_ms[n] by the mode's linear)"""
import csv
import glob
import json
import os
import sys

MODELS = ("pick", "pk")


def load(p):
    return json.load(open(p))


def bench(out):
    for m in MODELS:
        for mode in ("default", "det"):
            p = os.path.join(out, "bench", f"{m}_{mode}.json")
            if not os.path.exists(p):
                continue
            b = load(p)
            print(f"{m} {mode} (d{b['hidden']} L{b['layers']}, vocab {b['vocab']}, cache {b['max_seq']})")
            for r in b["prefixes"]:
                dc, df = r["decode_ms"]["cublas"]["p50"], r["decode_ms"]["fixed"]["p50"]
                hs = r["host_sample_ms"]["p50"]
                print(f"  P {r['prefix']}: decode step cublas {dc:.3f} ms, fixed {df:.3f} ms ({df / dc:.3f}x); "
                      f"host sampler {hs:.3f} ms")
                for lin, ref in (("rows", dc), ("fixed", df), ("smallm", dc)):
                    ms = [r["pass_ms"][lin][str(n)]["p50"] for n in range(1, 10)]
                    print(f"    pass {lin:6s} n=1..9 steps: " + " ".join(f"{x / ref:.2f}" for x in ms) +
                          f"   (ms: {ms[0]:.3f} .. {ms[-1]:.3f})")
                for dl in ("cublas", "fixed"):
                    s = " ".join(f"{lin} {v['rows_bit_identical']}/{v['rows']}" for lin, v in r["bits_n9"][dl].items())
                    print(f"    bits n=9, decode on {dl}: {s}")


def cat(name):
    n = name.lower()
    if "attn" in n:
        return "attention"
    if any(x in n for x in ("gemv", "splitkreduce", "linear_smallm", "linear_fixed", "gemm", "sgemm")):
        return "linear"
    return "rest"


def nsys(out):
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, "nsys", "*.json"))):
        j = load(f)
        tag = os.path.basename(f)[:-5]
        csvp = os.path.join(out, "nsys", tag + "_cuda_gpu_kern_sum.csv")
        if not os.path.exists(csvp):
            continue
        calls = j["profile"]["calls"]
        sums = {"linear": 0.0, "attention": 0.0, "rest": 0.0}
        launches = 0
        for r in csv.DictReader(open(csvp)):
            inst = int(r["Instances"])
            if inst < calls:
                continue  # the prefill's kernels
            sums[cat(r["Name"])] += float(r["Total Time (ns)"]) / calls / 1000.0
            launches += inst / calls
        rows[tag] = (sums, launches, j)
    for m in MODELS:
        bd = {}
        for mode in ("det",):
            p = os.path.join(out, "bench", f"{m}_{mode}.json")
            if os.path.exists(p):
                bd = {r["prefix"]: r for r in load(p)["prefixes"]}
        print(f"{m} (EMBER_DETERMINISTIC=1; kernel us per call inside the graph; wall from the bench, no nsys)")
        for tag in sorted(t for t in rows if t.startswith(m + "_")):
            sums, launches, j = rows[tag]
            pr = j["profile"]
            P = pr["prefix"]
            wall = None
            if P in bd:
                if pr["what"] == "decode":
                    wall = bd[P]["decode_ms"][pr["decode_linear"]]["p50"] * 1000
                else:
                    wall = bd[P]["pass_ms"][pr["linear"]][str(pr["n"])]["p50"] * 1000
            k = sum(sums.values())
            ws = f"wall {wall:7.1f}, outside kernels {wall - k:7.1f}" if wall else ""
            print(f"  {tag[len(m) + 1:]:20s} kernels {launches:6.0f}: linear {sums['linear']:7.1f}  attention "
                  f"{sums['attention']:6.1f}  rest {sums['rest']:6.1f}  sum {k:7.1f}  {ws}")


def summ(out, m, tag):
    p = os.path.join(out, "e2e", f"{m}_{tag}.jsonl.meta.json")
    return load(p)["summary"] if os.path.exists(p) else None


def e2e(out):
    for m in MODELS:
        base = {"h": summ(out, m, "h_off"), "hd": summ(out, m, "h_det_off"), "nd": summ(out, m, "n_det_off"),
                "s": summ(out, m, "s_off"), "sd": summ(out, m, "s_det_off")}
        if not base["h"] and not base["s"]:
            continue
        tags = [("s_off", "s"), ("s_off2", "s"), ("s_table_k8", "s"), ("s_merge_k8", "s"), ("s_det_off", "sd"),
                ("s_det_table_k8", "sd"), ("s_det_merge_k8", "sd")]
        if base["s"]:
            print(f"{m}: self-output tables, decode tok/s (x plain)")
            for tag, ref in tags:
                s = summ(out, m, tag)
                if s:
                    tpp = (s["accepted"] + s["passes"]) / s["passes"] if s["passes"] else 0
                    print(f"  {tag:16s} {s['decode_tok_per_s']:7.1f} ({s['decode_tok_per_s'] / base[ref]['decode_tok_per_s']:.3f})"
                          f"  acc {s['acceptance'] or 0:.3f}  tpp {tpp:.2f}  positions {s['decode_tokens'] - s['accepted']}"
                          f"/{s['decode_tokens']}  draft us/call {s['draft_us_per_call'] or 0:.1f}")
            for f in sorted(glob.glob(os.path.join(out, "e2e", f"{m}_s_*_vs_*.json"))):
                c = load(f)
                print(f"  ids {os.path.basename(f)[len(m) + 1:-5]:34s} {c['ids_identical']}/{c['n']}")
        if not base["h"]:
            continue
        print(f"{m}: decode tok/s (x the same mode and binary's plain) | acceptance | tokens per pass | positions")
        for tag, ref in (("h_off", "h"), ("h_off2", "h"), ("h_pld_k8", "h"), ("h_table_k8", "h"), ("n_off", "h"),
                         ("n_table_k8", "h"), ("h_det_off", "hd"), ("h_det_pld_k4", "hd"), ("h_det_pld_k8", "hd"),
                         ("h_det_table_k4", "hd"), ("h_det_table_k8", "hd"), ("n_det_off", "nd"),
                         ("n_det_off2", "nd"), ("n_det_pld_k4", "nd"), ("n_det_pld_k8", "nd"),
                         ("n_det_table_k4", "nd"), ("n_det_table_k8", "nd")):
            s = summ(out, m, tag)
            if not s:
                continue
            r = s["decode_tok_per_s"] / base[ref]["decode_tok_per_s"]
            extra = ""
            if ref == "nd":
                extra = f"  (x main's det plain {s['decode_tok_per_s'] / base['hd']['decode_tok_per_s']:.3f})"
            tpp = (s["accepted"] + s["passes"]) / s["passes"] if s["passes"] else 0
            pos = s["decode_tokens"] - s["accepted"]
            print(f"  {tag:16s} {s['decode_tok_per_s']:7.1f} ({r:.3f}){extra}  acc {s['acceptance'] or 0:.3f}  "
                  f"tpp {tpp:.2f}  positions {pos}/{s['decode_tokens']}  warm=rec {s['warm_pass_same']}/{s['requests']}")
        for f in sorted(glob.glob(os.path.join(out, "e2e", f"{m}_*_vs_*.json"))):
            c = load(f)
            print(f"  ids {os.path.basename(f)[len(m) + 1:-5]:34s} {c['ids_identical']}/{c['n']}"
                  + (f" differ {c['differ'][:6]}" if c["differ"] else ""))


def bits(out):
    for f in sorted(glob.glob(os.path.join(out, "bits", "*_vs_*.json"))):
        d = load(f)
        print(f"{os.path.basename(f)[:-5]:44s} bit_identical {d.get('bit_identical')}  ids {d['ids_identical']}/"
              f"{d['requests']}  rows {d['rows_bit_identical']}/{d['rows_compared']}  max|d| {d['max_abs_diff']:.3g}  "
              f"spec {d['other_spec']}")


def rows(out):
    for f in sorted(glob.glob(os.path.join(out, "rows", "*.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            run = os.path.basename(d["run"])[:-6]
            print(f"{os.path.basename(f)[:-6]:16s} {run:22s} req {d['requests_compared']}/{d['of']} rows {d['rows']} "
                  f"bit-equal {d['rows_bit_equal']}  abs p50 {d['abs']['p50']:.2g} p99 {d['abs']['p99']:.2g} max "
                  f"{d['abs']['max']:.3g}  rel max {d['rel']['max']:.3g}  worst {d['worst_row']}  by index "
                  + " ".join(f"{k}:{v:.2g}" for k, v in d["mean_abs_by_row_index"].items()))


def costs(out, m, mode, dst, P=384):
    b = load(os.path.join(out, "bench", f"{m}_{mode}.json"))
    r = next(x for x in b["prefixes"] if x["prefix"] == P)
    lin = "fixed" if mode == "det" else "smallm"
    dl = "fixed" if mode == "det" else "cublas"
    pm = {str(n): r["pass_ms"][lin][str(n)]["p50"] for n in range(1, 10)}
    # beyond 9 rows: linear from the last two
    for n in range(10, 17):
        pm[str(n)] = pm["9"] + (pm["9"] - pm["8"]) * (n - 9)
    c = {"source": f"{out} bench {m} {mode} P {P}", "step_ms": r["decode_ms"][dl]["p50"] + r["host_sample_ms"]["p50"],
         "pass_ms": pm}
    json.dump(c, open(dst, "w"), indent=1)
    print(json.dumps(c))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "costs":
        costs(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 384)
    else:
        {"bench": bench, "nsys": nsys, "e2e": e2e, "bits": bits, "rows": rows}[cmd](sys.argv[2])
