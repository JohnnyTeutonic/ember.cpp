#!/usr/bin/env python3
"""Merge banked TTFT cells into experiment records and compare variants.

    python3 tools/ttft_report.py export --bank /mnt/c/ml_artifacts/ttft/bank \
        --variant baseline --out experiments/ttft_2026-10-17
    python3 tools/ttft_report.py compare experiments/ttft_2026-10-17/baseline \
        experiments/ttft_2026-10-17/flash_v1

export writes <out>/<variant>/summary.jsonl (one docs/TTFT.md record per
(model, run, length); a cell measured twice, e.g. by a relaunched VM, keeps
the first-banked row and the duplicate is listed in the manifest),
manifest.json (launch dirs, commits, GPU UUIDs, duplicates, missing cells)
and table.md (per length: mean, std, min, max of the per-run p50s).

compare reads two exported variants and applies the docs/TTFT.md rules per
(model, length): gain_i = 1 - variant_p50_i / baseline_p50_i over the runs
both have (run i of each lane is an independent fresh-server run; pairing by
index only orders them). Verdict: "within noise" if std(gain) > 50% of
|mean(gain)|; else "faster" / "slower". The fusion falsifier is reported
separately: |mean(gain)| <= 5% means "no TTFT change (+-5%)".
"""
import argparse
import glob
import json
import os
import statistics
import sys


def load_bank(bank):
    """{(model, run, len): row}, first-banked wins (launch dirs sort by time)."""
    cells, dups, launches = {}, [], []
    for launch in sorted(glob.glob(os.path.join(bank, "*"))):
        found = False
        for path in sorted(glob.glob(os.path.join(launch, "*", "summary.jsonl"))):
            model = os.path.basename(os.path.dirname(path))
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["_model_dir"] = model
                    row["_launch"] = os.path.basename(launch)
                    key = (model, row["run"], row["input_len"])
                    found = True
                    if key in cells:
                        dups.append({"cell": list(key), "kept": cells[key]["_launch"],
                                     "dropped": row["_launch"]})
                    else:
                        cells[key] = row
        if found:
            launches.append(os.path.basename(launch))
    return cells, dups, launches


def stats(xs):
    return {"n_runs": len(xs), "mean": statistics.mean(xs),
            "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs)}


def export(args):
    cells, dups, launches = load_bank(args.bank)
    out = os.path.join(args.out, args.variant)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "summary.jsonl")
    if os.path.exists(path) and not args.replace:
        sys.exit(f"{path} exists; pass --replace to regenerate it from the bank")
    with open(path, "w") as f:
        for key in sorted(cells):
            row = {k: v for k, v in cells[key].items() if not k.startswith("_")}
            row["model_dir"] = key[0]
            row["bank_launch"] = cells[key]["_launch"]
            f.write(json.dumps(row) + "\n")
    by = {}
    for (model, run, n), row in cells.items():
        by.setdefault((model, n), []).append(row["ttft_ms"]["p50"])
    lines = [f"# TTFT {args.variant}: per-run p50 (ms) across runs", "",
             "| model | tokens | runs | mean | std | min | max | deviation |",
             "|---|---|---|---|---|---|---|---|"]
    for (model, n) in sorted(by):
        s = stats(by[(model, n)])
        dev = {r["protocol_deviation"] for (m, _, ln), r in cells.items()
               if m == model and ln == n and r.get("protocol_deviation")}
        lines.append(f"| {model} | {n} | {s['n_runs']} | {s['mean']:.2f} | {s['std']:.2f} | "
                     f"{s['min']:.2f} | {s['max']:.2f} | {'reduced reps' if dev else ''} |")
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest = {
        "variant": args.variant, "bank": args.bank, "launches": launches,
        "commits": sorted({r["commit"] for r in cells.values()}),
        "gpu_uuids": sorted({r["gpu_uuid"] for r in cells.values()}),
        "kernel_variants": sorted({r["config"]["kernel_variant"] for r in cells.values()}),
        "duplicates_dropped": dups, "cells": len(cells),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("\n".join(lines))
    print(f"\n{len(cells)} cells from {len(launches)} launches; {len(dups)} duplicates dropped")


def compare(args):
    def read(d):
        rows = {}
        with open(os.path.join(d, "summary.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                rows[(r["model_dir"], r["run"], r["input_len"])] = r
        return rows

    base, var = read(args.baseline), read(args.variant)
    keys = sorted({(m, n) for (m, _, n) in base} & {(m, n) for (m, _, n) in var})
    print("| model | tokens | runs | baseline p50 | variant p50 | mean gain | std gain | verdict | ±5% falsifier |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model, n in keys:
        b = [base[k]["ttft_ms"]["p50"] for k in sorted(base) if k[0] == model and k[2] == n]
        v = [var[k]["ttft_ms"]["p50"] for k in sorted(var) if k[0] == model and k[2] == n]
        m = min(len(b), len(v))
        gains = [1 - v[i] / b[i] for i in range(m)]
        mg = statistics.mean(gains)
        sg = statistics.stdev(gains) if m > 1 else float("nan")
        # >= so a zero gain with zero spread counts as noise, not "slower".
        verdict = ("within noise (not recommended)" if m < 2 or sg >= 0.5 * abs(mg)
                   else ("faster" if mg > 0 else "slower"))
        fals = "no change" if abs(mg) <= 0.05 else "changed"
        print(f"| {model} | {n} | {m} | {statistics.mean(b):.2f} | {statistics.mean(v):.2f} | "
              f"{100 * mg:.1f}% | {100 * sg:.2f}% | {verdict} | {fals} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--bank", required=True)
    e.add_argument("--variant", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--replace", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("variant")
    args = ap.parse_args()
    {"export": export, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
