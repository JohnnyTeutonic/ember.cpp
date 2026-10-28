#!/usr/bin/env python3
"""Tables from tools/kv_quant_eval's receipts (R2, experiments/kv_quant_r2_2026-10-28).

    python3 tools/kvq_report.py RECEIPTS_DIR [--models qwen3,qwen25] > tables.md

RECEIPTS_DIR holds eval_<model>/{meta,tf_decode,tf_prefill,greedy,bench}.json. Per model, context C and
format: teacher-forced error over positions [0, C) (max |dlogit| / row range, its mean, top-1 agreement,
perplexity change; book and chat inputs), greedy prompts at C identical over 200 tokens (and the first
differing index of each), decode tokens/s and the KV cache's size at C.
"""
import argparse
import json
import os

NAMES = {"qwen3": "Qwen3-0.6B (Q8_0 file, f32 on the device)", "qwen25": "Qwen2.5-0.5B-Instruct (f16 file, f32 on the device)"}


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def tf_cells(tf, fmt, C):
    out = []
    for inp in tf or []:
        c = inp["formats"][fmt]["contexts"].get(str(C))
        if c:
            a = c["all"]
            out.append((inp["name"], a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--models", default="qwen3,qwen25")
    a = ap.parse_args()
    for m in a.models.split(","):
        d = os.path.join(a.dir, "eval_" + m)
        meta = load(d, "meta.json")
        if not meta:
            continue
        tfd, tfp, greedy, bench = (load(d, f + ".json") for f in ("tf_decode", "tf_prefill", "greedy", "bench"))
        fmts = [f for f in meta["kv"] if f != "f32"]
        print(f"### {NAMES.get(m, m)}\n")
        print(f"{meta['layers']} layers, kv capacity {meta['max_len']} positions; KV bytes per position: " +
              ", ".join(f"{f} {meta['kv'][f]['bytes_per_position']:.0f}" for f in meta["kv"]) + "\n")
        for label, tf in (("decode path (token by token, the captured graph)", tfd), ("prefill path (512-token chunks)", tfp)):
            if not tf:
                continue
            print(f"Teacher forcing, {label}: book / chat\n")
            print("| C | format | max err / range | mean err / range | top-1 agree | ppl f32 | ppl change |")
            print("|---|---|---|---|---|---|---|")
            for C in (2048, 4096, 8192):
                for f in fmts:
                    cells = tf_cells(tf, f, C)
                    if not cells:
                        continue
                    mx = " / ".join(f"{c['max_rel']:.2e}" for _, c in cells)
                    mean = " / ".join(f"{c['mean_rel']:.2e}" for _, c in cells)
                    top1 = " / ".join(f"{c['top1']}/{c['positions']}" for _, c in cells)
                    ppl = " / ".join(f"{c.get('ppl_f32', float('nan')):.3f}" for _, c in cells)
                    dppl = " / ".join(f"{100 * c.get('ppl_rel_diff', float('nan')):+.3f}%" for _, c in cells)
                    print(f"| {C} | {f} | {mx} | {mean} | {top1} | {ppl} | {dppl} |")
            print()
        if greedy:
            print("Greedy, 200 tokens after a prompt of about C tokens (identical to f32 / prompts; first differing index per prompt, - when identical)\n")
            print("| C | " + " | ".join(fmts) + " |")
            print("|---|" + "---|" * len(fmts))
            for C in (2048, 4096, 8192):
                rows = [g for g in greedy if g["ctx"] == C]
                if not rows:
                    continue
                cells = []
                for f in fmts:
                    same = sum(1 for g in rows if g["formats"][f]["identical"])
                    fd = ", ".join("-" if g["formats"][f]["first_diff"] < 0 else str(g["formats"][f]["first_diff"]) for g in rows)
                    cells.append(f"{same}/{len(rows)} ({fd})")
                print(f"| {C} | " + " | ".join(cells) + " |")
            print()
        if bench:
            print("Decode speed (median of 120 steps after a C-token prefill, tokens/s) and KV cache size at C\n")
            print("| C | " + " | ".join(["f32"] + fmts) + " |")
            print("|---|" + "---|" * (len(fmts) + 1))
            for b in bench:
                base = b["formats"]["f32"]["decode_tok_s"]
                cells = []
                for f in ["f32"] + fmts:
                    x = b["formats"][f]
                    sp = "" if f == "f32" else f", {x['decode_tok_s'] / base:.2f}x"
                    cells.append(f"{x['decode_tok_s']:.1f} tok/s{sp}; {x['kv_bytes_at_ctx'] / 2**20:.0f} MiB; prefill {x['prefill_ms']:.0f} ms")
                print(f"| {b['ctx']} | " + " | ".join(cells) + " |")
            print()
            print("Scratch per format (prefill's dequantised rows of one layer + decode partials): " +
                  ", ".join(f"{f} {meta['kv'][f]['scratch_bytes'] / 2**20:.0f} MiB" for f in fmts) + "\n")


if __name__ == "__main__":
    main()
