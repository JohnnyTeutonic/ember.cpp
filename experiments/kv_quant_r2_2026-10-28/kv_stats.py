#!/usr/bin/env python3
"""Channel statistics of real Qwen keys and values, and the reconstruction error of three quantisation
layouts on them (R2). Input: tools/kvq_dump_kv's file (the CPU interpreter's f32 kv rows after a prefill).

    python3 kv_stats.py KV.bin HEAD_DIM [--json OUT.json]

Layouts, each at 8 and 4 bits:
  token32   what EMBER_KV ships: blocks of 32 consecutive channels of one position (q8_0 / q4_1 rules, half
            scales);
  channel   one scale per (head, channel) over all positions (KIVI's key layout; InnerQ normalises the key
            channels before grouping), symmetric for 8 bits, asymmetric for 4;
  token128  one block per head per position (all head_dim channels).
Error: ||X - X'||_F / ||X||_F per layer, the mean over layers and the worst layer. Outliers: per (layer,
head) the largest channel's max |x| over the median channel's, its median and maximum over heads.
"""
import argparse
import json

import numpy as np


def half(x):
    return x.astype(np.float16).astype(np.float32)


def q_sym(x, axis, bits):
    qmax = 2 ** (bits - 1) - 1
    d = half(np.abs(x).max(axis=axis, keepdims=True) / np.float32(qmax))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.clip(np.rint(x / d), -qmax, qmax), 0)
    return (q * d).astype(np.float32)


def q_asym(x, axis, bits):
    qmax = 2 ** bits - 1
    mn, mx = x.min(axis=axis, keepdims=True), x.max(axis=axis, keepdims=True)
    d, m = half((mx - mn) / np.float32(qmax)), half(mn)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.clip(np.rint((x - m) / d), 0, qmax), 0)
    return (q * d + m).astype(np.float32)


def quant(X, layout, bits):
    """X [n, heads, hd]."""
    n, H, hd = X.shape
    f = q_sym if bits == 8 else q_asym
    if layout == "token32":
        return f(X.reshape(n, H, hd // 32, 32), 3, bits).reshape(n, H, hd)
    if layout == "token128":
        return f(X, 2, bits)
    if layout == "channel":
        return f(X, 0, bits)
    raise ValueError(layout)


def rel(X, Y):
    return float(np.linalg.norm(X - Y) / np.linalg.norm(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kv")
    ap.add_argument("head_dim", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    raw = open(a.kv, "rb").read()
    L, n, row = np.frombuffer(raw[:12], np.int32)
    data = np.frombuffer(raw[12:], np.float32).reshape(L, 2, n, row)
    H, hd = row // a.head_dim, a.head_dim
    out = {"layers": int(L), "positions": int(n), "kv_heads": int(H), "head_dim": int(hd)}
    for which, name in ((0, "K"), (1, "V")):
        ratios = []
        errs = {}
        for l in range(L):
            X = data[l, which].reshape(n, H, hd)
            cmax = np.abs(X).max(axis=0)  # [H, hd]
            ratios += list(cmax.max(1) / np.median(cmax, 1))
            for layout in ("token32", "token128", "channel"):
                for bits in (8, 4):
                    errs.setdefault(f"{layout}_q{bits}", []).append(rel(X, quant(X, layout, bits)))
        out[name] = {"outlier_ratio_median": float(np.median(ratios)), "outlier_ratio_max": float(np.max(ratios)),
                     "rel_error": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in errs.items()}}
        print(name, "channel max/median over heads: median %.1f, max %.1f" % (out[name]["outlier_ratio_median"],
                                                                             out[name]["outlier_ratio_max"]))
        for k, v in out[name]["rel_error"].items():
            print("  %-12s rel error mean %.4f max %.4f" % (k, v["mean"], v["max"]))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
