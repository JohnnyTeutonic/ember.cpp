"""Two ir_device_dump files (HEAD's build and this work's) on one model: prefill, decode and greedy ids.

    python3 dump_compare.py HEAD.bin NEW.bin N LEN GEN     (prints one JSON object)

Each file holds, per prompt, the device's prefill logits (LEN rows), its token-by-token decode logits (LEN rows,
the captured graph after the first step) and GEN greedy ids (tools/ir_device_dump.cpp). The vocabulary follows
from the size. Per file: decode against prefill, worst |d| and worst |d| / max(1, max|prefill row|) (IR_SPEC 4.3's
ROUNDING scale; 1e-5 passes). Across the files: prefill bit-identical, decode rows bit-identical, worst |d|
between the two decodes, greedy ids identical.
"""
import json
import os
import sys

import numpy as np


def load(path, n, ln, gen):
    raw = np.fromfile(path, dtype=np.uint8)
    per = raw.size // n
    v = (per - 4 * gen) // (8 * ln)
    assert n * (8 * ln * v + 4 * gen) == raw.size, (path, raw.size)
    out = []
    for i in range(n):
        b = raw[i * per:(i + 1) * per]
        pre = b[:4 * ln * v].view(np.float32).reshape(ln, v)
        dec = b[4 * ln * v:8 * ln * v].view(np.float32).reshape(ln, v)
        ids = b[8 * ln * v:].view(np.int32)
        out.append((pre, dec, ids))
    return out, v


def dvp(recs):
    worst = worst_rel = 0.0
    for pre, dec, _ in recs:
        d = np.abs(dec.astype(np.float64) - pre)
        worst = max(worst, float(d.max()))
        scale = np.maximum(1.0, np.abs(pre.astype(np.float64)).max(axis=1))
        worst_rel = max(worst_rel, float((d.max(axis=1) / scale).max()))
    return {"decode_vs_prefill_worst_abs": worst, "decode_vs_prefill_worst_rounding_ratio": worst_rel}


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    n, ln, gen = (int(x) for x in sys.argv[3:6])
    a, va = load(a_path, n, ln, gen)
    b, vb = load(b_path, n, ln, gen)
    assert va == vb
    rows = sum(x[1].shape[0] for x in a)
    rep = {"head": os.path.basename(a_path), "new": os.path.basename(b_path), "prompts": n, "len": ln, "gen": gen,
           "vocab": va, "head_self": dvp(a), "new_self": dvp(b),
           "prefill_bit_identical": all(np.array_equal(x[0].view(np.uint32), y[0].view(np.uint32)) for x, y in zip(a, b)),
           "decode_rows": rows,
           "decode_rows_bit_identical": int(sum((x[1].view(np.uint32) == y[1].view(np.uint32)).all(axis=1).sum()
                                                for x, y in zip(a, b))),
           "decode_head_vs_new_worst_abs": float(max(np.abs(x[1].astype(np.float64) - y[1]).max() for x, y in zip(a, b))),
           "decode_argmax_equal": int(sum((x[1].argmax(axis=1) == y[1].argmax(axis=1)).sum() for x, y in zip(a, b))),
           "greedy_ids_identical": int(sum(np.array_equal(x[2], y[2]) for x, y in zip(a, b)))}
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
