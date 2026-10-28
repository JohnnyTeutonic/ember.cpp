#!/usr/bin/env python3
"""Compare two dumps of float32 logit rows (e4_device_gate's DUMP.bin): bit-equal rows and the worst max |d| over
the reference row's range.  python3 e4_rows_cmp.py REF.bin OTHER.bin VOCAB OTHER_REPORT.json"""
import json
import struct
import sys

ref, oth, V, rep = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
a = open(ref, "rb").read()
b = open(oth, "rb").read()
n = len(a) // (4 * V)
assert len(a) == len(b) == n * 4 * V, (len(a), len(b))
equal = 0
worst = 0.0
for r in range(n):
    ra = a[r * 4 * V:(r + 1) * 4 * V]
    rb = b[r * 4 * V:(r + 1) * 4 * V]
    if ra == rb:
        equal += 1
        continue
    x = struct.unpack("<%df" % V, ra)
    y = struct.unpack("<%df" % V, rb)
    rng = max(x) - min(x)
    d = max(abs(p - q) for p, q in zip(x, y))
    worst = max(worst, d / rng if rng else d)
r = json.load(open(rep))
print(json.dumps({"rows": n, "bit_equal_rows": equal, "worst_rel_vs_native": worst,
                  "wasm_greedy_vs_hf": r.get("dev_greedy_vs_hf"), "wasm_worst_first_rel_hf": r.get("dev_worst_first_rel_hf")}))
