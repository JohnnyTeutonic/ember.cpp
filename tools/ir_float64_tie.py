#!/usr/bin/env python3
"""Each top-1 flip in an ir_device_gate receipt, settled against float64: which of the two tokens a
float64 evaluation of the same IR prefers, and by how much.

    python3 tools/ir_float64_tie.py --gguf MODEL.gguf --paperkiln PAPERKILN --gate receipt.json

For a flip at position t of prompt ids, the IR (gguf_ir.build, as ember builds it) runs on ids[:t+1]
in paperkiln's tools/ir_numpy.py twice: in float32 (the reference engine) and with every float32 read
as float64. Printed per flip: the CPU's and the device's top tokens, the CPU's margin and the device's
gap from the receipt, and the float64 and numpy-f32 logit differences (CPU's token minus the
device's). A flip is a near-tie when the float64 difference is within the engines' spread; the side
float64 agrees with is reported, not assumed.
"""
import argparse
import json
import os
import sys
import types

import numpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--paperkiln", required=True)
    ap.add_argument("--gate", required=True, help="an ir_device_gate receipt with rows[].flips")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.paperkiln, "tools"))
    import gguf_ir
    import ir_numpy
    meta, shapes = gguf_ir.read_meta(a.gguf)
    doc = gguf_ir.build(meta, shapes)
    W32 = ir_numpy.load_weights(a.gguf)
    W64 = {k: (v.astype(numpy.float64) if hasattr(v, "astype") else v) for k, v in W32.items()}
    np32 = ir_numpy.np
    np64 = types.ModuleType("np64")
    np64.__dict__.update({k: getattr(numpy, k) for k in dir(numpy)})
    np64.float32 = numpy.float64
    out = []
    for row in json.load(open(a.gate))["rows"]:
        for f in row.get("flips", []):
            ids = row["ids"][: f["pos"] + 1]
            ir_numpy.np = np32
            l32 = ir_numpy.forward(doc, W32, ids, last_only=True)[0].astype(numpy.float64)
            ir_numpy.np = np64
            l64 = ir_numpy.forward(doc, W64, ids, last_only=True)[0]
            ir_numpy.np = np32
            c, d = f["cpu_top"], f["dev_top"]
            rng = float(l64.max() - l64.min())
            out.append({"prompt": row["i"], "pos": f["pos"], "cpu_top": c, "dev_top": d,
                        "cpu_margin": f["cpu_margin"], "dev_logit_gap": f["dev_logit_gap"],
                        "float64_cpu_minus_dev": float(l64[c] - l64[d]),
                        "numpy32_cpu_minus_dev": float(l32[c] - l32[d]),
                        "float64_top": int(l64.argmax()), "float64_range": rng,
                        "float64_prefers": "cpu" if l64[c] > l64[d] else "device" if l64[d] > l64[c] else "tie"})
    print(json.dumps({"gguf": os.path.basename(a.gguf), "flips": out}, indent=1))


if __name__ == "__main__":
    main()
