"""Smoke test of ember's `scale` op on gpt2_tiny: a scale node after each block's attention LayerNorm,
gain and bias borrowed from the block's ffn_norm (both [d]). Checks that it loads and runs on the
CPU, that prefill equals decode, and that ember's logits equal a numpy evaluation of the same IR
(paperkiln's ir_numpy with `scale` added here as (x * g) + b in float32)."""
import json
import os
import subprocess
import sys

import numpy as np

PK = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch"
GGUF = "/mnt/c/ml_artifacts/gpt2_tiny/gpt2_tiny.gguf"
BIN = os.path.expanduser("~/ember_wt/build_cpu/ir_logits")
sys.path.insert(0, os.path.join(PK, "tools"))
import gguf_ir  # noqa: E402
import ir_numpy  # noqa: E402

doc = json.load(open("/mnt/c/ml_artifacts/gpt2_tiny/gpt2_tiny.ir.json"))
nodes = doc["block"]["nodes"]
i = next(k for k, nd in enumerate(nodes) if nd["op"] == "layernorm")
h = nodes[i]["out"]
nodes.insert(i + 1, {"op": "scale", "in": [h], "weight": "blk.{i}.ffn_norm.weight", "bias": "blk.{i}.ffn_norm.bias",
                     "out": h})
path = os.path.expanduser("~/ember_wt/scale_smoke.ir.json")
json.dump(doc, open(path, "w"))

# numpy: ir_numpy without `scale`, so run it node by node with scale done here.
W = ir_numpy.load_weights(GGUF)
orig = ir_numpy.run_nodes


def run_nodes(ns, env, W_, i=None, no_window=False):
    for nd in ns:
        if nd["op"] == "scale":
            g = W_[nd["weight"].replace("{i}", str(i))]
            b = W_[nd["bias"].replace("{i}", str(i))]
            x = env[nd["in"][0]]
            env[nd["out"]] = ((x * g).astype(np.float32) + b).astype(np.float32)
        else:
            orig([nd], env, W_, i, no_window=no_window)


ir_numpy.run_nodes = run_nodes
seqs = [[5, 17, 300, 42, 9, 77, 128, 3], [11, 2, 250, 99, 1]]
r = subprocess.run([BIN, GGUF], input=json.dumps(seqs), capture_output=True, text=True,
                   env={**os.environ, "EMBER_IR": path})
if r.returncode:
    raise SystemExit(r.stderr[-1500:])
ember = json.loads(r.stdout)
r0 = subprocess.run([BIN, GGUF], input=json.dumps(seqs[:1]), capture_output=True, text=True)
base = json.loads(r0.stdout)[0]["prefill"]
out = {}
for k, ids in enumerate(seqs):
    ref = ir_numpy.forward(doc, W, ids)
    pre, dec = np.array(ember[k]["prefill"]), np.array(ember[k]["decode"])
    rng = ref.max(axis=-1) - ref.min(axis=-1)
    out[k] = {"vs_numpy_rel": float((np.abs(pre - ref).max(axis=-1) / rng).max()),
              "decode_eq_prefill": bool(np.array_equal(dec, pre)),
              "top1": int((pre.argmax(1) == ref.argmax(1)).sum()), "positions": len(ids)}
out["changed_vs_no_scale"] = float(np.abs(np.array(ember[0]["prefill"]) - np.array(base)).max())
print(json.dumps(out, indent=1))
