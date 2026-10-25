"""gpt2_tiny's IR with every gelu node replaced by expr (the tanh GELU as a formula, as
experiments/expr_2026-10-22), and a second variant whose final layernorm is a runtime-defined norm
over the row (rowmean, sqrt) followed by scale: the expr row reductions.   python3 expr_doc.py OUT_DIR"""
import json
import os
import sys

out = sys.argv[1]
doc = json.load(open("/mnt/c/ml_artifacts/gpt2_tiny/gpt2_tiny.ir.json"))
g = "0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))"
n = 0
for sec in ("embed", "block", "head"):
    for i, nd in enumerate(doc[sec]["nodes"]):
        if nd["op"] == "gelu":
            doc[sec]["nodes"][i] = {"op": "expr", "in": list(nd["in"]), "expr": g, "out": nd["out"]}
            n += 1
os.makedirs(out, exist_ok=True)
json.dump(doc, open(os.path.join(out, "gpt2_tiny_expr.ir.json"), "w"))
# The head's layernorm as expr + scale (same tensors: weight and bias of output_norm).
head = doc["head"]["nodes"]
for i, nd in enumerate(head):
    if nd["op"] == "layernorm":
        e = {"op": "expr", "in": list(nd["in"]),
             "expr": "(x - rowmean(x)) / sqrt(rowmean((x - rowmean(x))^2) + 1e-5)", "out": nd["out"] + "_n"}
        s = {"op": "scale", "in": [nd["out"] + "_n"], "weight": nd["weight"], "bias": nd["bias"], "out": nd["out"]}
        head[i:i + 1] = [e, s]
        break
json.dump(doc, open(os.path.join(out, "gpt2_tiny_exprnorm.ir.json"), "w"))
print("gelu nodes swapped:", n)
