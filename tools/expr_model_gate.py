#!/usr/bin/env python3
"""Whole-model gate for the IR's `expr` op: the same weights with an op swapped for its expr formula.

    python3 tools/expr_model_gate.py --gguf M.gguf --ir M.ir.json --ref mtstudio_greedy.jsonl \
        --op gelu --expr "0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))" --bin build/tinyllama_server --out DIR

Every node with op --op (and, with --attr k=v, that attribute) becomes {"op": "expr", "in": ..., "expr":
--expr, "out": ...}. ember serves the model twice on the interpreter (EMBER_IR: the original IR, then the
swapped one) and records the next-token logits on the first 40 eval prompts (tools/quant_gate.py probe).
The swapped model must then reply exactly as the reference (--ref rows: prompt, mtstudio, mts_tokens,
mts_stop) on every prompt, with first-token logits within ROUNDING (max |a - b| / range <= --tol) of the
original op's. Writes DIR/ir_expr.json, the two logit dumps and DIR/gate.json; exit 1 unless it passes.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_gate  # noqa: E402
from release_chat import compare, ember, greedy_vs  # noqa: E402


def swap(ir, op, attr, text):
    n_swapped = 0
    for sub in ("embed", "block", "head"):
        part = ir.get(sub)
        nodes = part.get("nodes", []) if isinstance(part, dict) else (part or [])
        for i, n in enumerate(nodes):
            if n.get("op") != op or any(str(n.get(k)) != v for k, v in attr):
                continue
            nodes[i] = {"op": "expr", "in": list(n["in"]), "expr": text, "out": n["out"]}
            n_swapped += 1
    return n_swapped


def logits(binary, gguf, port, ir_path, dump):
    if os.path.exists(dump):
        os.remove(dump)
    env = {"EMBER_IR": ir_path, "EMBER_PREFIX_REUSE": "0", "TINYLLAMA_LOGITS_DUMP": dump}
    with ember(binary, gguf, port, 0, env):
        quant_gate.probe(port, 40)
    return quant_gate.load(dump)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--attr", action="append", default=[], help="k=v: only nodes with this attribute value")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ir = json.load(open(a.ir))
    n = swap(ir, a.op, [kv.split("=", 1) for kv in a.attr], a.expr)
    if n == 0:
        sys.exit(f"no '{a.op}' node to swap in {a.ir}")
    swapped = os.path.join(a.out, "ir_expr.json")
    json.dump(ir, open(swapped, "w"), indent=1)
    base = logits(a.bin, a.gguf, a.port, os.path.abspath(a.ir), os.path.join(a.out, "logits_op.txt"))
    mine = logits(a.bin, a.gguf, a.port, os.path.abspath(swapped), os.path.join(a.out, "logits_expr.txt"))
    rel = float(max(np.abs(r - s).max() / (r.max() - r.min()) for r, s in zip(base, mine)))
    top1 = int(sum(r.argmax() == s.argmax() for r, s in zip(base, mine)))
    ref = [json.loads(line) for line in open(a.ref) if line.strip()]
    with ember(a.bin, a.gguf, a.port, 0, {"EMBER_IR": os.path.abspath(swapped), "EMBER_PREFIX_REUSE": "0"}):
        greedy = compare(greedy_vs(a.port, ref))
    result = {"op": a.op, "expr": a.expr, "nodes_swapped": n, "first_token_max_rel": rel,
              "first_token_top1": f"{top1}/{len(base)}", "greedy_vs_ref": greedy,
              "pass": greedy["pass"] and rel <= a.tol and top1 == len(base)}
    json.dump(result, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
