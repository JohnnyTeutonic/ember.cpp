"""Synthetic recurrence op cases beside keramos's ops/recurrence.json, in its format (IR_SPEC 11.2: 8 hex digits
per f32 value), for the WebGPU recurrence gate (tools/webgpu_parity/recurrence_gate.mjs). They cover what the
suite's nine cases leave out: every function and operator in one node, an init body with bare indices, offsets
in both directions on a 3-D state, constant positions inside and outside a tensor, nested sums, numbers whose
f32 values overflow inside a body (exp(100)), the ^ forms (literal and non-literal exponents, negative bases)
and an index space wider than one workgroup. Expected values: keramos tools/ir_recurrence.py (`run`, float32),
used as a black-box oracle.

    python make_cases.py OUT.json
"""
import json
import os
import sys

import numpy as np

KERAMOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "keramos", "tools")
sys.path.insert(0, KERAMOS)
import ir_recurrence  # noqa: E402

T = 6
rng = np.random.default_rng(20261027)


def hexf(a):
    return "".join(f"{int(b):08x}" for b in np.asarray(a, np.float32).reshape(-1).view(np.uint32))


def tensor(a):
    a = np.asarray(a, np.float32)
    return {"shape": list(a.shape), "f32": hexf(a)}


CASES = [
    ("grammar_all", {
        "op": "recurrence", "in": ["x", "z"], "state": "g", "index": {"c": 3, "j": 4, "k": 2, "o": 1},
        "inputs": ["x[c]", "z[j]"], "weights": {"W[c,j]": "blk.{i}.w", "V[k,c,j]": "blk.{i}.v", "b[o]": "blk.{i}.b"},
        "init": ["h[c] = b[0] * c + 0.5"],
        "let": ["a[c,j] = W[c,j] * z[j] - x[c] / (abs(z[j]) + 1)", "m[c] = sum(j, sum(k, V[k,c,j] * a[c,j]))"],
        "update": ["h[c] = tanh(h[c - 1] + m[c]) * sigmoid(x[c]) + relu(-x[c]) ^ 2", "n[c] = n[c] + 1",
                   "q[j,c] = where(x[c] - z[j], erf(z[j]), q[j,c] * 0.5) + max(z[j], x[c]) - min(z[j], x[c])"],
        "readout": "y[c] = h[c] / sqrt(n[c]) + sum(j, q[j,c]) + pi * x[2] + z[3] + x[5] + min(exp(100), 1) * 1.0001 ^ n[c]"
                   " + log(n[c]) + exp(-abs(x[c]))",
        "out": "y"}, {"x": 3, "z": 4}, {"blk.0.w": (3, 4), "blk.0.v": (2, 3, 4), "blk.0.b": (1,)}),
    ("wide_2d", {
        "op": "recurrence", "in": ["x", "bb"], "state": "w", "index": {"c": 100, "n": 7},
        "inputs": ["x[c]", "b[n]"],
        "update": ["s[c,n] = 0.875 * s[c,n] + x[c] * b[n]"],
        "readout": "y[c] = sum(n, s[c,n] * b[n])",
        "out": "y"}, {"x": 100, "bb": 7}, {}),
    ("shift3d_init", {
        "op": "recurrence", "in": ["u"], "state": "sh", "index": {"a": 2, "b": 3, "c": 4},
        "inputs": ["u[c]"], "weights": {"S0[a,b,c]": "blk.{i}.s0"},
        "init": ["s[a,b,c] = S0[a,b,c] * 0.5 + a - c"],
        "update": ["s[a,b,c] = s[a,b + 1,c - 1] + u[c] * (a + 1) - s[a - 1,b,c + 2] / 4"],
        "readout": "y[a,c] = sum(b, s[a,b,c]) + s[1,2,3] + s[2,0,0] + u[c - 3]",
        "out": "y"}, {"u": 4}, {"blk.0.s0": (2, 3, 4)}),
    ("overflow_constants", {
        "op": "recurrence", "in": ["x"], "state": "ov", "index": {"c": 3},
        "inputs": ["x[c]"],
        "update": ["m[c] = m[c] + x[c]"],
        "readout": "y[c] = m[c] + min(exp(100), 1) - 1 / exp(100) + max(-exp(100), x[c]) + 3.4e38 * 0",
        "out": "y"}, {"x": 3}, {}),
    ("pow_forms", {
        "op": "recurrence", "in": ["x"], "state": "pw", "index": {"c": 5},
        "inputs": ["x[c]"],
        "update": ["p[c] = 0.5 * p[c] + x[c]"],
        "readout": "y[c] = p[c] ^ 2 + p[c] ^ 3 + abs(p[c]) ^ 0.5 + 2 ^ x[c] + (-2) ^ 3 + p[c] ^ -1 + (-1.5) ^ (2 + 0 * x[c]) + (-1.5) ^ (3 + 0 * x[c]) + p[c] ^ 4 ^ 1",
        "out": "y"}, {"x": 5}, {}),
    ("deep_sums", {   # 30 nested sums (the numpy reference broadcasts at most 32 dimensions; ember parses up to 50: refusal_check.mjs)
        "op": "recurrence", "in": ["x"], "state": "ds", "index": dict({"c": 2}, **{f"z{j}": 1 for j in range(30)}),
        "inputs": ["x[c]"],
        "update": ["m[c] = " + "".join(f"sum(z{j}, " for j in range(30)) + "m[c] * 0.5 + x[c]" + ")" * 30],
        "readout": "y[c] = m[c]",
        "out": "y"}, {"x": 2}, {}),
]


def main(out):
    cases = []
    for name, node, ins, ws in CASES:
        prog = ir_recurrence.compile_node(node)
        xs = {k: rng.standard_normal((T, w)).astype(np.float32) for k, w in ins.items()}
        wts = {k: rng.standard_normal(s).astype(np.float32) for k, s in ws.items()}
        decl = {t: d.split("[")[0] for d, t in node.get("weights", {}).items()}
        try:   # weights by declared name, else by tensor name
            y, st = ir_recurrence.run(prog, [xs[k] for k in node["in"]], {decl[t.replace("0", "{i}", 1)]: v for t, v in wts.items()},
                                      dtype=np.float32)
        except KeyError:
            y, st = ir_recurrence.run(prog, [xs[k] for k in node["in"]], {t.replace("0", "{i}", 1): v for t, v in wts.items()},
                                      dtype=np.float32)
        slot = ir_recurrence.slot_for(node, node["state"])
        cases.append({"name": name, "node": node, "slot": slot,
                      "inputs": {k: tensor(v) for k, v in xs.items()}, "weights": {k: tensor(v) for k, v in wts.items()},
                      "expect": {node["out"]: tensor(y)},
                      "class": "EXACT" if prog_exact(node) else "ROUNDING", "tol": 1e-5,
                      "expect_state": {node["state"]: {k: tensor(v) for k, v in st.items()}}})
    json.dump({"op": "recurrence", "_comment": __doc__.strip().splitlines()[0], "cases": cases},
              open(out, "w", encoding="utf-8"), indent=1)
    print(out, len(cases), "cases")


def prog_exact(node):
    text = " ".join(node.get("init", []) + node.get("let", []) + node["update"] + [node["readout"]])
    return not any(f + "(" in text.replace(" ", "") for f in ("exp", "log", "tanh", "erf", "sigmoid")) and "^" not in text


if __name__ == "__main__":
    main(sys.argv[1])
