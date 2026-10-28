"""The WebAssembly leg's numbers (e22_wasm_leg.sh's outputs) -> JSON on stdout.

    python3 e22_wasm_cmp.py LEG_DIR SERVER_ROWS.jsonl
"""
import json
import os
import sys

import numpy as np

O, ROWS = sys.argv[1], sys.argv[2]
rows = lambda p: [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]  # noqa: E731
F = ("reply", "tokens", "stop_reason", "prompt_ids", "ids", "stop_id")


def cmp(a, b):
    out = {f: sum(x.get(f) == y.get(f) for x, y in zip(a, b)) for f in F}
    out["n"] = len(a)
    out["differing_rows"] = [x["i"] for x, y in zip(a, b) if any(x.get(f) != y.get(f) for f in F)]
    return out


nat, was, srv = rows(os.path.join(O, "native", "greedy.jsonl")), rows(os.path.join(O, "wasm", "greedy.jsonl")), rows(ROWS)
S = {"wasm_vs_native_page_gate": cmp(was, nat), "native_page_gate_vs_server": cmp(nat, srv)}
expect = sum(r["tokens"] + (r["stop_reason"] != "length") for r in nat)
n_rows = same = 0
worst_abs = worst_rel = 0.0
with open(os.path.join(O, "native", "logits.txt")) as fa, open(os.path.join(O, "wasm", "logits.txt")) as fb:
    for la, lb in zip(fa, fb):
        n_rows += 1
        if la == lb:
            same += 1
            continue
        a, b = np.array(la.split(), dtype=np.float32), np.array(lb.split(), dtype=np.float32)
        d = np.abs(a.astype(np.float64) - b).max()
        worst_abs = max(worst_abs, float(d))
        worst_rel = max(worst_rel, float(d / (a.max() - a.min())))
S["decode_logits_wasm_vs_native"] = {"rows": n_rows, "rows_expected": expect, "rows_bit_equal": same,
                                     "max_abs_where_different": worst_abs, "max_over_row_range_where_different": worst_rel}
pp = os.path.join(O, "page_parity.json")
if os.path.exists(pp):
    d = json.load(open(pp))
    d.pop("replies", None)
    S["space_ember_js_vs_native_server"] = d
print(json.dumps(S, indent=1))
