#!/usr/bin/env python3
"""ember's third-party GGUF gate on the CPU (LOCAL_LLM_STACK.md section 7), against the HF reference
written by tools/thirdparty_hf_reference.py.

    python3 tools/thirdparty_gate.py --ref REF.jsonl --port 8110 --out-dir DIR \
        [--hf-logits DIR --ember-logits DIR] [--ir-ref gguf_ir.json --ir-ember ember.ir.json]

Per prompt: ember's /chat (greedy: top_k 1, seed 0, return_ids, max_new_tokens = the reference's)
prompt ids against HF's templated ids, and its reply ids (+ the stop id) against HF's greedy ids;
with the logits directories, the first-token logits max |d| and its share of HF's logit range, and
top-1 agreement. With the two IR files, whether ember's built IR equals paperkiln's tools/gguf_ir.py
(every field but _comment). Writes DIR/gate.jsonl (one row per prompt, appended as each finishes;
a rerun resumes after the rows already there) and DIR/summary.json over all of them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def ir_diff(a, b, path=""):
    """Paths where two JSON values differ (numbers compared as floats)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k == "_comment" and not path:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'ember' if k in a else 'reference'}")
            else:
                out += ir_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += ir_diff(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return [] if float(a) == float(b) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-logits")
    ap.add_argument("--ember-logits")
    ap.add_argument("--ir-ref")
    ap.add_argument("--ir-ember")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    import numpy as np
    os.makedirs(a.out_dir, exist_ok=True)
    rows = [json.loads(x) for x in open(a.ref, encoding="utf-8")]
    url = f"http://127.0.0.1:{a.port}/chat"
    gate_path = os.path.join(a.out_dir, "gate.jsonl")
    out_rows = [json.loads(x) for x in open(gate_path, encoding="utf-8")] if os.path.exists(gate_path) else []
    done = {r["i"] for r in out_rows}
    f_out = open(gate_path, "a", encoding="utf-8")
    for r in rows:
        if r["i"] in done:
            continue
        max_new = 32
        e = post(url, {"user_input": r["prompt"], "max_new_tokens": max_new, "temperature": 1.0, "top_k": 1,
                       "top_p": 1.0, "seed": 0, "return_ids": True})
        ember_ids = list(e["ids"]) + ([e["stop_id"]] if e.get("stop_id", -1) >= 0 else [])
        row = {"i": r["i"], "prompt": r["prompt"], "prompt_ids_same": e["prompt_ids"] == r["prompt_ids"],
               "ember_prompt_tokens": len(e["prompt_ids"]), "hf_prompt_tokens": len(r["prompt_ids"]),
               "greedy_same": ember_ids == r["hf_ids"], "ember_ids": ember_ids, "hf_ids": r["hf_ids"],
               "first_diff": next((k for k, (x, y) in enumerate(zip(ember_ids, r["hf_ids"])) if x != y),
                                  None if len(ember_ids) == len(r["hf_ids"]) else min(len(ember_ids), len(r["hf_ids"]))),
               "stop_reason": e["stop_reason"], "ember_reply": e["reply"], "hf_reply": r["hf_reply"]}
        if a.hf_logits and a.ember_logits:
            h = np.fromfile(os.path.join(a.hf_logits, f"first_{r['i']}.f32"), dtype=np.float32)
            m = np.fromfile(os.path.join(a.ember_logits, f"first_{r['i']}.f32"), dtype=np.float32)
            d = float(np.abs(h - m).max())
            rng = float(h.max() - h.min())
            row.update({"first_max_dlogit": d, "first_range": rng, "first_rel": d / rng,
                        "top1_same": int(h.argmax()) == int(m.argmax())})
        out_rows.append(row)
        f_out.write(json.dumps(row) + "\n")
        f_out.flush()
        print(f"{r['i']}: ids {row['prompt_ids_same']} greedy {row['greedy_same']} "
              f"({len(ember_ids)} vs {len(r['hf_ids'])} tok) {row.get('first_rel', float('nan')):.2e}", flush=True)
    f_out.close()
    out_rows.sort(key=lambda r: r["i"])
    summ = {"label": a.label, "prompts": len(out_rows),
            "prompt_ids_same": sum(r["prompt_ids_same"] for r in out_rows),
            "greedy_same": sum(r["greedy_same"] for r in out_rows),
            "greedy_differs_at": {r["i"]: r["first_diff"] for r in out_rows if not r["greedy_same"]}}
    if a.hf_logits and a.ember_logits:
        summ.update({"top1_same": sum(r["top1_same"] for r in out_rows),
                     "worst_first_max_dlogit": max(r["first_max_dlogit"] for r in out_rows),
                     "worst_first_rel": max(r["first_rel"] for r in out_rows),
                     "median_first_rel": float(np.median([r["first_rel"] for r in out_rows]))})
    if a.ir_ref and a.ir_ember:
        diffs = ir_diff(json.load(open(a.ir_ember, encoding="utf-8")), json.load(open(a.ir_ref, encoding="utf-8")))
        summ["ir_equal_to_gguf_ir_py"] = not diffs
        summ["ir_differences"] = diffs
    json.dump(summ, open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
