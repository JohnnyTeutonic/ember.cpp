"""E2.2 gate numbers from e22_gate.sh's outputs (~/e22rel/run) -> summary.json (and stdout).

    python3 e22_compare.py RUN_DIR MTSTUDIO_PARITY_CPU.jsonl OUT_SUMMARY.json
"""
import json
import os
import sys

import numpy as np

R, MTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def rows(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def greedy_cmp(a, b, fields=("reply", "tokens", "stop_reason", "prompt_ids", "ids", "stop_id")):
    n = len(a)
    out = {"n": n}
    for f in fields:
        out[f] = sum(x.get(f) == y.get(f) for x, y in zip(a, b))
    out["all_fields_40"] = all(out[f] == n for f in fields) and n == 40
    out["differing_rows"] = [x["i"] for x, y in zip(a, b) if any(x.get(f) != y.get(f) for f in fields)]
    return out


S = {}
f32 = rows(os.path.join(R, "f32_server.jsonl"))
q8 = rows(os.path.join(R, "q8_server.jsonl"))
mts = rows(MTS)

# 1. F32 ember (this build) against the recorded mtstudio greedy replies of the 25 Oct gate
S["f32_ember_vs_mtstudio_recorded"] = {
    "n": len(f32),
    "reply": sum(a["reply"] == m["mtstudio"] for a, m in zip(f32, mts)),
    "tokens": sum(a["tokens"] == m["mts_tokens"] for a, m in zip(f32, mts)),
    "stop_reason": sum(a["stop_reason"] == m["mts_stop"] for a, m in zip(f32, mts)),
    "prompt_tokens_vs_llama3_bpe": sum(a["prompt_tokens"] == m["bpe_py_prompt_tokens"] for a, m in zip(f32, mts)),
    "reply_vs_recorded_ember_cpu_aa50a61": sum(a["reply"] == m["ember"] for a, m in zip(f32, mts)),
    "distinct_replies": len({a["reply"] for a in f32}),
}

# 2. Q8 ember against F32 ember, greedy
S["q8_vs_f32_greedy"] = greedy_cmp(q8, f32)

# 3. teacher-forced logits on F32's trajectories
V = None
tf = {}
for tag in ("f32", "q8"):
    meta = json.load(open(os.path.join(R, f"tf_{tag}.f32.json")))
    V = meta[0]["vocab"]
    tf[tag] = (np.fromfile(os.path.join(R, f"tf_{tag}.f32"), dtype=np.float32).reshape(-1, V), meta)
A, meta = tf["f32"]
B, _ = tf["q8"]
assert A.shape == B.shape, (A.shape, B.shape)
rng = A.max(1) - A.min(1)
d = np.abs(A.astype(np.float64) - B.astype(np.float64))
rel_max = d.max(1) / rng
rel_mean = d.mean(1) / rng
srt = np.sort(A, axis=1)
margin = (srt[:, -1] - srt[:, -2]).astype(np.float64)
top1 = A.argmax(1) == B.argmax(1)
S["teacher_forced_logits_q8_vs_f32"] = {
    "positions": int(A.shape[0]), "vocab": V,
    "prompts": len(meta),
    "max_abs_dlogit": float(d.max()), "mean_abs_dlogit": float(d.mean()),
    "max_abs_over_row_range": float(rel_max.max()), "mean_abs_over_row_range": float(rel_mean.mean()),
    "max_abs_over_global_range": float(d.max() / (A.max() - A.min())),
    "top1_agree": int(top1.sum()),
    "f32_min_top1_margin": float(margin.min()),
    "f32_top1_margin_p01": float(np.quantile(margin, 0.01)),
    "max_abs_dlogit_at_smallest_margin_row": float(d[margin.argmin()].max()),
}
# per prompt offsets into the teacher-forced rows
off, o = [], 0
for m in meta:
    off.append(o)
    o += m["rows"]
dis = []
for k in S["q8_vs_f32_greedy"]["differing_rows"]:
    a, b = f32[k]["ids"], q8[k]["ids"]
    j = next((t for t in range(min(len(a), len(b))) if a[t] != b[t]), min(len(a), len(b)))
    r = off[k] + j   # the row predicting reply token j
    fa = int(a[j]) if j < len(a) else f32[k]["stop_id"]
    qb = int(b[j]) if j < len(b) else q8[k]["stop_id"]
    dis.append({"i": k, "first_divergent_token": j, "f32_token": fa, "q8_token": qb,
                "f32_margin_f32choice_minus_q8choice": float(A[r, fa] - A[r, qb]) if qb >= 0 and fa >= 0 else None,
                "q8_margin_q8choice_minus_f32choice": float(B[r, qb] - B[r, fa]) if qb >= 0 and fa >= 0 else None,
                "f32_top1_top2_margin": float(margin[r]), "max_abs_dlogit_there": float(d[r].max()),
                "f32_reply": f32[k]["reply"], "q8_reply": q8[k]["reply"]})
S["q8_vs_f32_divergences"] = dis

# 4. the page's entry points: WebAssembly against native, Q8
nat_p, wasm_p = os.path.join(R, "native_q8"), os.path.join(R, "wasm_q8")
if os.path.exists(os.path.join(wasm_p, "greedy.jsonl")):
    nat, was = rows(os.path.join(nat_p, "greedy.jsonl")), rows(os.path.join(wasm_p, "greedy.jsonl"))
    S["page_gate_wasm_vs_native_q8"] = greedy_cmp(was, nat)
    S["page_gate_native_vs_server_q8"] = greedy_cmp(nat, q8)
    ln = [line.split() for line in open(os.path.join(nat_p, "logits.txt"))]
    lw = [line.split() for line in open(os.path.join(wasm_p, "logits.txt"))]
    expect = sum(r["tokens"] + (r["stop_reason"] != "length") for r in nat)
    same_text = sum(x == y for x, y in zip(ln, lw))
    worst = 0.0
    for x, y in zip(ln, lw):
        if x != y:
            ax, ay = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
            worst = max(worst, float(np.abs(ax.astype(np.float64) - ay).max() / (ax.max() - ax.min())))
    S["page_gate_decode_logits_wasm_vs_native_q8"] = {
        "rows_native": len(ln), "rows_wasm": len(lw), "rows_expected": expect,
        "rows_bit_equal": same_text, "max_abs_over_row_range_where_different": worst}
pp = os.path.join(R, "page_parity_q8.json")
if os.path.exists(pp):
    d2 = json.load(open(pp))
    d2.pop("replies", None)
    S["space_ember_js_vs_native_server_q8"] = d2
json.dump(S, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(S, indent=1, ensure_ascii=False))
