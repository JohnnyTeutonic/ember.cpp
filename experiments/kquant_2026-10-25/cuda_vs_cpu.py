#!/usr/bin/env python3
"""CUDA (Colab T4, cuda_t4/) replies against the CPU receipts here, and the first-token flips against F32.
    cuda_vs_cpu.py [WORK]   WORK holds the CPU logit dumps (default ~/kq/work) for the flip table."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/kq/work")


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


out = {"replies_cuda_vs_cpu": {}, "first_token_flips_vs_f32": {}}
for tag in ["q4_k_m", "q4_k_m_ref", "q4_k_s", "q4_k_s_ref"]:
    c, p = rows(f"{HERE}/cuda_t4/cuda_{tag}_replies.jsonl"), rows(f"{HERE}/{tag}_replies.jsonl")
    out["replies_cuda_vs_cpu"][tag] = {
        "reply_identical": sum(a["reply"] == b["reply"] for a, b in zip(c, p)),
        "token_count_identical": sum(a["tokens"] == b["tokens"] for a, b in zip(c, p)),
        "stop_identical": sum(a["stop"] == b["stop"] for a, b in zip(c, p)), "n": len(c)}


def load(tag):
    return np.array([np.array(l.split(), dtype=np.float32) for l in open(f"{WORK}/{tag}.logits") if l.strip()])


if os.path.exists(f"{WORK}/f32.logits"):
    f = load("f32")
    srt = np.sort(f, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    for tag in ["q4_k_m", "q4_k_s"]:
        q = load(tag)
        flips = np.nonzero(f.argmax(1) != q.argmax(1))[0]
        out["first_token_flips_vs_f32"][tag] = [
            {"prompt": int(i), "f32_top1_margin": round(float(margin[i]), 4),
             "f32_rank_of_quant_top1": int((f[i] > f[i, q[i].argmax()]).sum()) + 1} for i in flips]
    out["f32_top1_margin_quantiles"] = {k: round(float(np.quantile(margin, k / 100)), 4) for k in (0, 10, 25, 50)}
json.dump(out, open(f"{HERE}/cuda_vs_cpu.json", "w"), indent=1)
print(json.dumps(out, indent=1))
