"""Inputs of tools/mor_device_gate for the CUDA leg of Mixture-of-Recursions (run in WSL):
  mor3_in.json, mor2p0_in.json  paperkiln's MoR gate models (C:/ml_artifacts/mor_gate, the models of
                                ../recursion_2026-10-23's mor gate): the 40 score_in rows (prompt, continuation)
                                with mtstudio score's depth trace and log-probabilities (score_out) as references;
  tc_mor3_in.json               the tiny-chat MoR3 model (vocab 16384, ctx 512): 16 seeded random prompts of 24 ids
                                with 32-id random continuations (no reference; device against CPU only).
Usage: python3 make_inputs.py OUT_DIR"""
import json
import os
import sys

import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def rows(path):
    return [json.loads(x) for x in open(path) if x.strip()]


for m in ["mor3", "mor2p0"]:
    d = os.path.join("/mnt/c/ml_artifacts/mor_gate", m)
    ins, outs = rows(os.path.join(d, "score_in.jsonl")), rows(os.path.join(d, "score_out.jsonl"))
    assert [b["continuation_ids"] for b in outs] == [a["continuation_ids"] for a in ins]
    doc = {"prompt": [a["prompt_ids"] for a in ins], "continuation": [a["continuation_ids"] for a in ins],
           "depth": [b["depth"] for b in outs], "logprobs": [b["logprobs"] for b in outs]}
    json.dump(doc, open(os.path.join(out, f"{m}_in.json"), "w"))

rng = np.random.default_rng(1025)
V = 16384
doc = {"prompt": rng.integers(0, V, size=(16, 24)).tolist(), "continuation": rng.integers(0, V, size=(16, 32)).tolist()}
json.dump(doc, open(os.path.join(out, "tc_mor3_in.json"), "w"))
print("inputs in", out, sorted(os.listdir(out)))
