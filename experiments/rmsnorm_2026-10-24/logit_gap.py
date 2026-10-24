"""Logits at the first divergent step: ember's TINYLLAMA_LOGITS_DUMP rows (one per sampled position, 9 digits)
from the VM for the first four spot-check items' first turns, against paperkiln's numpy reference
(tools/ir_numpy.py on the SFT GGUF and its IR) on the same ids.

    logit_gap.py OUT_DIR GGUF IR.json > logit_gap.json

For each request: the first step k where ember CUDA at the pinned commit (pin_cuda) and ember CUDA with the
fix (fix_cuda) differ from ember CPU (fix_cpu, the CPU path, which matches mtstudio's records), and at k and
at step 0 (the prompt's last position, the prefill row): every engine's max |logit - numpy| over the vocab
and its top-1 minus top-2 margin."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import ir_numpy  # noqa: E402

out, gguf, irp = sys.argv[1:4]
doc = json.load(open(irp))
W = ir_numpy.load_weights(gguf)
ENG = ("pin_cuda", "fix_cuda", "fix_cpu")


def margin(v):
    t = np.sort(v)[-2:]
    return float(t[1] - t[0])


res = []
for k in range(4):
    req, dump = {}, {}
    for e in ENG:
        req[e] = json.loads(open(os.path.join(out, f"dumpreq_{k}_{e}.jsonl")).readline())
        dump[e] = [np.array(x.split(), dtype=np.float64) for x in open(os.path.join(out, f"dump_{k}_{e}.txt")) if x.strip()]
    P = req["fix_cpu"]["prompt_ids"]
    assert all(req[e]["prompt_ids"] == P for e in ENG)
    ref_ids = req["fix_cpu"]["ids"]

    def first_div(ids):
        j = next((j for j, (a, b) in enumerate(zip(ids, ref_ids)) if a != b), None)
        if j is None and len(ids) != len(ref_ids):
            j = min(len(ids), len(ref_ids))  # one reply stopped (end-of-text) where the other went on
        return j

    row = {"item": k, "prompt_tokens": len(P), "replies": {e: req[e]["reply"] for e in ENG},
           "first_divergent_step_vs_cpu": {e: first_div(req[e]["ids"]) for e in ("pin_cuda", "fix_cuda")}}
    steps = sorted({0} | {s for s in row["first_divergent_step_vs_cpu"].values() if s is not None})
    row["at"] = {}
    for s in steps:
        ids = P + ref_ids[:s]
        num = ir_numpy.forward(doc, W, ids, last_only=True)[-1].astype(np.float64)
        at = {"numpy_top1": int(num.argmax()), "numpy_margin": margin(num)}
        for e in ENG:
            if s < len(dump[e]):
                v = dump[e][s]
                at[e] = {"max_abs_vs_numpy": float(np.abs(v - num).max()), "top1": int(v.argmax()), "margin": margin(v)}
        if s < len(dump["pin_cuda"]) and s < len(dump["fix_cpu"]):
            at["max_abs_pin_cuda_vs_fix_cpu"] = float(np.abs(dump["pin_cuda"][s] - dump["fix_cpu"][s]).max())
        if s < len(dump["fix_cuda"]) and s < len(dump["fix_cpu"]):
            at["max_abs_fix_cuda_vs_fix_cpu"] = float(np.abs(dump["fix_cuda"][s] - dump["fix_cpu"][s]).max())
        row["at"][str(s)] = at
    res.append(row)
print(json.dumps(res, indent=1))
