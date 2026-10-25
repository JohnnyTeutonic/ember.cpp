"""Log-probs of ar_x5's 40 continuations from state_wasm_gate's WebAssembly prefill (DIR/run.json, DIR/logits.f32)
against mtstudio score's (score_out.jsonl): worst |d| / max(1, max |ref|), and ember's argmax = continuation.
    python3 x5_wasm_logprobs.py WASM_RUN_DIR"""
import json
import sys

import numpy as np

D = "/mnt/c/ml_artifacts/attnres_gate_x5/ar_x5"
d = sys.argv[1]
run = json.load(open(d + "/run.json"))
lg = np.fromfile(d + "/logits.f32", dtype=np.float32).reshape(-1, run["vocab"]).astype(np.float64)
ins = [json.loads(x) for x in open(D + "/score_in.jsonl") if x.strip()]
outs = [json.loads(x) for x in open(D + "/score_out.jsonl") if x.strip()]
worst, greedy = 0.0, 0
for r, a, b in zip(run["rows"], ins, outs):
    o, n = r["prefill_on"]
    pre = lg[o:o + n]
    lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
    P, cont = len(a["prompt_ids"]), a["continuation_ids"]
    lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
    ref = np.array(b["logprobs"])
    worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
    greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
print(json.dumps({"engine": run["engine"], "logprob_worst_rel_vs_mtstudio": worst, "greedy": f"{greedy}/{len(ins)}"}))
