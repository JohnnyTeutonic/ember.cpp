"""Gate of ember's block weight_index on paperkiln's recursion gate models (/mnt/c/ml_artifacts/recursion_gate,
paperkiln b1b620dd, tests/test_recursion_ir.py): for each model, ember's CPU interpreter (the IR embedded in
r.gguf) scores the 40 rows of score_in.jsonl. Gate: per-token log-probabilities of the continuation within
ROUNDING (1e-5 of max(1, max |ref|)) of mtstudio score's (score_out.jsonl), ember's argmax at every scored
position equal to the continuation (numpy's greedy, which mtstudio's argmax also equals): 40/40, and ember's
decode logits equal to its prefill logits."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = "/mnt/c/ml_artifacts/recursion_gate"
BIN = os.path.expanduser("~/ember_wt/build_cpu/ir_logits")
MODELS = ["p1_b2x3_c1", "p1_b2x3_c1_inject", "b2x4"]


def rows(path):
    return [json.loads(x) for x in open(path) if x.strip()]


report = {"bin": BIN, "models": {}}
ok_all = True
for m in MODELS:
    d = os.path.join(ROOT, m)
    gguf = os.path.join(d, "r.gguf")
    ins, outs = rows(os.path.join(d, "score_in.jsonl")), rows(os.path.join(d, "score_out.jsonl"))
    seqs = [r["prompt_ids"] + r["continuation_ids"] for r in ins]
    env = {k: v for k, v in os.environ.items() if k not in ("EMBER_IR",)}
    r = subprocess.run([BIN, gguf], input=json.dumps(seqs), capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(m + ": " + r.stderr[-1500:])
    em = json.loads(r.stdout)
    worst, greedy, dec_eq = 0.0, 0, True
    for k, (a, b) in enumerate(zip(ins, outs)):
        assert b["continuation_ids"] == a["continuation_ids"]
        pre = np.array(em[k]["prefill"], dtype=np.float64)
        dec_eq = dec_eq and bool(np.array_equal(np.array(em[k]["decode"]), np.array(em[k]["prefill"])))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P = len(a["prompt_ids"])
        cont = a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        am = [int(pre[P - 1 + j].argmax()) for j in range(len(cont))]
        greedy += am == cont
    ir = json.load(open(os.path.join(d, "r.ir.json")))
    passed = worst <= 1e-5 and greedy == len(ins) and dec_eq
    ok_all = ok_all and passed
    report["models"][m] = {"gguf_sha256": hashlib.sha256(open(gguf, "rb").read()).hexdigest(),
                           "rows": len(ins), "logprob_worst_rel": worst, "greedy": f"{greedy}/{len(ins)}",
                           "decode_eq_prefill": dec_eq, "applications": ir["block"]["repeat"],
                           "weight_index": ir["block"]["weight_index"]["per_layer"], "pass": passed}
report["pass"] = ok_all
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recursion_gate_cpu.json"), "w"),
          indent=1)
sys.exit(0 if ok_all else 1)
