"""Gate of ember's Mixture-of-Recursions (depth_route, causal token choice, kv_rows "active") on paperkiln's MoR
gate models (/mnt/c/ml_artifacts/mor_gate, paperkiln c4427679): for each model, ember's CPU interpreter (the IR
embedded in the GGUF) runs the 40 contexts (prompt + continuation without its last token) as one prefill and
token by token. Gate: per-token log-probabilities of the continuation within ROUNDING (1e-5 of max(1, max |ref|))
of mtstudio score's (score_out.jsonl); ember's argmax at every scored position equal to the continuation (40/40
greedy); ember's depth at every position equal to mtstudio's depth trace, in prefill and in decode; decode logits
equal to prefill logits to ROUNDING. A depth mismatch is reported with the prompt's smallest router margin."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = "/mnt/c/ml_artifacts/mor_gate"
BIN = os.path.expanduser("~/ember_wt/build_cpu/ir_logits")
MODELS = ["mor3", "mor2p0"]


def rows(path):
    return [json.loads(x) for x in open(path) if x.strip()]


report = {"bin": BIN, "models": {}}
ok_all = True
for m in MODELS:
    d = os.path.join(ROOT, m)
    gguf = os.path.join(d, f"{m}.gguf")
    ins, outs = rows(os.path.join(d, "score_in.jsonl")), rows(os.path.join(d, "score_out.jsonl"))
    ctx = [a["prompt_ids"] + a["continuation_ids"][:-1] for a in ins]
    env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}
    r = subprocess.run([BIN, gguf], input=json.dumps(ctx), capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(m + ": " + r.stderr[-1500:])
    em = json.loads(r.stdout)
    worst, worst_dec, greedy, depth_pre, depth_dec = 0.0, 0.0, 0, 0, 0
    mismatches = []
    for k, (a, b) in enumerate(zip(ins, outs)):
        pre = np.array(em[k]["prefill"], dtype=np.float64)
        dec = np.array(em[k]["decode"], dtype=np.float64)
        rng = pre.max(axis=-1) - pre.min(axis=-1)
        worst_dec = max(worst_dec, float((np.abs(dec - pre).max(axis=-1) / rng).max()))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
        want = b["depth"]
        depth_pre += em[k].get("prefill_depth") == want
        depth_dec += em[k].get("decode_depth") == want
        if em[k].get("prefill_depth") != want or em[k].get("decode_depth") != want:
            mismatches.append({"row": k, "depth_margin_min": b["depth_margin_min"], "mtstudio": want,
                               "ember_prefill": em[k].get("prefill_depth"), "ember_decode": em[k].get("decode_depth")})
    n = len(ins)
    passed = worst <= 1e-5 and greedy == n and depth_pre == n and depth_dec == n and worst_dec <= 1e-5
    ok_all = ok_all and passed
    ir = json.load(open(os.path.join(d, f"{m}.ir.json")))
    report["models"][m] = {"gguf_sha256": hashlib.sha256(open(gguf, "rb").read()).hexdigest(),
                           "recursion": ir["block"]["_recursion"], "rows": n, "logprob_worst_rel": worst,
                           "greedy": f"{greedy}/{n}", "depth_prefill": f"{depth_pre}/{n}",
                           "depth_decode": f"{depth_dec}/{n}", "decode_vs_prefill_rel": worst_dec,
                           "smallest_router_margin": min(b["depth_margin_min"] for b in outs),
                           "depth_mismatches": mismatches, "pass": passed}
report["pass"] = ok_all
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mor_gate_cpu.json"), "w"), indent=1)
sys.exit(0 if ok_all else 1)
