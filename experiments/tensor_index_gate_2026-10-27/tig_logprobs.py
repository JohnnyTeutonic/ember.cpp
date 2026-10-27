"""Log-probs of ar_rec's 40 continuations from a prefill's logits against mtstudio score's (score_out.jsonl):
worst |d| / max(1, max |ref|), and the argmax at every scored position = continuation (n/40); also against ember's
CPU prefill of the same file (tig_gate_cpu.py's ~/tig/work/cpu_<variant>_<doc>.json): max |dlogit| over the CPU
row's range, top-1 agreement at every position.
    python3 tig_logprobs.py wasm RUN_DIR CPU_JSON        (state_wasm_gate's run.json + logits.f32)
    python3 tig_logprobs.py device DEVICE_JSON CPU_JSON  (x5_device_logits' output: prefill, decode, free)"""
import json
import sys

import numpy as np

D = "/mnt/c/ml_artifacts/attnres_gate_rec/ar_rec"
kind, src, cpu_path = sys.argv[1:4]
ins = [json.loads(x) for x in open(D + "/score_in.jsonl") if x.strip()]
outs = [json.loads(x) for x in open(D + "/score_out.jsonl") if x.strip()]
cpu = [np.array(e["prefill"], dtype=np.float64) for e in json.load(open(cpu_path))]
res = {"kind": kind, "src": src, "cpu": cpu_path}
if kind == "wasm":
    run = json.load(open(src + "/run.json"))
    lg = np.fromfile(src + "/logits.f32", dtype=np.float32).reshape(-1, run["vocab"]).astype(np.float64)
    pres = []
    for r in run["rows"]:
        o, n = r["prefill_on"]
        pres.append(lg[o:o + n])
    res["engine"] = run["engine"]
else:
    dev = json.load(open(src))
    pres = [np.array(e["prefill"], dtype=np.float64) for e in dev]
    decs = [np.array(e["decode"], dtype=np.float64) for e in dev]
worst, greedy, vs_cpu, top1, n_rows, free, dec_rel = 0.0, 0, 0.0, 0, 0, 0, 0.0
for k, (a, b) in enumerate(zip(ins, outs)):
    pre = pres[k]
    lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
    P, cont = len(a["prompt_ids"]), a["continuation_ids"]
    lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
    ref = np.array(b["logprobs"])
    worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
    greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
    c = cpu[k]
    rng = c.max(1) - c.min(1)
    vs_cpu = max(vs_cpu, float((np.abs(pre - c).max(1) / rng).max()))
    top1 += int((pre.argmax(1) == c.argmax(1)).sum())
    n_rows += len(c)
    if kind == "device":
        free += dev[k]["free"] == cont
        dec_rel = max(dec_rel, float((np.abs(decs[k] - pre).max(1) / (pre.max(1) - pre.min(1))).max()))
res.update({"logprob_worst_rel_vs_mtstudio": worst, "greedy": f"{greedy}/{len(ins)}",
            "logits_worst_rel_vs_cpu": vs_cpu, "top1_vs_cpu": f"{top1}/{n_rows}"})
if kind == "device":
    res.update({"free_running_greedy": f"{free}/{len(ins)}", "decode_vs_prefill_rel": dec_rel})
print(json.dumps(res))
