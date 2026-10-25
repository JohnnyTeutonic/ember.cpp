"""The receipt of the X5 gate's Colab session (run at home in WSL on the VM's out/): cuda_gate.json in RECEIPTS.

    python3 x5_analyze.py OUT_DIR CTL_OUT_DIR CPU_WORK_DIR RECEIPTS_DIR

  - the device and its compute capability; the pin's build (e115fd3, ensure_ember's cmake and make lines, sm_89);
  - ar_x5 and ar_x5_q8_0 on the device, e115fd3 and cbdcb915 + ce41ffb (x5_device_logits): log-probs of the 40
    continuations against mtstudio score's (worst |d| / max(1, max |ref|)), ember's argmax at every scored
    position equal to the continuation and to mtstudio's argmax (teacher-forced greedy), free-running greedy,
    decode against prefill, and the device against ember's CPU (max |dlogit| over the CPU row's range; argmax);
    state_device_gate and ir_device_gate summaries;
  - plain llama: x5_plain_compare.py's CUDA comparison (e115fd3 vs cbdcb915, same GPU), and each CUDA run against
    the same build's CPU run (replies; informative: CUDA is ROUNDING against the CPU);
  - the second session (CTL_OUT_DIR, x5_vm2.sh): each build twice, every pair compared at every sampled position,
    so the new-vs-old difference is set against each build's own run-to-run spread."""
import json
import os
import subprocess
import sys

import numpy as np

OUT, CTL, CPUW, REC = sys.argv[1:5]
D = "/mnt/c/ml_artifacts/attnres_gate_x5/ar_x5"
HERE = os.path.dirname(os.path.abspath(__file__))
ins = [json.loads(x) for x in open(os.path.join(D, "score_in.jsonl")) if x.strip()]
outs = [json.loads(x) for x in open(os.path.join(D, "score_out.jsonl")) if x.strip()]
N = len(ins)


def rel(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    rng = ref.max(1) - ref.min(1)
    return float((np.abs(x - ref).max(1) / np.where(rng > 0, rng, 1)).max())


def score(dev, cpu):
    worst, tf, tf_mts, free, dec_bit, dec_rel, vs_cpu, top1, rows = 0.0, 0, 0, 0, True, 0.0, 0.0, 0, 0
    for k, (a, b) in enumerate(zip(ins, outs)):
        pre = np.array(dev[k]["prefill"], dtype=np.float64)
        dec = np.array(dev[k]["decode"], dtype=np.float64)
        dec_bit = dec_bit and bool(np.array_equal(np.float32(dec), np.float32(pre)))
        dec_rel = max(dec_rel, rel(dec, pre))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        am = [int(pre[P - 1 + j].argmax()) for j in range(len(cont))]
        tf += am == cont
        tf_mts += am == b["argmax"]
        if "free" in dev[k]:
            free += dev[k]["free"] == cont
        c = np.array(cpu[k]["prefill"], dtype=np.float64)
        vs_cpu = max(vs_cpu, rel(pre, c))
        top1 += int((pre.argmax(1) == c.argmax(1)).sum())
        rows += len(c)
    return {"logprob_worst_rel_vs_mtstudio": worst, "greedy_teacher_forced": f"{tf}/{N}",
            "argmax_eq_mtstudio_argmax": f"{tf_mts}/{N}", "greedy_free_running": f"{free}/{N}",
            "decode_bit_equal_prefill": dec_bit, "decode_vs_prefill_worst_rel": dec_rel,
            "prefill_vs_cpu_worst_rel": vs_cpu, "top1_vs_cpu": f"{top1}/{rows}"}


rep = {"session_out": OUT}
hw = open(os.path.join(OUT, "hardware.txt")).read()
rep["gpu"] = next((x.split("|")[1].strip() for x in hw.splitlines() if "Tesla" in x or "NVIDIA L4" in x), None)
vm = open(os.path.join(OUT, "..", "vm.log")).read() if os.path.exists(os.path.join(OUT, "..", "vm.log")) else ""
rep["vm_log_build_lines"] = [x for x in vm.splitlines() if "built" in x or "build" in x.lower() and "rc=" in x
                             or "compute capability" in x or "cuBLAS" in x or "CMAKE_" in x or "new_server = pin" in x]
rep["pin_build"] = json.load(open(os.path.join(OUT, "pin_build.json")))
rep["pin_build"]["server_sha256"] = open(os.path.join(OUT, "pin_server.sha256")).read().split()[0]
rep["pin_build"]["cuda_found"] = "CUDA toolkit and cuBLAS found" in open(os.path.join(OUT, "cmake_pin.log")).read()
rep["pin_build"]["elf"] = open(os.path.join(OUT, "pin_server_elf.txt")).read().strip().splitlines()
rep["bins_sha256"] = open(os.path.join(OUT, "bins.sha256")).read().strip().splitlines()

a = {}
for t in ("new", "oldce"):
    for g in ("ar_x5", "ar_x5_q8_0"):
        p = os.path.join(OUT, f"dev_{t}_{g}.json")
        try:
            dev = json.load(open(p))
        except (OSError, ValueError):
            a[f"{t}_{g}"] = {"error": open(p.replace(".json", ".log")).read()[-500:]}
            continue
        cpu = json.load(open(os.path.join(CPUW, f"cpu_{t}_{g}.json")))
        a[f"{t}_{g}"] = score(dev, cpu)
for g in ("ar_x5", "ar_x5_q8_0"):
    d = json.load(open(os.path.join(OUT, f"sdg_{g}.json")))
    a[f"state_device_gate_{g}"] = {"summary": d["summary"], "pass": d["pass"]}
for t in ("new", "oldce"):
    d = json.load(open(os.path.join(OUT, f"dg_{t}_ar_x5.json")))
    d.pop("rows", None)
    a[f"ir_device_gate_{t}"] = d
a["new_vs_oldce_device_bit_equal"] = {g: open(os.path.join(OUT, f"dev_new_{g}.json")).read() ==
                                      open(os.path.join(OUT, f"dev_oldce_{g}.json")).read() for g in ("ar_x5", "ar_x5_q8_0")}
f = a["new_ar_x5"]
a["pass"] = bool(f["logprob_worst_rel_vs_mtstudio"] <= 1e-5 and f["greedy_teacher_forced"] == f"{N}/{N}"
                 and f["greedy_free_running"] == f"{N}/{N}" and f["decode_vs_prefill_worst_rel"] <= 1e-4
                 and f["prefill_vs_cpu_worst_rel"] <= 1e-4 and a["new_ar_x5_q8_0"]["greedy_teacher_forced"] == f"{N}/{N}"
                 and a["state_device_gate_ar_x5"]["pass"] and a["ir_device_gate_new"]["pass"])
rep["a_ar_x5_device"] = a

b = {"cuda_new_vs_old": json.load(open(os.path.join(OUT, "plain_cuda_new_vs_old.json")))}
for t in ("new", "old"):
    r = subprocess.run([sys.executable, os.path.join(HERE, "x5_plain_compare.py"), os.path.join(OUT, "plain", f"{t}_cuda"),
                        os.path.expanduser(f"~/x5g/plain/{t}_cpu")], capture_output=True, text=True)
    d = json.loads(r.stdout)
    b[f"cuda_vs_cpu_{t}"] = {s: {k: v[k] for k in ("n", "replies_identical", "ids_identical", "first_token_logits_worst_abs")}
                             for s, v in d["sets"].items()}
for s, v in b["cuda_new_vs_old"]["sets"].items():
    v.pop("divergent", None)  # every request is listed: the decode positions differ in rounding


def summary(v):
    e = v.get("every_position", {})
    return {k: v[k] for k in ("n", "replies_identical", "ids_identical", "prompt_ids_identical",
                              "first_token_logits_bit_equal", "all_positions_logits_bit_equal")} | \
        {"position_rows_bit_equal": f"{e['bit_equal_rows']}/{e['rows']}" if e else None,
         "worst_abs_every_position": e.get("worst_abs"), "top1_every_position": e.get("top1_agree")}


# The second session (x5_vm2.sh): each build run twice; every pair compared at every sampled position.
c = {"session_out": CTL}
for f in sorted(os.listdir(CTL)):
    if f.startswith("ctl_") and f.endswith(".json"):
        d = json.load(open(os.path.join(CTL, f)))
        c[f[4:-5]] = {s: summary(v) for s, v in d["sets"].items()}
pairs = [k for k in c if k != "session_out"]
same = [k for k in pairs if k.split("_")[0] == k.split("_vs_")[1].split("_")[0]]
cross = [k for k in pairs if k not in same]
worst = lambda ks: max(c[k][s]["worst_abs_every_position"] for k in ks for s in ("tinychat", "carry5"))  # noqa: E731
c["worst_abs_same_build"] = worst(same)
c["worst_abs_new_vs_old"] = worst(cross)
c["replies_and_ids_identical_every_pair"] = all(c[k][s]["replies_identical"] == c[k][s]["n"] == c[k][s]["ids_identical"]
                                                for k in pairs for s in ("tinychat", "carry5"))
c["fresh_prefill_first_token_bit_equal_every_pair"] = all(c[k]["tinychat"]["first_token_logits_bit_equal"] == 40 for k in pairs)
c["same_build_bit_reproducible"] = all(c[k][s]["all_positions_logits_bit_equal"] == c[k][s]["n"] for k in same
                                       for s in ("tinychat", "carry5"))
rep["c_cuda_run_to_run"] = c
# e115fd3 serves plain llama as cbdcb915 did: identical replies and ids in every pair, fresh-prompt prefill logits
# bit-equal, and the two builds no further apart than either is from itself (decode is not bit-reproducible).
b["pass"] = (c["replies_and_ids_identical_every_pair"] and c["fresh_prefill_first_token_bit_equal_every_pair"]
             and c["worst_abs_new_vs_old"] <= c["worst_abs_same_build"]
             and all(b[f"cuda_vs_cpu_{t}"][s]["replies_identical"] == b[f"cuda_vs_cpu_{t}"][s]["n"]
                     for t in ("new", "old") for s in ("tinychat", "carry5")))
rep["b_plain_llama_cuda"] = b
rep["pass"] = a["pass"] and b["pass"]
os.makedirs(REC, exist_ok=True)
json.dump(rep, open(os.path.join(REC, "cuda_gate.json"), "w"), indent=1)
print(json.dumps(rep, indent=1)[:6000])
