"""The CUDA receipt of the third-party-ops training gate (run at home in WSL on the VM's out/): cuda_gate.json.

    python3 tp_analyze.py VM_OUT_DIR CPU_WORK_DIR RECEIPT.json

Per GGUF (F32 and Q8_0), from tp_session_logits at gpu_layers -1 (the path and device the server takes, EMBER_GEMM_TF32=0):
  - path and device (cases 2-5: the interpreter on CUDA; tp_gqa: whichever the CPU leg took, on CUDA);
  - log-probs of the 40 continuations against mtstudio score's (worst |d| / max(1, max |ref|)), argmax at every scored
    position = continuation (teacher-forced greedy) and = mtstudio's argmax, free-running greedy = continuation;
  - device vs ember's CPU on the same path (tp_session_logits at gpu_layers 0, ~/tptrain/work): prefill and decode
    logits, max |dlogit| over the CPU row's range, top-1 at every position; device decode vs device prefill;
  - tools/state_device_gate and tools/ir_device_gate summaries (the interpreter, device vs CPU in one process);
  - tp_gqa: the interpreter forced on the device (EMBER_INTERPRETER=1) against the CPU interpreter, and the default
    TF32 GEMMs (reported, not gated).
Gate (ROUNDING 1e-5): log-probs and every logits comparison within 1e-5, greedy 40/40 both ways, top-1 complete,
state_device_gate and ir_device_gate pass; Q8_0: the same against the CPU on the Q8_0 file (log-probs reported)."""
import json
import os
import re
import sys

import numpy as np

OUT, CPUW, REC = sys.argv[1:4]
D = "/mnt/c/ml_artifacts/thirdparty_train_gate"
NAMES = ["tp_gqa", "tp_qwen", "tp_mqa_partial", "tp_neox_partial", "tp_mamba1"]
TOL = 1e-5


def rows(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def rel(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    rng = ref.max(1) - ref.min(1)
    return float((np.abs(x - ref).max(1) / np.where(rng > 0, rng, 1)).max())


def score(dev, cpu, ins, outs):
    n = len(ins)
    worst, tf, tf_m, free, dec_rel, pre_cpu, dec_cpu, top1, top1d, nrows = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0
    for k, (a, b) in enumerate(zip(ins, outs)):
        pre = np.array(dev[k]["prefill"], np.float64)
        dec = np.array(dev[k]["decode"], np.float64)
        dec_rel = max(dec_rel, rel(dec, pre))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        am = [int(pre[P - 1 + j].argmax()) for j in range(len(cont))]
        tf += am == cont
        tf_m += am == b["argmax"]
        free += dev[k]["free"] == cont
        c = np.array(cpu[k]["prefill"], np.float64)
        pre_cpu = max(pre_cpu, rel(pre, c))
        dec_cpu = max(dec_cpu, rel(dec, c))
        top1 += int((pre.argmax(1) == c.argmax(1)).sum())
        top1d += int((dec.argmax(1) == c.argmax(1)).sum())
        nrows += len(c)
    return {"logprob_worst_rel_vs_mtstudio": worst, "greedy_teacher_forced": f"{tf}/{n}",
            "argmax_eq_mtstudio": f"{tf_m}/{n}", "greedy_free_running": f"{free}/{n}",
            "decode_vs_prefill_worst_rel": dec_rel, "prefill_vs_cpu_worst_rel": pre_cpu,
            "decode_vs_cpu_worst_rel": dec_cpu, "top1_vs_cpu_prefill": f"{top1}/{nrows}",
            "top1_vs_cpu_decode": f"{top1d}/{nrows}"}


def ok(s, n, q8):
    full = f"{n}/{n}"
    r = (s["greedy_teacher_forced"] == full and s["greedy_free_running"] == full and s["decode_vs_prefill_worst_rel"] <= TOL
         and s["prefill_vs_cpu_worst_rel"] <= TOL and s["decode_vs_cpu_worst_rel"] <= TOL
         and s["top1_vs_cpu_prefill"].split("/")[0] == s["top1_vs_cpu_prefill"].split("/")[1]
         and s["top1_vs_cpu_decode"].split("/")[0] == s["top1_vs_cpu_decode"].split("/")[1])
    return bool(r and (q8 or s["logprob_worst_rel_vs_mtstudio"] <= TOL))


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


rep = {"session_out": OUT}
hw = open(os.path.join(OUT, "hardware.txt")).read()
rep["gpu"] = next((m.group(0) for m in (re.search(r"Tesla T4|NVIDIA L4", x) for x in hw.splitlines()) if m), None)
rep["nvcc"] = next((x.strip() for x in hw.splitlines() if "release" in x), None)
rep["bins_sha256"] = open(os.path.join(OUT, "bins.sha256")).read().strip().splitlines()
cpu_report = json.load(open(os.path.join(CPUW, "..", "gate_cpu.json"))) if os.path.exists(
    os.path.join(CPUW, "..", "gate_cpu.json")) else None
rep["models"] = {}
for name in NAMES:
    ins, outs = rows(os.path.join(D, name, "score_in.jsonl")), rows(os.path.join(D, name, "score_out.jsonl"))
    n = len(ins)
    m = {}
    for tag in (name, name + "_q8_0"):
        dev = load(os.path.join(OUT, f"dev_session_{tag}.json"))
        if dev is None and not os.path.exists(os.path.join(D, name, tag + ".gguf")):
            continue
        e = {}
        if dev is None:
            e["error"] = open(os.path.join(OUT, f"dev_session_{tag}.log")).read()[-600:]
            e["pass"] = False
            m[tag] = e
            continue
        cpu = json.load(open(os.path.join(CPUW, f"cpu_session_{tag}.json")))
        cpu_path = cpu_report["models"][name][tag]["path"] if cpu_report else None
        e["path"], e["device"], e["cpu_path"] = dev["path"], dev["device"], cpu_path
        e["served"] = score(dev["seqs"], cpu, ins, outs)
        q8 = tag.endswith("_q8_0")
        sdg = load(os.path.join(OUT, f"sdg_{tag}.json"))
        dg = load(os.path.join(OUT, f"dg_{tag}.json"))
        e["state_device_gate"] = {"summary": sdg["summary"], "pass": sdg["pass"]} if sdg else {"error": True}
        if dg:
            dg.pop("rows", None)
        e["ir_device_gate"] = dg or {"error": True}
        path_ok = dev["device"] == "cuda" and (dev["path"] == cpu_path if cpu_path else True) and (
            name == "tp_gqa" or dev["path"] == "interpreter")
        e["pass"] = bool(path_ok and ok(e["served"], n, q8) and sdg and sdg["pass"] and dg and dg["pass"])
        if name == "tp_gqa":
            di = load(os.path.join(OUT, f"dev_session_interp_{tag}.json"))
            ci = json.load(open(os.path.join(CPUW, f"cpu_ir_{tag}.json")))
            if di:
                e["interpreter_forced_on_device"] = {"path": di["path"], "device": di["device"],
                                                     **score(di["seqs"], ci, ins, outs)}
            dt = load(os.path.join(OUT, f"dev_session_tf32_{tag}.json"))
            if dt:
                e["default_tf32_reported"] = {"path": dt["path"], "device": dt["device"], **score(dt["seqs"], cpu, ins, outs)}
        m[tag] = e
        print(name, tag, e["path"], e["device"], json.dumps(e["served"]), "sdg", e["state_device_gate"].get("pass"),
              "dg", e["ir_device_gate"].get("pass"), "pass", e["pass"])
    m["pass"] = all(v["pass"] for v in m.values())
    rep["models"][name] = m
rep["pass"] = all(rep["models"][x]["pass"] for x in NAMES)
os.makedirs(os.path.dirname(os.path.abspath(REC)), exist_ok=True)
json.dump(rep, open(REC, "w"), indent=1)
print("pass", rep["pass"])
