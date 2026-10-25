"""The receipt of the MoR CUDA session (run at home on the VM's out/): cuda_gate.json beside this file.

    python3 analyze.py /mnt/c/ml_artifacts/ember_mor_cuda/<session>/out [first session's vm.log]

(a) tools/mor_kernels_test.cu's verdict; (b) mor_device_gate on mor3, mor2p0 (F32 and Q8_0) and the tiny-chat
MoR3 model (its timings), ir_device_gate on the three; (c) ir_device_gate on the device-gate set, and the device
dumps of the new and the old build (identical files)."""
import json
import os
import sys

OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))


def last_line(name):
    p = os.path.join(OUT, name)
    return open(p).read().strip().splitlines()[-1] if os.path.exists(p) else None


def load(name):
    p = os.path.join(OUT, name)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


rep = {"session_out": OUT}
hw = open(os.path.join(OUT, "hardware.txt")).read()
rep["gpu"] = next((x.split("|")[1].strip() for x in hw.splitlines() if "Tesla" in x or "NVIDIA L4" in x), None)
rep["a_kernel_test"] = {"mor_kernels_test": last_line("mor_kernels_test.txt")}
rep["a_kernel_test"]["pass"] = (rep["a_kernel_test"]["mor_kernels_test"] or "").startswith("mor_kernels_test: PASS")


def dg(name):
    d = load(f"dg_{name}.json")
    if d is None:
        return {"missing": True, "pass": False}
    return {k: d.get(k) for k in ("greedy_identical", "top1_agree", "worst_prefill_rel", "worst_decode_rel", "prompts",
                                  "len", "gen", "pass")}


def device_vs_cpu(s, tol=1e-4, lp_tol=1e-5):
    """mor_device_gate's verdict without its reference terms (depth_reference, logprob vs reference)."""
    full = lambda k: s[k].split("/")[0] == s[k].split("/")[1]  # noqa: E731
    return (all(s[k] <= tol for k in ("prefill_worst_rel", "decode_worst_rel", "two_call_worst_rel",
                                      "decode_vs_prefill_worst_rel", "rewind_worst_rel_vs_cpu",
                                      "rewind_worst_rel_vs_fresh_cpu")) and s["logprob_worst_rel_vs_cpu"] <= lp_tol and
            all(full(k) for k in ("top1_vs_cpu", "depth_prefill", "depth_decode", "depth_two_call",
                                  "depth_cpu_decode_equal_cpu_prefill", "rows_held_equal_cpu",
                                  "greedy_free_running_device_equal_cpu", "rewinds")) and
            s["greedy_teacher_forced_device"] == s["greedy_teacher_forced_cpu"] and
            s["greedy_free_running_device"] == s["greedy_free_running_cpu"])


# The Q8_0 files ran with the F32 models' mtstudio references (score_out is the F32 model's), which the CPU
# misses too (its log-probabilities move by the quantisation); they are gated device against CPU only.
b = {}
for name in ("mor3", "mor3_q8_0", "mor2p0", "mor2p0_q8_0", "tc_mor3"):
    d = load(f"mdg_{name}.json")
    if d is None:
        b[name] = {"missing": True, "pass": False}
        continue
    b[name] = {"summary": d["summary"], "bench": d.get("bench"), "tool_pass": d["pass"],
               "device_vs_cpu_pass": device_vs_cpu(d["summary"])}
    b[name]["pass"] = b[name]["device_vs_cpu_pass"] if name.endswith("q8_0") else d["pass"]
b["ir_device_gate"] = {n: dg(n) for n in ("mor3", "mor2p0", "tc_mor3")}
rep["b_mor"] = b

c = {"ir_device_gate": {n: dg(n) for n in ("syn_mistral_swa", "syn_gptneox", "syn_gpt2", "syn_mamba", "syn_gemma3",
                                           "syn_gemma3_freq_scale", "syn_gemma3_weight_offset", "gpt2_tiny_expr",
                                           "state4", "ar3", "rec_p1_b2x3_c1", "rec_p1_b2x3_c1_inject", "rec_b2x4")}}
dumps = [x.split() for x in open(os.path.join(OUT, "dumps.txt")).read().split("\n") if x.strip()]
c["dumps_new_vs_old"] = {n: v for n, v in dumps}

# The third-party legs, from the first session's log (its VM was lost before its out/ came home; the second
# session ran the local legs only). ir_device_gate by the real-model rule of ../thirdparty_2026-10-22/device_gate:
# greedy identical and top-1 at every position decide, not the flat 1e-4 of the pass field.
TP = ("qwen3_06b", "pythia_160m", "gemma3_270m", "qwen25_05b")
if len(sys.argv) > 2:
    log = open(sys.argv[2]).read().splitlines()
    tp = {}
    for line in log:
        w = line.split()
        if line.startswith("ir_device_gate ") and w[1] in TP:
            if len(w) >= 7:
                full = lambda f: f.split("/")[0] == f.split("/")[1]  # noqa: E731
                tp[w[1]] = {"greedy_identical": w[3], "top1_agree": w[4], "worst_prefill_rel": float(w[5]),
                            "worst_decode_rel": float(w[6]), "pass_field": w[8] == "True",
                            "pass": full(w[3]) and full(w[4])}
            else:
                tp[w[1]] = {"rc": w[2].rstrip(":"), "pass": None,
                            "note": "not run: the CPU half was killed by the VM's memory (rc 137); dump below"}
        if line.startswith("dump ") and w[1].rstrip(":") in TP:
            c["dumps_new_vs_old"][w[1].rstrip(":")] = w[2]
    c["thirdparty_session1"] = {"source": sys.argv[2], "ir_device_gate": tp}
c["dumps_all_identical"] = len(dumps) > 0 and all(v == "identical" for v in c["dumps_new_vs_old"].values())
rep["c_device_gate_set"] = c
tp_ok = all(v["pass"] is not False for v in c.get("thirdparty_session1", {}).get("ir_device_gate", {}).values())
rep["pass"] = (rep["a_kernel_test"]["pass"] and all(v["pass"] for k, v in b.items() if k != "ir_device_gate") and
               all(v["pass"] for v in b["ir_device_gate"].values()) and
               all(v["pass"] for v in c["ir_device_gate"].values()) and c["dumps_all_identical"] and tp_ok)
json.dump(rep, open(os.path.join(HERE, "cuda_gate.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
