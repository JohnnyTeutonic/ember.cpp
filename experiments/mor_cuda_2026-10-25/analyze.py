"""The receipt of the MoR CUDA session (run at home on the VM's out/): cuda_gate.json beside this file.

    python3 analyze.py /mnt/c/ml_artifacts/ember_mor_cuda/<session>/out

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


b = {}
for name in ("mor3", "mor3_q8_0", "mor2p0", "mor2p0_q8_0", "tc_mor3"):
    d = load(f"mdg_{name}.json")
    b[name] = {"missing": True, "pass": False} if d is None else {"summary": d["summary"], "bench": d.get("bench"),
                                                                    "pass": d["pass"]}
b["ir_device_gate"] = {n: dg(n) for n in ("mor3", "mor2p0", "tc_mor3")}
rep["b_mor"] = b

c = {"ir_device_gate": {n: dg(n) for n in ("syn_mistral_swa", "syn_gptneox", "syn_gpt2", "syn_mamba", "syn_gemma3",
                                           "syn_gemma3_freq_scale", "syn_gemma3_weight_offset", "gpt2_tiny_expr",
                                           "state4", "ar3", "rec_p1_b2x3_c1", "rec_p1_b2x3_c1_inject", "rec_b2x4",
                                           "qwen3_06b", "pythia_160m", "gemma3_270m", "qwen25_05b")}}
dumps = [x.split() for x in open(os.path.join(OUT, "dumps.txt")).read().split("\n") if x.strip()]
c["dumps_new_vs_old"] = {n: v for n, v in dumps}
c["dumps_all_identical"] = len(dumps) > 0 and all(v == "identical" for _, v in dumps)
rep["c_device_gate_set"] = c
rep["pass"] = (rep["a_kernel_test"]["pass"] and all(v["pass"] for k, v in b.items() if k != "ir_device_gate") and
               all(v["pass"] for v in b["ir_device_gate"].values()) and
               all(v["pass"] for v in c["ir_device_gate"].values()) and c["dumps_all_identical"])
json.dump(rep, open(os.path.join(HERE, "cuda_gate.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
