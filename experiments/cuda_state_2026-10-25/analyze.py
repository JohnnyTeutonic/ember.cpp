"""The receipt of the CUDA session (run at home on the VM's out/): cuda_gate.json beside this file.

    python3 analyze.py /mnt/c/ml_artifacts/ember_cuda_state/<session>/out

(a) the kernel tests' verdicts; the prefill test's outputs, new vs old cuda_kernels.cu, compared per head
dim bit for bit (equal at the powers of two, where the launch is unchanged); rmsnorm outputs new = old;
(b) state_device_gate summaries, state4's kv rows held against mtstudio's (manifest kv_rows_evict and
kv_rows_keep), ir_device_gate on state4 and ar3; (c) ir_device_gate on the synthetic families, and the
device dumps of the new and the old build (identical files); (d) SFT L32 (d 192) on CUDA through the new
server against mtstudio's CPU records and the T4 CUDA records of the rmsnorm fix (ember 69d7ca2)."""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
RMS = os.path.join(os.path.dirname(HERE), "rmsnorm_2026-10-24", "receipts")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rmsnorm_2026-10-24"))
from sp_compare import compare  # noqa: E402


def last_line(name):
    p = os.path.join(OUT, name)
    return open(p).read().strip().splitlines()[-1] if os.path.exists(p) else None


def lines(name):
    p = os.path.join(OUT, name)
    return open(p).read().strip().splitlines() if os.path.exists(p) else []


rep = {"session_out": OUT}
hw = open(os.path.join(OUT, "hardware.txt")).read()
rep["gpu"] = next((x.split("|")[1].strip() for x in hw.splitlines() if "Tesla" in x or "NVIDIA L4" in x), None)

# (a)
a = {}
for t in ("new", "head"):
    a[f"prefill_attention_test_{t}"] = last_line(f"prefill_attention_test_{t}.txt")
    a[f"prefill_attention_failed_head_dims_{t}"] = [int(x.split()[1]) for x in lines(f"prefill_attention_test_{t}.txt")
                                                    if x.startswith("head_dim") and x.endswith("FAIL")]
    a[f"rmsnorm_batch_test_{t}"] = last_line(f"rmsnorm_batch_test_{t}.txt")
HDS = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 200, 256]
new = np.fromfile(os.path.join(OUT, "prefill_out_new.f32"), dtype=np.uint32)
old = np.fromfile(os.path.join(OUT, "prefill_out_head.f32"), dtype=np.uint32)
eq, off = {}, 0
for hd in HDS:
    n = sum(T * hq * hd for hq in (4, 3) for _ in (0, 13) for T in (1, 7, 40))
    eq[hd] = bool(np.array_equal(new[off:off + n], old[off:off + n]))
    off += n
assert off == new.size == old.size
a["prefill_outputs_bit_equal_new_vs_old"] = eq
a["prefill_pow2_unchanged"] = all(eq[h] for h in HDS if h & (h - 1) == 0)
rn = open(os.path.join(OUT, "rmsnorm_out_new.f32"), "rb").read()
ro = open(os.path.join(OUT, "rmsnorm_out_head.f32"), "rb").read()
a["rmsnorm_outputs_bit_equal_new_vs_old"] = rn == ro
a["turn_state_attention_test"] = last_line("turn_state_attention_test.txt")
a["turn_state_attention_lines"] = lines("turn_state_attention_test.txt")
a["decode_attention_test"] = last_line("decode_attention_test.txt")
a["pass"] = (a["prefill_attention_test_new"].startswith("prefill_attention_test: PASS") and
             a["rmsnorm_batch_test_new"].startswith("rmsnorm_batch_test: PASS") and a["prefill_pow2_unchanged"] and
             a["rmsnorm_outputs_bit_equal_new_vs_old"] and
             a["turn_state_attention_test"].startswith("turn_state_attention_test: PASS"))
rep["a_kernel_tests"] = a

# (b)
man = json.load(open("/mnt/c/ml_artifacts/state_gate/state4/manifest.json"))["numpy_vs_mtstudio"]
b = {}
for name in ("state4", "state4_q8_0", "ar3", "synth_k4"):
    d = json.load(open(os.path.join(OUT, f"sdg_{name}.json")))
    b[name] = {"summary": d["summary"], "pass": d["pass"], "evicts_on": d["evicts_on"], "evicts_off": d["evicts_off"]}
    if name.startswith("state4"):
        on = [r["kv_rows_held"]["device_on"] for r in d["rows"]]
        offr = [r["kv_rows_held"]["device_off"] for r in d["rows"]]
        b[name]["kv_rows_evict_equal_mtstudio"] = on == man["kv_rows_evict"]
        b[name]["kv_rows_keep_equal_mtstudio"] = offr == man["kv_rows_keep"]
        b[name]["kv_rows_mean"] = {"evict": float(np.mean(on)), "keep": float(np.mean(offr))}
        b[name]["pass"] = b[name]["pass"] and b[name]["kv_rows_evict_equal_mtstudio"] and b[name]["kv_rows_keep_equal_mtstudio"]
for name in ("state4", "ar3"):
    d = json.load(open(os.path.join(OUT, f"dg_{name}.json")))
    d.pop("rows", None)
    b[f"ir_device_gate_{name}"] = d
b["pass"] = all(b[n]["pass"] for n in ("state4", "state4_q8_0", "ar3", "synth_k4")) and \
    all(b[f"ir_device_gate_{n}"]["pass"] for n in ("state4", "ar3"))
rep["b_state_and_attnres"] = b

# (c)
c = {}
for name in ("syn_mistral_swa", "syn_gptneox", "syn_gpt2", "syn_mamba", "syn_gemma3", "syn_gemma3_freq_scale",
             "syn_gemma3_weight_offset", "gpt2_tiny_expr"):
    d = json.load(open(os.path.join(OUT, f"dg_{name}.json")))
    c[name] = {k: d[k] for k in ("greedy_identical", "top1_agree", "worst_prefill_rel", "worst_decode_rel", "pass")}
dumps = dict(x.split() for x in lines("dumps.txt"))
c["device_dumps_new_vs_old"] = dumps
c["thirdparty_sha256"] = lines("thirdparty.sha256")
c["pass"] = all(c[n]["pass"] for n in c if isinstance(c[n], dict) and "pass" in c[n]) and \
    all(v == "identical" for v in dumps.values())
rep["c_device_gate_set"] = c

# (d)
d = {}
for v in ("spot", "p1"):
    mine = os.path.join(OUT, f"l32_new_cuda_{v}.jsonl")
    d[f"vs_mtstudio_cpu_{v}"] = compare(mine, os.path.join(RMS, "home", f"sft_L32_mtstudio_cpu_{v}.jsonl"))
    d[f"vs_fix_cuda_T4_{v}"] = compare(mine, os.path.join(RMS, "vm", f"tc_sft_fix_cuda_{v}.jsonl"))
    for k in list(d):
        d[k].pop("a", None)
        d[k].pop("b", None)
d["pass"] = all(d[k]["replies_identical"] == 40 for k in d)
rep["d_rmsnorm_L32"] = d
rep["pass"] = a["pass"] and b["pass"] and c["pass"] and d["pass"]
json.dump(rep, open(os.path.join(HERE, "cuda_gate.json"), "w"), indent=1)
print(json.dumps({k: rep[k]["pass"] if isinstance(rep[k], dict) else rep[k] for k in rep}, indent=1))
