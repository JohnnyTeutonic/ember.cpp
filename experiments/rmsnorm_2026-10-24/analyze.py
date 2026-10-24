"""Receipt of the d=192 CUDA prefill rmsnorm check: the VM's outputs (OUT) against the home references.
    analyze.py OUT HOME_RUNS SPOT_HOME ITEMS > receipt.json"""
import json
import os
import subprocess
import sys

import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
out, runs, spot_home, items = sys.argv[1:5]
sys.path.insert(0, S)
import sp_compare  # noqa: E402

R = {}

# 1. rmsnorm_batch_cuda vs a float64 reference, cuda_kernels.cu at each commit (T4, sm_75).
widths = [32, 64, 96, 128, 160, 192, 200, 255, 256, 320, 384, 448, 512, 640, 704, 832, 896, 1024]
k = {}
for c in ("b72ade8", "ef4b30f", "cbdcb91", "2886a97", "FIX"):
    txt = open(os.path.join(out, f"rmsnorm_test_{c}.txt")).read().splitlines()
    k[c] = {"summary": txt[-1], "failed_widths": [int(x.split()[1]) for x in txt if x.endswith("FAIL") and x.startswith("d ")]}
# bit equality of the outputs, per width, old vs fixed launcher
raw = {c: np.fromfile(os.path.join(out, f"rmsnorm_out_{c}.f32"), dtype=np.uint32) for c in ("2886a97", "FIX")}
off, eq = 0, {}
for d in widths:
    n = d * (1 + 7 + 40)
    eq[d] = bool(np.array_equal(raw["2886a97"][off:off + n], raw["FIX"][off:off + n]))
    off += n
k["bit_equal_2886a97_vs_fix"] = eq
R["kernel_test"] = k

# 2. The spot-check itself on the VM (carry5_eval ember leg, gpu_layers -1, EMBER_GEMM_TF32=0) vs mtstudio's
#    records from the post job's L4 (tiny_chat_run.compare_gens).
cmp = {}
for tag in ("pin", "pinfix", "head", "fix"):
    p = subprocess.run([sys.executable, os.path.join(S, "cmp_spot.py"), os.path.join(out, f"spot_{tag}", "m", "gen.jsonl"),
                        os.path.join(spot_home, "mtstudio", "m", "gen.jsonl"), items], capture_output=True, text=True)
    cmp[tag] = json.loads(p.stdout)
p = subprocess.run([sys.executable, os.path.join(S, "cmp_spot.py"), os.path.join(out, "spot_pin", "m", "gen.jsonl"),
                    os.path.join(spot_home, "ember_default", "m", "gen.jsonl"), items], capture_output=True, text=True)
cmp["pin_vs_post_job_ember_L4"] = json.loads(p.stdout)
R["spot_check_T4"] = cmp

# 3. The 40 frozen tinychat prompts, both request variants: VM engines vs mtstudio (CPU, home) and vs each other.
tc = {}
for model, home in (("sft", "run_sft_L32"), ("L16", "run_sp_L16")):
    for V in ("spot", "p1"):
        mts = os.path.join(runs, home, f"mtstudio_cpu_{V}.jsonl")
        row = {}
        for e in ("pin_cuda", "fix_cuda", "fix_cpu"):
            c = sp_compare.compare(os.path.join(out, f"tc_{model}_{e}_{V}.jsonl"), mts)
            row[f"{e}_vs_mtstudio"] = f"{c['replies_identical']}/{c['n']}"
        home_cpu = sp_compare.compare(os.path.join(runs, home, f"ember_cpu_{V}.jsonl"), mts)
        row["home_ember_cpu_vs_mtstudio"] = f"{home_cpu['replies_identical']}/{home_cpu['n']}"
        for a, b in (("pin_cuda", "fix_cuda"), ("fix_cuda", "fix_cpu")):
            c = sp_compare.compare(os.path.join(out, f"tc_{model}_{a}_{V}.jsonl"), os.path.join(out, f"tc_{model}_{b}_{V}.jsonl"))
            row[f"{a}_vs_{b}_ids"] = f"{c['ids_identical']}/{c['with_ids']}"
        tc[f"{model}_{V}"] = row
R["tinychat_prompts"] = tc
print(json.dumps(R, indent=1))
