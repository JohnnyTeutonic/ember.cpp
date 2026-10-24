"""Gather the scripts and the small receipts for ember.cpp/experiments/rmsnorm_2026-10-24 (written on ext4, then
copied into the repo from Windows) and write the combined receipt rmsnorm_gate.json."""
import glob
import json
import os
import shutil
import sys

S = os.path.dirname(os.path.abspath(__file__))
E_REPO = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/rmsnorm_2026-10-24"
E = os.path.expanduser("~/ember_wt/rmsnorm_exp")  # ext4; copied into the repo from Windows
O = "/mnt/c/ml_artifacts/ember_rmsnorm/ember-rms/out"
W = os.path.expanduser("~/ember_wt/shape_parity")
H = "/mnt/c/ml_artifacts/tiny_chat/e1/post_e1_L32_r1"
os.makedirs(os.path.join(E, "receipts", "vm"), exist_ok=True)
os.makedirs(os.path.join(E, "receipts", "home"), exist_ok=True)

# scripts, with the scratch folder they ran from rewritten to this folder
for f in ("vm.sh", "drive.sh", "stage.sh", "dumpprec.py", "sp_compare.py", "cmp_spot.py", "spot_cpu.sh", "parity_cpu.sh",
          "make_specs.py", "train.sh", "emulate_bug.py", "logit_gap.py", "analyze.py", "finalize.py"):
    s = open(os.path.join(S, f), encoding="utf-8").read().replace(S, E_REPO)
    open(os.path.join(E, f), "w", encoding="utf-8", newline="\n").write(s)

vm = os.path.join(E, "receipts", "vm")
for f in glob.glob(os.path.join(O, "rmsnorm_test_*.txt")) + glob.glob(os.path.join(O, "tc_*.jsonl")) + \
        glob.glob(os.path.join(O, "dumpreq_*.jsonl")) + [os.path.join(O, x) for x in ("hardware.txt", "models.sha256", "spot_first4.jsonl")]:
    shutil.copy(f, vm)
shutil.copy("/mnt/c/ml_artifacts/ember_rmsnorm/ember-rms/vm.log", vm)
for tag in ("pin", "pinfix", "head", "fix"):
    shutil.copy(os.path.join(O, f"spot_{tag}", "m", "gen.jsonl"), os.path.join(vm, f"spot_{tag}_gen.jsonl"))

home = os.path.join(E, "receipts", "home")
shutil.copy(os.path.join(W, "spot_cpu", "m", "gen.jsonl"), os.path.join(home, "spot_cpu_gen.jsonl"))
for run in ("run_sft_L32", "run_sp_L16", "run_L32_e1", "run_L8_e5"):
    for f in glob.glob(os.path.join(W, run, "*_cpu_*.jsonl")):
        shutil.copy(f, os.path.join(home, f"{run[4:]}_{os.path.basename(f)}"))
shutil.copy(os.path.join(W, "sp_L16.json"), os.path.join(home, "sp_L16_spec.json"))

main = json.load(open(os.path.join(W, "receipt_main.json")))
spot_cpu = json.load(open(os.path.join(W, "spot_cpu_vs_mtstudio.json")))
cpu_tc = {}
for run in ("sft_L32", "sp_L16", "L32_e1", "L8_e5"):
    for V in ("spot", "p1"):
        d = json.load(open(os.path.join(W, f"run_{run}", f"cmp_cpu_{V}.json")))
        cpu_tc[f"{run}_{V}"] = f"{d['replies_identical']}/{d['n']}"
emu = json.load(open(os.path.join(W, "emulate.json")))
gap = json.load(open(os.path.join(W, "logit_gap.json")))
sha = {}
for line in open(os.path.join(O, "models.sha256")):
    h, n = line.split()
    sha[n] = h
receipt = {
    "models": {"tc_sft_L32_r1.gguf": {"sha256": sha["tc_sft_L32_r1.gguf"], "shape": "d 192, 32 layers, 3 heads, d_ff 832",
                                      "source": H + "/sft (the post job's file; ember ran it in the failing spot-check)"},
               "sp_L16.gguf": {"sha256": sha["sp_L16.gguf"], "shape": "d 320, 16 layers, 5 heads, d_ff 864",
                               "source": "shape_20m_L16.json's arch, 2 CPU steps of mtstudio (receipts/home/sp_L16_spec.json)"},
               "tc_e1_L32_r1.gguf": {"source": "/mnt/c/ml_artifacts/tiny_chat/e1/pt_e1_L32_r1 (E1 L32 base)"},
               "tc_e5_s0_r1.gguf": {"source": "/mnt/c/ml_artifacts/tiny_chat/e5/pt_e5_s0_r1 (E5 L8 base, d 448)"}},
    "cpu_home": {"spot_check_ember_cpu_vs_mtstudio_records": {k: spot_cpu[k] for k in ("items", "items_identical", "replies", "replies_identical")},
                 "tinychat_40_ember_cpu_vs_mtstudio_cpu": cpu_tc},
    "cuda_T4": main,
    "numpy_emulation_of_the_bug": {k: emu[k] for k in ("n", "bug_eq_ember_cuda", "clean_eq_mtstudio")},
    "logits_at_divergence": gap,
}
json.dump(receipt, open(os.path.join(E, "rmsnorm_gate.json"), "w"), indent=1)
json.dump(emu, open(os.path.join(home, "emulate.json"), "w"), indent=1)
print(json.dumps({k: receipt[k] for k in ("cpu_home", "numpy_emulation_of_the_bug")}, indent=1))
