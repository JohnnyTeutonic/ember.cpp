r"""The 26 Oct WebGPU IR gate list again (Windows), on the working tree's ir_webgpu.js: tools/webgpu_parity/ir_gate.mjs
per family through swiftshader_run.py, against ember CPU references made at ember HEAD by
../webgpu_ir_2026-10-26/ref.sh (BIN=~/e6rec/bin WORK=~/e6rec/regress), the families and documents of
../webgpu_ir_2026-10-26/gates.py (kimi_tiny and swa_tiny as copies relabelled "paperkiln" by keramos's
gguf_relabel.py: ember HEAD refuses their gpt2-labelled files, IR_SPEC 3.1 rule 38). Receipts into regress/<family>/ (summary.json, check.jsonl, run.log).
    WEBGPU_NODE=... python regress_gates.py FAM[:K] [...]     (K: the first K prompts; default the 26 Oct count)
"""
import json
import os
import shutil
import subprocess
import sys

R = r"C:\Users\jonat\OneDrive\Documents\research_portfolio_complete\ember.cpp"
H = os.path.dirname(os.path.abspath(__file__))
U = r"\\wsl.localhost\Ubuntu\home\jonat"
G = U + r"\e6rec\regress"
W = os.environ.get("WG_OUT", U + r"\e6rec\regress_runs")
A = r"C:\ml_artifacts"
T = A + r"\thirdparty"
FAM = {
    "kimi_tiny": (U + r"\e6rec\models\kimi_tiny.gguf", None, 40),
    "srd_tiny": (A + r"\srd_tiny\srd_tiny.gguf", None, 40),
    "attnres_tiny": (A + r"\attnres_tiny\attnres_tiny.gguf", None, 40),
    "flex_highway_tiny": (A + r"\flex_highway_tiny\flex_highway_tiny.gguf", None, 40),
    "mamba_tiny": (A + r"\mamba_tiny\mamba_tiny.gguf", None, 40),
    "gpt2_tiny": (A + r"\gpt2_tiny\gpt2_tiny.gguf", None, 40),
    "swa_tiny": (U + r"\e6rec\models\swa_tiny.gguf", None, 40),
    "moe_tiny_v3": (A + r"\moe_tiny\moe_tiny_v3.gguf", None, 40),
    "rec_b2x4": (A + r"\recursion_gate\b2x4\r.gguf", None, 40),
    "rec_p1_b2x3_c1": (A + r"\recursion_gate\p1_b2x3_c1\r.gguf", None, 40),
    "rec_p1_b2x3_c1_inject": (A + r"\recursion_gate\p1_b2x3_c1_inject\r.gguf", None, 40),
    "mor3": (A + r"\mor_gate\mor3\mor3.gguf", None, 40),
    "mor2p0": (A + r"\mor_gate\mor2p0\mor2p0.gguf", None, 40),
    "state4": (A + r"\state_gate\state4\state4.gguf", None, 40),
    "state4_q8_0": (A + r"\state_gate\state4\state4_q8_0.gguf", None, 40),
    "ar3": (A + r"\attnres_gate\ar3\ar3.gguf", None, 40),
    "ar3_q8_0": (A + r"\attnres_gate\ar3\ar3_q8_0.gguf", None, 40),
    "synthetic-gpt2": (U + r"\ember_wt\bcheck_gpt2\gpt2.gguf", U + r"\ember_wt\bcheck_gpt2\gpt2.ember_built.ir.json", 40),
    "synthetic-mistral-swa": (U + r"\ember_wt\bcheck_mistral_swa\mistral_swa.gguf",
                              U + r"\ember_wt\bcheck_mistral_swa\mistral_swa.ember_built.ir.json", 40),
    "pythia-160m": (T + r"\pythia-160m\pythia-160m-f32.gguf", U + r"\ember_wt\bcheck_pythia-160m\gptneox.ember_built.ir.json", 40),
    "mamba-130m": (T + r"\mamba-130m\mamba-130m-f32.gguf", U + r"\ember_wt\bcheck_mamba-130m\mamba.ember_built.ir.json", 40),
    "gpt2_tiny_expr": (A + r"\gpt2_tiny\gpt2_tiny.gguf", G + r"\docs\gpt2_tiny_expr.ir.json", 40),
    "gpt2_tiny_exprnorm": (A + r"\gpt2_tiny\gpt2_tiny.gguf", G + r"\docs\gpt2_tiny_exprnorm.ir.json", 40),
}
env = dict(os.environ)
for arg in sys.argv[1:]:
    fam, _, kk = arg.partition(":")
    gguf, doc, k = FAM[fam]
    k = int(kk) if kk else k
    out = os.path.join(W, fam)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(R, r"tools\webgpu_parity\swiftshader_run.py"), "--", "node",
           os.path.join(R, r"tools\webgpu_parity\ir_gate.mjs"), gguf, G + rf"\{fam}\ref", out, "--first", str(k)]
    if doc:
        cmd += ["--doc", doc]
    with open(os.path.join(W, fam + ".log"), "a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    s = {}
    try:
        s = json.load(open(os.path.join(out, "summary.json")))
    except (OSError, ValueError):
        pass
    dst = os.path.join(H, "regress", fam)
    os.makedirs(dst, exist_ok=True)
    for f in ("summary.json", "check.jsonl"):
        if os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(out, f), os.path.join(dst, f))
    shutil.copy(os.path.join(W, fam + ".log"), os.path.join(dst, "run.log"))
    print(fam, "rc", rc, {x: s.get(x) for x in ("greedy_identical", "top1_agree", "worst_rel", "median_rel", "pass",
                                               "dispatches_per_token", "ms_per_prompt")}, flush=True)
