r"""The WebGPU IR gates (Windows): tools/webgpu_parity/ir_gate.mjs per family through swiftshader_run.py (Dawn on
SwiftShader only), against ref.sh's ember CPU references (read from WSL as \\wsl.localhost\Ubuntu\home\jonat\wgate).
Resumes per prompt; the receipts (summary.json, check.jsonl) are copied into this folder per family.
    WEBGPU_NODE=...\webgpu\index.js [SWIFTSHADER_ICD=dir] [WG_OUT=dir] python gates.py FAM [FAM ...]
Run it under the CPU budget: python scripts/cpu_budget.py run --session ember --cores 2 -- python gates.py ...
"""
import json
import os
import shutil
import subprocess
import sys

R = r"C:\Users\jonat\OneDrive\Documents\research_portfolio_complete\ember.cpp"
H = os.path.dirname(os.path.abspath(__file__))
W = os.environ.get("WG_OUT", os.path.join(H, "runs"))
U = r"\\wsl.localhost\Ubuntu\home\jonat"
A = r"C:\ml_artifacts"
T = A + r"\thirdparty"
FAM = {
    "kimi_tiny": (A + r"\kimi_tiny\kimi_tiny.gguf", None, 40),
    "srd_tiny": (A + r"\srd_tiny\srd_tiny.gguf", None, 40),
    "attnres_tiny": (A + r"\attnres_tiny\attnres_tiny.gguf", None, 40),
    "flex_highway_tiny": (A + r"\flex_highway_tiny\flex_highway_tiny.gguf", None, 40),
    "mamba_tiny": (A + r"\mamba_tiny\mamba_tiny.gguf", None, 40),
    "gpt2_tiny": (A + r"\gpt2_tiny\gpt2_tiny.gguf", None, 40),
    "swa_tiny": (A + r"\swa_tiny\swa_tiny.gguf", None, 40),
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
    "gemma-3-270m-it": (T + r"\gemma-3-270m-it\gemma-3-270m-it-f32.gguf",
                        U + r"\ember_wt\bcheck_gemma3_270m\gemma3.ember_built.ir.json", 40),
    "qwen2.5-0.5b-instruct": (T + r"\qwen2.5-0.5b-instruct\qwen2.5-0.5b-instruct-fp16.gguf",
                              U + r"\wgate\qwen2.5-0.5b-instruct\doc_built.json", 10),
    "gpt2_tiny_expr": (A + r"\gpt2_tiny\gpt2_tiny.gguf", U + r"\wgate\docs\gpt2_tiny_expr.ir.json", 40),
    "gpt2_tiny_exprnorm": (A + r"\gpt2_tiny\gpt2_tiny.gguf", U + r"\wgate\docs\gpt2_tiny_exprnorm.ir.json", 40),
    "qwen3-0.6b": (T + r"\qwen3-0.6b\Qwen3-0.6B-Q8_0.gguf", U + r"\wgate\qwen3-0.6b\doc_built.json", 10),
}
env = dict(os.environ)
for fam in sys.argv[1:]:
    gguf, doc, k = FAM[fam]
    out = os.path.join(W, fam)
    cmd = [sys.executable, os.path.join(R, r"tools\webgpu_parity\swiftshader_run.py"), "--", "node",
           os.path.join(R, r"tools\webgpu_parity\ir_gate.mjs"), gguf, U + rf"\wgate\{fam}\ref", out, "--first", str(k)]
    if doc:
        cmd += ["--doc", doc]
    with open(os.path.join(W, fam + ".log"), "a", encoding="utf-8") as log:
        os.makedirs(out, exist_ok=True)
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    s = {}
    try:
        s = json.load(open(os.path.join(out, "summary.json")))
    except (OSError, ValueError):
        pass
    os.makedirs(os.path.join(H, fam), exist_ok=True)
    for f in ("summary.json", "check.jsonl"):
        if os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(out, f), os.path.join(H, fam, f))
    print(fam, "rc", rc, {k2: s.get(k2) for k2 in ("greedy_identical", "top1_agree", "worst_rel", "median_rel", "pass", "ms_per_prompt")},
          flush=True)
