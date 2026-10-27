r"""The WebGPU recurrence gate on whole documents (Windows): tools/webgpu_parity/ir_gate.mjs per document through
swiftshader_run.py (Dawn on SwiftShader only), against ref.sh's two references (read from WSL as
\\wsl.localhost\Ubuntu\home\jonat\e6rec\gate): ember CPU (ref/) and keramos's numpy reference (np/). Receipts
(summary.json, check.jsonl) into <name>/cpu/ and <name>/numpy/ in this folder. Resumes per prompt.
    WEBGPU_NODE=...\webgpu\index.js [WG_OUT=dir] python gates.py NAME[:cpu|:numpy] [...]
Under the CPU budget: python scripts/cpu_budget.py run --session ember --cores 2 -- python gates.py ...
"""
import json
import os
import shutil
import subprocess
import sys

R = r"C:\Users\jonat\OneDrive\Documents\research_portfolio_complete\ember.cpp"
KC = r"C:\Users\jonat\OneDrive\Documents\research_portfolio_complete\keramos\conformance"
H = os.path.dirname(os.path.abspath(__file__))
U = r"\\wsl.localhost\Ubuntu\home\jonat\e6rec"
W = os.environ.get("WG_OUT", U + r"\wg_runs")
A = r"C:\ml_artifacts"
D = os.path.join(H, "docs")
DOCS = {   # name: (GGUF, document or None for the file's own, prompts)
    "pk_recurrence": (KC + r"\models\pk_recurrence\pk_recurrence.gguf", None, 40),
    "pk_kimi_rec": (KC + r"\models\pk_kimi\pk_kimi.gguf", KC + r"\recurrence\pk_kimi.ir.json", 40),
    "pk_srd_rec": (KC + r"\models\pk_srd\pk_srd.gguf", KC + r"\recurrence\pk_srd.ir.json", 40),
    "pk_mamba_rec": (KC + r"\models\pk_mamba\pk_mamba.gguf", KC + r"\recurrence\pk_mamba.ir.json", 40),
    "tp_mamba1_rec": (KC + r"\models\tp_mamba1\tp_mamba1.gguf", KC + r"\recurrence\tp_mamba1.ir.json", 40),
    "kimi_tiny_rec": (U + r"\models\kimi_tiny.gguf", D + r"\kimi_tiny_rec.ir.json", 40),
    "srd_tiny_rec": (A + r"\srd_tiny\srd_tiny.gguf", D + r"\srd_tiny_rec.ir.json", 40),
    "mamba_tiny_rec": (A + r"\mamba_tiny\mamba_tiny.gguf", D + r"\mamba_tiny_rec.ir.json", 40),
    "kimi_tiny_custom": (U + r"\models\kimi_tiny.gguf", D + r"\kimi_tiny_custom.ir.json", 40),
    "mamba-130m_rec": (A + r"\thirdparty\mamba-130m\mamba-130m-f32.gguf", D + r"\mamba-130m_rec.ir.json", 10),
}
env = dict(os.environ)
for arg in sys.argv[1:]:
    name, _, which = arg.partition(":")
    gguf, doc, k = DOCS[name]
    for kind, sub in (("cpu", "ref"), ("numpy", "np")):
        if which and which != kind:
            continue
        ref = U + rf"\gate\{name}\{sub}"
        n = sum(1 for x in open(os.path.join(ref, "ref.jsonl")) if x.strip())
        out = os.path.join(W, name, kind)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(R, r"tools\webgpu_parity\swiftshader_run.py"), "--", "node",
               os.path.join(R, r"tools\webgpu_parity\ir_gate.mjs"), gguf, ref, out, "--first", str(min(k, n))]
        if doc:
            cmd += ["--doc", doc]
        with open(os.path.join(W, name, kind + ".log"), "a", encoding="utf-8") as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        s = {}
        try:
            s = json.load(open(os.path.join(out, "summary.json")))
        except (OSError, ValueError):
            pass
        dst = os.path.join(H, name, kind)
        os.makedirs(dst, exist_ok=True)
        for f in ("summary.json", "check.jsonl"):
            if os.path.exists(os.path.join(out, f)):
                shutil.copy(os.path.join(out, f), os.path.join(dst, f))
        shutil.copy(os.path.join(W, name, kind + ".log"), os.path.join(dst, "run.log"))
        print(name, kind, "rc", rc, {x: s.get(x) for x in ("greedy_identical", "top1_agree", "worst_rel", "median_rel", "pass",
                                                           "dispatches_per_token", "ms_per_prompt")}, flush=True)
