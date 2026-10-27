r"""The WebGPU leg (Windows), ../tensor_index_gate_2026-10-27/tig_webgpu.py's method: tools/webgpu_parity/ir_gate.mjs
(deploy/wasm/web/webgpu/ir_webgpu.js) through swiftshader_run.py (Dawn on SwiftShader only; a monitor kills the run
if the process opens a context on any hardware adapter), per model (pk_flex_mdha, pk_flex_shaw from keramos v1.4.0's
git archive on ext4), against two references wasm_gate.sh wrote on WSL (\\wsl.localhost\Ubuntu\home\jonat\d2e\wasm):
  <m>_vs_native   ember's native CPU interpreter (ir_wasm_gate ref: 40 prompts of 8 ids, 8 greedy)
  <m>_vs_numpy    keramos's numpy reference on the same prompts (ref_numpy.py), judged at --tol 1e-5
With "chain" after OUT_JSON: the arXiv chain's Shaw model instead (chain_legs.sh's references in ...\wasm\chain, 40
prompts of 16 ids and 16 greedy, ctx 64).
Receipts (summary.json, check.jsonl) are copied into webgpu/<run>/ beside this file; OUT_JSON sums them.
    set WEBGPU_NODE=...\webgpu\index.js; python webgpu_gate.py OUT_JSON [chain]
Under the CPU budget: python scripts/cpu_budget.py run --session ember --cores 2 -- python webgpu_gate.py ...
"""
import json
import os
import shutil
import subprocess
import sys

R = r"C:\Users\jonat\OneDrive\Documents\research_portfolio_complete\ember.cpp"
H = os.path.dirname(os.path.abspath(__file__))
U = r"\\wsl.localhost\Ubuntu\home\jonat\d2e"
W = os.environ.get("WG_OUT", os.path.join(os.environ.get("TEMP", H), "d2_webgpu_runs"))
RUNS = []
CHAIN = len(sys.argv) > 2 and sys.argv[2] == "chain"
CTX = "64" if CHAIN else "16"
for m in (("chain",) if CHAIN else ("pk_flex_mdha", "pk_flex_shaw")):
    g = (r"C:\ml_artifacts\arxiv_chain_2026-10-27_d2\arxiv_1803_02155.gguf" if CHAIN else
         U + r"\keramos_v1.4.0\conformance\models" + "\\" + m + "\\" + m + ".gguf")
    RUNS += [(m + "_vs_native", g, U + r"\wasm" + "\\" + m + r"\native", "1e-4"),
             (m + "_vs_numpy", g, U + r"\wasm" + "\\" + m + r"\numpy", "1e-5")]
out_json = sys.argv[1]
report = {"runs": {}}
ok_all = True
for name, gguf, ref, tol in RUNS:
    out = os.path.join(W, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    cmd = [sys.executable, os.path.join(R, r"tools\webgpu_parity\swiftshader_run.py"), "--", "node",
           os.path.join(R, r"tools\webgpu_parity\ir_gate.mjs"), gguf, ref, out, "--first", "40", "--ctx", CTX,
           "--tol", tol]
    with open(os.path.join(W, name + ".log"), "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    lines = open(os.path.join(W, name + ".log"), encoding="utf-8").read().strip().splitlines()
    s = {}
    try:
        s = json.load(open(os.path.join(out, "summary.json")))
    except (OSError, ValueError):
        pass
    os.makedirs(os.path.join(H, "webgpu", name), exist_ok=True)
    for f in ("summary.json", "check.jsonl"):
        if os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(out, f), os.path.join(H, "webgpu", name, f))
    keep = {k: s.get(k) for k in ("adapter", "prompts", "len", "gen", "dispatches_per_token", "greedy_identical",
                                  "top1_agree", "worst_rel", "median_rel", "worst_spec_rel", "tol", "pass",
                                  "ms_per_prompt")}
    keep.update({"rc": rc, "monitor": lines[-1] if lines else "", "log_tail": lines[-4:]})
    keep["as_expected"] = bool(s) and bool(s.get("pass")) and "swiftshader" in str(s.get("adapter", "")).lower()
    ok_all = ok_all and keep["as_expected"]
    report["runs"][name] = keep
    print(name, json.dumps(keep), flush=True)
report["pass"] = ok_all
json.dump(report, open(out_json, "w"), indent=1)
sys.exit(0 if ok_all else 1)
