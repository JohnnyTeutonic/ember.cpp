r"""The WebGPU leg of the tensor_index gate (Windows), ../webgpu_ir_2026-10-26/gates.py's method: tools/webgpu_parity/
ir_gate.mjs (deploy/wasm/web/webgpu/ir_webgpu.js) through swiftshader_run.py (Dawn on SwiftShader only; a monitor
kills the run if the process opens a context on any hardware adapter), against tig_wgref.sh's ember CPU references
(read from WSL as \\wsl.localhost\Ubuntu\home\jonat\tig\wgate). Runs, per variant (ar_rec, ar_rec_q8_0):
  NAME            the file's embedded (tagged) document against the tagged reference          (must pass)
  NAME_untagged   tig_gate_cpu.py's untagged document (--doc) against the untagged reference  (must pass)
  NAME_crossed    the untagged document against the tagged reference: the WebGPU path's logits move by the
                  untagged gap, so ir_gate.mjs must fail on the logit bound                  (must fail)
Receipts (summary.json, check.jsonl) are copied into webgpu/RUN/ beside this file; OUT/webgpu_gate.json sums them.
    set WEBGPU_NODE=...\webgpu\index.js; python tig_webgpu.py OUT_JSON
Under the CPU budget: python scripts/cpu_budget.py run --session ember --cores 2 -- python tig_webgpu.py ...
"""
import json
import os
import shutil
import subprocess
import sys

R = r"C:\Users\jonat\OneDrive\Documents\research_portfolio_complete\ember.cpp"
H = os.path.dirname(os.path.abspath(__file__))
U = r"\\wsl.localhost\Ubuntu\home\jonat\tig\wgate"
D = r"C:\ml_artifacts\attnres_gate_rec\ar_rec"
W = os.environ.get("WG_OUT", os.path.join(os.environ.get("TEMP", H), "tig_webgpu_runs"))
UT = U + r"\ar_rec_untagged.ir.json"
RUNS = []
for v in ("ar_rec", "ar_rec_q8_0"):
    g = D + "\\" + v + ".gguf"
    RUNS += [(v, g, v, None, True), (v + "_untagged", g, v + "_untagged", UT, True), (v + "_crossed", g, v, UT, False)]
out_json = sys.argv[1]
report = {"runs": {}}
ok_all = True
for name, gguf, ref, doc, must_pass in RUNS:
    out = os.path.join(W, name)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(R, r"tools\webgpu_parity\swiftshader_run.py"), "--", "node",
           os.path.join(R, r"tools\webgpu_parity\ir_gate.mjs"), gguf, U + "\\" + ref + r"\ref", out, "--first", "40"]
    if doc:
        cmd += ["--doc", doc]
    with open(os.path.join(W, name + ".log"), "a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    tail = open(os.path.join(W, name + ".log"), encoding="utf-8").read().strip().splitlines()[-1:]
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
    keep.update({"rc": rc, "reference": ref, "doc": "untagged" if doc else "embedded (tagged)",
                 "must_pass": must_pass, "monitor": tail[0] if tail else ""})
    keep["as_expected"] = bool(s) and bool(s.get("pass")) == must_pass and "swiftshader" in str(s.get("adapter", "")).lower()
    ok_all = ok_all and keep["as_expected"]
    report["runs"][name] = keep
    print(name, json.dumps(keep), flush=True)
report["pass"] = ok_all
json.dump(report, open(out_json, "w"), indent=1)
sys.exit(0 if ok_all else 1)
