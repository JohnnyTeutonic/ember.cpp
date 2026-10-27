r"""The page's chat path (Windows): tools/webgpu_parity/chat_test.mjs per model through swiftshader_run.py, WebGPU
(the path the worker picks: ir_webgpu.js for every document here but chat_sftfinal_v3's, which ember serves on its
tuned llama path) against WebAssembly (ember.js built at ember HEAD by deploy/wasm/build.sh: EMBER_JS) and, where
there is one, a CPU fixture. Greedy.
- Recurrence: kimi_tiny, srd_tiny and mamba_tiny with their recurrent ops as recurrences, and kimi_tiny_custom
  (make_docs.py), each document embedded in a copy of its GGUF (gguf_doc.py put; the page's WebAssembly reads the
  file's own document): the 40 page prompts (60 tokens; the rewrites against mtstudio's greedy replies of the
  original files, which ember CPU equals 40/40, ../families_2026-10-20) and 20 multi-turn requests
  (chat_multiturn.jsonl, make_chats.py).
- Regression (../webgpu_ir_2026-10-26's chat list): chat_sftfinal_v3 (llama_webgpu.js), state4's multi-turn
  requests with state blocks, and the seven paperkiln families against mtstudio.
Receipts into chat/<name>.json and regress/chat/<name>.json.
    WEBGPU_NODE=... EMBER_JS=...\ember.js PAGE_DIR=dir (prompts.json, greedy.json) python chats.py NAME [NAME ...]
"""
import json
import os
import shutil
import subprocess
import sys

R = r"C:\Users\jonat\OneDrive\Documents\research_portfolio_complete\ember.cpp"
H = os.path.dirname(os.path.abspath(__file__))
U = r"\\wsl.localhost\Ubuntu\home\jonat"
W = os.environ.get("WG_OUT", U + r"\e6rec\chat_runs")
A = r"C:\ml_artifacts"
EMBER = os.environ["EMBER_JS"]
PAGE = os.environ["PAGE_DIR"]
os.makedirs(W, exist_ok=True)


def mts_fixture(fam):   # mtstudio's 40 greedy replies of the original file (ember CPU's equal them 40/40)
    out = os.path.join(W, fam + "_mtstudio.json")
    rows = [json.loads(x) for x in open(os.path.join(A, fam, "mtstudio_greedy.jsonl"), encoding="utf-8") if x.strip()]
    json.dump({"replies": [{"reply": r["mtstudio"].strip()} for r in rows]}, open(out, "w", encoding="utf-8"))
    return out


PROMPTS = os.path.join(PAGE, "prompts.json")
MULTI = os.path.join(H, "chat_multiturn.jsonl")
M = U + r"\e6rec\models"
RELABELLED = ("kimi_tiny", "swa_tiny")   # copies relabelled "paperkiln" (gguf_relabel.py): ember HEAD refuses the gpt2 label (rule 38)
MODELS = {   # name: (GGUF, requests, fixture or None, receipt folder)
    "chat_sftfinal_v3": (A + r"\chat_base\sftfinal\chat_sftfinal_v3.gguf", PROMPTS, os.path.join(PAGE, "greedy.json"), "regress"),
    "state4": (A + r"\state_gate\state4\state4.gguf", U + r"\wgate\state4_requests.jsonl",
               R + r"\experiments\wasm_state_2026-10-25\chat_state4.json", "regress"),
}
for fam in ("kimi_tiny", "srd_tiny", "attnres_tiny", "flex_highway_tiny", "mamba_tiny", "gpt2_tiny", "swa_tiny"):
    MODELS[fam] = (os.path.join(A, fam, fam + ".gguf") if fam not in RELABELLED else M + rf"\{fam}.gguf", PROMPTS, "mts:" + fam, "regress")
for fam in ("kimi_tiny", "srd_tiny", "mamba_tiny"):
    MODELS[fam + "_rec"] = (M + rf"\{fam}_rec.gguf", PROMPTS, "mts:" + fam, ".")
    MODELS[fam + "_rec_multiturn"] = (M + rf"\{fam}_rec.gguf", MULTI, None, ".")
MODELS["kimi_tiny_custom"] = (M + r"\kimi_tiny_custom.gguf", PROMPTS, None, ".")
MODELS["kimi_tiny_custom_multiturn"] = (M + r"\kimi_tiny_custom.gguf", MULTI, None, ".")

env = dict(os.environ)
for name in sys.argv[1:]:
    gguf, prompts, fx, where = MODELS[name]
    if fx and fx.startswith("mts:"):
        fx = mts_fixture(fx[4:])
    out = os.path.join(W, name + ".json")
    cmd = [sys.executable, os.path.join(R, r"tools\webgpu_parity\swiftshader_run.py"), "--", "node",
           os.path.join(R, r"tools\webgpu_parity\chat_test.mjs"), EMBER, gguf, prompts, out]
    if fx:
        cmd += ["--fixture", fx]
    with open(os.path.join(W, name + ".log"), "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    s = {}
    try:
        s = json.load(open(out))
    except (OSError, ValueError):
        pass
    dst = os.path.join(H, where, "chat") if where != "." else os.path.join(H, "chat")
    os.makedirs(dst, exist_ok=True)
    if os.path.exists(out):
        shutil.copy(out, os.path.join(dst, name + ".json"))
    shutil.copy(os.path.join(W, name + ".log"), os.path.join(dst, name + ".log"))
    print(name, "rc", rc, {k: s.get(k) for k in ("backend", "n", "webgpu_vs_wasm", "webgpu_vs_ember_cpu", "wasm_vs_ember_cpu",
                                               "tokens", "ms_per_reply", "errors")}, flush=True)
