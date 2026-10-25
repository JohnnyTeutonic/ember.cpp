r"""The page's chat path (Windows): tools/webgpu_parity/chat_test.mjs per model through swiftshader_run.py,
WebGPU against WebAssembly (deploy/wasm/build.sh's ember.js, EMBER_JS) and a CPU fixture: the page test's
greedy.json for chat_sftfinal_v3 (ember CPU /chat ids, 1fdaaa5), state4's receipt
../wasm_state_2026-10-25/chat_state4.json (equal to the native server's 40/40) for its multi-turn requests
(chat_state4.py requests), and mtstudio's greedy replies for the families of ../families_2026-10-20 (ember
CPU's equal them 40/40). Receipts <name>.json into this folder.
    WEBGPU_NODE=... EMBER_JS=...\ember.js PAGE_DIR=dir (prompts.json, greedy.json) python chats.py NAME [NAME ...]
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
EMBER = os.environ["EMBER_JS"]
C = W
PAGE = os.environ.get("PAGE_DIR", W)
os.makedirs(C, exist_ok=True)


def mts_fixture(fam):   # mtstudio's 40 greedy replies (ember CPU's equal them 40/40, families_2026-10-20)
    out = os.path.join(C, fam + "_mtstudio.json")
    rows = [json.loads(x) for x in open(os.path.join(A, fam, "mtstudio_greedy.jsonl"), encoding="utf-8") if x.strip()]
    json.dump({"replies": [{"reply": r["mtstudio"].strip()} for r in rows]}, open(out, "w", encoding="utf-8"))
    return out


MODELS = {
    "chat_sftfinal_v3": (A + r"\chat_base\sftfinal\chat_sftfinal_v3.gguf", os.path.join(PAGE, "prompts.json"), os.path.join(PAGE, "greedy.json")),
    "state4": (A + r"\state_gate\state4\state4.gguf", U + r"\wgate\state4_requests.jsonl",
               R + r"\experiments\wasm_state_2026-10-25\chat_state4.json"),
}
for fam in ("kimi_tiny", "srd_tiny", "attnres_tiny", "flex_highway_tiny", "mamba_tiny", "gpt2_tiny", "swa_tiny"):
    MODELS[fam] = (os.path.join(A, fam, fam + ".gguf"), os.path.join(PAGE, "prompts.json"), None)

env = dict(os.environ)
for name in sys.argv[1:]:
    gguf, prompts, fx = MODELS[name]
    if fx is None:
        fx = mts_fixture(name)
    out = os.path.join(C, name + ".json")
    cmd = [sys.executable, os.path.join(R, r"tools\webgpu_parity\swiftshader_run.py"), "--", "node",
           os.path.join(R, r"tools\webgpu_parity\chat_test.mjs"), EMBER, gguf, prompts, out, "--fixture", fx]
    with open(os.path.join(C, name + ".log"), "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    s = {}
    try:
        s = json.load(open(out))
    except (OSError, ValueError):
        pass
    if os.path.exists(out):
        shutil.copy(out, os.path.join(H, name + ".json"))
    print(name, "rc", rc, {k: s.get(k) for k in ("backend", "n", "webgpu_vs_wasm", "webgpu_vs_ember_cpu", "wasm_vs_ember_cpu",
                                               "tokens", "ms_per_reply", "errors")}, flush=True)
