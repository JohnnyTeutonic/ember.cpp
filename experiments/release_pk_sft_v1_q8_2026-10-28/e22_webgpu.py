r"""The page's WebGPU leg on pk_sft_v1 F16 (Windows): tools/webgpu_parity/chat_test.mjs from the ember 9d993f4
source tree on WSL ext4 (~/e22rel/src, so deploy/wasm/web/webgpu is that commit's llama_webgpu.js), through
swiftshader_run.py (SwiftShader only, hardware GPU contexts monitored and fatal), with the 2gb page build's
ember.js, the 40 frozen prompts and the native server's replies as the fixture. Run under the CPU budget:
    python scripts/cpu_budget.py run --session ember --cores 1 --job ... --resume ... -- python e22_webgpu.py
"""
import os
import subprocess
import sys

U = r"\\wsl.localhost\Ubuntu\home\jonat\e22rel"
SP = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, WEBGPU_NODE=os.path.join(os.path.dirname(SP), "webgpu_ir", "wg_node", "node_modules", "webgpu", "index.js"))
leg = U + r"\wasmleg\f16"
out = os.path.join(SP, "webgpu_f16.json")
cmd = [sys.executable, U + r"\src\tools\webgpu_parity\swiftshader_run.py", "--cores", "1", "--", "node",
       U + r"\src\tools\webgpu_parity\chat_test.mjs", U + r"\page2gb\ember.js", U + r"\variants\pk_sft_v1_f16.gguf",
       leg + r"\prompts.json", out, "--fixture", leg + r"\greedy.json"]
with open(os.path.join(SP, "webgpu_f16.log"), "w", encoding="utf-8") as log:
    rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print("rc", rc)
sys.exit(rc)
