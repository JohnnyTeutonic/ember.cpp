"""Run a node WebGPU script (ir_gate.mjs, chat_test.mjs) on Dawn with SwiftShader, the CPU Vulkan
device, as the ONLY driver: the laptop's GPUs (Intel and NVIDIA) are never opened. Windows.

    python swiftshader_run.py [--cores 2] [--icd DIR] -- node SCRIPT ARGS...

Run it under the CPU budget (scripts/cpu_budget.py run --session ember --cores 2 -- python ...).

- WEBGPU_NODE names the `webgpu` npm package's index.js (Dawn's node bindings, 0.6.2 used here);
  Dawn loads the Vulkan loader from beside dawn.node, so this copies System32's vulkan-1.dll there
  once if it is missing.
- The loader sees one driver: VK_DRIVER_FILES / VK_ICD_FILENAMES name SwiftShader's ICD
  (vk_swiftshader.dll as Chrome ships it: --icd, $SWIFTSHADER_ICD, else the newest Chrome install's),
  and every layer is off (VK_LOADER_LAYERS_DISABLE=~all~, DISABLE_LAYER_NV_OPTIMUS_1=1). The scripts
  create Dawn with backend=vulkan and stop unless the adapter names SwiftShader.
- A monitor reads Windows' per-process GPU counters (GPU Engine) every 5 s: an engine instance owned
  by the node process (a context on any WDDM adapter, Intel or NVIDIA) kills it at once and the run
  fails (exit 3). The last line reports the checks made and the contexts seen.
- The process tree is pinned to the last --cores logical cores.
- node gets --max-opt=1 unless the command sets a --max-opt: dawn.node 0.6.2 crashes (access
  violation) when V8's Maglev-optimised code calls into it (26 Oct 2026; --no-maglev also avoids it).
"""
import argparse
import ctypes
import glob
import os
import shutil
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--cores", type=int, default=2)
ap.add_argument("--icd", default=os.environ.get("SWIFTSHADER_ICD"))
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
if not cmd:
    sys.exit("no command")
if os.path.basename(cmd[0]).lower() in ("node", "node.exe") and not any(c.startswith("--max-opt") for c in cmd[1:]):
    cmd = [cmd[0], "--max-opt=1"] + cmd[1:]

n = os.cpu_count() or 4
mask = sum(1 << (n - 1 - i) for i in range(a.cores))
ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.c_size_t(mask))

icd_dir = a.icd
if not icd_dir:
    found = sorted(glob.glob(r"C:\Program Files\Google\Chrome\Application\*\vk_swiftshader_icd.json"))
    icd_dir = os.path.dirname(found[-1]) if found else ""
icd = os.path.join(icd_dir, "vk_swiftshader_icd.json")
if not os.path.exists(icd):
    sys.exit("no SwiftShader ICD (vk_swiftshader_icd.json): pass --icd DIR")
wn = os.environ.get("WEBGPU_NODE")
if not wn or not os.path.exists(wn):
    sys.exit("set WEBGPU_NODE to the webgpu package's index.js")
dawn_dir = os.path.join(os.path.dirname(wn), "dist", "win32-x64")
if not os.path.exists(os.path.join(dawn_dir, "vulkan-1.dll")):
    shutil.copy(r"C:\Windows\System32\vulkan-1.dll", dawn_dir)

env = dict(os.environ, VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd, VK_LOADER_DRIVERS_SELECT="*swiftshader*",
           VK_LOADER_LAYERS_DISABLE="~all~", DISABLE_LAYER_NV_OPTIMUS_1="1", VK_INSTANCE_LAYERS="")


def gpu_instances(pid):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue)"
                          ".CounterSamples | ForEach-Object { $_.InstanceName }"],
                         capture_output=True, text=True, timeout=60).stdout
    return [x.strip() for x in out.splitlines() if x.startswith(f"pid_{pid}_")]


proc = subprocess.Popen(cmd, env=env)
violation, checks = [], [0]


def monitor():
    while proc.poll() is None:
        hits = gpu_instances(proc.pid)
        checks[0] += 1
        if hits:
            violation.extend(hits)
            print("GPU CONTEXT ON A HARDWARE ADAPTER: killing the process", hits[:4], flush=True)
            proc.kill()
            return
        time.sleep(5)


threading.Thread(target=monitor, daemon=True).start()
t0 = time.time()
rc = proc.wait()
print(f"# rc {rc}; GPU-counter checks {checks[0]}, hardware GPU contexts {len(violation)}; wall {time.time() - t0:.1f} s",
      flush=True)
sys.exit(3 if violation else rc)
