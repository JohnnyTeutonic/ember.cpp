#!/usr/bin/env python3
"""/health's "provenance" (model_provenance.h) on a running server, against hashlib.

    python3 tools/provenance_server_check.py SERVER_BIN MODEL.gguf PORT OUT.json [ALT.gguf]

Starts SERVER_BIN on MODEL (CPU, gpu_layers 0) and checks: weights.status "sha256" with sha256 and bytes equal
to hashlib's over the file; the executable's sha256 equal to SERVER_BIN's; build.git_commit 40 hex digits or
"unknown", git_dirty a boolean or null, backend "cpu" or "cuda"; instance_id 32 hex digits; gpu_layers_requested
0. A second start on the same file: the same weights and build, a new instance_id. EMBER_WEIGHTS_DIGEST=0:
status "disabled", sha256 null, and EMBER_WEIGHTS_DIGEST named (without its value) in runtime.ember_env.
ALT.gguf (another file that loads, e.g. MODEL with bytes appended): its own digest, not MODEL's. Writes
OUT.json; exit 1 when a check fails."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

SERVER, MODEL, PORT, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ALT = sys.argv[5] if len(sys.argv) > 5 else None
HEX = re.compile(r"^[0-9a-f]+$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def start(model, extra_env=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **(extra_env or {}))
    log = open(OUT + f".server_{int(time.time() * 1000)}.log", "w")
    proc = subprocess.Popen([SERVER, model, str(PORT), "127.0.0.1", "/nonexistent-www", "0"], env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} (log {log.name})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                return proc, json.load(r)
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not answer /health")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


checks, report = {}, {}
model_sha, exe_sha = sha256(MODEL), sha256(os.path.realpath(SERVER))
proc, h1 = start(MODEL)
stop(proc)
p1 = h1.get("provenance", {})
report["first"] = p1
w, b = p1.get("weights", {}), p1.get("build", {})
checks["schema"] = p1.get("schema") == "ember.provenance/1"
checks["weights digest = hashlib"] = w.get("status") == "sha256" and w.get("sha256") == model_sha
checks["weights bytes = file size"] = w.get("bytes") == os.path.getsize(MODEL)
checks["executable digest = hashlib"] = p1.get("executable", {}).get("sha256") == exe_sha
checks["git_commit"] = b.get("git_commit") == "unknown" or (len(b.get("git_commit", "")) == 40 and bool(HEX.match(b["git_commit"])))
checks["git_dirty"] = b.get("git_dirty") in (True, False, None)
checks["backend"] = b.get("backend") in ("cpu", "cuda")
checks["instance_id"] = len(p1.get("instance_id", "")) == 32 and bool(HEX.match(p1["instance_id"]))
checks["gpu_layers_requested"] = p1.get("runtime", {}).get("gpu_layers_requested") == 0
checks["model and existing fields kept"] = h1.get("model") == MODEL and "chat_template" in h1 and "context" in h1

proc, h2 = start(MODEL)
stop(proc)
p2 = h2.get("provenance", {})
checks["restart: same weights and build"] = p2.get("weights") == p1.get("weights") and p2.get("build") == p1.get("build")
checks["restart: new instance_id"] = p2.get("instance_id") != p1.get("instance_id")

proc, h3 = start(MODEL, {"EMBER_WEIGHTS_DIGEST": "0"})
stop(proc)
p3 = h3.get("provenance", {})
report["disabled"] = p3
checks["disabled: no digest"] = p3.get("weights", {}).get("status") == "disabled" and p3["weights"].get("sha256") is None
env_names = p3.get("runtime", {}).get("ember_env", [])
checks["disabled: env named without value"] = "EMBER_WEIGHTS_DIGEST" in env_names and all("=" not in n for n in env_names)

if ALT:
    proc, h4 = start(ALT)
    stop(proc)
    p4 = h4.get("provenance", {})
    report["alt"] = p4
    checks["other file: its own digest"] = p4.get("weights", {}).get("sha256") == sha256(ALT) != model_sha

report["checks"] = checks
report["model_sha256"] = model_sha
with open(OUT, "w") as f:
    json.dump(report, f, indent=2)
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
sys.exit(0 if all(checks.values()) else 1)
