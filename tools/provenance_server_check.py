#!/usr/bin/env python3
"""/health's "provenance" (model_provenance.h) on a running server, against hashlib.

    python3 tools/provenance_server_check.py SERVER_BIN MODEL.gguf PORT OUT.json [ALT.gguf [MUTABLE.gguf]]

Starts SERVER_BIN on MODEL (CPU, gpu_layers 0) and checks: weights.status "sha256" with sha256 and bytes equal
to hashlib's over the file; the executable's sha256 equal to SERVER_BIN's; build.git_commit 40 hex digits or
"unknown", git_dirty a boolean or null, backend "cpu" or "cuda"; instance_id 32 hex digits; gpu_layers_requested
0. A second start on the same file: the same weights and build, a new instance_id. With EMBER_WEIGHTS_DIGEST=0,
EMBER_DEQUANT_AT_LOAD=0, EMBER_TEST_SCRIPTED_REPLIES (a side file) and an unknown EMBER_AUDIT_UNKNOWN: status
"disabled" with sha256 null; the two settings with their values; the side file by its SHA-256 (no path); the
unknown variable by name only. EMBER_PUBLIC=1: "public" true, the weights digest, size, git commit and
executable digest kept; no paths, compiler or EMBER_* record. ALT.gguf (another file that loads, e.g. MODEL with bytes appended): its own
digest, not MODEL's. MUTABLE.gguf (a scratch copy; this check appends to it): hashed and monitored, then
"modified_since_load" with no digest after an in-place append. Writes OUT.json; exit 1 when a check fails."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

SERVER, MODEL, PORT, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ALT = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
MUTABLE = sys.argv[6] if len(sys.argv) > 6 else None
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

scripted = OUT + ".scripted.json"
with open(scripted, "w") as f:
    json.dump([[5, 6, 7]], f)
proc, h3 = start(MODEL, {"EMBER_WEIGHTS_DIGEST": "0", "EMBER_DEQUANT_AT_LOAD": "0",
                         "EMBER_TEST_SCRIPTED_REPLIES": scripted, "EMBER_AUDIT_UNKNOWN": "secret-value"})
stop(proc)
p3 = h3.get("provenance", {})
report["disabled"] = p3
checks["disabled: no digest"] = p3.get("weights", {}).get("status") == "disabled" and p3["weights"].get("sha256") is None
rt = p3.get("runtime", {})
env_names = rt.get("ember_env", [])
checks["env: names listed"] = {"EMBER_WEIGHTS_DIGEST", "EMBER_AUDIT_UNKNOWN", "EMBER_TEST_SCRIPTED_REPLIES"} <= set(env_names)
checks["env: settings with values"] = rt.get("ember_settings", {}).get("EMBER_DEQUANT_AT_LOAD") == "0" and \
    rt.get("ember_settings", {}).get("EMBER_WEIGHTS_DIGEST") == "0"
checks["env: unknown variable without value"] = "secret-value" not in json.dumps(h3) and \
    "EMBER_AUDIT_UNKNOWN" not in rt.get("ember_settings", {})
checks["env: side file by digest"] = rt.get("ember_side_files", {}).get("EMBER_TEST_SCRIPTED_REPLIES", {}).get(
    "sha256") == sha256(scripted) and scripted not in json.dumps(rt)

proc, hp = start(MODEL, {"EMBER_PUBLIC": "1", "EMBER_KV": "f32"})
stop(proc)
pp = hp.get("provenance", {})
report["public"] = pp
checks["public: marked"] = pp.get("public") is True
checks["public: digest kept"] = pp.get("weights", {}).get("status") == "sha256" and \
    pp["weights"].get("sha256") == model_sha and pp["weights"].get("bytes") == os.path.getsize(MODEL)
checks["public: commit and executable digest kept"] = pp.get("build", {}).get("git_commit") == b.get("git_commit") \
    and pp.get("executable", {}).get("sha256") == exe_sha
checks["public: no paths, compiler or settings"] = "path" not in pp.get("weights", {}) and \
    "path" not in pp.get("executable", {}) and "compiler" not in pp.get("build", {}) and \
    not ({"ember_env", "ember_settings", "ember_side_files"} & set(pp.get("runtime", {}))) and \
    os.path.realpath(SERVER) not in json.dumps(pp)

if ALT:
    proc, h4 = start(ALT)
    stop(proc)
    p4 = h4.get("provenance", {})
    report["alt"] = p4
    checks["other file: its own digest"] = p4.get("weights", {}).get("sha256") == sha256(ALT) != model_sha

if MUTABLE:
    # A served file rewritten in place after the load: the next /health has no digest.
    proc, h5 = start(MUTABLE)
    try:
        w5 = h5.get("provenance", {}).get("weights", {})
        checks["mutable: hashed and monitored"] = w5.get("status") == "sha256" and w5.get("monitored") is True
        with open(MUTABLE, "ab") as f:
            f.write(b"in-place-append")
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
            w6 = json.load(r).get("provenance", {}).get("weights", {})
        report["modified"] = w6
        checks["mutable: modified_since_load"] = w6.get("status") == "modified_since_load" and \
            w6.get("sha256") is None and w6.get("sha256_at_load") == w5.get("sha256")
    finally:
        stop(proc)

report["checks"] = checks
report["model_sha256"] = model_sha
with open(OUT, "w") as f:
    json.dump(report, f, indent=2)
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
sys.exit(0 if all(checks.values()) else 1)
