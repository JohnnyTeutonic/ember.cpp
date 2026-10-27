"""Inputs of the Carry-12 CUDA gates (run in WSL), into OUT_DIR:
  the earlier gates' own inputs, by their own scripts: ../cuda_state_2026-10-25/make_inputs.py (state4_in.json with
    the session script, state4q8_in.json, ar3_in.json, synth_in.json) and ../attnres_x5_2026-10-25/x5_inputs.py
    (ar_x5_in.json);
  the long leg: ar3 and ar_x5's IR documents with model.ctx and the kv state's max_len raised from 16 to 1024 and
    nothing else changed
    ({m}_ctx1024.ir.json, served through EMBER_IR), the unchanged documents ({m}.ir.json, the EMBER_IR control),
    and long_in.json: 8 seeded random sequences of 600 ids below the vocab (15), prompt length 300, so the decode
    attention reads up to 600 keys (19 of the tiled kernel's 32-key chunks) where the files' own 16 positions
    give one chunk.
Every file of the three model folders is checked against its manifest's sha256 first.
    python3 c12_inputs.py OUT_DIR"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
H = os.path.dirname(os.path.abspath(__file__))
X = os.path.join(H, "..")
A = "/mnt/c/ml_artifacts"
DIRS = {"state4": f"{A}/state_gate/state4", "ar3": f"{A}/attnres_gate/ar3", "ar_x5": f"{A}/attnres_gate_x5/ar_x5"}
rep = {"manifests": {}}
for m, d in DIRS.items():
    files = json.load(open(os.path.join(d, "manifest.json")))["files"]
    ok = {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() == h for f, h in files.items()}
    rep["manifests"][m] = f"{sum(ok.values())}/{len(ok)}"
    if not all(ok.values()):
        raise SystemExit(f"{m}: sha256 mismatch {ok}")
for script in ("cuda_state_2026-10-25/make_inputs.py", "attnres_x5_2026-10-25/x5_inputs.py"):
    subprocess.run([sys.executable, os.path.join(X, script), out], check=True)
for m in ("ar3", "ar_x5"):
    raw = open(os.path.join(DIRS[m], f"{m}.ir.json"), "rb").read()
    open(os.path.join(out, f"{m}.ir.json"), "wb").write(raw)
    doc = json.loads(raw)
    assert doc["model"]["ctx"] == 16, doc["model"]
    doc["model"]["ctx"] = 1024
    kv = [s for s in doc["block"]["state"] if s.get("kind") == "kv"]
    assert len(kv) == 1 and kv[0]["max_len"] == 16, kv
    kv[0]["max_len"] = 1024  # the kv state's positions (the interpreter's max_len and rope table)
    json.dump(doc, open(os.path.join(out, f"{m}_ctx1024.ir.json"), "w"))
    rep[f"{m}_ctx1024"] = {"vocab": doc["model"]["vocab"], "ctx": 1024, "kv_max_len": 1024}
V = 15
rng = np.random.default_rng(1027)
seqs = [[int(x) for x in rng.integers(0, V, 600)] for _ in range(8)]
assert all(rep[f"{m}_ctx1024"]["vocab"] == V for m in ("ar3", "ar_x5"))
json.dump({"seqs": seqs, "prompt_len": [300] * 8}, open(os.path.join(out, "long_in.json"), "w"))
rep["long"] = {"sequences": 8, "len": 600, "prompt_len": 300, "seed": 1027}
print(json.dumps(rep))
