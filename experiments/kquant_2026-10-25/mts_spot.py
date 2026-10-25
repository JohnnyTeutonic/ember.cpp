#!/usr/bin/env python3
"""Spot check that the recorded mtstudio F32 replies (../pk_base_v1_2026-10-24/pk_base_v1_parity.jsonl) still
reproduce: mtstudio chat (paperkiln's CPU build in ~/pk_cpu_1024, c8f0b8af) on pk_base_v1, the first N prompts.
    mts_spot.py N OUT.jsonl
The spec is the run's base_spec.json re-pointed at the run folder, as ../pk_base_v1_2026-10-24/gate.py does."""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "/mnt/c/ml_artifacts/next_base/base"
MTS = os.path.expanduser("~/pk_cpu_1024/build/mtstudio")
PORT = 8123
n, out = int(sys.argv[1]), sys.argv[2]
ref = [json.loads(l) for l in open(os.path.join(HERE, "..", "pk_base_v1_2026-10-24", "pk_base_v1_parity.jsonl"))][:n]
spec = json.load(open(f"{RUN}/base_spec.json"))
spec["out_dir"], spec["name"] = RUN, "pk_base_v1"
spec.setdefault("data", {})["vocab"] = f"{RUN}/pk_base_v1.gguf"
spec_path = os.path.expanduser("~/kq/work/mts_spot_spec.json")
json.dump(spec, open(spec_path, "w"), indent=1)


def get(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


p = subprocess.Popen([MTS, "chat", spec_path, "--port", str(PORT)], stdout=open(os.path.expanduser("~/kq/work/mts_spot.log"), "w"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
try:
    for _ in range(900):
        try:
            h = get("/health")
            break
        except Exception:
            if p.poll() is not None:
                sys.exit("mtstudio exited")
            time.sleep(1)
    assert h.get("model") == "pk_base_v1", h
    same = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for r in ref:
            m = get("/chat", {"user_input": r["prompt"], "max_new_tokens": 60, "temperature": 1.0, "top_k": 1,
                              "top_p": 1.0, "seed": 0})
            same += m["reply"] == r["mtstudio"]
            f.write(json.dumps({"i": r["i"], "mtstudio_now": m["reply"], "tokens": m["tokens"],
                                "recorded": r["mtstudio"], "same": m["reply"] == r["mtstudio"]}) + "\n")
    print(f"mtstudio /health {json.dumps(h)}; first {n} prompts: reply identical to the 24 Oct record {same}/{n}; "
          f"{time.time() - t0:.0f} s")
finally:
    p.terminate()
    p.wait(30)
