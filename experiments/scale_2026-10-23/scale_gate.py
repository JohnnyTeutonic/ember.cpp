"""ember gate for paperkiln's scale re-export (arxiv_expr_norm_gate, 23 Oct 2026):
1. ember CPU logits on the scale GGUF vs the diag GGUF (same weights) on token sequences: equal?
2. ember's server on the scale GGUF replays the 40 recorded prompts (tools/chat_parity.py's request:
   max_new_tokens 60, top_k 1, seed 0) against mtstudio's recorded replies in parity.jsonl.
"""
import json
import os
import subprocess
import time
import urllib.request

import numpy as np

G = "/home/jonat/arxiv_chain_runs/expr_norm_gate"
SCALE, DIAG = f"{G}/export_scale/arxiv_expr_norm_gate.gguf", f"{G}/export/arxiv_expr_norm_gate.gguf"
B = os.path.expanduser("~/ember_wt/build_cpu")
out = {}

# 1. logits, scale vs diag
seqs = [[2, 536, 48, 384, 427, 307, 67, 3], [2, 100, 200, 300, 400, 500, 3], list(range(2, 60))]
logits = {}
for name, g in (("scale", SCALE), ("diag", DIAG)):
    r = subprocess.run([f"{B}/ir_logits", g], input=json.dumps(seqs), capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ir_logits {name}: {r.stderr[-800:]}")
    logits[name] = json.loads(r.stdout)
diffs = [float(np.abs(np.array(a[k]) - np.array(b[k])).max())
         for a, b in zip(logits["scale"], logits["diag"]) for k in ("prefill", "decode")]
out["scale_vs_diag_max_abs"] = max(diffs)

# 2. the server against mtstudio's recorded replies
refs = [json.loads(line) for line in open(f"{G}/parity.jsonl")]
port = 8117
srv = subprocess.Popen([f"{B}/tinyllama_server", SCALE, str(port), "127.0.0.1", f"{G}/www" if os.path.isdir(f"{G}/www") else ".", "0"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=os.path.expanduser("~/ember_wt"))
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    same = 0
    rows = []
    for r in refs:
        body = {"user_input": r["prompt"], "max_new_tokens": 60, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        e = json.loads(urllib.request.urlopen(req, timeout=120).read())
        ok = e["reply"] == r["mtstudio"] and e.get("tokens") == r["mts_tokens"] and e.get("stop_reason") == r["mts_stop"]
        same += ok
        rows.append({"i": r["i"], "same": ok, "ember": e["reply"], "mtstudio": r["mtstudio"],
                     "ember_tokens": e.get("tokens"), "mts_tokens": r["mts_tokens"]})
    out["replies_identical_to_mtstudio"] = f"{same}/{len(refs)}"
    out["differing"] = [x for x in rows if not x["same"]][:5]
finally:
    srv.terminate()
print(json.dumps(out, indent=1))
