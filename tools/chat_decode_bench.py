"""Decode speed on an ember chat server: 20 eval prompts, greedy, 64 new
tokens max. Per-token decode = (wall - TTFT) / (tokens - 1)."""
import json
import statistics
import sys
import time
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
P = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
rows = [json.loads(l) for l in open(P, encoding="utf-8")][:20]


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


cfg = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "max_new_tokens": 64, "seed": 0, "return_ids": True}
post({"user_input": "warm up", **cfg})
per_tok, ids = [], []
for r in rows:
    t0 = time.perf_counter()
    d = post({"user_input": r["prompt"], **cfg})
    wall = (time.perf_counter() - t0) * 1000
    n = d["tokens"] + (1 if d["stop_id"] >= 0 else 0)  # sampled steps incl. the stop token
    if n > 1:
        per_tok.append((wall - d["ttft"]["ttft_ms"]) / (n - 1))
    ids.append(d["ids"])
print(f"{tag}: decode p50 {statistics.median(per_tok):.2f} ms/token "
      f"(min {min(per_tok):.2f}, max {max(per_tok):.2f}, n={len(per_tok)})")
json.dump({"tag": tag, "per_tok_ms": per_tok, "ids": ids},
          open(f"{tag}.bench.json", "w"))
