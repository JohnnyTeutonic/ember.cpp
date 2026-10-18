"""Chat-shaped TTFT on the running ember server: first 20 eval prompts,
greedy, 1 new token (TTFT) then 60 (full reply wall time)."""
import json
import statistics
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8090/chat"
P = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
tag = sys.argv[1] if len(sys.argv) > 1 else "run"


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


rows = [json.loads(l) for l in open(P, encoding="utf-8")][:20]
base = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}
post({"user_input": "warm up", "max_new_tokens": 4, **base})
ttft, wall, ntok, replies = [], [], [], []
for r in rows:
    d = post({"user_input": r["prompt"], "max_new_tokens": 1, **base})
    ttft.append(d["ttft"]["ttft_ms"])
    t0 = time.perf_counter()
    d = post({"user_input": r["prompt"], "max_new_tokens": 60, **base})
    wall.append((time.perf_counter() - t0) * 1000)
    ntok.append(d["tokens"])
    replies.append(d["reply"])
q = statistics.quantiles(ttft, n=10)
print(f"{tag}: ttft p50 {statistics.median(ttft):.2f} ms p90 {q[8]:.2f} | reply wall p50 "
      f"{statistics.median(wall):.1f} ms, {sum(ntok)/len(ntok):.1f} tok avg, "
      f"{sum(wall)/sum(ntok):.2f} ms/tok")
json.dump({"tag": tag, "ttft": ttft, "wall": wall, "tokens": ntok, "replies": replies},
          open(f"/home/jonat/ember_ttft/chat/ttft_{tag}.json", "w"))
