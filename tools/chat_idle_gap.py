"""Does an idle gap before a request (a person reading and typing) cost
latency? The same 2-turn chat requests sent back to back and after a pause.
Usage: chat_idle_gap.py PORT [N] [GAP_S]
"""
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
port = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]


def ask(i):
    hist = []
    for j in range(2):
        r = rows[(i + 1 + j) % len(rows)]
        hist += [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["reference"]}]
    body = {"user_input": rows[i]["prompt"], "history": hist, "max_new_tokens": 64, "temperature": 0.7,
            "top_k": 40, "top_p": 0.9, "seed": 1000 + i}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d["ttft"]["ttft_ms"], (time.perf_counter() - t0) * 1000


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    return xs[int(k)] + (xs[min(int(k) + 1, len(xs) - 1)] - xs[int(k)]) * (k - int(k))


for _ in range(3):
    ask(0)
for label, gap in (("back to back", 0.0), (f"after {GAP:g} s idle", GAP), ("back to back again", 0.0)):
    out = []
    for i in range(N):
        if gap:
            time.sleep(gap)
        out.append(ask(i))
    t = [x[0] for x in out]
    w = [x[1] for x in out]
    print(f"{label:>20}: TTFT p50 {statistics.median(t):6.1f} p90 {pct(t, 90):6.1f} max {max(t):6.1f} ms | "
          f"reply wall p50 {statistics.median(w):6.1f} p90 {pct(w, 90):6.1f} max {max(w):6.1f} ms", flush=True)
