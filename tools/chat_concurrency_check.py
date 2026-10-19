"""Concurrent /chat requests must give the same replies as sequential ones.

Greedy (top_k=1) replies for N prompts, first one at a time, then from K
parallel clients; counts mismatches and reports throughput and latency.
Usage: chat_concurrency_check.py PORT [N] [K]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
port = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")][:N]


def ask(i):
    body = {"user_input": rows[i]["prompt"], "max_new_tokens": 64, "temperature": 1.0, "top_k": 1,
            "top_p": 1.0, "seed": 0, "return_ids": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d["ids"], (time.perf_counter() - t0) * 1000


t0 = time.perf_counter()
seq = [ask(i) for i in range(N)]
t_seq = time.perf_counter() - t0
par = [None] * N


def worker(w):
    for i in range(w, N, K):
        par[i] = ask(i)


t0 = time.perf_counter()
ths = [threading.Thread(target=worker, args=(w,)) for w in range(K)]
for t in ths:
    t.start()
for t in ths:
    t.join()
t_par = time.perf_counter() - t0
bad = [i for i in range(N) if seq[i][0] != par[i][0]]
print(f"sequential: {N / t_seq:.2f} req/s, wall p50 {statistics.median(x[1] for x in seq):.1f} ms")
print(f"{K} clients: {N / t_par:.2f} req/s, wall p50 {statistics.median(x[1] for x in par):.1f} ms, "
      f"max {max(x[1] for x in par):.1f} ms")
print(f"replies differing from sequential: {len(bad)}/{N} {bad[:10]}")
