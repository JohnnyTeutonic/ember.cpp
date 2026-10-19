"""Chat latency analysis for an ember chat server (BPE chat mode).

Requests are shaped like the chat page's: sampling t=0.7, top_k 40,
top_p 0.9, max_new_tokens 64, a seed per request. Conversation length is
varied by sending 0..N earlier turns as history (taken from the frozen
TinyChat prompts and their reference replies), so prompt length covers what
a real chat reaches before the server starts dropping old turns.

For every request: server TTFT (ttft.ttft_ms), request wall time from the
client, generated tokens, and per-token decode
(wall - TTFT) / (sampled steps - 1). Reports p50/p90/p99 per level, and a
concurrency leg with K parallel clients.

Usage: chat_latency.py PORT OUT_JSON [--reps N] [--levels 0,1,2,4,6,8]
       [--concurrency K] [--conc-requests M]
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(xs):
    return {"p50": pct(xs, 50), "p90": pct(xs, 90), "p99": pct(xs, 99), "mean": statistics.fmean(xs) if xs else float("nan"),
            "n": len(xs)}


def one(port, rows, i, turns, seed):
    hist = []
    for j in range(turns):
        r = rows[(i + 1 + j) % len(rows)]
        hist += [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["reference"]}]
    body = {"user_input": rows[i % len(rows)]["prompt"], "history": hist, "max_new_tokens": 64,
            "temperature": 0.7, "top_k": 40, "top_p": 0.9, "seed": seed, "return_ids": True}
    t0 = time.perf_counter()
    d = post(port, body)
    wall = (time.perf_counter() - t0) * 1000
    steps = d["tokens"] + (1 if d.get("stop_id", -1) >= 0 else 0)
    ttft = d["ttft"]["ttft_ms"]
    return {"turns": turns, "turns_used": d["history_turns_used"], "prompt_tokens": d["prompt_tokens"],
            "ttft_ms": ttft, "wall_ms": wall, "tokens": d["tokens"], "steps": steps,
            "decode_ms_per_token": (wall - ttft) / (steps - 1) if steps > 1 else None,
            "overhead_ms": wall - d["ttft"]["request_to_first_token_ms"] - (0 if steps <= 1 else 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--levels", default="0,1,2,4,6,8")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--conc-requests", type=int, default=40)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]
    for k in range(3):  # warm up
        one(a.port, rows, k, 2, 900 + k)
    report = {"levels": {}, "records": []}
    for turns in [int(x) for x in a.levels.split(",")]:
        recs = [one(a.port, rows, i, turns, 1000 + i) for i in range(a.reps)]
        report["records"] += recs
        dec = [r["decode_ms_per_token"] for r in recs if r["decode_ms_per_token"] is not None]
        lvl = {"prompt_tokens": summary([r["prompt_tokens"] for r in recs]),
               "turns_used": summary([r["turns_used"] for r in recs]),
               "ttft_ms": summary([r["ttft_ms"] for r in recs]),
               "wall_ms": summary([r["wall_ms"] for r in recs]),
               "tokens": summary([r["tokens"] for r in recs]),
               "decode_ms_per_token": summary(dec)}
        report["levels"][str(turns)] = lvl
        print(f"turns {turns}: prompt {lvl['prompt_tokens']['p50']:.0f} tok (used {lvl['turns_used']['p50']:.0f}) | "
              f"TTFT p50 {lvl['ttft_ms']['p50']:.2f} p90 {lvl['ttft_ms']['p90']:.2f} p99 {lvl['ttft_ms']['p99']:.2f} ms | "
              f"decode p50 {lvl['decode_ms_per_token']['p50']:.2f} ms/tok | reply {lvl['tokens']['p50']:.0f} tok, "
              f"wall p50 {lvl['wall_ms']['p50']:.1f} p90 {lvl['wall_ms']['p90']:.1f} p99 {lvl['wall_ms']['p99']:.1f} ms",
              flush=True)
    if a.concurrency > 1:
        out, lock = [], threading.Lock()

        def worker(w):
            for i in range(w, a.conc_requests, a.concurrency):
                r = one(a.port, rows, i, 2, 5000 + i)
                with lock:
                    out.append(r)
        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker, args=(w,)) for w in range(a.concurrency)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        el = time.perf_counter() - t0
        report["concurrency"] = {"clients": a.concurrency, "requests": len(out), "seconds": el,
                                 "requests_per_s": len(out) / el, "wall_ms": summary([r["wall_ms"] for r in out]),
                                 "ttft_ms": summary([r["ttft_ms"] for r in out])}
        c = report["concurrency"]
        print(f"concurrency {a.concurrency} clients, {len(out)} requests (2 turns): {c['requests_per_s']:.1f} req/s, "
              f"wall p50 {c['wall_ms']['p50']:.1f} p90 {c['wall_ms']['p90']:.1f} p99 {c['wall_ms']['p99']:.1f} ms "
              f"(server TTFT p50 {c['ttft_ms']['p50']:.2f} ms)", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
