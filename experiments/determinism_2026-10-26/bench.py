"""Decode and prefill cost of one tinyllama_server process; deterministic mode is on or off by the caller's
environment (EMBER_DETERMINISTIC).

    python3 bench.py BIN GGUF PORT PROMPTS.jsonl CARRY5.jsonl OUT.json

The server runs with EMBER_GEMM_TF32=0 (the post jobs' setting) and EMBER_PREFIX_REUSE=0 (every request prefills
its whole prompt), gpu_layers -1. After three warm-up requests:
  - decode: the 40 tinychat prompts, greedy (temperature 1, top_k 1, top_p 1, seed 7), max_new_tokens 64, three
    rounds; per request (wall - ttft_ms) / (steps - 1), steps = generated tokens + the stop token (as
    tools/chat_decode_bench.py);
  - prefill: the 40 Carry-5 items' gold conversations (the turns before the target with their gold replies as
    history, then the target's user turn), max_new_tokens 1, two rounds; the server's prefill_wall_ms and ttft_ms,
    with the prompt length.
Prints one summary line; OUT.json holds every sample."""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

BIN, GGUF, PORT, PROMPTS, CARRY5, OUT = sys.argv[1:7]
URL = f"http://127.0.0.1:{PORT}"
run = "/tmp/ember_bench_run_" + PORT
os.makedirs(os.path.join(run, "www"), exist_ok=True)
dl = os.path.join(run, "debugging.log")
if not os.path.lexists(dl):
    os.symlink(os.devnull, dl)
env = {**os.environ, "EMBER_GEMM_TF32": "0", "EMBER_PREFIX_REUSE": "0"}
env.pop("TINYLLAMA_LOGITS_DUMP", None)
proc = subprocess.Popen([BIN, GGUF, PORT, "127.0.0.1", os.path.join(run, "www"), "-1"], cwd=run, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
health = None
for _ in range(600):
    try:
        health = json.loads(urllib.request.urlopen(URL + "/health", timeout=2).read())
        break
    except OSError:
        if proc.poll() is not None:
            sys.exit("server exited before serving")
        time.sleep(1)
if health is None:
    proc.kill()
    sys.exit("server did not come up")


def chat(history, user, max_new):
    body = {"user_input": user, "history": [{"user": u, "assistant": r} for u, r in history], "temperature": 1.0,
            "top_k": 1, "top_p": 1.0, "seed": 7, "max_new_tokens": max_new, "return_ids": True}
    req = urllib.request.Request(URL + "/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        e = json.loads(r.read())
    return e, (time.perf_counter() - t0) * 1000.0


decode, prefill = [], []
try:
    ps = [json.loads(x)["prompt"] for x in open(PROMPTS, encoding="utf-8") if x.strip()][:40]
    items = [json.loads(x) for x in open(CARRY5, encoding="utf-8") if x.strip()][::26][:40]
    for p in ps[:3]:
        chat([], p, 64)
    for rnd in range(3):
        for i, p in enumerate(ps):
            e, wall = chat([], p, 64)
            steps = e["tokens"] + (1 if e.get("stop_id", -1) >= 0 else 0)
            if steps > 1:
                decode.append({"round": rnd, "i": i, "steps": steps, "wall_ms": wall, "ttft_ms": e["ttft"]["ttft_ms"],
                               "ms_per_token": (wall - e["ttft"]["ttft_ms"]) / (steps - 1)})
    for rnd in range(2):
        for it in items:
            tt = it["target_turn"]
            hist = [(t["user"], t["gold"]) for t in it["turns"][:tt - 1]]
            e, wall = chat(hist, it["turns"][tt - 1]["user"], 1)
            prefill.append({"round": rnd, "id": it["id"], "prompt_tokens": e["ttft"]["num_prompt_tokens"],
                            "prefill_wall_ms": e["ttft"]["prefill_wall_ms"], "ttft_ms": e["ttft"]["ttft_ms"],
                            "wall_ms": wall})
finally:
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


mpt = [d["ms_per_token"] for d in decode]
pw = [d["prefill_wall_ms"] for d in prefill]
tt = [d["ttft_ms"] for d in prefill]
summary = {"deterministic": bool(health.get("deterministic", False)),
           "decode_ms_per_token_p50": statistics.median(mpt), "decode_ms_per_token_mean": statistics.fmean(mpt),
           "decode_ms_per_token_p90": q(mpt, 0.9), "decode_requests": len(mpt),
           "decode_tokens_mean": statistics.fmean(d["steps"] for d in decode),
           "prefill_wall_ms_p50": statistics.median(pw), "prefill_wall_ms_mean": statistics.fmean(pw),
           "prefill_ttft_ms_p50": statistics.median(tt), "prefill_requests": len(pw),
           "prefill_prompt_tokens_mean": statistics.fmean(d["prompt_tokens"] for d in prefill)}
json.dump({"bin": BIN, "gguf": GGUF, "health": health, "summary": summary, "decode": decode, "prefill": prefill},
          open(OUT, "w"), indent=1)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}))
