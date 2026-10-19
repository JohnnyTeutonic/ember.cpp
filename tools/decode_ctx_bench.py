"""Decode speed against context length on an ember server running a
word-level timing model (paperkiln's make_timing_gguf; see docs/TTFT.md).

For each prompt length, REPS distinct prompts (ttft_bench's "once upon" word
prompts, calibrated to the exact token count), greedy, NEW new tokens.
Per-token decode = (wall - TTFT) / (sampled steps - 1); the median over the
prompts is reported, with the number of steps actually taken (a random model
can sample its stop token early).
Usage: decode_ctx_bench.py PORT MODEL.gguf [LENGTHS=128,1024,2048,4096] [NEW=64] [REPS=5]
"""
import json
import random
import statistics
import sys
import time
import urllib.request

from ttft_bench import prompt_words, read_gguf_tokens, make_prompt


def post(port, prompt, new):
    body = {"user_input": prompt, "max_new_tokens": new, "temperature": 1.0, "top_k": 1, "top_p": 1.0,
            "seed": 0, "return_ids": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return (time.perf_counter() - t0) * 1000, d


def main():
    port, model = int(sys.argv[1]), sys.argv[2]
    lengths = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "128,1024,2048,4096").split(",")]
    new = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    words = prompt_words(read_gguf_tokens(model))
    post(port, make_prompt(words, 16, random.Random("warm")), 8)
    for length in lengths:
        _, d = post(port, make_prompt(words, length, random.Random(f"cal:{length}")), 1)
        n_words = length - (d["ttft"]["num_prompt_tokens"] - length)
        per_tok, steps = [], []
        for i in range(reps):
            prompt = make_prompt(words, n_words, random.Random(f"m:{length}:{i}"))
            wall, d = post(port, prompt, new)
            if d["ttft"]["num_prompt_tokens"] != length:
                raise SystemExit(f"length {length}: prompt has {d['ttft']['num_prompt_tokens']} tokens")
            n = d["tokens"] + (0 if d["stop_reason"] == "length" else 1)  # sampled steps incl. a stop token
            steps.append(n)
            if n > 1:
                per_tok.append((wall - d["ttft"]["ttft_ms"]) / (n - 1))
        med = statistics.median(per_tok) if per_tok else float("nan")
        print(f"context {length:5d}: decode {med:7.2f} ms/token (median of {len(per_tok)}; steps {min(steps)}-{max(steps)})",
              flush=True)


if __name__ == "__main__":
    main()
