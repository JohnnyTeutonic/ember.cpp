#!/usr/bin/env python3
"""Check the prefix-cache hit-rate receipt (docs/TTFT.md) against a server.

    python3 tools/test_prefix_logging.py --server build/tinyllama_server --model m.gguf [--gpu-layers 0]

Sequence and expected receipts (word-level model, BOS counted):
  1. A (n tokens), fresh server     -> reusable 0 (nothing cached)
  2. A extended by k words          -> reusable n: A plus the 1 generated
     token were cached, but A's tokens are the common prefix (the generated
     token differs from the first extension word except by coincidence, so
     the test asserts the range [n, n+1])
  3. an unrelated prompt B          -> reusable 1 at most (the shared BOS
     "once upon" prefix is 3 tokens; accept <= 3)
  4. B again, identical             -> reusable len(B) - 1 (cap: the last
     prompt token must be recomputed), hit rate (len-1)/len
"""
import argparse
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttft_bench as tb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-layers", type=int, default=-1)
    args = ap.parse_args()
    words = tb.prompt_words(tb.read_gguf_tokens(args.model))
    rng = random.Random(7)
    srv = argparse.Namespace(server=args.server, model=args.model, gpu_layers=args.gpu_layers,
                             env=[], wrapper="", load_timeout=300)
    run_dir = tempfile.mkdtemp(prefix="prefix_test_")
    port = tb.free_port()
    proc, log = tb.start_server(srv, run_dir, port)
    failures = []

    def ask(prompt):
        _, pay = tb.post(port, prompt, 300)
        return pay["ttft"]

    def check(name, ok, tt):
        print(f"{'PASS' if ok else 'FAIL'} {name}: prompt {tt['num_prompt_tokens']}, "
              f"cached {tt['prefix_prev_cached_tokens']}, reusable {tt['prefix_reusable_tokens']}, "
              f"hit {tt['prefix_hit_rate']:.3f}")
        if not ok:
            failures.append(name)

    try:
        a = tb.make_prompt(words, 40, rng)
        t1 = ask(a)
        check("fresh server reuses nothing", t1["prefix_reusable_tokens"] == 0, t1)
        n = t1["num_prompt_tokens"]
        a_ext = a + " " + " ".join(rng.choice(words) for _ in range(20))
        t2 = ask(a_ext)
        check("extension reuses the previous prompt",
              t2["prefix_prev_cached_tokens"] == n and
              n <= t2["prefix_reusable_tokens"] <= n + 1, t2)
        b = tb.make_prompt(words, 50, random.Random(99))
        t3 = ask(b)
        check("unrelated prompt reuses at most the shared 'once upon' prefix",
              t3["prefix_reusable_tokens"] <= 3, t3)
        t4 = ask(b)
        m = t4["num_prompt_tokens"]
        check("identical prompt is capped at len - 1",
              t4["prefix_reusable_tokens"] == m - 1 and
              abs(t4["prefix_hit_rate"] - (m - 1) / m) < 1e-9, t4)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()
    print("PREFIX_LOGGING_PASS" if not failures else f"PREFIX_LOGGING_FAIL {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
