#!/usr/bin/env python3
"""TTFT benchmark for ember.cpp serve mode (protocol: docs/TTFT.md).

Starts tinyllama_server once per independent run, sends fixed prompts of
exact token lengths to /chat with max_new_tokens=1, and records the server's
own TTFT receipt (the "ttft" object in the response) next to the client
round trip. Standard library only, so it runs unchanged on a Colab VM.

    python3 tools/ttft_bench.py --server build/tinyllama_server \
        --model story3k.gguf --lengths 128 --runs 5 \
        --out-dir experiments/ttft_2026-10-16/story3k --commit $(git rev-parse HEAD)

Design choices that matter for the numbers:
  * Prompts are word-level text built from the model's own vocabulary and
    starting with "once upon", which the server passes through verbatim (no
    template tokens). Every response's num_prompt_tokens is checked against
    the requested length; a mismatch aborts the run.
  * Warmup and measured prompts are distinct, seeded, and the same across
    runs and configs, so every config sees the identical workload.
  * The order of lengths is shuffled per run (seeded by the run index), so
    slow drift over a run (clocks, thermals) does not line up with length.
  * Each run is a fresh server process: run-to-run spread includes load,
    allocation and cache state, which is what "independent run" means here.

Outputs in --out-dir (existing files are refused, never overwritten):
  raw.jsonl      one row per request, warmup included (phase field)
  summary.jsonl  one row per (run, input_len): the docs/TTFT.md record
"""
import argparse
import json
import os
import random
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.request

PREFIX = ["once", "upon"]  # server.cpp passes prompts starting this way through verbatim


# ---------------------------------------------------------------- vocabulary
def read_gguf_tokens(path):
    """tokenizer.ggml.tokens from a GGUF v3 header (metadata only)."""
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack("<" + fmt, f.read(n))[0]

        def rd_str():
            return f.read(rd("Q")).decode("utf-8", errors="replace")

        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                  10: "Q", 11: "q", 12: "d"}
        if rd("I") != 0x46554747:
            raise ValueError(f"{path} is not a GGUF file")
        rd("I")  # version
        rd("Q")  # tensor count
        tokens = None
        for _ in range(rd("Q")):
            key, vt = rd_str(), rd("I")
            if vt == 8:
                rd_str()
            elif vt == 9:
                et, n = rd("I"), rd("Q")
                items = [rd_str() if et == 8 else rd(scalar[et]) for _ in range(n)]
                if key == "tokenizer.ggml.tokens":
                    tokens = items
            else:
                rd(scalar[vt])
        if not tokens:
            raise ValueError(f"{path} has no tokenizer.ggml.tokens")
        return tokens


def prompt_words(tokens):
    """Plain lowercase words: each is exactly one word-level token."""
    words = sorted({t for t in tokens if t.isalpha() and t.islower()})
    if len(words) < 100:
        raise ValueError("vocabulary has too few plain words for prompts")
    return words


def make_prompt(words, n_words, rng):
    body = [rng.choice(words) for _ in range(max(0, n_words - len(PREFIX)))]
    return " ".join(PREFIX + body)


# ---------------------------------------------------------------- hardware
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def gpu_snapshot():
    """GPU identity and the clocks visible right now ('' fields if no GPU)."""
    q = ("name,driver_version,uuid,clocks.sm,clocks.max.sm,clocks.mem,"
         "temperature.gpu,power.draw,pstate")
    row = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits").splitlines()
    vals = [v.strip() for v in row[0].split(",")] if row else [""] * 9
    keys = ["gpu", "driver_version", "gpu_uuid", "sm_clock_mhz", "sm_clock_max_mhz",
            "mem_clock_mhz", "temp_c", "power_w", "pstate"]
    snap = dict(zip(keys, vals))
    nvcc = sh("nvcc --version | grep release")
    snap["cuda_version"] = nvcc.split("release")[-1].split(",")[0].strip() if nvcc else ""
    smi = sh("nvidia-smi | grep 'CUDA Version'")
    snap["cuda_driver_api"] = smi.split("CUDA Version:")[-1].split("|")[0].strip() if smi else ""
    return snap


# ---------------------------------------------------------------- server
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, prompt, timeout):
    body = json.dumps({"user_input": prompt, "max_new_tokens": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000.0, payload


def start_server(args, run_dir, port):
    # cwd = run_dir: ember's Logger writes debugging.log into the cwd.
    log = open(os.path.join(run_dir, "server.log"), "w")
    cmd = shlex.split(args.wrapper.replace("{run_dir}", run_dir)) if args.wrapper else []
    cmd += [os.path.abspath(args.server), os.path.abspath(args.model),
            str(port), "127.0.0.1", "/nonexistent_www"]
    proc = subprocess.Popen(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + args.load_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during load (see {run_dir}/server.log)")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return proc, log
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not start listening in time")


# ---------------------------------------------------------------- stats
def percentile(xs, q):
    """Linear interpolation between closest ranks (numpy's default)."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def pcts(xs):
    return {f"p{q}": round(percentile(xs, q), 3) for q in (50, 90, 99)}


def median(xs):
    return round(percentile(xs, 50), 3)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kernel-variant", default="baseline")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wrapper", default="",
                    help="command prefix for the server, e.g. an nsys profile line; "
                         "{run_dir} expands to the run directory")
    ap.add_argument("--load-timeout", type=float, default=900)
    ap.add_argument("--request-timeout", type=float, default=900)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "raw.jsonl")
    sum_path = os.path.join(args.out_dir, "summary.jsonl")
    for p in (raw_path, sum_path):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}: results are immutable, use a new --out-dir")

    words = prompt_words(read_gguf_tokens(args.model))
    model_label = args.model_label or os.path.basename(args.model)
    config = {"kernel_variant": args.kernel_variant, "chunk_size": args.chunk_size,
              "prefix_cache": args.prefix_cache}

    # Fixed prompt sets: identical for every run and config. Separate RNG
    # streams per (length, phase) so changing --reps never shifts warmups.
    def prompt_set(length, phase, count, n_words):
        rng = random.Random(f"{args.seed}:{length}:{phase}")
        return [make_prompt(words, n_words, rng) for _ in range(count)]

    with open(raw_path, "w") as raw, open(sum_path, "w") as summ:
        for run in range(1, args.runs + 1):
            run_dir = os.path.join(args.out_dir, f"run{run}")
            os.makedirs(run_dir, exist_ok=True)
            port = free_port()
            hw_start = gpu_snapshot()
            proc, log = start_server(args, run_dir, port)
            order = list(args.lengths)
            random.Random(f"order:{args.seed}:{run}").shuffle(order)
            print(f"run {run}/{args.runs}: lengths in order {order}", flush=True)
            try:
                for length in order:
                    # Calibrate words -> tokens once (BOS etc.), then hold it exact.
                    probe = prompt_set(length, "calibrate", 1, length)[0]
                    _, pay = post(port, probe, args.request_timeout)
                    n_words = length - (pay["ttft"]["num_prompt_tokens"] - length)
                    rows = {"warmup": prompt_set(length, "warmup", args.warmup, n_words),
                            "measure": prompt_set(length, "measure", args.reps, n_words)}
                    measured = []
                    for phase in ("warmup", "measure"):
                        for i, prompt in enumerate(rows[phase]):
                            rtt, pay = post(port, prompt, args.request_timeout)
                            tt = pay["ttft"]
                            if not tt["first_token"]:
                                raise RuntimeError(f"length {length}: no first token emitted "
                                                   f"(engine error; see {run_dir}/debugging.log)")
                            if tt["num_prompt_tokens"] != length:
                                raise RuntimeError(f"length {length}: prompt became "
                                                   f"{tt['num_prompt_tokens']} tokens")
                            row = {"run": run, "input_len": length, "phase": phase, "rep": i,
                                   "client_rtt_ms": round(rtt, 3), **tt}
                            raw.write(json.dumps(row) + "\n")
                            if phase == "measure":
                                measured.append(row)
                    raw.flush()
                    hw_now = gpu_snapshot()
                    summary = {
                        "config": config,
                        "input_len": length,
                        "ttft_ms": pcts([r["request_to_first_token_ms"] for r in measured]),
                        "client_rtt_ms": pcts([r["client_rtt_ms"] for r in measured]),
                        "phase_median_ms": {k: median([r[k] for r in measured]) for k in (
                            "server_pre_generate_ms", "tokenize_ms", "kv_reset_ms", "embed_ms",
                            "prefill_wall_ms", "prefill_gpu_span_ms", "sample_ms")},
                        "batch_prefill": all(r["batch_prefill"] for r in measured),
                        "n": len(measured), "warmup_discarded": args.warmup,
                        "run": run, "model": model_label,
                        "gpu": hw_start["gpu"], "gpu_uuid": hw_start["gpu_uuid"],
                        "driver_version": hw_start["driver_version"],
                        "cuda_version": hw_start["cuda_version"],
                        "clocks_after": {k: hw_now[k] for k in (
                            "sm_clock_mhz", "sm_clock_max_mhz", "mem_clock_mhz",
                            "temp_c", "power_w", "pstate")},
                        "commit": args.commit, "seed": args.seed,
                    }
                    summ.write(json.dumps(summary) + "\n")
                    summ.flush()
                    print(f"  len {length:5d}: p50 {summary['ttft_ms']['p50']:.2f} ms  "
                          f"p90 {summary['ttft_ms']['p90']:.2f}  p99 {summary['ttft_ms']['p99']:.2f}",
                          flush=True)
            finally:
                # A profiler wrapper (nsys) finalises its report on SIGINT and
                # can take minutes doing it; a bare server just stops.
                if args.wrapper:
                    proc.send_signal(signal.SIGINT)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=600 if args.wrapper else 30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()


if __name__ == "__main__":
    main()
