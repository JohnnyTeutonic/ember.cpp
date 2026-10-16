#!/usr/bin/env python3
"""First-token logit parity between two serve-mode configurations.

The numerical gate for TTFT work (docs/TTFT.md): a faster path only counts
if it computes the same thing. Both configurations receive the identical
request sequence (same seeded prompts as ttft_bench.py, calibration
included); each server dumps the full first-token logit vector per request
via TINYLLAMA_LOGITS_DUMP, and the vectors are compared pairwise.

    python3 tools/ttft_parity.py --server build/tinyllama_server --model m.gguf \
        --ref gpu_fp32 --test gpu_bf16 --lengths 128 1024 --prompts 2 \
        --max-rel-diff 0.05 --min-cosine 0.999 --out parity.jsonl

Configurations: cpu (n_gpu_layers 0), gpu_fp32 (all layers on GPU,
EMBER_DISABLE_BF16=1), gpu_bf16 (all layers on GPU, engine default). A
kernel variant is added here as one more named configuration.

Per pair it records max |a-b|, that divided by max |ref| (rel_diff), cosine
similarity, and whether the argmax agrees. Exit status 1 if any pair breaks
--max-rel-diff or --min-cosine; argmax disagreement is reported, not gated,
because near-ties legitimately flip under reduced precision.
"""
import argparse
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttft_bench as tb  # noqa: E402  (shared prompt + server plumbing)

CONFIGS = {
    "cpu": (0, []),
    "gpu_fp32": (-1, ["EMBER_DISABLE_BF16=1"]),
    "gpu_bf16": (-1, []),
    # Kernel variants: the reference for each is the same config on the
    # baseline kernel (gpu_fp32 for gpu_fp32_flash, gpu_bf16 for gpu_flash).
    "gpu_fp32_flash": (-1, ["EMBER_DISABLE_BF16=1", "EMBER_PREFILL_ATTN=flash"]),
    "gpu_flash": (-1, ["EMBER_PREFILL_ATTN=flash"]),
}


def collect(args, role, name, lengths, words, workdir):
    """Run one configuration over every length; return its logit rows."""
    gpu_layers, env = CONFIGS[name]
    run_dir = os.path.join(workdir, f"{role}_{name}")
    os.makedirs(run_dir, exist_ok=True)
    dump = os.path.join(run_dir, "logits.txt")
    srv = argparse.Namespace(server=args.server, model=args.model, gpu_layers=gpu_layers,
                             env=env + [f"TINYLLAMA_LOGITS_DUMP={dump}"], wrapper="",
                             load_timeout=args.load_timeout)
    port = tb.free_port()
    proc, log = tb.start_server(srv, run_dir, port)
    meta = []
    try:
        for length in lengths:
            rng = random.Random(f"{args.seed}:{length}:calibrate")
            probe = tb.make_prompt(words, length, rng)
            _, pay = tb.post(port, probe, args.request_timeout)
            meta.append({"input_len": length, "kind": "calibrate"})
            n_words = length - (pay["ttft"]["num_prompt_tokens"] - length)
            rng = random.Random(f"{args.seed}:{length}:measure")
            for i in range(args.prompts):
                _, pay = tb.post(port, tb.make_prompt(words, n_words, rng), args.request_timeout)
                tt = pay["ttft"]
                if not tt["first_token"] or tt["num_prompt_tokens"] != length:
                    raise RuntimeError(f"{name} len {length}: bad receipt {tt}")
                if gpu_layers != 0 and tt["prefill_gpu_span_ms"] <= 0:
                    raise RuntimeError(f"{name} len {length}: prefill did not run on the GPU")
                meta.append({"input_len": length, "kind": "measure", "prompt": i})
    finally:
        proc.terminate()
        proc.wait(timeout=60)
        log.close()
    with open(dump) as f:
        rows = [[float(x) for x in line.split()] for line in f if line.strip()]
    if len(rows) != len(meta):
        raise RuntimeError(f"{name}: {len(rows)} logit rows for {len(meta)} requests")
    return [(m, r) for m, r in zip(meta, rows) if m["kind"] == "measure"]


def compare(a, b):
    diff = max(abs(x - y) for x, y in zip(a, b))
    scale = max(abs(x) for x in a) or 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return {"max_abs_diff": diff, "rel_diff": diff / scale,
            "cosine": dot / (na * nb) if na and nb else 0.0,
            "argmax_ref": max(range(len(a)), key=a.__getitem__),
            "argmax_test": max(range(len(b)), key=b.__getitem__)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", choices=CONFIGS, required=True)
    ap.add_argument("--test", choices=CONFIGS, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128])
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rel-diff", type=float, required=True)
    ap.add_argument("--min-cosine", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-timeout", type=float, default=900)
    ap.add_argument("--request-timeout", type=float, default=1800)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite {args.out}")

    words = tb.prompt_words(tb.read_gguf_tokens(args.model))
    workdir = tempfile.mkdtemp(prefix="ttft_parity_")
    ref = collect(args, "ref", args.ref, args.lengths, words, workdir)
    test = collect(args, "test", args.test, args.lengths, words, workdir)

    ok = True
    with open(args.out, "w") as out:
        for (m, a), (_, b) in zip(ref, test):
            row = {"model": os.path.basename(args.model), "ref": args.ref, "test": args.test,
                   **m, **compare(a, b),
                   "max_rel_diff_gate": args.max_rel_diff, "min_cosine_gate": args.min_cosine}
            row["pass"] = row["rel_diff"] <= args.max_rel_diff and row["cosine"] >= args.min_cosine
            ok &= row["pass"]
            out.write(json.dumps(row) + "\n")
            print(f"{args.ref} vs {args.test} len {m['input_len']:5d} #{m['prompt']}: "
                  f"rel_diff {row['rel_diff']:.2e} cosine {row['cosine']:.6f} "
                  f"argmax {row['argmax_ref']}/{row['argmax_test']} "
                  f"{'PASS' if row['pass'] else 'FAIL'}", flush=True)
    print("PARITY_PASS" if ok else "PARITY_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
