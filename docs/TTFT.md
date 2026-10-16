# TTFT in ember.cpp serve mode

The aim is a repeatable loop for measuring time to first token (TTFT) and
optimising it: measure first, then change one thing, then measure again under
the same protocol. Results that don't help are recorded alongside the ones that
do. This file covers the measurement method, the baseline results, and the
conclusions drawn from them.

## What is measured

TTFT is the time from the server owning a `/chat` request to the first output
token being emitted. The server and `generate()` together return a receipt in
the response (`"ttft"` object):

| field | window |
|---|---|
| `request_to_first_token_ms` | handler entry -> first token emitted. **The TTFT metric.** |
| `server_pre_generate_ms` | handler entry -> `generate()` entry (JSON parse, prompt templating) |
| `tokenize_ms` | template selection + encode |
| `kv_reset_ms` | `kv_cache_.clear_data()` (zero-fills the host K/V vectors) |
| `embed_ms` | host embedding lookup for the whole prompt |
| `prefill_wall_ms` | embeddings -> last-token logits on the host: CPU-offload layers, device alloc + upload, GPU forward, free |
| `prefill_gpu_span_ms` | CUDA-event span of the GPU forward on the default stream |
| `sample_ms` | repetition penalty + sampling + first-token decode |

The phases tile the window (their sum matches `ttft_ms` to within tens of
microseconds in the CPU smoke). `prefill_gpu_span_ms` runs from the first to
the last GPU operation of the forward. It therefore includes time the GPU sat
idle waiting on the host inside the forward. It is **not** kernel busy time.
Kernel busy time comes from a separate nsys pass: the forward is the NVTX
range `ttft_prefill_forward`, and the run reports `nvtx_gpu_proj_sum` /
`cuda_gpu_kern_sum`. Those profiled runs never enter the TTFT records, because
profiling perturbs timing.

The client round trip (`client_rtt_ms`) is recorded too. It adds the HTTP
transport and the rest of `generate()` after the first token (with
`max_new_tokens=1`, only the final decode and response).

## Models

Both are GGUFs exported by paperkiln.

* **story3k**: a trained paperkiln export (d=128, 2 layers, 4 heads, ctx 256,
  4096-word vocabulary; `/mnt/c/ml_artifacts/paperkiln_story3k.zip`). With a
  256-token context, the only grid length it can hold is 128.
* **timing**: `microtorch/tools/make_timing_gguf` at the TinyLlama-1.1B shape
  (d=2048, 22 layers, 32 query heads / 4 KV heads (GQA), ff=5632, ctx 4096),
  seeded N(0, 0.02) weights, story3k's vocabulary, and an explicit LM head. It
  exists for timing only: prefill cost depends on shapes, not on weight
  values, and its output text means nothing. The same seed and shape give a
  byte-identical file. It is generated on the VM (4.4 GB at F32).

## Hardware

Colab **L4**, for every record that carries a claim (microtorch/AGENTS.md:
never mix GPU types within a comparison). Each summary row records the GPU
name, UUID, driver, CUDA version and the clocks read after the cell
(SM/memory clock, max SM clock, temperature, power, P-state).
`hardware.txt` in each result folder holds the full `nvidia-smi -q` clock and
performance dump.

## Protocol

* Lengths: {128, 512, 1024, 2048, 4096} prompt tokens, including BOS, for the
  timing model; {128} for story3k. Every response's `num_prompt_tokens` is
  checked against the requested length, and a mismatch aborts the run.
* Prompts: word-level text drawn from the model's vocabulary, starting with
  `once upon` (`server.cpp` passes such prompts through without template
  tokens). They are seeded per (seed, length, phase), so warmup and measured
  prompts are distinct and identical across runs and configs.
* Per length per run: 10 warmup requests (all discarded), then 50 measured
  requests. Reported: p50, p90, p99 by linear interpolation. With n=50, p99 is
  close to the sample maximum, so read it as "worst seen".
* Independence: 5 runs, each a fresh server process. The order of lengths is
  shuffled per run, so slow drift does not line up with length.
* `max_new_tokens=1`. Sampling uses the server defaults (T=0.1, top-k 40,
  top-p 0.9), which do not affect TTFT.
* Noise rule (paperkiln's seed-lottery rule applied to runtime): an
  optimisation's gain is computed per run from the per-run p50s. If the
  standard deviation of the gain across the 5 runs is more than 50% of the
  mean gain, it is recorded as "within noise, not recommended".
* Code identity: `tools/colab_ttft_drive.sh` refuses uncommitted changes and
  ships `git archive HEAD` of ember.cpp and paperkiln, so the `commit` field
  is exactly the code that ran.

Running it, from WSL, as one foreground process (it stops the session on
every exit path):

```bash
bash tools/colab_ttft_drive.sh probe <session> 45      # sizing + nsys availability
bash tools/colab_ttft_drive.sh full  <session> 110 RUNS=5 WARMUP=10 REPS=50
```

Results land in `/mnt/c/ml_artifacts/ttft/<session>/`, and the summaries are
copied into `experiments/ttft_<date>/`.

## Records

`summary.jsonl`, one row per (run, input_len):

```json
{"config": {"kernel_variant": "baseline", "chunk_size": null, "prefix_cache": false},
 "input_len": 1024, "ttft_ms": {"p50": 0, "p90": 0, "p99": 0},
 "client_rtt_ms": {...}, "phase_median_ms": {...}, "n": 50, "warmup_discarded": 10,
 "run": 1, "model": "...", "gpu": "NVIDIA L4", "gpu_uuid": "...", "driver_version": "...",
 "cuda_version": "...", "clocks_after": {...}, "commit": "...", "seed": 42}
```

`raw.jsonl` has one row per request (warmup included, marked by `phase`), with
the full receipt. Neither file is ever overwritten: the bench refuses an
existing output directory.

## Structural facts found while instrumenting

These shape which directions are worth pursuing, before any number is in.

1. **Serve mode is single-session and unguarded.** `server.cpp` shares one
   `TinyLlamaSession` across httplib's thread pool, with no mutex. Concurrent
   requests race on one KV cache. Chunked prefill (Direction 3) only pays off
   under concurrency, so it would first need a request queue. Until serve
   mode is multi-request, that direction is premature.
2. **No KV reuse.** Every `generate()` calls `kv_cache_.clear_data()` and
   prefills from position 0. Prefix caching (Direction 2) would also need to
   remove that reset.
3. **The prefill attention kernel is serial in context length.**
   `attention_batch_prefill_cuda_kernel` uses one block per (token, query
   head) with `blockDim = head_dim`. For every context position it runs two
   `__syncthreads()`, and thread 0 sums the head_dim products serially in
   FP64. GQA is handled by indexing (`q_head / (H_q / H_kv)`), but K/V are
   re-read once per query head.
4. **The prefill forward does host work inside the GPU timeline.** There are
   per-layer `Logger::info` calls, synchronous device-to-host copies for
   debug logging on the first GPU layer, `cudaStreamSynchronize` after RoPE
   and after the KV update in every layer, and a `cudaMalloc`/`cudaFree` of
   the logits buffer on every call.
5. **Short prompts take a different path.** Below 32 tokens, `generate()`
   runs token by token (`batch_prefill: false` in the receipt). Every grid
   length is at or above 128.
6. **The LM head is not tied automatically.** A GGUF without `output.weight`
   loads but has no LM head, and prefill fails. paperkiln's mtstudio exports
   write it explicitly. This is outside TTFT, but `generate()` now flags the
   failure (`first_token: false`) instead of reporting a zero TTFT.

## Engine bugs the probes found (fixed before the baseline)

The probes on Colab L4 found these defects before any baseline could be
taken. Each one either changes what "the baseline" means or makes long
prompts fail.

* **Every default GGUF run was CPU-only** (fixed in `3c4bbe9`). The session
  turned the documented `n_gpu_layers = -1` ("all on GPU") into an offload
  count of 0. The GGUF constructor reads that field as a raw GPU-layer
  request, where 0 means all CPU. The server and the CLI both used the
  default, so serve mode had never run a GGUF on the GPU. The first probe
  measured 140 ms (story3k, 128 tokens) and 68 s (timing model, 128 tokens)
  this way, with 0 MiB of GPU memory in use. Those numbers are CPU numbers
  and are not part of any baseline.
* **Prompts over 2048 tokens overflowed the prefill buffers** (fixed in
  `41fec65`). `MAX_BATCH_TOKENS = 2048` capped the persistent batch buffers,
  and the resize function only logged a warning. A 4096-token prefill then
  failed with a CUDA invalid-argument error in layer 0. The buffers now grow
  to fit. Lengths up to 2048 were always in bounds: the residual buffer is
  allocated at twice the size.
* **The BF16 tensor-core GEMM path never succeeds on this stack** (L4,
  CUDA 13.0, driver 580.82). Every `cublasGemmEx` BF16 call returns status
  15 (not supported) or 7 (invalid value), and the GEMM falls back to FP32.
  In the probe the timing model logged 2,471 such failures and no successes.
  As a result, the engine's default and `EMBER_DISABLE_BF16=1` give
  bit-identical logits. The path also doubled its per-layer weight offsets
  (fixed in `3c4bbe9`); that bug was latent, since the call never completes.
  The failed attempt still runs before every batched GEMM, so its cost is
  part of the baseline. It is not yet measured separately.

The GPU path is numerically verified against the CPU path (FP32 on both,
first-token logits; probe 3):

| model | length | max abs diff | rel diff | cosine | argmax |
|---|---|---|---|---|---|
| story3k | 128 | 1.0e-5 | 1.4e-6, 9.7e-7 | 1.000000 | equal (2/2) |
| timing (22 layers, GQA) | 128 | | ≤ 3.1e-6 | 1.000000 | equal (2/2) |

A max abs diff of 1e-5 is the precision of the dump (6 significant digits).

## Protocol deviation: seconds-scale lengths

The baseline prefill takes 33 s at 2048 tokens and about 130 s at 4096. At
those lengths the full protocol (5 runs × 61 requests) would cost about
15 L4-hours. The run-to-run spread there is under 0.05% of p50 (probe 3:
p90 − p50 = 12 ms on 33.3 s), so 50 repetitions add almost no precision.
At 2048 and 4096 the baseline therefore uses 2 warmup + 5 measured requests
per run, still over 5 fresh-server runs. Those rows carry
`"protocol_deviation"`. Every other length, and story3k, follows the full
protocol. The 5 runs are split over 3 L4 VMs (runs 1–2, 3–4, 5). Each VM
passes the parity gate before timing. `gpu_uuid` records which physical GPU
served each run.

## Results

Pending: the baseline sweep.

## Directions and falsifiers

The order and falsifiers follow the original brief. Each optimisation is
measured as a new `kernel_variant` (or `prefix_cache` / `chunk_size`), under
the protocol above, against the baseline rows from the same commit lineage.

| direction | falsifier (recorded as a negative if it holds) |
|---|---|
| Prefix-cache hit-rate logging + reuse | < 10% TTFT gain at > 50% hit rate |
| Attention fusion / online softmax / GQA-aware tiles | TTFT change within ±5% |
| Chunked prefill | premature until serve mode is multi-request (fact 1) |
| Multi-stream heads | no gain beyond noise; stream overhead is larger than per-head work |
