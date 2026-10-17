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
The baseline therefore uses 2 warmup + 5 measured requests per run at 2048
tokens, and 1 + 3 at 4096, so that a 4096 cell (about 11 minutes) fits
inside one short-lived VM. Both lengths still run over 5 fresh-server runs.
Variants fast enough to need no deviation use the full protocol at every
length. Those rows carry
`"protocol_deviation"`. Every other length, and story3k, follows the full
protocol. The 5 runs are split over 3 L4 VMs (runs 1–2, 3–4, 5). Each VM
passes the parity gate before timing. `gpu_uuid` records which physical GPU
served each run.

## Results

### Baseline (ember `dbadee2`, Colab L4, 17 Oct 2026)

These are 30 cells (model × run × length) over 5 fresh-server runs, served by
12 different physical L4s (driver 580.82.07, CUDA 13.0). Records are in
`experiments/ttft_2026-10-17/baseline/`: `summary.jsonl`, `manifest.json`
and `table.md`. Two story3k cells came from a session whose commit,
`e506934`, differs from `dbadee2` only under `tools/`, so the engine code
is identical for all 30 cells.

The TTFT columns give the mean over runs of each run's p50, with the
standard deviation of those per-run p50s.

| model | tokens | TTFT p50 | run-to-run std | p90 | worst p99 | requests per run |
|---|---|---|---|---|---|---|
| story3k | 128 | 12.96 ms | 0.07 ms | 13.11 ms | 14.44 ms | 10 + 50 |
| timing | 128 | 233.4 ms | 0.63 ms | 234.4 ms | 237.1 ms | 10 + 50 |
| timing | 512 | 2,297.6 ms | 3.6 ms | 2,300.8 ms | 2,306.3 ms | 10 + 50 |
| timing | 1024 | 8,568 ms | 16 ms | 8,579 ms | 8,602 ms | 10 + 50 |
| timing | 2048 | 33,219 ms | 40 ms | 33,227 ms | 33,290 ms | 2 + 5 (deviation) |
| timing | 4096 | 131,123 ms | 58 ms | 131,125 ms | 131,223 ms | 1 + 3 (deviation) |

Phase medians for the timing model, averaged over runs:

| tokens | KV reset | embedding | prefill wall | GPU span | GPU span / TTFT |
|---|---|---|---|---|---|
| 128 | 25.8 ms | 0.3 ms | 206.6 ms | 205.7 ms | 0.881 |
| 512 | 25.9 ms | 1.4 ms | 2,269.6 ms | 2,267.5 ms | 0.987 |
| 1024 | 25.9 ms | 2.8 ms | 8,538.4 ms | 8,534.3 ms | 0.996 |
| 2048 | 26.0 ms | 11.3 ms | 33,180.9 ms | 33,167.6 ms | 0.998 |
| 4096 | 25.9 ms | 24.2 ms | 131,071.2 ms | 131,042.5 ms | 0.999 |

Tokenisation, server parsing and sampling stay under 1 ms at every length.
The client round trip adds 0.5–5 ms to the server-side TTFT.

What the baseline says:

1. **Prefill is quadratic and almost entirely on the GPU timeline.** From
   512 tokens upward, each doubling of length multiplies TTFT by 3.7–3.9.
   The GPU span is 99%+ of TTFT from 512 tokens. The matrix work of a
   2048-token prefill at this shape is about 4.5 TFLOP, roughly 0.15 s at
   the L4's FP32 rate. The measured 33 s is about 200× that. The cost has
   the quadratic signature of `attention_batch_prefill_cuda_kernel`
   (fact 3 above). The GPU span cannot separate kernels from GPU idle
   time, though, and the nsys pass that should have done so captured
   nothing (below). So "attention is the cost" is an inference here. The
   flash_v1 comparison tests it directly.
2. **A constant 26 ms per request goes on zero-filling the host KV
   vectors** (`kv_cache_.clear_data()`). That is 11% of TTFT at 128
   tokens, and the vectors are not read on the GPU path. It is the
   cheapest fix in the profile, and a separate experiment.
3. **The noise floor is tiny.** The run-to-run standard deviation of p50
   is 0.04–0.5% of the mean, so any change above about 1% clears the
   50%-of-improvement rule easily. The reduced repetitions at 2048 and
   4096 cost no meaningful precision: p90 − p50 is 0.02% and 0.002% of
   p50 there.
4. **story3k is 13 ms, almost all of it on the GPU timeline** (12.35 of
   12.96 ms) for a 2-layer, d=128 model. At this size the cost cannot be
   FLOPs. It points to per-layer fixed costs: synchronising copies,
   logging, allocations and the per-position loop of the attention
   kernel.

The nsys kernel pass (`nsys_timing_128`, `nsys_timing_1024`, Nsight Systems
2025.3.1) produced reports with no CUDA kernel and no NVTX data. A report
opened locally with nsys 2024.5.1 contains three tables and "does not
contain CUDA kernel data". TTFT under the profiler equalled TTFT without it
(8,569 ms vs 8,568 ms at 1024 tokens). The profiler therefore never hooked
the server's CUDA activity. The split between kernel time and launch or
synchronisation overhead is still open; it is recorded as a failed
measurement, not inferred.

### flash_v1: tiled prefill attention (ember `8b40023`, 17 Oct 2026)

The variant changes only the prefill attention kernel: tiling, online
softmax and causal tile skip (see the comment above
`attention_prefill_flash_kernel`). The QKV and output projections are not
fused, so this is the attention part of Direction 1, not full fusion. It
is selected with `EMBER_PREFILL_ATTN=flash`; the default is still the
baseline kernel.

**Parity gate (passed before any timing counted):** baseline kernel vs
flash_v1, FP32, first-token logits. Max relative difference was ≤ 1e-5
(1.4e-6 to 8.5e-6) at 128, 1024 and 4096 tokens on the timing model and at
128 on story3k, with cosine 1.00000000 and an identical argmax in every
pair. Every VM also passed a 128-token check before its cells.

**TTFT** (full protocol at every length, 10 + 50 requests × 5 runs; 3
L4s). Records are in `experiments/ttft_2026-10-17/flash_v1/`. Gains are
per-run, `1 - flash_p50 / baseline_p50`, judged by the noise rule.

| model | tokens | baseline p50 | flash_v1 p50 | speedup | gain (mean ± std) | verdict |
|---|---|---|---|---|---|---|
| story3k | 128 | 12.96 ms | 12.06 ms | 1.07× | 6.9% ± 0.92% | faster |
| timing | 128 | 233.4 ms | 94.5 ms | 2.5× | 59.5% ± 0.23% | faster |
| timing | 512 | 2,297.6 ms | 243.4 ms | 9.4× | 89.4% ± 0.08% | faster |
| timing | 1024 | 8,568 ms | 470.0 ms | 18.2× | 94.5% ± 0.02% | faster |
| timing | 2048 | 33,219 ms | 986.5 ms | 33.7× | 97.0% ± 0.03% | faster |
| timing | 4096 | 131,123 ms | 2,490 ms | 52.7× | 98.1% ± 0.01% | faster |

The attention falsifier did not hold: TTFT moved far beyond ±5% at every
length on the timing model. Swapping only this one kernel turned a
quadratic 131 s into 2.5 s. That is the direct confirmation the baseline's
GPU-span data could only suggest: the per-position attention kernel was
nearly all of the baseline's TTFT. story3k moves only 6.9%, just past the
line. At d=128 with 2 layers, its 12 ms was never dominated by attention.

What remains (flash_v1 phase medians):

| tokens | TTFT | KV reset | GPU span |
|---|---|---|---|
| 128 | 94.5 ms | 25.9 ms | 66.7 ms |
| 512 | 243.4 ms | 25.9 ms | 213.2 ms |
| 1024 | 470.0 ms | 25.9 ms | 436.7 ms |
| 2048 | 986.5 ms | 25.9 ms | 945.9 ms |
| 4096 | 2,490 ms | 25.9 ms | 2,409.7 ms |

Up to 2048 tokens, TTFT now roughly doubles per doubling of length, so the
linear-in-length projections dominate. Attention's quadratic share shows
again at 4096 (×2.5). At FP32 peak, the 2048-token projection work would
take about 0.15 s, so 0.95 s of GPU span leaves headroom of several times.
The candidates, in the order the data supports:

1. **The failed BF16 attempt before each of the 154 GEMMs per prefill.**
   Each attempt makes two `cudaMalloc`, a conversion, two `cudaFree`
   (each a device sync) and two error log lines. Since the attempt always
   fails here, switching it off (`EMBER_DISABLE_BF16=1`) gives
   bit-identical output, so the `flash_v1_fp32` lane measures exactly
   what these attempts cost.
2. **The 26 ms host KV zero-fill.** It is 27% of TTFT at 128 tokens and
   is never read on the GPU path.
3. **Per-layer fixed costs:** the debug device-to-host copies, the device
   syncs after RoPE, after the KV update and around attention, and the
   per-layer logging.
4. **GEMM precision** (TF32 or working BF16 tensor cores). This changes
   numerics, so it needs its own looser parity gate.
5. **Attention v2:** sharing each K/V tile across a GQA group of 8 query
   heads, which matters most at 4096.

### How the baseline was collected (Colab)

Busy L4 sessions on this account were reclaimed about 22–26 minutes after
creation. At first this was blamed on running parallel sessions; it then
recurred with a single session. So the baseline ran as a resumable lane on
the repo's supervisor (`tools/ttft_supervise.py`, path B of
`microtorch/AGENTS.md`), working as follows:

* The unit of work is one cell.
* Each VM gets the not-yet-banked cells, packed longest first to fit about
  880 s of measurement, and passes a quick parity gate.
* Finished cells are relayed home every minute.
* The lane stops the VM when its batch is done, and relaunches after a
  reclaim.

The baseline took 12 VM launches. No banked cell was lost or duplicated,
and engine-level parity passed on every VM.

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
