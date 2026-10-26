# The speculative verification pass at about one decode step (27 Oct 2026)

`EMBER_SPEC=k` (prompt-lookup drafts, greedy; `../spec_2026-10-20`) was exact but not faster: its verification pass, a
forward over `n = 1 + k` rows, cost about 2.5 decode steps. This work measures where that time went and adds a pass
that costs **0.89-1.07 decode steps at a 32-token context and 0.91-1.34 at 384 cached tokens** on an L4, for k = 2, 4
and 8. With it, prompt-lookup speculation is **1.11-1.14x plain decode on the 40 frozen chat prompts and 1.73-5.8x on
a text-repeating workload**, with replies and ids identical to plain decode on every run (CPU and CUDA, both models,
with and without `EMBER_DETERMINISTIC=1`). Off by default, as before; with `EMBER_SPEC` unset nothing runs
differently.

## The new pass (`speculative_verify.cpp`, `verify_kernels.cu`)

`forward_device_verify_graph`: the pass for n rows is one CUDA graph per n, captured on first use and replayed with
the position and the ids in device memory (as `forward_device_graph` replays the decode step). Every row goes
through the decode step's own kernels with a row index added (embedding, rmsnorm (its deterministic kernel under
the mode), RoPE, KV write, the split-K decode attention chunked for the cache exactly as the graph decode chunks
it, residual adds, SwiGLU). The linear layers are a choice, `EMBER_VERIFY_LINEAR`:

| option | what | rows against the decode step |
|---|---|---|
| `smallm` (default) | one warp per output, all n rows at once (the weights read once per pass) | not bit-equal (max \|d\| 3.3e-6 to 5.7e-6 at n = 9); argmax 9/9 |
| `gemm` | cuBLAS SGEMM with n columns (the legacy pass's call) | max \|d\| 8.6e-6 to 1.7e-5 |
| `gemvb` | `cublasSgemvStridedBatched`, the weight shared | max \|d\| 3.5e-6 to 6.2e-6 |
| `rows` | the decode step's own `cublasSgemm(n = 1)`, once per row | **bit-identical** under `EMBER_DETERMINISTIC=1`: 9/9 rows, \|d\| 0, both models, both prefixes |

`rows` is the exact construction (every kernel is the decode step's arithmetic); it costs n linear launches per
layer. Off the deterministic mode no option can be bit-equal, as the decode step's own rmsnorm is not (atomics;
`../determinism_2026-10-26`).

The greedy pick is made on the device (`select_rows`: per row the largest logit, its first and last index, the
runner-up and a NaN flag), so the host reads n 20-byte records instead of n vocabulary rows. api.cpp reproduces
`sample_top_k_top_p_temperature`'s top-k-1 answer from the record (first index of the maximum below temperature
0.05; else the last index, which is the sampler's (probability, index) ordering, valid when the runner-up's scaled
logit is below -1e-5, so no smaller logit can round to the same float probability) and makes the sampler's one RNG
draw. A record that cannot decide (a near tie, a NaN, `repetition_penalty_` > 1) fetches its row and runs the
sampler itself. The path applies with every layer on the GPU on the tuned llama path, dense FFN, an FP32 KV cache, the
decode graph on and n <= 16; anything else (MoE, CPU layers, a quantised cache, the interpreter, `EMBER_VERIFY=legacy`)
takes the previous passes unchanged.

## 1. Where the 2.5 steps went (L4, `vf_bench.cpp`)

`receipts/cuda/bench_*.json`; p50 of 200 calls, host wall clock to the result, in decode steps (the engine's decode
step: graph replay, logits to pinned memory, sync). Decode step: pick_e1_L16 (d320 L16, vocab 16,384) 1.007 ms at a
32-token prefix, 1.473 ms at 384; pk_base_v1 (d768 L12) 2.252 / 2.941 ms. The host sampler adds 0.23 ms per token to
both (measured apart).

The ladder at n = 5 (k = 4), each step one change:

| | pick, P 32 | pick, P 384 | pk, P 32 | pk, P 384 |
|---|---|---|---|---|
| legacy pass (`forward_device_verify`) | 2.165 | 3.351 | 1.424 | 1.792 |
| the new path's kernels, no graph, cuBLAS GEMMs, attention row by row, picks | 2.012 | 3.252 | 1.350 | 1.739 |
| + one CUDA graph | 1.704 | 3.035 | 1.257 | 1.647 |
| + attention for all rows in one launch | 1.157 | 1.395 | 1.096 | 1.061 |
| + `smallm` linear layers (the new pass) | **1.005** | **1.293** | **0.913** | **0.933** |
| the new pass, then every row's logits to the host | 1.174 | 1.408 | 0.981 | 0.985 |

So the 2.5 steps were, in order: **attention run row by row** (n launches of the decode attention kernel per layer,
each ~35.6 µs at 384 keys: 74% of the legacy pass's kernel time on pick_e1_L16 and 54% on pk_base_v1 under nsys,
`receipts/cuda/nsys_*_nvtx_kern_sum.csv`); **launch overhead** (no graph: 0.1-0.3 steps); **cuBLAS GEMMs with n
columns** (`gemmSN`/`ampere_sgemm_32x32`, 0.1-0.2 steps over `smallm`); and **the rows' logits copied to the host**
(0.05-0.17 steps). The accept loop's host sampler costs the same per accepted token as in plain decode.

Verify cost in decode steps, before (legacy) and after (default), p50:

| | n | pick P 32 | pick P 384 | pk P 32 | pk P 384 |
|---|---|---|---|---|---|
| k = 2 | 3 | 1.813 -> **0.984** | 2.488 -> **1.286** | 1.209 -> **0.886** | 1.397 -> **0.910** |
| k = 4 | 5 | 2.165 -> **1.005** | 3.351 -> **1.293** | 1.424 -> **0.913** | 1.792 -> **0.933** |
| k = 8 | 9 | 3.006 -> **1.065** | 5.141 -> **1.335** | 1.694 -> **0.973** | 2.455 -> **0.982** |

(Default mode; `EMBER_DETERMINISTIC=1` gives the same within 0.01.) Below one step: the pass returns picks where the
decode step returns a vocabulary row, and `smallm` beats the decode step's own cuBLAS GEMV on pk_base_v1. Above one
step at 384 tokens on pick_e1_L16: the decode attention kernel, the same per row in both, takes 35.6 µs per layer
there (2 chunks of 256 keys for a 512-position cache), and n rows of it start more blocks than one; it is also the
largest part of the decode step itself at that length. The other options, all n = 2..9: `ladder_ms` in the bench
receipts (`rows` 1.3-4.7 steps; `gemvb` grows with n on pk_base_v1, to 2.0 at n = 9).

## 2. End to end (L4, `vf_e2e.py`)

`receipts/cuda/e2e/`, `receipts/cuda/e2e_summary.txt`. One server process per configuration, greedy, every request
sent twice (the second is the record; its reply equals the first's on every request). Request sets (`vf_prompts.py`):
**chat40**, the 40 frozen chat prompts, 60 new tokens; **copy20**, 20 text-repeating requests from the held-out
TinyStories panel (stories 0-19, first three sentences): the assistant has told the passage twice and the user asks
for it again word for word, 80 new tokens (the tiny models follow no bare "repeat this" instruction; told twice,
both copy). Decode tokens per second = tokens after the first over the time after the first token.

| | k | chat40 tok/s (x plain) | accept | copy20 tok/s (x plain) | accept | tokens per pass (chat / copy) |
|---|---|---|---|---|---|---|
| pick_e1_L16 | plain | 738 | | 656 | | |
| | 2 | 825 (**1.118**) | 0.49 | 1134 (**1.729**) | 0.71 | 1.98 / 2.42 |
| | 4 | 836 (**1.133**) | 0.37 | 1293 (**1.972**) | 0.59 | 2.47 / 3.25 |
| | 8 | 836 (**1.132**) | 0.22 | 1395 (**2.126**) | 0.44 | 2.58 / 4.09 |
| | 4, legacy pass | 746 (1.010) | | 706 (1.076) | | |
| | 8, legacy pass | 722 (0.978) | | 665 (1.014) | | |
| pk_base_v1 | plain | 379 | | 344 | | |
| | 2 | 423 (**1.114**) | 0.63 | 1033 (**3.003**) | 0.92 | 2.26 / 2.85 |
| | 4 | 428 (**1.128**) | 0.49 | 1466 (**4.260**) | 0.89 | 2.88 / 4.50 |
| | 8 | 432 (**1.139**) | 0.36 | 2001 (**5.815**) | 0.80 | 3.66 / 7.28 |
| | 4, legacy pass | 407 (1.074) | | 759 (2.205) | | |
| | 8, legacy pass | 411 (1.083) | | 880 (2.558) | | |

Acceptance = accepted / drafted. Plain decode measured twice (start and end of each model's runs): within 0.3%.
Other linear options at k = 4: `gemm` 1.116 / 1.881 (pick), 1.118 / 3.782 (pk); `gemvb` 1.117 / 1.875, 1.096 / 3.087;
`rows` (exact) 1.019 / 1.388, 1.019 / 1.801. Under `EMBER_DETERMINISTIC=1` the speed-ups are the same within 0.02.

Chat gains little because drafts are rare and short (a pass yields 2-3.7 tokens); the pass no longer costs more than
the step it replaces, so speculation now never loses on chat (k = 8 was 0.91x before) and repays every accepted
token on text it can copy.

## 3. Exactness

- **CUDA** (both models, chat40 + copy20, 60 requests per run): replies, ids and stop ids identical to plain decode
  **60/60 in every configuration**: k = 2, 4, 8, each linear option, the legacy pass; and under
  `EMBER_DETERMINISTIC=1`, k = 2, 4, 8 and `rows` at k = 4 and 8 against plain decode in the mode (ids identical).
- **CPU** (WSL, 2 threads, `receipts/cpu/`): k = 2, 4, 8 against plain on both models: replies and ids 60/60 (the CPU
  pass is the existing `forward_cpu_batch` path, unchanged). Draft, accept and pass counts equal the CUDA runs' for
  every configuration, and CPU and CUDA plain decode give the same ids, 60/60 on both models. CPU speed is a shared laptop's, indicative only (pick 1.01-1.24x at k = 2-4, 0.89-0.95x at
  k = 8; pk 0.72-1.41x).
- The default (`smallm`) rows are not the decode step's bits (max \|d\| ~5e-6); their identity of ids is measured on
  these 120 requests per model, not proved. `EMBER_VERIFY_LINEAR=rows` is bit-identical by construction and by
  measurement under `EMBER_DETERMINISTIC=1`.

## 4. Speculation off: unchanged

- **SASS** (`receipts/cuda/sass.json`, L4 sm_89; `sass_sm89_wsl_check.json`, built in WSL): every one of 70ed063's 80
  kernels has the same instructions in this build; 24 kernels are added (`ember_verify::`). `sass_compare.py` reports
  `graph_ir_cuda.cu`'s `embedding_kernel` as differing only because its parse keeps the blank lines before the next
  fatbin section, which is now `verify_kernels.cu`'s; `vf_sass_check.py` (the same parse without blank lines): 80/80.
- **Tuned llama path, pick_e1_L16** (`x5_plain_collect.py`, 360 requests, every sampled position's logits):
  `EMBER_DETERMINISTIC=1`, 70ed063 against this build: **bit-identical at every position, 360/360**. Default mode:
  replies, stop and ids 360/360, first-token logits bit-equal 40/40 on fresh prompts (decode positions spread as
  default-mode decode does run to run, `../determinism_2026-10-26`).
- **Interpreter** (`ir_device_dump`, prefill and graph decode): 70ed063 = this build, file for file, on the 16 local
  device-gate models (`receipts/cuda/dumps.txt`; the third-party checkpoints and ar_x5's F32 file were not uploaded).
- **CPU receipts** (`receipts/cpu/cpu_regress.txt`, `../cuda_state_2026-10-25/cpu_regress.sh`): recursion, MoR, MoR
  refusals, attnres, state4, regress_parity and G5 third party byte-identical to the committed receipts.
  state_tokens_gate equals 70ed063's own CPU build's receipt byte for byte (both differ from the committed file in
  the numpy distances only, as before). regress_bits, old side 70ed063's ir_logits: bit-equal on every case both
  run; swa_tiny and kimi_tiny abort in both builds (70ed063 refuses them), the receipt's only differences. The IR
  adapter self-test stops at the same rule-38 refusal with 70ed063's own ir_adapter, output identical: a state of
  70ed063, not of this change.

## Files

- `vf_bench.cpp` (built as `tools/vf_bench` by `build.sh`), `vf_e2e.py`, `vf_prompts.py`, `vf_summary.py`,
  `vf_sass_check.py`.
- `build.sh`, `stage.sh`, `drive.sh`, `vm.sh`: the Colab run (microtorch/AGENTS.md path C; session `ember-vf-1027`,
  an L4, 27 min; builds overlapping the model uploads; stopped). `cpu_legs.sh`: the CPU legs.
- `receipts/cuda/`: the VM log, hardware, binaries' and models' sha256, benches, end-to-end records and summary,
  nsys kernel tables, SASS, the 70ed063 comparisons; `receipts/cpu/`: CPU end-to-end records and the regression.
  The full VM output is in `C:/ml_artifacts/ember_verify/ember-vf-1027/`.

Run: `bash build.sh cpu|cuda`; `git archive 70ed063` to `<in>/ember_head.tgz`; `bash stage.sh <in> ~/vf/src
~/vf/stage && bash drive.sh <session> 115 ~/vf/stage`; `bash cpu_legs.sh <CPU build> <70ed063 CPU build> <work>
<REF>`; `python3 vf_summary.py bench|e2e <out>`.
