# Speculative decoding (prompt lookup), 20 Oct 2026

`EMBER_SPEC=k` (off by default): greedy (`top_k` 1) decoding with prompt-lookup drafts.
Each step looks for the most recent earlier occurrence of the last 3 (then 2) tokens in the
prompt plus reply, proposes up to k of the tokens that followed it, and verifies them in one
batched forward from the current position (all-GPU: `forward_device_batch_prefill` with
`all_logits`, the LM head in FP32 on every row; all-CPU: `forward_cpu_batch` +
`forward_cpu_logits_batch`). Tokens are taken while the greedy choice equals the draft, plus
the first that does not. No draft model. `/chat` reports `spec_drafted`, `spec_accepted` and
`spec_passes` in its `ttft` object; `tools/spec_bench.py` runs and compares.

## Results

Exactness: replies identical to `EMBER_SPEC` off on all 40 frozen chat prompts
(`chat_sftfinal_v2`, greedy, 60 tokens) with k = 4 and 8, on CUDA and CPU; 8/8 on the timing
model at context 2048 (CUDA).

| Run | drafted | accepted | passes | total wall vs off |
|---|---|---|---|---|
| CUDA, k=4 | 76 | 21 (28%) | 19 | 0.90x |
| CUDA, k=8 | 144 | 22 (15%) | 18 | 0.91x |
| CPU, k=4 | 76 | 21 (28%) | 19 | 0.97x |
| CPU, k=8 | 144 | 22 (15%) | 18 | 1.22x (within CPU run-to-run noise) |

636 reply tokens in all: chat replies rarely repeat earlier text, so drafts are rare. The
random-weight timing model produced no drafts at all (no repeated bigrams; repetition
penalty 1.3 on the word-level path).

## Why there is no speed-up yet

A verification pass costs more than the decode steps it replaces. After a cached 2048-token
prefix (timing model, CUDA, via prefix reuse) a batched forward of 1, 2, 5, 9 or 17 tokens
takes 26-35 ms, against 6.3 ms for one graph-captured decode step: a fixed cost of the batch
prefill path at small batch sizes. Speculation pays only once a pass costs about one decode
step, which needs a small-batch verification path built like the graph decode (batched
matvec, per-row decode attention), and workloads that repeat text (editing, extraction,
code). Until then it stays opt-in.
