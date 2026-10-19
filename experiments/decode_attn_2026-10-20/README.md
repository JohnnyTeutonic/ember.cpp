# Decode attention, 20 Oct 2026

## Correctness

`tools/decode_attention_test.cu` checks the three CUDA decode-attention entry points
(per-launch, graph/device-position, basic) against a double-precision CPU reference:
32 query / 4 KV heads, head_dim 64, cache of 8192 positions, lengths 1 to 8192 across chunk
boundaries; pass is max |error| < 1e-4.

The one-block-per-head kernel (`attention_optimized_kernel`, ember's CUDA decode
attention for every model until now) failed 16 of 24 cases, errors 0.3-1.6, from 255
tokens up, varying from run to run. Cause: a shared-memory race. Every thread read the
block maximum from a shared slot that thread 0 then reset for the sum without a barrier,
so some warps normalised with a different maximum. Short chat prompts usually won the
race, which is why the chat parity gates (contexts under 256 tokens) stayed 40/40. The
basic `attention_cuda` path launched with no dynamic shared memory for its score array
and faulted. Both fixed; all three paths now pass at every length (max error 2e-7 to 1e-6).

## Split-K kernel

Decode attention now runs as two kernels (`cuda_kernels.cu`, `decode_attn_split`): a grid
of (heads x key chunks), warps computing q.k with coalesced K rows and a chunk-local
softmax, then a per-head merge of the chunks' (max, sum, partial output). The scratch is a
static device array, so CUDA graph capture is unchanged; chunks past the current length
exit early. `EMBER_DECODE_ATTN=legacy` selects the (race-fixed) one-block-per-head kernel;
shapes beyond the scratch (more than 128 heads, head_dim above 256, chunks above 2048 keys)
fall back to it.

Timing model (paperkiln `make_timing_gguf`, random weights: d=2048, 4 layers, 32/4 heads,
ff 5632, context 8192), RTX 4060 laptop, graph decode, `tools/decode_ctx_bench.py`, three
interleaved rounds (another process shared the GPU, so absolute numbers are noisy):

| Context | split | legacy (race-fixed) |
|---|---|---|
| 128 | 5.5-7.6 ms/token | 5.6-8.6 |
| 1024 | 6.1-6.7 | 7.5-9.3 |
| 4096 | 8.1-10.9 | 15.4-19.8 |
| 8000 | 12.2-16.2 | 24.7-33.5 |

The context cost falls from about 2.5 to about 1.2 ms per 1,000 tokens; a run on an idle
GPU measured 9.3 ms/token at 8000 with split. Greedy output, split vs race-fixed legacy,
8 prompts at context 4096: 8/8 identical (before the race fix: 0/8, diverging within 1-4
steps, `first_gate_before_race_fix.log`). Chat `chat_sftfinal_v2` on CUDA: 40/40 vs
mtstudio with graph and with per-launch decode.
