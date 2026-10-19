# Sliding-window attention with sinks (IR interpreter, CPU), 20 Oct 2026

The rule, from paperkiln `src/ops.cpp` `swa_attention` (line ~1416; `tests/test_swa.cpp` pins it):
for query position p (from the start of the sequence), window W >= 1 and sinks S,
`win_lo = max(0, p+1-W)`, `sink_hi = min(S, p+1, win_lo)`; visible keys
`[0, sink_hi) U [win_lo, p]`: the last W positions including p, plus the first S positions,
counted once. Scores are scaled, softmaxed over the visible keys (sinks first, then the
window), masked keys exactly zero. W >= T with S = 0 is exact causal attention.

ember: `cpu_attention.cpp` `attention_window_batch_cpu` scores, softmaxes and accumulates the
visible keys in ascending position with the arithmetic of `attention_batch_cpu`; window 0 calls
the unchanged kernel. The interpreter reads window/sinks from the attention node or its kv
state, refuses a node that disagrees with a windowed state, causal=false, or negative values,
and any IR with a window always runs on the interpreter (the tuned llama path has none).

| Check | Result |
|---|---|
| kernel vs paperkiln's own `ops::swa_attention` (12 cases; full prefill, prefill + decode, chunks of 5) | max abs diff 3.58e-7 |
| kernel vs an independent double-precision rule (12 cases, grouped-query heads) | max abs diff 4.77e-7 |
| window 0, and windows covering every query (sinks 0-8), vs the unchanged kernel (30 runs) | bitwise equal |
| gpt2_tiny via the IR (window 0) | 40/40 vs mtstudio |
| chat_sftfinal_v3 via the IR (window 0) | 40/40 vs mtstudio |
| end to end: gpt2_tiny weights with a windowed IR vs paperkiln ParityLM(SWA) mtstudio, (W, S) = (8,2), (16,0), (4,6), (32,4) | 40/40 each (replies, tokens, stops, prompt tokens) |

In the end-to-end runs the windowed replies differ from the window-0 replies in 40, 37, 40 and
29 of 40 prompts, so the window is active. `swa_attention_test.txt`, `*.jsonl` and `e2e/` (IR
side files, specs, logs) are the receipts; the kernel test is `tools/swa_attention_test.cpp`,
the paperkiln-op dumper `tools/swa_reference_dump.cpp` (build lines in their headers).

Not yet: paperkiln's trained swa_tiny with its exported IR; a CUDA kernel; a ring-buffer cache
(the cache keeps every position up to max_len).
