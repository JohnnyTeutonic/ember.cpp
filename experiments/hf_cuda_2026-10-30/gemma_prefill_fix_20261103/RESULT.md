# Gemma3-270M prefill re-measure: result (run `f2-20261103`)

Claude Opus 5.5, 3 Nov 2026. Packet commit `0feeff1d` (Codex round 2 RUN), engine source `5794343c` (tree
`c4926553`). One Colab L4. The driver ran from 17:52:17 to 18:15:12 UTC+11, about 23 L4-minutes, and exited 0.
`status.json` matched the run, commit, HEAD and source; all 13 gates were 0; cleanup was certified. Receipts are in
`receipts/f2-20261103/`.

Run `f1-20261103` stopped at the driver's local preflight before any session existed (it was started from an ember
checkout, and the driver calls paperkiln's `tools/colab_reap_orphans.py`). No VM was created and nothing was spent.

## Outcome: gap narrowed

Every parity gate passed (`gate_summary.json`: `COMPLETE_PARITY_GATED_CUDA_MEASUREMENT`). Medians follow, with g6
(`../gemma_colab_20261103/receipts/g6-20261103/`) for reference.

| prompt tokens | ember prefill, g6 | ember prefill, f2 | HF prefill, f2 | HF/ember prefill, f2 | HF/ember decode, f2 |
|---:|---:|---:|---:|---:|---:|
| 16 | 7.0 ms | 7.52 ms | 34.07 ms | 4.53x | 5.45x |
| 128 | 20.1 ms | 15.96 ms | 34.39 ms | 2.15x | 5.15x |
| 512 | 174.0 ms | 47.23 ms | 34.03 ms | 0.72x | 4.46x |

- **At 512 tokens:** ember's prefill is 3.7x faster than in g6, but still 1.39x slower than HF's in the same run.
- **Decode:** unchanged in character. Ember is 5.6-7.0 ms per token against HF's 30.7-31.2.
- **Cross-run caveat:** g6 ran the older frozen engine tree. A change from g6 belongs to this engine source, not to
  the patch alone. HF's figures, 34.0-34.4 ms, are the within-run reference and match g6's 34.1-34.3.

## What the kernel test adds

`kernel_test.txt` passed: 2,928 cases with 0 failures. Every kernel was within 1.4e-6 of the FP64 reference, so the
Codex round-2 bounds hold. At Gemma 3 270M's window-layer shape (4 q heads over 1 kv head, head_dim 256, 512 tokens,
window 512) the per-call medians are:

| kernel | ms per call | error vs FP64 |
|---|---:|---:|
| the IR's window kernel (before the fix) | 0.590 | 2.0e-7 |
| tiled kernel, window 512 (the fix's routing) | 1.786 | 4.8e-7 |
| tiled kernel, causal | 1.787 | 4.8e-7 |

So on Gemma's sliding-window layers the tiled kernel is about 3x slower than the kernel it replaced: the routing of
window layers made those layers slower. The prefill gain therefore came from elsewhere. The candidates are the
global layers, which at head_dim 256 moved from the baseline kernel to the tiled one, and other engine changes since
g6's tree. This run does not separate the two.

The amendment's description of the suspected cause was half right. The global layers' baseline kernel was a
bottleneck worth removing; the window kernel was not.

**Implied next step, not measured here.** Keep the tiled kernel for head_dim 256 causal layers, and send
sliding-window layers back to the window kernel. A rough estimate from the per-call timings puts that near 29 ms
at 512 tokens, below HF's 34, but it is arithmetic on component timings, not a measurement. It needs its own
reviewed run.

## Qualifications from Codex round 2 (non-blocking)

- The test tool's `cudaFree` calls are not checked.
- Timing repetitions are not NaN-filled or scored; the three correctness launches before them are.
