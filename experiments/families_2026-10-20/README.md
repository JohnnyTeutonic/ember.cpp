# paperkiln's remaining model families through the IR interpreter, 20 Oct 2026

Trained tiny models from paperkiln (branch `families` e3f909d; d128 L2, ctx 128, 400 CPU steps;
GGUF with the IR embedded; 40-prompt mtstudio greedy references, 60 new tokens; hashes in each
folder's SHA256SUMS), served by ember's interpreter with the gating / attention-residual ops
(`graph_ir_ops.{h,cpp}`, this commit) and the recurrent state (2eda1f5).

| Model (architecture) | CPU | CUDA build | WASM | distinct reference replies |
|---|---|---|---|---|
| kimi_tiny (gpt2; linear attention) | 40/40 | 40/40 | 40/40 | 38 |
| srd_tiny (paperkiln; kv + recurrent, router) | 40/40 | 40/40 | 40/40 | 38 |
| attnres_tiny (paperkiln; list / depth_attend) | 40/40 | 40/40 | 40/40 | 34 |
| flex_highway_tiny (paperkiln; rmsnorm, swiglu, sinusoidal) | 40/40 | 40/40 | 40/40 | 9 |
| mamba_tiny (paperkiln; ssm_scan) | 40/40 | 40/40 | 40/40 | 1 |

These families have no device kernels yet: the CUDA build runs them on the CPU interpreter
(`device_supported`). mamba_tiny's 40 references are one repeated text, so its replies are weak
evidence on their own; first-token logits against paperkiln's numpy IR interpreter
(`logits_vs_paperkiln.txt`): mamba max abs diff 8.6e-6 (8.0e-7 of the logit range), top-1
40/40; flex_highway 8.3e-6 (6.6e-7), top-1 40/40 (paperkiln's own check of its numpy
interpreter against its forward: 4.2e-7 / 4.7e-7).

Gating ops, op level (sub-agent, `../gating_2026-10-20/op_gate.log`): EXACT ops, sigmoid and
rms_row bitwise equal to paperkiln's; depth_attend within ROUNDING (worst 1.07e-6, inside the
spec's 1e-5 x max(1, max|ref|) per tensor). The interpreter TU builds with -ffp-contract=off
(0 FMA instructions). Regressions (`regress_*.jsonl`): swa_tiny, gpt2, MoE and llama via the
IR on the GPU, and the llama tuned path, 40/40 each.

Not yet: flex_plain_tiny (being retrained); CUDA kernels for these ops and for recurrent state.
