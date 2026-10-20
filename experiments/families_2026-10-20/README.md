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
| flex_plain_tiny (gpt2; residual "plain", no residual adds) | 40/40 | 40/40 (GPU) | 40/40 | 8 |

The CUDA-build column for the first five rows ran on the CPU interpreter (`device_supported`);
the device kernels for these ops came in 1232431 (`../device_ops_2026-10-20/`), and
flex_plain_tiny ran on them (`[IR] CUDA device path`). mamba_tiny's 40 references are one repeated text, so its replies are weak
evidence on their own; first-token logits against paperkiln's numpy IR interpreter
(`logits_vs_paperkiln.txt`): mamba max abs diff 8.6e-6 (8.0e-7 of the logit range), top-1
40/40; flex_highway 8.3e-6 (6.6e-7), top-1 40/40 (paperkiln's own check of its numpy
interpreter against its forward: 4.2e-7 / 4.7e-7).

Gating ops, op level (sub-agent, `../gating_2026-10-20/op_gate.log`): EXACT ops, sigmoid and
rms_row bitwise equal to paperkiln's; depth_attend within ROUNDING (worst 1.07e-6, inside the
spec's 1e-5 x max(1, max|ref|) per tensor). The interpreter TU builds with -ffp-contract=off
(0 FMA instructions). Regressions (`regress_*.jsonl`): swa_tiny, gpt2, MoE and llama via the
IR on the GPU, and the llama tuned path, 40/40 each.

flex_plain_tiny (`flex_plain/`; d128 L2 T128, 3,000 Colab steps, val 5.68; GGUF da076a9f...,
reference 62ebeba9..., hashes checked before the run; ember at 1232431): paperkiln's numpy IR
interpreter 40/40 against the same reference. With only 8 distinct replies, first-token logits
are the stronger check: against the numpy interpreter, CPU max abs diff 1.07e-5 (6.7e-7 of the
logit range) and GPU 1.24e-5 (7.8e-7), top-1 40/40 each, smallest reference top-1 margin 0.179
(`flex_plain/gate.log`). WASM 543 ms per reply in node.
