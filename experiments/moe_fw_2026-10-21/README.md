# A trained 70M mixture of experts (fw_moe), 21 Oct 2026

paperkiln's FineWeb MoE: llama family, d384 L6, 8 experts top-2 with renormalised gates,
sparse-upcycled from fw_base and trained 10,000 steps (best val 3.3516; dense control 3.3735),
70.4M parameters. GGUF f32 (294 MB, IR embedded) and `q8_0_all` (expert stacks stay f32).
Reference: mtstudio greedy on the GPU (device ops + cuBLAS, FP32, no TF32), 40 frozen prompts,
60 new tokens, 34 distinct replies, served from the step-10,000 weights with rope_heads "all".
Hashes in `SHA256SUMS`; ember 2030ca5. This is the first trained model on the routed E=8 k=2
decode kernel, with uneven routing (the final step's per-layer expert shares run 9.4% to 16.9%).

## Greedy replies vs mtstudio (`gate.log`)

| Path | replies identical | token count | stop reason | wall, 40 replies |
|---|---|---|---|---|
| CPU, tuned llama path | 40/40 | 40 | 40 | 291 s |
| CUDA, routed experts | 40/40 | 40 | 40 | 49 s |
| CUDA, dense experts (`EMBER_MOE_DENSE=1`) | 40/40 | 40 | 40 | 78 s |
| CUDA, IR interpreter (`EMBER_INTERPRETER=1`) | 40/40 | 40 | 40 | 64 s |
| CPU, IR interpreter | 40/40 | 40 | 40 | 147 s |
| WebAssembly (`EMBER_WASM_MAX_MEMORY=4gb` build) | 40/40 | 40 | - | 5.0 s per reply (node) |
| `q8_0_all`, CUDA | 25/40 | 35 | 36 | 29 s |
| `q8_0_all`, CPU | 25/40 | 35 | 36 | 148 s |

First-token logits over the 40 prompts: CPU tuned vs CUDA IR interpreter max abs diff 1e-4
(4.4e-6 of the logit range); CPU vs CUDA tuned 0.15 (6.8e-3; the tuned CUDA prefill uses TF32
by default, `EMBER_GEMM_TF32=0` pins FP32); top-1 40/40 in every pair. The smallest f32
top-1 margin is 0.0048. paperkiln's numpy IR interpreter (`tools/ir_run.py`, no shared code)
gives ember's replies on the 5 prompts run (`ir5.jsonl`).

## Q8_0

Against f32 on the CPU: max abs dlogit 0.25 (1.1e-2 of the range), mean 0.016, first-token
top-1 38/40. The two flips are prompts 13 and 16 (f32 margins 0.082 and 0.025, overturned by
0.12 and 0.04). Mean error equals `v2_q8all` in `../quant_2026-10-19/` (0.016); the maximum
is higher, and this base model's top-1 margins are smaller (0.0048 against 0.035), so whole
replies meet near-ties sooner: 25/40 identical, the rest plausible continuations that part at
a near-tie (prompt 9 repeats "?" after the f32 reply's own first "?"). CPU and CUDA give the
same Q8 replies. This model misses the quantised gate's first-token criterion (40/40, error
<= 1% of range); reported, not passed.

## The first reference

`gate_first_reference.log` is the same gate against an earlier mtstudio recording: 0/40 on
every path, while all ember paths and the numpy interpreter agreed with each other. That
recording was served from a directory without `events.jsonl` whose spec did not state
`rope_heads`, so mtstudio fell back to the legacy "first" RoPE heads for weights trained with
"all". `mtstudio chat` refuses bare llama weights without `arch.rope_heads` (paperkiln
67da303d).
