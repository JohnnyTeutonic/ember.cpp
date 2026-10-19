# Mixture-of-experts inference, 20 Oct 2026

Model: paperkiln's `moe_tiny.gguf` (sha256 cd17e559..., d128 L2, E=4 experts, top-2,
renormalised gate), llama.cpp layout: `ffn_gate_inp` (router, ne {d, E}),
`ffn_{gate,up}_exps` (ne {d, d_ff, E}), `ffn_down_exps` (ne {d_ff, d, E}),
`llama.expert_count` / `llama.expert_used_count`.

ember's MoE FFN (`utils.cpp`, `moe_ffn_f32_cpu`), paperkiln's IR op `moe_ffn`: the top-k
of the E router logits (ties to the lower index), gates = a softmax over the k selected
logits (renorm; or the E-way softmax's entries, raw), output the gate-weighted sum of the
selected experts' SwiGLU FFNs in ascending expert index. It runs on the CPU paths (single
token and batch prefill), and on the GPU through `moe_ffn_cuda` (below).

## Gate

`moe_gate.log`: greedy replies vs paperkiln's recorded mtstudio reference
(`mtstudio_greedy.jsonl`, 40 frozen prompts, up to 60 tokens): **40/40 identical on CPU,
CUDA and WASM**, token counts and stop reasons identical.

`dense_regression.log`: with MoE in, the dense `chat_sftfinal_v2` model stays 40/40 vs
mtstudio on CUDA, CPU and WASM; `v2_q8all` stays at 36/40 (as in `../quant_2026-10-19`).

## CUDA (`cuda_moe_gate.log`)

`moe_ffn_cuda` (`cuda_kernels.cu`): router logits by cuBLAS, a routing kernel (top-k, ties to
the lower index, gates as the CPU kernel computes them), every expert's SwiGLU densely over
all T tokens (cuBLAS FP32), and a combine over the selected experts in ascending index. Calls
with T * k <= 32 (decode, short prefills) are routed instead: the routing kernel writes the
k selected experts per token in ascending order with their gates and fills device pointer
arrays, and `cublasSgemmBatched` runs only those experts. Neither path needs a host sync, so
both run in batch prefill, per-launch decode and the captured CUDA-graph decode.
`EMBER_MOE_DENSE=1` forces the dense path. Router and expert stacks are uploaded per GPU layer.

`tools/moe_ffn_test.cu` (`cuda_moe_routed.log`): E = 8, k = 2 (Mixtral-style, the FineWeb
upcycle shape), renorm and raw gates, T = 1-16 routed and 17-40 dense, against a
double-precision reference: max error <= 2.8e-5 everywhere; dense forced at every T also
passes.

`moe_tiny_v3` (c9ed2a99), 40 frozen prompts vs mtstudio's greedy reference:

| Run | replies = mtstudio | decode p50 |
|---|---|---|
| CPU | 40/40 | 10.1 ms/token |
| CUDA, graph decode (dense only, 7f01834) | 40/40 | 2.1 ms/token |
| CUDA, per-launch decode | 40/40 | 6.1 ms/token |
| CUDA, routed, graph decode (two rounds) | 40/40 | 1.8 / 1.3 ms/token |
| CUDA, dense forced, graph decode (two rounds) | 40/40 | 1.8 / 1.7 ms/token |

First-token logits CUDA vs CPU: max 1.2e-3 (7.8e-5 of range), top-1 40/40. Dense
`chat_sftfinal_v3` on CUDA stays 40/40.

moe_tiny (E = 4, k = 2) is launch-bound, so routing gains little there; at E = 8, k = 2 the
routed decode reads a quarter of the expert weights the dense one does.
