# Mixture-of-experts inference, 20 Oct 2026

Model: paperkiln's `moe_tiny.gguf` (sha256 cd17e559..., d128 L2, E=4 experts, top-2,
renormalised gate), llama.cpp layout: `ffn_gate_inp` (router, ne {d, E}),
`ffn_{gate,up}_exps` (ne {d, d_ff, E}), `ffn_down_exps` (ne {d_ff, d, E}),
`llama.expert_count` / `llama.expert_used_count`.

ember's MoE FFN (`utils.cpp`, `moe_ffn_f32_cpu`): softmax over the E router logits, top-k
by probability (ties to the lower index), probabilities renormalised over the k, output
the weighted sum of the selected experts' SwiGLU FFNs. It runs on the CPU paths (single
token and batch prefill); a CUDA build runs MoE models' layers on the CPU, with a warning.

## Gate

`moe_gate.log`: greedy replies vs paperkiln's recorded mtstudio reference
(`mtstudio_greedy.jsonl`, 40 frozen prompts, up to 60 tokens): **40/40 identical on CPU,
CUDA and WASM**, token counts and stop reasons identical.

`dense_regression.log`: with MoE in, the dense `chat_sftfinal_v2` model stays 40/40 vs
mtstudio on CUDA, CPU and WASM; `v2_q8all` stays at 36/40 (as in `../quant_2026-10-19`).
