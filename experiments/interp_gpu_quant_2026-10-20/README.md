# IR interpreter on the GPU, and quantised weights, 20 Oct 2026

Both by sub-agents (each in its own source copy, delivering a patch), reviewed, merged onto
the sliding-window work and re-gated on the merged tree (`merged_gate.log`).

## CUDA interpreter (`graph_ir_cuda.{h,cu}`, `graph_ir.cpp`)

In a CUDA build with GPU layers requested the interpreter runs on the device: weights uploaded
once at bind, KV state and RoPE table on the device, persistent per-name device buffers (no
allocation in steady state), one stream, token ids up and logits down once per call. Kernels:
cuBLAS FP32 GEMMs, the existing rmsnorm / decode-attention / prefill-attention / moe_ffn
kernels, and new embedding, pos_embedding, layernorm, bias, add/mul, silu, gelu and rope.
`EMBER_IR_DEVICE=cpu` keeps it on the CPU. Sliding-window IRs stay on the CPU interpreter (no
windowed device kernel yet; the device attention also refuses a window).

| Run | replies = mtstudio | decode p50 |
|---|---|---|
| gpt2_tiny, GPU / CPU interpreter | 40/40 / 40/40 | - |
| moe_tiny_v3 via the IR, GPU | 40/40 | - |
| chat_sftfinal_v3 via the IR, GPU / CPU | 40/40 / 40/40 | 8.5 / 14.7 ms/token |
| windowed gpt2 IRs (w8 s2, w4 s6) on the CUDA build | 40/40 (CPU interpreter, as designed) | - |

First-token logits GPU vs CPU interpreter (agent's run, `cuda_agent_gates.log`): gpt2 8.1e-7,
MoE 9.3e-7, llama 4.6e-6 of the logit range, top-1 40/40 each.

## Quantised weights (`graph_ir.cpp`, `utils.cpp` matmul_rows_f32_batch_cpu, `weight_management.cpp`)

`linear` and `embedding` weights can stay in their GGUF blocks (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1,
F16); each call dequantises one row into a scratch buffer with the load-time routines and dots
it in the f32 kernel's order, so results equal dequantising at load bit for bit. Kept quantised
by default only for models over 64M parameters on the CPU (as the llama path's
dequantize_all_if_small); small models and the CUDA path dequantise at load. EMBER_IR_QUANT=1
/ EMBER_IR_DEQUANT=1 force either.

- Forced quantised vs default on q4_k_s and q8all (via the IR): first-token logits
  byte-identical, replies 40/40 identical (23/40 and 36/40 vs mtstudio, as before).
- Agent's full run (`quant_agent_summary.log`): q4_0, q4_k_m, q4_k_s, q8all, f32 all
  byte-identical; the interpreter's weights 67 MiB f32 -> 12-18 MiB; decode 1.2-1.8x slower on
  these small models (per-row dequantisation), the reason for the 64M threshold.

WASM builds; gpt2 via the IR 40/40 in WASM.
