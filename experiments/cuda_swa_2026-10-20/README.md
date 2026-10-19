# Sliding-window attention on CUDA; compiled device plan, 20 Oct 2026

By a sub-agent (own source copy, patch), reviewed, merged and re-gated (`merge_gate.log`).

- `ir_window_attention_cuda` (`graph_ir_cuda.cu`): one block per (token, query head), decode and
  prefill alike; visits only the visible keys (sinks, then the window, ascending; the CPU's
  `swa_visible_ranges` rule), online softmax over 1024-key tiles (no window-length limit),
  head_dim <= 1024. Window 0 keeps the split-K decode and prefill kernels unchanged. Windowed
  IRs now run on the GPU interpreter.
- A compiled per-layer plan for the device path (`compile_device`): op enum, buffer slots,
  resolved device weight pointers and numeric attributes, built once at load.

| Check | Result |
|---|---|
| kernel vs `attention_window_batch_cpu` (`window_attention_cuda_test.txt`: decode past the window, window < sinks, window >= T, window 1, head_dim 96, >1024 visible keys; unread cache positions NaN) | max abs diff 5.1e-6 |
| swa_tiny on the GPU interpreter, 200 tokens | 40/40 vs mtstudio |
| windowed gpt2 IRs (w8 s2, w32 s4) on the GPU | 40/40 each |
| gpt2, MoE, llama via the IR on the GPU; llama tuned path | 40/40 each |

Decode speed did not measurably change with the plan (load average 7-16 throughout; see
`ir_decode_bench.txt`). The nsys API trace shows the device decode bound by launch count (~135
kernel launches, 14 copies and ~180 cuBLAS capture queries per token), not attribute lookups:
the next step is capturing the T = 1 step as a CUDA graph with a device-side position, as the
llama path does. At swa_tiny's size the CPU interpreter is faster than the GPU one.
