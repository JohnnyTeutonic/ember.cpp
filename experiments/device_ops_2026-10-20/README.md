# Every interpreter op on the GPU; CUDA-graph decode for the interpreter, 20 Oct 2026

By a sub-agent (own source copy, patch), reviewed, merged and re-gated (`merge_gate.log`).

- Device kernels for sub, relu, sigmoid, affine, mul_col, rms_row, list/append, depth_attend,
  linear_attention and ssm_scan, in the CPU kernels' order of operations, each multiply and add
  rounded separately (no FMA). Lists are values (each list slot owns copies of its tensors).
- Recurrent state on the device: one live buffer per state (restores copy into it, so a captured
  graph always reads the live state); snapshots in a position -> buffer map with a free pool,
  mirroring the CPU's reset / restore / drop / eviction / resume_point logic.
- Every interpreter op now has a CUDA kernel; `device_supported()` sends an IR to the CPU only
  past two kernel limits (ssm_scan n > 6144, more than 128 depth_attend sources).
  EMBER_IR_DEVICE=cpu still forces the CPU.
- CUDA graph of the T = 1 step: captured once; token and position live in a device buffer;
  pos_embedding, rope, the window kernel and a KV-write kernel read the position there; full
  attention uses attention_devpos_cuda. Recaptured after any buffer reallocation; a failed capture
  runs that step per-launch, and four in a row switch the graph off; EMBER_IR_GRAPH=0 disables it.

| Check | Result |
|---|---|
| op test (`ir_device_ops_test.txt`): EXACT ops, rms_row, ssm_scan bitwise; depth_attend 2.4e-7; linear_attention 9.5e-7; recurrent prefill == step decode | PASS |
| five trained families on the GPU, graph and EMBER_IR_GRAPH=0 | 40/40 each |
| GPU vs CPU interpreter logits, all five families (40 first-token + 160 decode rows) | worst 0.155 of the 5e-5 x max(1, max abs) tolerance; mamba 9.3e-7 of range, top-1 40/40 |
| recurrent state on the GPU: repeat requests, interleaved prompts, two-turn chats, kimi EMBER_SPEC=4 | identical to from scratch; reused-token counts equal the CPU's |
| regressions: swa_tiny, gpt2, MoE and llama via the IR on the GPU (graph and not), llama tuned path, gpt2 on the CPU build, srd in WASM | 40/40 each |

Decode, llama via the IR on the GPU (chat_decode_bench p50): agent's six rounds at load
1.9-2.45, graph 2.07-2.35 ms/token vs per-launch 3.45-4.99 and the tuned path 2.03-2.40 (one
outlier each from the other session's GPU job); merge gate 2.03 ms/token.
