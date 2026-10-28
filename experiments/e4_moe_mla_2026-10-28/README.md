# E4: third-party MoE and DeepSeek MLA in ember (28 October 2026)

ember on the later families of `LOCAL_LLM_WORK.md` E4 that are wiring and a new op: Qwen-MoE (qwen2moe, qwen3moe),
Mixtral (the llama builder's MoE form) and DeepSeek's multi-head latent attention (deepseek2, and DeepSeek-V3's
router on the same graph). Code: ember **4290c91** (keramos v1.5.0's split groups, mla_attention, the qwen2moe,
qwen3moe and deepseek2 builders, their validation and ir_adapter's run_node), then **f393955** (keramos v1.6: apply,
V3's router, YaRN by value, D.11's deepseek2 graph; `../e4_gemma2_phi3_2026-10-28/`, which holds the suite receipts
at v1.5.0 and v1.6.1). Every result below is of ember **8769fb3** (GitHub main), a pinned CPU build from its git
archive (`-DHAS_CUDA=OFF`, Release, GCC 13.3, WSL ext4; `receipts/synth/bins.sha256`).

The work was written from keramos's IR_SPEC.md at the tags (5.5, 5.16, 5.20, 9.3, Appendix D.7, D.10 and D.11) and
ember's own code; paperkiln's sources and keramos's `cleanroom/` were not read. keramos v1.6.1's `tools/gguf_ir.py`
(a git archive of the tag, b74a410d) was run as a black box through its command line, `gguf_ir.py FILE OUT`. HF's
greedy continuations and logits are paperkiln's recorded ones in each file's `.cases.json`.

## What ember does

- **split groups** (5.5): x is g consecutive groups of sum(sizes) columns; output k is the k-th part of every group,
  in order. Rules 6, 10 and 37; D.7's default 1. CPU `cpu::split_groups`; CUDA one strided copy per part.
- **mla_attention** (5.20) in the absorbed form, in the reference's order (`graph_ir_ops.h` `cpu::mla_attention`):
  qa = K[h] qn_h, s = ((qa . c_j) + (qr_h . kr_j)) * scale over section 5.15's visible keys, the max-shifted
  softmax, u = sum a c, out = V[h] u. The kv slot holds [c | kr] per position (kv_heads 1, head_dim r + dr), so
  decode, truncate, snapshots and rewinds are the ordinary kv slot's; the node's rope rotates qr and kr at its own
  head_dim. Refused: rules 7, 8, 12, 14, 37 and 42; mla_attention under depth_route by name.
- **Builders** (D.10, D.11): qwen2moe (qwen2's attention, moe_ffn raw, the shared expert times
  sigmoid(ffn_gate_inp_shexp h2) through linear, sigmoid and mul_col), qwen3moe (qwen3's attention, moe_ffn renorm),
  deepseek2 (the query from attn_q or the low-rank attn_q_a / attn_q_b, split groups H into qn | qr, attn_kv_a_mqa
  split into c | kr, mla_attention, moe_ffn and the shared experts; from f393955 also the dense leading layers by
  `apply`, V3's sigmoid router with exp_probs_b and expert groups, and YaRN's divisors by value).
- **WebGPU**: `irSupport` returns false by name for mla_attention and for split with groups other than 1, so the
  page runs such a document in WebAssembly; no WGSL kernel was added.
- **CUDA**: `ir_mla_attention_cuda` and `ir_mla_kv_write_cuda` (one block per row and head, the CPU's sequential
  order with no contraction, no atomics) and the grouped split compile at 8769fb3 (nvcc 12.6, sm_75, compile only;
  `graph_ir.cpp` with HAS_CUDA under g++ -fsyntax-only); moe_ffn's raw gates,
  sigmoid and mul_col were already device kernels. On-device validation is pending: `e4_stage.sh`, `e4_drive.sh`
  and `e4_vm.sh` (microtorch/AGENTS.md path C) hold the leg, not run.

## paperkiln's synthetic files against HF

`synth_gate.sh 8769fb3 OUT` (`synth_gate.py`), seven files of `C:/ml_artifacts/thirdparty/e4_synth/`, 40 sequences
each. ember's interpreter (ir_adapter: one prefill, then 15 decode steps) against HF's 16-token greedy continuations
and HF's f32 last-position logits. The builder's document (as every loader builds it) against `gguf_ir.py`'s, field
by field except `_comment`, and in Appendix D.7's normal form at v1.6.1 (block nodes' `apply` true and moe_ffn's
`scoring` "softmax" filled in). Mixtral is a llama file: ember serves it on the tuned llama path, so that path is
gated too (`llama_path.cpp`, greedy by fresh prefills through `score_rows`), and its IR with EMBER_IR_BUILD=1.

| File (gguf / cases sha256) | Greedy vs HF | Top-1, first token | First token, worst; median max \|dlogit\| / HF range | Builder = gguf_ir.py |
|---|---:|---:|---|---|
| mixtral (52026828 / 317d9393), interpreter | 40/40 | 40/40 | 1.7e-6; 7.7e-7 | yes, both forms |
| mixtral, tuned llama path | 40/40 | 40/40 | 1.9e-6; 7.0e-7 | - |
| qwen2moe (1aa32ef3 / 09882b3a) | 40/40 | 40/40 | 3.0e-6; 9.7e-7 | yes, both forms |
| qwen3moe (b6e07bbb / 3582299c) | 40/40 | 40/40 | 8.3e-7; 4.9e-7 | yes, both forms |
| deepseek2 (521d3dd2 / 090d9542) | 40/40 | 40/40 | 2.9e-6; 6.4e-7 | yes, both forms |
| deepseek2_qlora (84d29dc4 / c3ad4e6c) | 40/40 | 40/40 | 2.0e-6; 6.0e-7 | yes, both forms |
| deepseek2_dense_yarn (d42b4fd2 / cd4f5408) | 40/40 | 40/40 | 9.0e-7; 4.8e-7 | yes, both forms |
| deepseek3 (2310e0cf / 4c276a5c) | 40/40 | 40/40 | 1.1e-6; 5.4e-7 | yes, both forms |

The files' sha256 are those paperkiln recorded. `receipts/synth/`: `synth_gate.json` (per sequence: ember's tokens,
distances, top-1), `ir/` (ember's and gguf_ir.py's documents per file), `bins.sha256`.

## WebAssembly against the CPU

`wasm_gate.sh` at 8769fb3: `tools/ir_wasm_gate` native (linked with the pinned build's `libtinyllama_core.a`) and
WebAssembly with `deploy/wasm/build.sh`'s sources and flags (-O3 -msimd128 -std=c++17 -DNDEBUG -fexceptions,
Emscripten's Boost port, the page's 1gb heap), Emscripten 6.0.6, node with NODERAWFS. ir_device_gate's 40 seeded
prompts of 24 ids: every prefill position, the token-by-token decode and 32 greedy tokens.

| File | Greedy (32) | Top-1 | Worst prefill; decode |
|---|---:|---:|---|
| mixtral (its IR, --build-ir) | 40/40 | 960/960 | 0; 0 |
| qwen2moe | 40/40 | 960/960 | 0; 0 |
| qwen3moe | 40/40 | 960/960 | 0; 0 |
| deepseek2 | 40/40 | 960/960 | 0; 0 |
| deepseek2_qlora | 40/40 | 960/960 | 0; 0 |
| deepseek2_dense_yarn | 40/40 | 960/960 | 0; 0 |
| deepseek3 | 40/40 | 960/960 | 0; 0 |

WebAssembly is bit-equal to the native CPU build at every position of every file (`receipts/wasm/`, the binaries'
sha256 in `bins.sha256`). The page serves Mixtral on the llama path.

## Files

`synth_gate.sh`, `synth_gate.py`, `llama_path.cpp`, `wasm_gate.sh`; the staged CUDA leg `e4_stage.sh`,
`e4_drive.sh`, `e4_vm.sh`; `receipts/synth/`, `receipts/wasm/`.
