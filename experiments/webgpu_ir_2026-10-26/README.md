# The IR families on WebGPU, 26 Oct 2026

The browser page's WebGPU forward pass ran llama and llama.cpp-layout MoE GGUFs only
(`deploy/wasm/web/webgpu/llama_webgpu.js`, `../webgpu_2026-10-21/`); every document ember's IR
interpreter runs went to WebAssembly. `deploy/wasm/web/webgpu/ir_webgpu.js` now runs those documents
in WGSL, and the worker follows ember's own choice of path: `ember_ir_doc()` (new WASM export) is ""
when ember serves the file on its tuned llama path, and the page then runs `llama_webgpu.js` exactly as
before; otherwise it runs the document on `ir_webgpu.js` when every op has a kernel, else WebAssembly.

## What `ir_webgpu.js` runs

Every op of the core, state, recursion and thirdparty profiles except `recurrence`: embedding (with
`state_weight`), pos_embedding, linear (bias, `tied_to`), scale, split, rmsnorm (`group`,
`weight_offset`), layernorm, gelu (tanh and erf), silu, sigmoid, relu, add, mul, sub, affine (per-layer
scale and shift), mul_col, rms_row, rope (interleaved, NEOX, partial `n_rot`, per-layer theta and
freq_scale, GQA), attention (GQA, window and sinks, the `turn_state` mask, MoR's compact rows), moe_ffn,
list / append / depth_attend, `expr` (compiled to WGSL, row reductions included), linear_attention,
ssm_scan, conv1d_causal, selective_scan, depth_route with `kv_rows: active`, and the block's
`weight_index` / `tensor_index`. Matrices stay F32, F16, BF16 or Q8_0 on the GPU (no dequantising to
f32 at load), stored transposed so one invocation computes one output row reading consecutive
addresses across the workgroup.

Per token: one command buffer of all the document's dispatches (a fresh buffer per write, so no
dispatch reads and writes one buffer), a 16-byte step buffer (position, token) and, for state tokens,
the position's turn_state code. Prefill runs one token per step, which the IR defines to equal prefill
(IR_SPEC 3.9, 6.3). Recurrent state, kv rows and MoR's per-application row counts stay on the GPU;
MoR's routing, the inactive-token pass-through and the step gate run on the GPU too (no readback).
`turn_state` keeps every kv row (eviction is a memory policy; the IR defines both to give the same
values). Refused (the page stays in WebAssembly): `recurrence`, `rope_heads` "first", lists that fork,
depth_attend with more than two extra sources, an `expr` with more inputs than the device binds, other
stored matrix types (the k-quants, Q4_0/Q5_x), kv contexts past the workgroup score array (4,096
positions at 32 KB of workgroup memory).

## The adapter: software only

GPU work is remote only (Jonathan, 26 Oct 2026; the laptop's NVIDIA and Intel GPUs both). The
21 Oct page tests ran headless Edge on the Intel iGPU; Chrome 156 on this laptop opens a D3D context on
the Intel GPU even with `--use-angle=swiftshader --use-webgpu-adapter=swiftshader --disable-gpu`, and
offers no SwiftShader WebGPU adapter, so no browser run happened here and `tools/webgpu_parity/run.py`
now refuses to run without `EMBER_WEBGPU_REMOTE=1`.

Every run below used Dawn's node bindings (npm `webgpu` 0.6.2) on SwiftShader, the CPU Vulkan device
Chrome ships, through `tools/webgpu_parity/swiftshader_run.py`: the Vulkan loader sees only
SwiftShader's ICD and no layers, the scripts refuse any adapter that does not name SwiftShader
(adapter: `google | swiftshader | swiftshader-device-subzero- | SwiftShader driver 5.0.0`), and a
monitor reading Windows' per-process GPU counters kills the run if the process opens a context on any
hardware adapter. Across the 36 completed runs the monitor read the counters 874 times and found no
context on a hardware adapter (Qwen3's stopped run had none either). Two cores (affinity), under the CPU budget.
node runs with `--max-opt=1`: dawn.node 0.6.2 crashes (access violation) when V8's Maglev-optimised code
calls into it.

The numbers say the WGSL is correct; they say nothing about speed on a GPU. SwiftShader runs WGSL on
two CPU cores: tens of milliseconds to seconds per token.

## Gate 1: logits against ember's CPU interpreter

`tools/webgpu_parity/ir_gate.mjs` on `tools/ir_wasm_gate`'s reference, so each family is gated on the
prompts of its WebAssembly and CUDA legs: 40 seeded random prompts of 24 ids (8 ids for the 16-position
recursion, MoR and ar3 models, as their WASM legs), the native CPU build's prefill logits at every
position, and 32 greedy tokens (8 for the 16-position models). The WebGPU path feeds the prompt one
token per step and compares each position's logits with the CPU's prefill row (max |dlogit| over the
row's range), the argmax at every position, then its greedy continuation. References: `ref.sh` (ember
CPU, the `ir_wasm_gate` build of `../wasm_families_2026-10-25/`, EMBER_IR_CTX 1024); third-party
documents as ember builds them (`*.ember_built.ir.json` of the builder checks, `gguf_ir.py` for Qwen3,
equal to ember's builder: 9aad4a2). Receipts: `<family>/summary.json`, `<family>/check.jsonl`; the
reference logits were deleted once each family passed.

| Family (model) | Prompts x ids + greedy | Greedy | Top-1 | Worst \|dlogit\| / range | Median | Dispatches per token | s per prompt (SwiftShader) |
|---|---|---:|---:|---:|---:|---:|---:|
| kimi (linear attention, recurrent) | 40 x 24 + 32 | 40/40 | 960/960 | 4.6e-07 | 3.5e-07 | 38 | 1.3 |
| srd (kv + recurrent, router) | 40 x 24 + 32 | 40/40 | 960/960 | 3.7e-07 | 3.2e-07 | 68 | 1.8 |
| attnres (list, depth_attend) | 40 x 24 + 32 | 40/40 | 960/960 | 4.9e-07 | 3.9e-07 | 46 | 2.5 |
| flex highway (rmsnorm, swiglu, sinusoidal) | 40 x 24 + 32 | 40/40 | 960/960 | 4.5e-07 | 3.7e-07 | 60 | 1.5 |
| paperkiln mamba (ssm_scan) | 40 x 24 + 32 | 40/40 | 960/960 | 6.8e-07 | 4.2e-07 | 34 | 1.3 |
| gpt2 | 40 x 24 + 32 | 40/40 | 960/960 | 4.2e-07 | 3.2e-07 | 40 | 1.5 |
| gpt2_swa (window, sinks) | 40 x 24 + 32 | 40/40 | 960/960 | 4.6e-07 | 3.2e-07 | 40 | 1.6 |
| expr: gpt2_tiny's GELU as a formula | 40 x 24 + 32 | 40/40 | 960/960 | 4.7e-07 | 3.2e-07 | 40 | 1.7 |
| expr: + its final norm as rowmean/sqrt + scale | 40 x 24 + 32 | 40/40 | 960/960 | 4.7e-07 | 3.2e-07 | 41 | 1.6 |
| llama MoE through the IR (moe_ffn) | 40 x 24 + 32 | 40/40 | 960/960 | 4.7e-07 | 3.7e-07 | 39 | 1.5 |
| recursion b2x4 (weight_index) | 40 x 8 + 8 | 40/40 | 320/320 | 2.1e-07 | 1.3e-07 | 139 | 0.5 |
| recursion p1_b2x3_c1 | 40 x 8 + 8 | 40/40 | 320/320 | 4.5e-07 | 1.3e-07 | 139 | 0.6 |
| recursion p1_b2x3_c1, inject add | 40 x 8 + 8 | 40/40 | 320/320 | 2.5e-07 | 1.2e-07 | 156 | 0.6 |
| Mixture-of-Recursions mor3 | 40 x 8 + 8 | 40/40 | 320/320 | 2.5e-07 | 1.3e-07 | 147 | 0.6 |
| Mixture-of-Recursions mor2p0 | 40 x 8 + 8 | 40/40 | 320/320 | 2.2e-07 | 1.5e-07 | 130 | 0.5 |
| state tokens state4 (turn_state) | 40 x 24 + 32 | 40/40 | 960/960 | 1.0e-06 | 4.3e-07 | 37 | 0.8 |
| state4 Q8_0 | 40 x 24 + 32 | 40/40 | 960/960 | 1.2e-06 | 4.2e-07 | 37 | 1.0 |
| llama_attnres ar3 | 40 x 8 + 8 | 40/40 | 320/320 | 2.5e-07 | 1.6e-07 | 69 | 1.0 |
| ar3 Q8_0 | 40 x 8 + 8 | 40/40 | 320/320 | 3.2e-07 | 1.5e-07 | 69 | 1.1 |
| GPT-2 (tiny random, third-party builder) | 40 x 24 + 32 | 40/40 | 960/960 | 8.5e-07 | 5.7e-07 | 58 | 1.2 |
| Mistral sliding window (tiny random) | 40 x 24 + 32 | 40/40 | 960/960 | 2.6e-06 | 1.2e-06 | 54 | 1.2 |
| Pythia-160m f32 | 40 x 24 + 32 | 40/40 | 960/960 | 1.6e-04 | 1.4e-05 | 243 | 47.4 |
| Mamba-130m f32 | 40 x 24 + 32 | 40/40 | 960/960 | 3.3e-04 | 4.4e-05 | 411 | 49.2 |
| Qwen3-0.6B Q8_0 | 1 of 10 run | 1/1 | 24/24 | 4.7e-06 | | | 360 |

Every greedy continuation is identical and every argmax agrees, on every family run. The paperkiln
families and the tiny random models agree within 2.6e-6 of the logit range (median 1e-7 to 1e-6), the
level of their WebAssembly legs. The two official checkpoints are past the receipts' flat 1e-4 bound
(`summary.json` `pass` false on the logit bound alone): Pythia-160m 1.6e-4 (median 1.4e-5) and Mamba-130m
3.3e-4 (median 4.4e-5). That is the float32 conditioning of deep real checkpoints, which the CUDA legs
show at the same level (RTX 4060: Pythia 2.0e-4, Mamba 1.0e-4; WebAssembly Pythia 3.6e-4): ember's CPU
accumulates its dot products in double, a GPU in f32. For official checkpoints the IR's gate is greedy
N/N and top-1 at every position, with closeness judged against float64 (IR_SPEC 4.3); both hold.
Qwen3-0.6B was stopped after its first prompt (greedy identical, 24/24, 4.7e-6): the SwiftShader run
committed 4.9 GB, the laptop had 1.6 GB of physical memory left, and one prompt took 6 minutes.

## Gate 2: the page's chat path

`tools/webgpu_parity/chat_test.mjs` runs the worker's own steps outside a browser: the page's
WebAssembly build (`deploy/wasm/build.sh`'s ember.js, working tree of this commit) prepares each request
and samples, and the GPU path the worker would pick computes the logits; the same request then goes
through `ember_chat` (WebAssembly, ember's CPU interpreter). Greedy, 60 new tokens. Fixtures: the 21 Oct
page test's ember CPU replies (chat_sftfinal_v3), state4's multi-turn requests with their state blocks
and the native server's replies (`../wasm_state_2026-10-25/`), and mtstudio's greedy replies for the
paperkiln families (ember CPU's equal them 40/40, `../families_2026-10-20/`).

| Model | GPU path | WebGPU = WASM | WebGPU = CPU fixture | WASM = CPU fixture | Reply tokens | s per reply (SwiftShader) |
|---|---|---:|---:|---:|---:|---:|
| chat_sftfinal_v3 (tuned llama path: llama_webgpu.js) | llama_webgpu | 40/40 | 40/40 | 40/40 | 636 | 33.5 |
| state4, multi-turn with state blocks | ir_webgpu | 40/40 | 40/40 | 40/40 | 200 | 1.1 |
| kimi_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2188 | 1.9 |
| srd_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2367 | 2.5 |
| attnres_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2400 | 3.8 |
| flex_highway_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2400 | 2.1 |
| mamba_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2400 | 1.8 |
| gpt2_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 1705 | 1.7 |
| swa_tiny | ir_webgpu | 40/40 | 11/40 | 11/40 | 2171 | 1.9 |

chat_sftfinal_v3 is the llama regression: `ember_ir_doc()` is "" for it, so the page runs the unchanged
`llama_webgpu.js`; replies equal the 21 Oct receipt's 40/40 (636 tokens, the same count). swa_tiny's
mtstudio file is not this request set's reference (WebAssembly matches it on the same 11 prompts); its
WebGPU replies equal WebAssembly's 40/40 and gate 1 passes.

## Not done here

- Speed on a GPU (replies per second against WASM SIMD): needs a remote GPU VM running headless Chrome
  (a runner entry); the SwiftShader times above are not GPU numbers.
- Gemma-3-270m (f32, 1.1 GB) and Qwen2.5-0.5B (f16, 1 GB): not run, and Qwen3-0.6B 1 of 10 prompts:
  the laptop had 1.6-3.5 GB of free memory, and a SwiftShader run holds the file, its transposed GPU
  copy and the load's temporaries. On a GPU VM, or after freeing the file once uploaded.
- The browser page itself on a GPU (`page_test.sh` / `page_test.html`): chat_test.mjs runs the
  worker's steps in node instead; the worker's routing (`ember_ir_doc`) is the same code.
- The 70M fw_moe on the llama path: not rerun (SwiftShader runs it at seconds per token);
  `llama_webgpu.js` is unchanged, and chat_sftfinal_v3's page path is the regression.
- `recurrence` documents stay in WebAssembly (no WGSL kernel; their bodies would need a compiler like
  `expr`'s).

## Files

- `ref.sh` (WSL): the CPU references; `expr_doc.py` (the two expr documents); `ir_ctx.py`.
- `gates.py`, `chats.py` (Windows): the two gates through `swiftshader_run.py`, receipts into this folder.
