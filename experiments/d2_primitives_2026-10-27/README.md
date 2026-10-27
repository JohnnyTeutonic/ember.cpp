# D2's primitives in ember: Shaw's relative key and value vectors and Primer's MDHA (27 October 2026)

ember at keramos **v1.4.0** (c1a5319784ed66f2f3d3e4e4e18903eeb49a8f08; revision 1.4). Code: ember **757e36b**
(CPU interpreter, load-time rule 40, adapter, vendored schema) and **210fc3c** (the WebGPU kernel). Written from
IR_SPEC.md 5.15 ("Relative key and value vectors"), rule 40 of 10.1, 6.9 and 12.6 N1/N2; paperkiln's
`ops::shaw_attention`, its `src/ir.cpp` and keramos's `cleanroom/` were not read. `tools/ir_validate.py` and
`tools/ir_numpy.py` were black-box oracles only (their verdicts, and `ir_numpy.forward`'s values through the runner's
`load_model`); the runner (`tools/ir_conformance.py`: `CmdAdapter`, `dec`, `load_model`) was read and used as a tool.
mtstudio (paperkiln) ran as a black box.

## What changed

- **attention.relative_kv** (`cpu_attention.cpp` `attention_relkv_batch_cpu`, `graph_ir.cpp`): head h computes
  s(p, j) = ((q_p . k_j) + (q_p . rk[n])) * scale with n = min(p - j, max_distance) on positions (absolute, as the
  window's; under `turn_state` the compact rows' positions), and out_p = (sum a v_j) + (sum a rv[n]); tables
  [K + 1, head_dim] (shared by the heads) or [K + 1, heads * head_dim] (head h at columns h * head_dim); grouped-query
  attention reads the query head's vectors and the kv head's k and v; masking unchanged (window, sinks,
  `turn_state`; a masked key adds exactly 0 to both sums). Without `value` the second sum is absent. The tables are
  bound at load like any tensor (`{i}` through `layer_name`, as `position_bias`). depth_route with relative_kv is
  refused by name (a capability refusal, 10.3), as position_bias is.
- **Rule 40 at load** (`graph_ir_validate.cpp`): the schema's members (`key` a name, `value` a name or null,
  `max_distance` an integer >= 0, nothing else; schema failures under `relative_kv` cite rule 40, or rule 6 when the
  value is a per-layer list), relative_kv beside position_bias on one node, and (container) each application's key
  and value tensors of ne {head_dim or heads * head_dim, max_distance + 1}. `relative_kv` is structural (3.8):
  a per-layer one is rule 6. The vendored schema is keramos v1.4.0's `ir.schema.json` (sha256 4de64f8f... of the
  Windows checkout's file; the document equals the tag's blob).
- **Adapter** (`tools/ir_adapter.cpp`): `run_node` on an attention node with relative_kv; hello names revision 1.4.
- **CUDA**: `device_supported` is false for a document with relative_kv, so a CUDA build serves it on the CPU
  interpreter (as position_bias). MDHA needs nothing: conv1d_causal and a recurrent slot beside the kv slot already
  run on the device.
- **WebGPU** (`deploy/wasm/web/webgpu/ir_webgpu.js`): a kernel `attn_rel` (the `attn` kernel with the tables, both in
  one storage buffer: nine per stage; with fewer `createIr` refuses and the page stays in WebAssembly);
  relative_kv under depth_route stays in WebAssembly (`irSupport`). MDHA needed nothing.
- **MDHA** (`pk_flex_mdha`: conv1d_causal over the fused qkv's 48 columns, kernel 3, `bias` null, slot `qkv_conv`
  x [2, 48] beside the kv slot, before rope) passed ember's interpreter as it was, at ec3201b, on every case
  (model, decode, snapshot, truncate, chunks, accept, digest); nothing was changed for it.

## The suite at v1.4.0 (CPU)

`run_suite.sh` (git archive of the tag on ext4, no EMBER_* variable) against the pinned build (`../../` git archive
of 757e36b, CPU only, HAS_CUDA off, Release, GCC 13.3, WSL ext4; `~/d2e/pin/build/ir_adapter`, sha256
685baa6c42c2ddbb74e4829751c45b4ac4a9f26abfc1a268ce14145e61ba8fff, `receipts/ir_adapter.sha256`):
**413 pass, 0 fail, 52 skip** of 465 (`suite_v1.4.0_cpu.json`).

| Kind | Pass | Fail | Skip |
|---|---:|---:|---:|
| model | 27 | 0 | 0 |
| trace | 80 | 0 | 0 |
| op | 27 | 0 | 41 |
| refuse | 105 | 0 | 0 |
| digest | 90 | 0 | 0 |
| dequant | 10 | 0 | 0 |
| grad | 0 | 0 | 11 |
| accept | 42 | 0 | 0 |
| chat | 32 | 0 | 0 |

Against ember's v1.3.5 receipt (keramos `conformance/receipts/ember_cpu.json`, 437 cases; `compare_receipt.py`,
`compare_v1.3.5_receipt.json`): **no earlier case changes verdict or detail** (437/437 identical), none missing.
The 28 new cases:

| Case | Verdict | Detail |
|---|---|---|
| model/pk_flex_mdha | PASS | log-probs worst 1.09e-07; greedy 4/4 |
| trace/pk_flex_mdha_decode, _snapshot, _truncate, _chunks | PASS | 6, 5, 3, 4 steps |
| model/pk_flex_shaw | PASS | log-probs worst 9.36e-08; greedy 4/4 |
| trace/pk_flex_shaw_decode, _chunks | PASS | 6, 4 steps |
| op/attention/relative_kv | PASS | a worst 7.38e-08 of max(1, \|ref\|) |
| op/attention/relative_kv_per_head | PASS | 4.19e-08 |
| op/attention/relative_kv_keys_only | PASS | 1.27e-07 |
| op/attention/relative_kv_window_sinks | PASS | 7.38e-08 |
| op/attention/relative_kv_grouped_query | PASS | 1.14e-07 |
| op/attention/relative_kv_distance_zero | PASS | 1.19e-07 |
| refuse/relative_kv_distance_negative, _without_key, _unknown_field, _key_shape, _value_shape, _with_position_bias | PASS | refused, citing IR_SPEC 5.15 (rule 40) |
| refuse/relative_kv_per_layer | PASS | refused (the interpreter's "a per-layer 'relative_kv' is not implemented"; the document rules alone cite 3.8 rule 6) |
| digest/vectors/document_pk_flex_mdha, document_pk_flex_shaw | PASS | sha256 and canonical form equal |
| accept/documents/model_pk_flex_mdha, model_pk_flex_shaw | PASS | accepted |
| grad/attention/relative_kv, _per_head_grouped_query, _keys_only | SKIP | no backward (an inference engine) |

`relkv_bits.json` lists, per refusal case, ember's full message with the case's gguf and with the document alone:
each cites the case's rule (rule 40, and 3.8 rule 6 for the per-layer case).

## Exactness of the relative_kv path

The CPU path follows 5.15's stated order: each dot product summed from +0 over the head's columns in ascending
order, rounded, then (q . k) + (q . rk[n]), then the product with scale; each output sum from +0 over the visible
keys in ascending order, then the two sums added. `relkv_order.py` states that order in f32 (with ember's softmax:
max, glibc expf, ascending sum, times 1 / (sum + 1e-9)) and ember's run_node output equals it **bit for bit on all
432 elements** of the six op cases (`relkv_order.jsonl`). It is **not bit-exact against the numpy reference**:
21-40 of each case's 72 elements are bit-equal, worst 82 ulp, within 1.3e-7 of max(1, |ref|) (`relkv_bits.json`).
The difference is the softmax and the reference's own summation (the attention op is ROUNDING, 4.2; ember's plain
attention cases sit at the same distance in the v1.3.5 receipt: 7.0e-8 to 9.9e-8 plain, 1.1e-7 to 1.6e-7 with
position_bias). `cpu_attention.cpp` is not
among the `-ffp-contract=off` translation units; on the x86-64 builds here (no -march) and in WebAssembly no FMA is
emitted, which the bit-for-bit check above confirms.

## CUDA (Colab T4, sm_75)

`d2_drive.sh` / `d2_vm.sh` (microtorch/AGENTS.md path C; session `ember-d2-1027`, 6 minutes from create to stop):
the CUDA build of 757e36b (HAS_CUDA on, `receipts/cuda/`):

| Model | Path | Greedy (8) | Top-1 | Worst prefill; decode (of range) |
|---|---|---:|---:|---|
| pk_flex_mdha | device kernels (conv1d_causal beside the kv slot) | 40/40 | 320/320 | 2.2e-7; 1.7e-7 |
| pk_flex_shaw | `device_supported` false: CUDA builds serve it on the CPU interpreter | - | - | - |
| pk_srd (kv + recurrent, control) | device | 40/40 | 320/320 | 1.6e-7; 1.1e-7 |
| pk_llama (control) | device | 40/40 | 320/320 | 1.8e-7; 1.7e-7 |

`ir_device_gate` (the interpreter on the device against the same interpreter on the CPU, one process, 40 seeded
prompts of 8 ids). The suite at v1.4.0 against the CUDA build's `ir_adapter` (its CPU interpreter): 413/0/52, all
465 verdicts and details equal to the CPU build's. The relative_kv fallback is the evidence the table gives; no
CUDA kernel was written.

## WebAssembly

`wasm_gate.sh`: `tools/ir_wasm_gate` native (the pinned build) and WebAssembly (deploy/wasm/build.sh's sources and
flags, Emscripten 6.0.6, node, 1gb heap), 40 seeded prompts of 8 ids and 8 greedy tokens, ctx 16, against two
references: ember's native CPU interpreter, and keramos's numpy reference on the same prompts (`ref_numpy.py`,
`ir_numpy.forward` through the runner's `load_model`; greedy by recomputing the prefix), at tol 1e-5
(`receipts/wasm/`):

| Model | vs native: greedy, top-1, worst | vs numpy: greedy, top-1, worst (of range) | native vs numpy (of max(1, \|ref\|)) |
|---|---|---|---|
| pk_flex_mdha | 40/40, 320/320, 0 (bit for bit) | 40/40, 320/320, 1.7e-7 | 2.1e-7; greedy 40/40 |
| pk_flex_shaw | 40/40, 320/320, 0 (bit for bit) | 40/40, 320/320, 1.7e-7 | 2.2e-7; greedy 40/40 |

## WebGPU (SwiftShader only)

`webgpu_gate.py` on Windows: `tools/webgpu_parity/ir_gate.mjs` through `swiftshader_run.py` (Dawn's node bindings,
npm `webgpu` 0.6.2; adapter "google | swiftshader | swiftshader-device-subzero- | SwiftShader driver 5.0.0"; the
GPU-counter monitor saw no hardware context), the same prompts and references (`webgpu_gate.json`, `webgpu/`):

| Model | vs native: greedy, top-1, worst (of range) | vs numpy |
|---|---|---|
| pk_flex_mdha (existing kernels) | 40/40, 320/320, 2.2e-7 | 40/40, 320/320, 2.3e-7 |
| pk_flex_shaw (`attn_rel`) | 40/40, 320/320, 2.2e-7 | 40/40, 320/320, 2.2e-7 |

`tools/webgpu_parity/run.py` (a browser on a hardware GPU) was not used: its `EMBER_WEBGPU_REMOTE` guard keeps it to
remote VMs.

## The chain model (paperkiln's arXiv chain on 1803.02155)

`C:/ml_artifacts/arxiv_chain_2026-10-27_d2/arxiv_1803_02155.gguf` (sha256 6f5d4a5b...; flex, d 64, 2 layers,
4 heads, T 64, layernorm, relu; relative_kv per head, values on, max_distance 16, per layer):

- **Greedy parity, ember CPU against mtstudio** (`chain_parity.sh`: ember's pinned `tinyllama_server` on the CPU,
  `mtstudio chat` (`~/d2_build/mtstudio`) from a copy of the chain's `sft/` run, `tools/chat_parity.py` at
  max_new_tokens 24, top_k 1): the 40 frozen `tinychat_hf_prompts_v1` prompts and the chain's 2 chat turns (each a
  one-turn request, as the chain sent them): **42/42** replies identical, 42/42 token counts, 42/42 stop reasons,
  42/42 prompt token counts equal to paperkiln's `bpe.py`; mtstudio's replies here equal the chain's recorded
  `mtstudio_greedy.jsonl` 42/42, and ember's equal them 42/42 (`receipts/chain/`). ember logged "Running the model
  through the paperkiln IR interpreter (CPU; family_hint 'flex')", 32 weights bound.
- **Logits against ir_numpy** (`chain_legs.sh`, 40 seeded prompts of 16 ids, 16 greedy, ctx 64): ember's CPU
  within 5.1e-7 of max(1, |ref|) at every position, argmax 640/640, greedy 40/40.
- **WebAssembly**: against ember's CPU bit for bit (40/40, 640/640, 0); against numpy 40/40, 640/640, 3.2e-7 of
  range.
- **WebGPU (SwiftShader)**: 40/40, 640/640, 3.7e-7 against ember's CPU; 40/40, 640/640, 4.2e-7 against numpy
  (`webgpu_gate_chain.json`).
- **CUDA** (Colab T4, session `ember-d2c-1027`, `d2_vm_chain.sh` at 210fc3c, `receipts/cuda_chain/`):
  `ir_device_gate` reports `device_supported` false, and the CUDA build's `tinyllama_server` with 99 GPU layers
  logs "Running the model through the paperkiln IR interpreter (CPU; ...)": the CPU fallback. Its 42 /chat replies
  (same requests) equal the CPU build's 42/42 (text, token counts, stop reasons) and mtstudio's 42/42.

## Regressions

`regress.sh` on the pinned build (`receipts/regress/`): recursion_gate, mor_gate, mor_refusals, attnres_gate,
state4_gate, regress_parity and g5_thirdparty byte-identical to the committed receipts; state_tokens_gate equal but
for its distances to paperkiln's numpy reference (as in earlier runs); regress_bits equal but for the path of the
binary compared. Adapter self-test (`../ir_adapter_2026-10-25/adapter_selftest.py`): **189/189**, 3072 requests,
all 189 check records identical to the committed receipt.

## Spend

Colab: two T4 sessions (`ember-d2-1027`, 6 minutes; `ember-d2c-1027`, 4 minutes), well under one compute
unit together; no session left running.

## Where the specification was silent or could be read two ways

1. `{i}` in relative_kv's tensor names under `weight_index`: 3.8 makes `relative_kv` structural and 7.1 maps a
   tensor name's `{i}` through weight_index, but 3.8's list of tensor-name keys (`weight`, `bias`, `router`, ...)
   does not name relative_kv's `key` and `value`. ember maps them as every tensor name (m_i under weight_index, i
   under `tensor_index: "application"`), as it does position_bias's weight; no case tests it.
2. Rule 40's container clause "in an application that reads it" with weight_index: ember checks the tensor each
   application reads (the mapped name), as for rule 39.
3. relative_kv under depth_route: 5.15 says p and j are positions there too; there is no case, and ember refuses the
   combination by name, as it does position_bias.
4. The softmax inside attention has no stated order, so no engine's attention is EXACT against the reference;
   relative_kv inherits that (above). Not a defect in the text, but the reason the CPU path is ROUNDING against the
   reference while following every order the text states.

## Files

`run_suite.sh`, `compare_receipt.py`, `relkv_bits.py`, `relkv_order.py`, `regress.sh`, `wasm_gate.sh`,
`ref_numpy.py`, `webgpu_gate.py`, `chain_parity.sh`, `chain_legs.sh`, `d2_stage.sh`, `d2_stage_chain.sh`,
`d2_drive.sh` (from `../tensor_index_gate_2026-10-27/tig_drive.sh`), `d2_vm.sh`, `d2_vm_chain.sh`. Builds and work
directories under `~/d2e` (WSL ext4); the pinned build `~/d2e/pin` is kept.
