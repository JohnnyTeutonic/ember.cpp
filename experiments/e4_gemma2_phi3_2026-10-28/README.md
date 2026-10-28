# E4 in ember: Gemma 2 and Phi-3, and keramos revision 1.6 (28 October 2026)

ember's side of `LOCAL_LLM_WORK.md` E4 for Gemma 2 (attention soft cap) and Phi-3 (LongRoPE) at keramos **v1.5.0**
(062c1513a3d5b1a3ae2055695da56474dffc92f8), and revision 1.6's lifts at **v1.6.1**
(b74a410d00cc2e2698d8353adaf0ca36bc95ed51). Code: ember 9e70cd4 (soft cap, LongRoPE, builders phi3 and gemma2) and
f393955 and 5cc541d (revision 1.6: `apply`, DeepSeek-V3's router, YaRN's `freq_factors` by value, deepseek2's
Appendix D.11 graph, `run_node` on `moe_ffn`), on top of 4290c91 (the third-party MoE and MLA). Everything was written from
IR_SPEC.md at the tags; no paperkiln source and nothing in keramos's `cleanroom/` was read. `tools/gguf_ir.py`,
`tools/ir_validate.py` and `tools/ir_numpy.py` were black-box oracles (run, outputs compared); the runner was read
for its protocol. HF transformers was not run here: the synthetic files' cases hold HF's greedy continuations and
f32 logits (paperkiln's export).

## What ember computes

- **attention `softcap`** c (IR_SPEC 5.15): the scaled score s becomes c * tanh(s / c), the division rounded, then
  tanhf, then the product, before any `position_bias`; per-layer values allowed. Every CPU attention kernel takes
  it (`cpu_attention.h` `softcap_score`; with c absent the arithmetic is unchanged, bit for bit).
- **rope `freq_factors` and `attn_factor`** (5.14): one table per distinct (theta, freq_scale, attn_factor,
  factors); pair j's angle is divided by f_j, and the table holds cosf and sinf times m. The factors are a tensor,
  two sets `{short, long, original_ctx}`, or (revision 1.6) the n_rot / 2 values themselves.
- **LongRoPE's set follows the sequence's length** (6.9): `Interpreter::forward` compares the length after the call
  with original_ctx; when the set changes, it runs the held ids plus the new ones again from position 0, so every
  kv row carries the set, and returns the same rows of logits; a `truncate` across original_ctx recomputes the
  first n ids the same way. The interpreter keeps the ids (and the set) in its state, so snapshots and the adapter's
  restore carry them. This is in `forward` itself, below every caller (the server, the CLI, the adapter, CUDA).
- **Builders** phi3 (Appendix D.8) and gemma2 (D.9), and deepseek2 at revision 1.6 (D.11: leading dense blocks under
  `apply`, `expert_gating_func` 2 as `scoring` sigmoid, `exp_probs_b` as `selection_bias`, `expert_group_count` as
  `groups`, YaRN's divisors in binary64 rounded to f32, mscale^2 in the MLA scale).
- **Revision 1.6** (3.8, 5.16): `apply` skips a block node in the applications its value names (bind and run);
  `moe_ffn` `scoring`, `selection_bias` and `groups` (`utils.cpp` `moe_ffn_routed_f32_cpu`; softmax scores without
  a bias or groups stay on `moe_ffn_f32_cpu`).
- **Rules** 6 (freq_factors, selection_bias and groups are structural), 10 (softcap, attn_factor, groups), 38 (D.7's
  defaults: attn_factor 1, apply true, scoring softmax), 41 (the factor forms; the container's ne {n_rot / 2}), 43
  (`apply`), 44 (selection_bias ne {E}); rule 36 checked per application under `apply`. The schema is vendored from
  keramos v1.6.1.
- **ir_adapter 6**: `run_node` on rope (the single-node length max(pos) + 1) and `moe_ffn` nodes; softcap on
  attention nodes.

## Results

Pinned build: `../ir_rev12_2026-10-26/build_pinned.sh 5cc541d` (git archive, CPU only, Release, GCC 13.3, WSL
ext4): `~/ember_adapter_5cc541d/build/ir_adapter`, sha256
8eae3a1d84482c229eed4b14f9733440a208798fed716ad04c655d779e7d2aac.

**The suite** (`receipts/`), no EMBER_* variable:

| Tag | Pass | Fail | Skip | model | trace | op | refuse | digest | dequant | grad | accept | chat |
|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|
| v1.5.0 | 534 | 0 | 44 | 32/0/0 | 96/0/0 | 47/0/33 | 119/0/0 | 95/0/0 | 10/0/0 | 0/0/11 | 49/0/0 | 86/0/0 |
| v1.6.1 | 554 | 0 | 44 | 33/0/0 | 99/0/0 | 51/0/33 | 129/0/0 | 96/0/0 | 10/0/0 | 0/0/11 | 50/0/0 | 86/0/0 |

No XFAIL, no XPASS. The new cases of this work: op `attention/softcap`, `_window_sinks`, `_position_bias` (within
1.25e-7 of max(1, |ref|)); op `rope/freq_factors`, `_short`, `_long`, `attn_factor`,
`freq_factors_partial_attn_factor`, `freq_factors_values` (within 1.4e-7); models `tp_gemma2` (log-probs 1.55e-7)
and `tp_phi3` (5.97e-7), each greedy 4/4, with their decode, chunks, rewind-into-window and truncate traces (tp_phi3's
original_ctx 6: the traces cross it, the truncate both ways); `tp_deepseek3` (5.89e-7) with its traces; op `moe_ffn/*` (5
cases, within 2.7e-7); the refusals `softcap_not_positive`, `architecture_gemma2_softcap_dropped`,
`attn_factor_not_positive`, `freq_factors_*` (4), the revision 1.6 refusals (10). Against the v1.4.2 receipt
(keramos af98187, 472/0/52): no case moved down; eight moved from skip to pass (op `rope/interleaved`, `neox`,
`neox_partial`, `freq_scale`, `grouped_query`, `moe_ffn/renorm`, `raw`: the new `run_node` paths; `split/three`:
4290c91's), 33 skips carry the reworded `run_node` message, and `refuse/architecture_without_graph` names the
suite's new fixture ("falcon"; it was a phi3 file at v1.4.2).

**HF on the synthetic files** (`e4_hf_gate.py`, `receipts/hf_gate_cpu.json`; through the adapter: one prefill, then
16 greedy tokens one forward of one id at a time). The sha256s of every F.gguf and F.cases.json equal paperkiln's.

| File | greedy vs HF | first token vs HF | max distance / row range | decode = fresh prefill |
|---|---|---|---|---|
| gemma2 (48 ids) | 40/40 | 40/40 | 1.04e-6 | bit-equal |
| phi3_longrope (12, 24, 48 ids; original_ctx 32) | 40/40 | 40/40 | 3.39e-6 | bit-equal |
| phi3_swa (48 ids) | 40/40 | 40/40 | 1.40e-6 | bit-equal |

The 24-id phi3_longrope sequences cross original_ctx at the ninth greedy step: the slot is recomputed there, and the
last decode step's row equals a fresh prefill of the same ids bit for bit.

**The builder** (`e4_builder.py`, `receipts/builder_equality.json`): on the three synthetic files, tp_gemma2 and
tp_phi3, and seven variants (gemma2's caps 0.1 / 0.3, 30 / 50 without a window key, 0 / 0; phi3's attn_factor 0, 1
and absent), ember's document equals gguf_ir.py's on every key but `_comment`; the oracle's rule 38 accepts each
file with ember's document embedded, and ember's rule 38 accepts each with gguf_ir.py's. tp_deepseek3 carries
gguf_ir.py's document and ember's rule 38 (Appendix D.7) accepts it: its model and trace cases pass.

**Regressions**: `tools/ir_logits` of 1456c0c and of 5cc541d on tp_gemma3, tp_qwen3, tp_qwen3_hd16, tp_llama_hd16,
tp_mistral, tp_mistral_hd16, tp_gptneox and tp_mamba1: byte-equal (`receipts/regress_bits.txt`); the suite's
earlier cases as above.

**WebAssembly** (ember 9e70cd4; deploy/wasm/build.sh's sources and flags, -O3 -msimd128, 1 GB heap, node with
NODERAWFS; Emscripten 6.0.6):
- HF's cases (`e4_wasm_hf.sh`): greedy 40/40 against HF on all three files; the rows against the native build are not
  bit-equal (Emscripten's libm), worst 6.1e-7 (gemma2, 0 of 640 rows bit-equal), 4.0e-7 (phi3_longrope, 123 of
  640), 2.9e-7 (phi3_swa, 88 of 640) of the row range; first token against HF within 1.32e-6, 3.41e-6 and 1.36e-6.
- tools/ir_wasm_gate (40 random prompts of 24 ids, 32 greedy; tp files 12 ids, 16 greedy): greedy 40/40 and top-1 at
  every position on gemma2 (8.7e-7), phi3_longrope (3.4e-7), phi3_swa (bit-equal), tp_gemma2 (1.2e-6) and tp_phi3
  (prefill bit-equal). The gate's decode measure compares token-by-token decode with the whole prompt's prefill;
  under two LongRoPE sets the prefill of a prompt past original_ctx rotates every row with the long set while a
  decode of t + 1 <= original_ctx tokens uses the short one (section 5.14), so on prompts crossing it (phi3_longrope
  at 40 ids, tp_phi3 at 12) that measure differs by design (1.04 and 0.46); greedy is 40/40 there too.

**WebGPU**: `irSupport` refuses `attention.softcap`, `rope.freq_factors`, `rope.attn_factor` other than 1, `apply`
and `moe_ffn`'s `scoring`, `selection_bias` and `groups`, so such a document stays in WebAssembly
(`receipts/webgpu_irsupport.jsonl`: tp_gemma2 and gemma2 "attention softcap", tp_phi3 and phi3_longrope "rope
freq_factors", tp_deepseek3 "split groups"; tp_gemma3 still runs on WebGPU). No kernel was added.

**CUDA**: not run (no GPU work this week). The soft cap is in the window kernel (an unwindowed capped node runs it
with window max_len) and the LongRoPE tables are switched per block with the captured decode graph captured again;
both compile (nvcc 12.6, sm_75, compile only) and wait for on-device validation (`e4_device_gate.cpp`: HF and device
against CPU on the synthetic files, default and EMBER_DETERMINISTIC=1, rows dumped for byte comparison). A capped
document under `turn_state` or `depth_route`, and any document with `apply` or DeepSeek-V3's router, is served on the
CPU by CUDA builds (`device_supported`).

## Files

- `e4_cpu_gates.sh COMMIT BASE_BUILD OUT`: pinned build, the v1.5.0 suite, the HF gate, the builder check, regression
  bits.
- `e4_hf_gate.py`, `e4_builder.py`, `e4_wasm.sh`, `e4_wasm_hf.sh`, `e4_rows_cmp.py`, `e4_irsupport.mjs`,
  `e4_device_gate.cpp`.
- `receipts/`: the suite runs at v1.5.0 and v1.6.1, `hf_gate_cpu.json`, `builder_equality.json`,
  `regress_bits.txt`, the WebAssembly summaries, the WebGPU support verdicts.
