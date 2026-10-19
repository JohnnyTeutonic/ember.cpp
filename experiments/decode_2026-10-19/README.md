# Decode speed, 19 Oct 2026

Model: paperkiln `chat_sft5k.gguf` (d384, 6 layers, BPE 8192, ctx 256) served by
`tinyllama_server` (CUDA build) on the local RTX 4060. Bench: `tools/chat_decode_bench.py`,
first 20 prompts of `tinychat_hf_prompts_v1.jsonl`, greedy, up to 64 new tokens;
per-token decode = (request wall − TTFT) / (sampled steps − 1).

| Build | Decode p50 (ms/token) | min | max |
|---|---|---|---|
| before (`bacdcd7`) | 12.44 | 10.20 | 34.21 |
| after, TF32 prefill (default) | **4.52** | 3.52 | 11.64 |
| after, `EMBER_GEMM_TF32=0` | 5.52 | 3.61 | 11.35 |

## Where the time went

`EMBER_DECODE_TIMING=1` (new) syncs the stream at each phase boundary and prints
per-step wall time per phase to stderr (`base_timed.decode_phases.log`; the syncs
make the timed run slower, 22 ms/step, so read the shares). Before the fixes:

| Phase (summed over 6 layers) | ms/step | Cause |
|---|---|---|
| LM head | 3.9 | `matvec_bf16_f32_cuda` allocated, converted and freed a full FP32 copy of the 8192 x 384 head every step (12.6 MB cudaMalloc/cudaFree) |
| RoPE + KV cache write | 3.1 | one `update_kv_cache` launch per KV head per tensor (12 per layer) |
| O-projection block, second copy | 1.9 | the O-proj / residual / copy / post-norm sequence ran twice per layer, recomputing identical values |
| sampling | 1.4 | a full sort of all 8192 logits on every step, even at top_k = 1 |
| weight check | 0.56 | `ensure_f32_concatenated_weights_loaded` logged a line before its early return, per layer per step |
| input D2H | 0.34 | an unused debug copy of the input back to the host, plus a sync |

## Fixes

- Decode LM head uses the FP32 device copy (`lm_head_f32_dev_`), as the batched prefill does;
  `EMBER_DECODE_LMHEAD=bf16` restores the old path.
- KV write: one device-to-device copy per K and per V (the cache is `[pos][kv_head][dim]`,
  the same layout as `k_dev_`/`v_dev_`); the per-head kernel stays as the fallback.
- The duplicated O-projection block is removed (`add_residual` writes `a + b` from unchanged
  inputs, so the second pass was idempotent).
- Sampling sorts only the top-k head (`std::partial_sort`): the same elements in the same
  order as the full sort.
- The weight check returns before logging; the unused input copy and its sync are gone.
- `EMBER_DECODE_OPT=0` selects the standard decode kernels (measurement knob).

## Greedy parity against mtstudio

`tools/chat_parity.py`, 40 prompts, ember on 8095, mtstudio chat on 8096
(specs `mtstudio_spec_*.json`, `rope_heads: "all"`):

| Model | Default (TF32 prefill) | `EMBER_GEMM_TF32=0` |
|---|---|---|
| sft1500 (`chat_sft_bpe.gguf`) | 39/40 | 39/40 |
| sft5250 (`chat_sft5k.gguf`) | 40/40 | 40/40 |

Prompt token counts equal `tools/bpe.py` and stop reasons match on all 80.

The sft1500 mismatch is prompt 22 ("I do not understand why my trousers feel so tight
after washing them"). Both replies agree for 13 tokens; at step 13 the top two tokens are
644 ("them") and 48 (","). Full logit dumps (`TINYLLAMA_LOGITS_DUMP`) on that prompt, all
with TF32 off, give logit[644] − logit[48] at step 13:

| Path | 644 − 48 | argmax |
|---|---|---|
| ember CPU, batched prefill | −0.00274 | 48 |
| ember CPU, token-by-token (`EMBER_PREFILL_MIN=1000`) | −0.00041 | 48 |
| ember CUDA, BF16 decode head (old) | −0.00040 | 48 |
| ember CUDA, FP32 decode head (new) | +0.00118 | 644 |
| ember CUDA, standard decode kernels | +0.00118 | 644 |

The logit vectors of ember's own two CPU paths differ by up to 5e-4 of the max logit
before step 13, the same size as CUDA against CPU (2e-4 to 9e-4), and the standard
kernels give bit-identical logits to the optimized ones. So the gap is a near-tie inside
the spread between valid FP32 summation orders of this model, not a decode defect; the
old 40/40 on this prompt came from the BF16 head landing on the tie's side by 0.0004.
