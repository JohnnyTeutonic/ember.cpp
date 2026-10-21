# ember interpreter on synthetic third-party models (22 Oct 2026, ember branch ir-ops)

The IR features paperkiln's `tools/gguf_ir.py` emits for standard llama.cpp GGUFs (LOCAL_LLM_STACK.md
section 7), run by ember's CPU interpreter and checked with `tools/thirdparty_ember_check.py`: a tiny
randomly initialised HF transformers model of the family, converted by llama.cpp b6000's
convert_hf_to_gguf.py (paperkiln `tools/thirdparty_synth.convert`, f32, a placeholder vocabulary), its
IR built by `gguf_ir`, then the same random token sequences through HF's PyTorch forward, paperkiln's
`tools/ir_numpy.py` and ember (`tools/ir_logits.cpp`: one prefill from position 0, and token by token
through the recurrent state as generation runs).

## Mamba-1 (`mamba_synth_check.json`, `mamba_synth.ir.json`)

New ops: `conv1d_causal` and `selective_scan` (`graph_ir_mamba.{h,cpp}`, -ffp-contract=off), both
recurrent (state slots "conv": x [K - 1, C], "ssm": s [D, N]).
Model: MambaConfig vocab 512, hidden 64, state 16, 3 layers, expand 2 (D = 128), conv kernel 4,
time-step rank 8, conv bias, tied embeddings, seed 0. Four random 40-token sequences:

| | worst max |dlogit| / range | argmax |
|---|---|---|
| ember (prefill and decode) vs numpy IR | 6.7e-7 | 160/160 |
| ember vs HF | 8.2e-7 | 160/160 |
| numpy IR vs HF | 7.7e-7 | |
| ember decode vs ember prefill | 0 (bit-identical) | |

Not yet: an official checkpoint (state-spaces/mamba-130m) through ember's server (needs paperkiln's
C++ IR builder for GGUFs without `paperkiln.ir`); CUDA and WASM kernels.

## Mistral-style sliding window (`mistral_swa_synth_check.json`)

paperkiln's synthetic MistralForCausalLM (window 8, grouped-query 4/2 heads, 3 layers; thirdparty_synth's
model): ember's interpreter runs it with grouped-query attention and the window: ember vs numpy IR
worst 1.4e-6 of the logit range, argmax agrees everywhere, decode equal to prefill.

## GPT-NeoX / Pythia (`gptneox_synth_check.json`)

A tiny random GPTNeoXForCausalLM (4 heads, 3 layers, rotary_pct 0.25, parallel residual, exact GELU,
untied head) through llama.cpp b6000's convert and gguf_ir, on ember ir-ops rebased onto paperkiln
369a54c (its NEOX and partial rotary): ember vs numpy IR worst 4.5e-7 of the logit range, vs HF 4.9e-7,
argmax agrees everywhere, decode equal to prefill. The parallel residual needed nothing new: it is two
adds in the graph.

All gates rerun on the rebased branch (22 Oct, ~08:40): expr gpt2_tiny 40/40 (8.1e-7), Mamba 6.7e-7,
Mistral SWA 1.4e-6, GPT-NeoX 4.5e-7.

## Gemma 3 (`gemma3_synth_check.json`)

paperkiln's synthetic Gemma3ForCausalLM (8 layers; window 8 on local layers, global layer 5 with
theta 1e6 against 10000; sandwich norms, per-head q/k norm, NEOX RoPE, GeGLU, embedding x sqrt(d),
tied head), regenerated through thirdparty_synth.convert, 4 sequences of 40 ids (longer than the
window, so the per-layer windows and bases decide the result). Needed in ember: per-layer attributes
({"per_layer": [...]} for the kv state's window, attention window/sinks and rope theta; one RoPE table
per distinct theta), rope freq_scale (scalar), rmsnorm weight_offset (read as w + c once at load).
ember vs numpy IR worst 1.1e-6 of the logit range, vs HF 6.9e-7, argmax agrees everywhere, decode
equal to prefill. The same model with weight_offset 0.25 on every rmsnorm, ember vs numpy IR: 1.1e-6.

Every gate on this commit (22 Oct, ~08:45): expr gpt2_tiny 40/40 (8.1e-7), Mamba 6.7e-7, Mistral SWA
1.4e-6, GPT-NeoX 4.5e-7, Gemma 3 1.1e-6.

## Still to do

- The C++ IR builder entries (GGUF without paperkiln.ir) for gemma3, gptneox, mamba, mistral sliding
  window and gpt2, so these load through ember's server like Qwen does; then the official checkpoints.
- CUDA kernels for the per-layer, NEOX, per-head RMSNorm, expr and Mamba ops (device graphs with any
  of them run on the CPU interpreter today), and the WASM gate.
