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
