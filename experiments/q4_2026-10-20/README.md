# 4-bit paperkiln GGUFs, 20 Oct 2026

Files: paperkiln's 4-bit exports of `chat_sftfinal_v2` (paperkiln 1d707e34; tensor bytes
identical to llama.cpp's own quantiser). At d=384 every k-quant falls back per tensor
(Q4_K -> Q5_0, Q5_K -> Q5_1, Q6_K -> Q8_0), so the files hold Q4_0, Q5_0, Q5_1, Q8_0 and F32:

| File | sha256 | Size |
|---|---|---|
| `chat_sftfinal_v2_q4_0.gguf` | 0ba44851... | 11.9 MB |
| `chat_sftfinal_v2_q4_k_m.gguf` | 0381b21a... | 14.2 MB |
| `chat_sftfinal_v2_q4_k_s.gguf` | 0eccbf10... | 13.7 MB |

ember dequantises Q4_0, Q4_1, Q5_0, Q5_1 and F16 to f32 at load
(`weight_management.cpp`, `dequant_legacy_to_f32`); any other unhandled tensor type now
fails the load with the tensor's name.

## Gate (`q4_gate.log`)

Two references per file. **Dequantiser:** the same file rewritten as F32 by gguf-py's
reference dequantiser (`tools/quantize_gguf.py IN OUT --dequantize`); ember should give
identical logits. **Quantisation error:** the f32 `chat_sftfinal_v2` model, and greedy
whole-reply agreement with mtstudio (40 frozen prompts, up to 60 tokens).

| File | logits vs gguf-py F32 dequant | max abs dlogit vs f32 | max / logit range | first-token top-1 | replies = mtstudio (CPU, CUDA, WASM) | llama.cpp |
|---|---|---|---|---|---|---|
| f32 v2 | - | - | - | - | 40/40 | - |
| q4_0 | identical (max 0.0000) | 1.28 | 4.6e-2 | 37/40 | 22/40 | 21/40 |
| q4_k_m | identical | 0.50 | 1.8e-2 | 38/40 | 23/40 | 24/40 |
| q4_k_s | identical | 0.60 | 2.2e-2 | 39/40 | 23/40 | 23/40 |

The dequantised references give the same reply counts as the quantised files on every
backend, and CPU and CUDA differ by under 1% on every logit number. The remaining gap to
f32 is 4-bit quantisation itself; ember matches llama.cpp's agreement on the same files
to within one reply. These files fail the Q8_0 gate's first-token criterion (40/40);
4-bit is for size (11.9-14.2 MB vs 19.2 MB for `v2_q8all`), not fidelity.
