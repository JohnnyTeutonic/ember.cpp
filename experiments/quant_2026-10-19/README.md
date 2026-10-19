# Q8_0 for paperkiln GGUFs, 19-20 Oct 2026

Source: `chat_sftfinal_v2.gguf` (paperkiln; eps 1e-5, ggml-order dims; sha256 1fc4ec39...).
Quantised with `tools/quantize_gguf.py` (llama.cpp's gguf-py Q8_0 blocks; every metadata
field copied; old-order files rewritten with ggml-order dims, bytes unchanged):

| Variant | Q8_0 tensors | Size |
|---|---|---|
| f32 (v2) | none | 71.5 MB |
| `v2_q8` | all `blk.*` 2-D weights | 37.7 MB |
| `v2_q8out` | + `output.weight` | 28.4 MB |
| `v2_q8all` | + `token_embd.weight` | 19.2 MB |

## Gate (`tools/quant_gate.py`, `tools/chat_parity.py`)

Logit error against the f32 model on the same backend, from each of the 40 frozen prompts'
next-token logits, and whole-reply greedy agreement with mtstudio (40 prompts, up to 60
tokens). CPU and CUDA agree to within 5% on every number (`dequant_check.log`):

| Variant | max abs dlogit | mean abs dlogit | max / logit range | first-token top-1 | greedy replies = mtstudio |
|---|---|---|---|---|---|
| f32 | - | - | - | - | 40/40 (CPU, CUDA, WASM) |
| `v2_q8` | 0.063-0.066 | 0.012 | 2.3e-3 | 40/40 | 38/40 |
| `v2_q8out` | 0.088 | 0.014 | 3.1e-3 | 40/40 | 37/40 |
| `v2_q8all` | 0.098-0.100 | 0.016 | 3.5e-3 | 40/40 | 36/40 |

The smallest f32 top-1 margin among the 40 first tokens is 0.035, so the first token never
flips; whole replies diverge in 2-4 of 40 where a later step meets a near-tie. The same
counts hold on CPU, CUDA and WebAssembly. Quantised gate as run here: first-token top-1
40/40, max |dlogit| <= 1% of the logit range, whole-reply agreement reported.

## Speed: dequantise at load

ember's CPU path already dequantised lazily, but kept the quantised copies too, and decode
ran 15-45% slower on Q8_0 files. `TinyLlamaModel::dequantize_all_if_small` (called at load
for models up to 64M parameters; `EMBER_DEQUANT_AT_LOAD=0` disables) dequantises every
weight once and frees the quantised copies. Interleaved A/B in node (WebAssembly build,
40 greedy replies per run, ms per reply):

| Round | f32 | `v2_q8all` | `v2_q8out` |
|---|---|---|---|
| 1 | 588 | 665 | 571 |
| 2 | 537 | 497 | 494 |
| 3 | 462 | 491 | 451 |
| 4 | 453 | 457 | 461 |

Within noise after the first (warm-up) round: a Q8_0 download (19 MB for `v2_q8all`)
replies at f32 speed and holds one copy of the weights in memory.
