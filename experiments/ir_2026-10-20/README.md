# paperkiln IR interpreter, milestone 1, 20 Oct 2026

`graph_ir.h` / `graph_ir.cpp` run paperkiln's block-structured IR v0 (LOCAL_LLM_STACK.md
section 4; op spec in `microtorch/include/microtorch/ir.hpp`) on the CPU: embed -> block x L
-> head over named [T, width] buffers, weights bound by GGUF name and dequantised to f32 at
load, one KV state slot per block owned by the interpreter. Ops: embedding, rmsnorm, linear,
rope (interleaved, all heads), attention (causal), add, mul, silu, moe_ffn (renorm or raw
gates). Any other op, state kind, rope variant, window or sinks fails at load, naming it.

Source: GGUF key `paperkiln.ir` (paperkiln 4010e12 writes it for every llama export) or a
side file named by `EMBER_IR`. An embedded IR whose `family_hint` is llama keeps the tuned
llama path (design point 6); other families, or `EMBER_IR`, run the interpreter;
`EMBER_INTERPRETER=1/0` forces it on or off. Prefill, decode, prefix reuse and speculative
verification all go through it when it is active.

## Gate 1 (`gate1_llama_in_ir.log`): llama in the IR vs the llama path

`chat_sftfinal_v2` with the hand-written `microtorch/ir/chat_sftfinal.ir.json`:

| Run | replies = mtstudio | replies = llama path | first-token logits vs llama path |
|---|---|---|---|
| interpreter, f32, CPU | 40/40 | 40/40 | max 1.0e-4 (3.6e-6 of range), top-1 40/40 |
| interpreter, q8all, CPU | 36/40 (llama path: 36/40) | 40/40 | max 1.0e-4 (3.6e-6 of range), top-1 40/40 |
| interpreter in the CUDA build | 40/40 | 40/40 | - |

## Gate 2 (`gate2_embedded_ir_moe.log`): embedded IR and MoE

| Model (sha256) | path | result |
|---|---|---|
| `chat_sftfinal_v3` (a9d5ba5e) | default (llama path; no interpreter) | 40/40 vs mtstudio |
| `chat_sftfinal_v3` | interpreter from the embedded key, CPU build / CUDA build | 40/40 / 40/40 |
| `chat_sftfinal_v3` | WASM (llama path) | 40/40 |
| `moe_tiny_v3` (c9ed2a99) | llama path, CPU / CUDA | 40/40 / 40/40 |
| `moe_tiny_v3` | interpreter (`moe_ffn`), CPU | 40/40 |
| `moe_tiny_v3` | WASM (llama path) | 40/40 |

The MoE kernel (`utils.cpp`, `moe_ffn_f32_cpu`, shared by both paths) now follows the op
spec exactly: renorm gates are a softmax over the k selected logits, and experts accumulate
in ascending index. An IR with an unknown op (`gelu_tanh`) is refused at load:
"IR op 'gelu_tanh' (block) has no CPU kernel in ember's interpreter".

## IR-only loading (`ir_only.log`)

A GGUF whose IR family is not llama, or any GGUF with `EMBER_IR_ONLY=1`, loads without the
llama weights or GPU state: the model's dimensions come from the IR and the interpreter holds
the only copy of the weights. `EMBER_INTERPRETER=0` on such a file is refused at load. Run on
the llama files with `EMBER_IR_ONLY=1`:

| Run | replies = mtstudio | RSS |
|---|---|---|
| `chat_sftfinal_v3`, CPU build, llama path | 40/40 | 156 MB |
| `chat_sftfinal_v3`, CPU build, IR-only | 40/40 | 161 MB |
| `chat_sftfinal_v3`, CUDA build, llama path / IR-only | 40/40 / 40/40 | 392 / 287 MB |
| `chat_sftfinal_v3`, IR-only, `EMBER_SPEC=4` | 40/40 | |
| `moe_tiny_v3`, IR-only | 40/40 | 39 MB |

## gpt2 (`gpt2_gate.log`)

`gpt2_tiny.gguf` (paperkiln 79b023ca, sha256 37a30e0a; d128 L2 H2, ctx 128, llama.cpp's gpt2
tensor layout) loads IR-only (family_hint gpt2). New ops, with paperkiln's numerics from
`ir.hpp`: `pos_embedding` (x + position_embd[pos], EXACT), `layernorm` (mean and biased
variance, affine with bias, eps 1e-5), `split` (fused qkv columns, EXACT), `gelu` (approx
`tanh`, or `none` for the erf form); `linear` biases throughout.

| Run | replies = mtstudio | first-token logits vs llama.cpp (gpt2 arch) |
|---|---|---|
| CPU build | 40/40 | max 1.1e-3 (6.6e-5 of range), top-1 40/40 |
| CUDA build | 40/40 | same |
| WASM (node) | 40/40 | - |

After the gpt2 change, llama via the IR (`chat_sftfinal_v3`, `EMBER_INTERPRETER=1`) and
`moe_tiny_v3` via the IR stay 40/40.

Next families, by new-op cost (section 4): sliding-window attention (windowed KV + sinks),
then gating ops (flex/highway/SRD), then recurrent state (Mamba, Kimi linear).
