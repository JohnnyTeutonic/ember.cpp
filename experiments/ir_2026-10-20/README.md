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

Next families, by new-op cost (section 4): gpt2 (layernorm, gelu, learned positions), then
sliding-window attention, then gating ops, then recurrent state.
