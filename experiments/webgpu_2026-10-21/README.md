# WebGPU forward pass for the browser chat, 21 Oct 2026

The browser chat page runs the model's forward pass on the visitor's GPU through WebGPU
(`deploy/wasm/web/webgpu/`) when the browser has it, and ember's WebAssembly keeps the request rules,
tokeniser, sampler and stops (`ember_prepare` / `ember_sample` / `ember_finish`). Without WebGPU, or
with `?backend=wasm`, the page runs everything in WebAssembly as before (`ember_chat`). Hardware: the
laptop's Intel Gen12 integrated GPU, Edge 155 headless through Playwright.

| Check (40 frozen prompts, greedy, 60 new tokens) | Result | Time per reply |
|---|---|---|
| chat_sftfinal_v3 (f32), WebGPU forward vs ember CPU ids (`tools/webgpu_parity/greedy.html`) | 40/40 ids and stops | 146-340 ms |
| chat_sftfinal_q8all, Q8_0 weights kept quantised on the GPU | 40/40 | ~400 ms |
| fw_moe (70M, E=8 top-2) | 40/40 | 0.5-1.1 s (SIMD WASM: ~5 s) |
| The page's own worker, WebGPU vs WASM vs ember CPU replies (`tools/webgpu_parity/page_test.sh` + `page_test.html`, ember 1fdaaa5 build) | 40/40 and 40/40 | WebGPU 166 ms, WASM 646 ms |

First-token logits (`index.html`) match ember CPU within 2.1e-6 of the logit range on 5 prompts (the
reference dump's 6-digit precision). The time ranges are from runs sharing the laptop with other
sessions' jobs; the page-test pair ran back to back under the same load.

Notes:
- Shapes come from GGUF metadata: files written before `tools/quantize_gguf.py`'s dims fix record 2-D
  tensors as [out, in] (the 19 Oct q8all does), so dims cannot be trusted for the vocabulary size.
- One command buffer per token (75 dispatches for d384 L6); only a 16-byte step uniform is written per
  token and the logits are read back once per call.
- Not yet: the IR families (gpt2, kimi, mamba, ...), batched prefill, a Space release (it goes out with
  the next model through `tools/release_chat.py`).
