# WebAssembly gated per third-party family, 25 Oct 2026

ember's IR interpreter compiled to WebAssembly is gated against ember's native CPU build on each
third-party family: Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Gemma-3-270m-it, Pythia-160m and Mamba-130m
(official checkpoints, `C:/ml_artifacts/thirdparty/`), and the synthetic Mistral sliding-window and
GPT-2 models of `../thirdparty_2026-10-22/` (LOCAL_LLM_STACK.md section 11, item 4.1). Sources: ember
d49c3e1.

## Method

- `tools/ir_wasm_gate.cpp`, one source for both sides. `ref` is the native build, linked with
  `libtinyllama_core.a` from the CPU build every CPU receipt uses. `check` is the WebAssembly build:
  `deploy/wasm/build.sh`'s sources and compile flags (`-O3 -msimd128`, `-fexceptions`, Emscripten's
  Boost port), with the gate in place of the page's `ember_wasm.cpp`, run in node (emsdk's 24.19.0)
  with NODERAWFS. The model is read without mmap, as the page reads it. Emscripten 6.0.6.
  `build_gate.sh` builds both.
- The prompts are the CUDA leg's (`tools/ir_device_gate`): seeded random prompts of 24 token ids, drawn
  by the same generator. For each one: the prefill logits at every position (max |dlogit| over the
  native row's range, and the argmax), WebAssembly's token-by-token decode against the native prefill,
  and 32 greedy tokens on each side. `EMBER_IR_CTX=1024` on both sides.
- Heap: `-sMAXIMUM_MEMORY` 4gb for every official checkpoint and 1gb for the synthetic models. Pythia
  and Mamba died at load in the 1gb build (an uncaught C++ exception after the tensors were read and
  the IR built, as the heap ran out). The 4gb build runs them.
- Prompts: 40 per family, and 10 for the two 0.5B models. Their WebAssembly forward on one core of a
  loaded laptop takes 55 s (Qwen2.5) and 150 s (Qwen3) per prompt (57 forward calls each).
  40 prompts would have taken 2.5 and 7 hours.
- `run_gate.sh` runs it under the CPU budget, one core per job, two families at a time, and resumes.
  Receipts per family: `ids.json` (all 40 prompts), `ref.jsonl` (native argmax per position and
  greedy tokens), `check.jsonl` (per prompt: distances, top-1 count, first greedy difference) and
  `summary.json`. The reference logits (15-25 MB per prompt) are deleted once a family passes.

## Results

| Family | Heap | Prompts | Greedy (32 tokens) | Top-1 | Worst max \|dlogit\| / range (prefill; decode) | Median | ms per prompt |
|---|---|---:|---:|---:|---|---:|---:|
| Qwen2.5-0.5B-Instruct f16 | 4gb | 10 | 10/10 | 240/240 | 9.0e-7; 9.0e-7 | 5.8e-7 | 55,400 |
| Qwen3-0.6B Q8_0 | 4gb | 10 | 10/10 | 240/240 | 9.8e-7; 9.8e-7 | 7.0e-7 | 152,500 |
| Gemma-3-270m-it f32 | 4gb | 40 | 40/40 | 960/960 | 1.4e-5; 1.4e-5 | 4.3e-6 | 10,800 |
| Pythia-160m f32 | 4gb | 40 | 40/40 | 960/960 | 3.6e-4; 3.6e-4 | 1.0e-5 | 10,600 |
| Mamba-130m f32 | 4gb | 40 | 40/40 | 960/960 | 2.0e-5; 2.0e-5 | 3.7e-6 | 8,300 |
| Mistral sliding window (synthetic) | 1gb | 40 | 40/40 | 960/960 | 0 (bit-identical) | 0 | 39 |
| GPT-2 (synthetic) | 1gb | 40 | 40/40 | 960/960 | 1.4e-7; 1.4e-7 | 1.2e-7 | 132 |

Every greedy continuation is identical and every argmax agrees. That is the real-model rule of the
CUDA leg (`../thirdparty_2026-10-22/device_gate/README.md`; LOCAL_LLM_STACK section 5.3), under which
closeness is judged against a float64 evaluation rather than the flat 1e-4 bound in the receipts'
`pass` field. Pythia is the one family past that bound (`summary.json` `pass` false). Three prompts pass
1e-4 (2, 22, 37), and the median is 1.0e-5. Its CUDA leg has the same profile: median 1.2e-5, and four
prompts over 1e-4 on each GPU (worst 2.0e-4 on the RTX 4060, 3.9e-4 on the L4). Prompts 22 and 37 are
among those four on both GPUs. It is the float32 conditioning of Pythia's deep positions that the
CUDA leg met, not something WebAssembly adds.

## The browser page

At these sources (d49c3e1) `deploy/wasm/build.sh` and `ember_wasm.cpp` pass no `EMBER_IR_CTX`, so the
page's interpreter allocates f32 KV state for the whole context each IR declares. From the declared shapes: Qwen2.5-0.5B
(24 layers, 2 KV heads x 64, 32,768 positions) 0.8 GB; Gemma-3-270m (18 x 1 x 256, 32,768) 1.2 GB;
Qwen3-0.6B (28 x 8 x 128, 40,960) 9.4 GB, beyond any 4 GB heap. The page also copies the file into
its in-memory filesystem before loading it. So a page serving a third-party model needs a context cap
first, at least for Qwen3. The gate above runs the same interpreter with the cap the native gates use.
The page's cap (default 2048, `EMBER_WASM_CTX`, `?ctx=N`) is gated in `../wasm_ctx_2026-10-25/`.
