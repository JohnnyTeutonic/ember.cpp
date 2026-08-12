# ember.cpp in WebAssembly

*Proven 12 Aug 2026 (night build): the inference core compiles to wasm
with ZERO source changes and runs real GGUF chat inference under node.
`chat7a.gguf` produced coherent chat completions; a known-bad toy
checkpoint produced the same salad it produces natively, i.e. the port
is numerically faithful.*

## Build (one command)

```bash
source ~/emsdk/emsdk_env.sh
em++ -O2 -std=c++17 \
  model.cpp model_config.cpp kv_cache.cpp weight_management.cpp \
  cpu_attention.cpp model_utils.cpp gpu_initialization.cpp \
  gpu_memory_optimization.cpp cpu_batch_processor.cpp utils.cpp \
  safetensors_loader.cpp tokenizer.cpp logger.cpp api.cpp \
  gguf_parser.cpp quantization.cpp main.cpp \
  -I. -I<dir with nlohmann/json.hpp single header> \
  -sUSE_BOOST_HEADERS=1 -sALLOW_MEMORY_GROWTH=1 -sNODERAWFS=1 \
  -sEXIT_RUNTIME=1 -Wno-unknown-pragmas \
  -o tinyllama.js
```

Artifacts: `tinyllama.js` (~79 KB glue) + `tinyllama.wasm` (~1.0 MB).

## Run (node)

```bash
node tinyllama.js chat7a.gguf chat7a.gguf 1 prompt "Once upon a time"
```

Notes:
- `-sNODERAWFS=1` lets the wasm binary read GGUF files from the real
  filesystem under node. A browser build swaps this for `--preload-file`
  or fetch-into-MEMFS, plus `-sEXPORTED_RUNTIME_METHODS` for a JS API —
  that is the remaining (small) step for the in-browser demo page.
- OpenMP pragmas compile away (`-Wno-unknown-pragmas`); single-threaded.
  `-pthread` + wasm workers is the perf follow-up if wanted.
- Boost.Regex (tokenizer) comes from Emscripten's boost-headers port
  (`-sUSE_BOOST_HEADERS=1`) — header-only, no compiled Boost needed.
- The CUDA/MPI paths are simply not compiled; CPU fallback is the path.
- Positional-arg parsing quirk: pass only up to the prompt; later
  positionals (max_tokens etc.) are rejected by the current CLI parser
  under this arg order (pre-existing behaviour, not wasm-related).
