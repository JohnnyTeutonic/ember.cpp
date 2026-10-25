#!/usr/bin/env bash
# Build ember chat for the browser: ember.cpp's CPU core + deploy/wasm/ember_wasm.cpp
# compiled by Emscripten into ember.js + ember.wasm (single-threaded; OpenMP
# pragmas compile away, CUDA and MPI are off). WebAssembly SIMD (-msimd128) is
# on: replies identical, about 2x faster per reply (experiments/wasm_simd_2026-10-20);
# EMBER_WASM_SIMD=0 builds without it.
#   bash deploy/wasm/build.sh OUT_DIR
# Needs: emsdk activated (emcc on PATH) and the nlohmann/json single header,
# passed through a private include directory so emcc never sees the host's
# system headers. Boost (regex, xpressive: header-only) comes from
# Emscripten's own port, as in WASM.md.
set -euo pipefail
OUT=${1:?out_dir}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
JSON_INC=${JSON_INC:?set JSON_INC to a directory containing nlohmann/json.hpp}
INC=$(mktemp -d)
ln -s "$JSON_INC/nlohmann" "$INC/nlohmann"
mkdir -p "$OUT"
SRC="model.cpp model_config.cpp kv_cache.cpp weight_management.cpp cpu_attention.cpp model_utils.cpp
     gpu_initialization.cpp gpu_memory_optimization.cpp cpu_batch_processor.cpp utils.cpp safetensors_loader.cpp
     tokenizer.cpp logger.cpp api.cpp gguf_parser.cpp quantization.cpp graph_ir.cpp graph_ir_ops.cpp graph_ir_expr.cpp graph_ir_mamba.cpp graph_ir_recurrence.cpp graph_ir_validate.cpp ir_digest.cpp parallel_for.cpp"
SIMD=-msimd128
[ "${EMBER_WASM_SIMD:-1}" = 0 ] && SIMD=
# EMBER_WASM_MAX_MEMORY (default 1gb, up to 4gb): the heap ceiling; models past about
# 150M f32 parameters need more than 1gb.
# EMBER_WASM_THREADS=N: a pthreads build (ember::parallel_for on N threads,
# workers pre-spawned); it needs a cross-origin-isolated page (COOP/COEP).
# EMBER_WASM_NAME=ember-mt names the outputs (pthread workers load the main
# script by its own name, so a threaded build is built under its final name).
THREADS=
[ -n "${EMBER_WASM_THREADS:-}" ] && THREADS="-pthread -sPTHREAD_POOL_SIZE=$EMBER_WASM_THREADS -DEMBER_WASM_THREADS=$EMBER_WASM_THREADS"
cd "$REPO"
em++ -O3 $SIMD $THREADS -std=c++17 -DNDEBUG -I"$REPO" -I"$INC" -Wno-unknown-pragmas \
    $SRC deploy/wasm/ember_wasm.cpp \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=${EMBER_WASM_MAX_MEMORY:-1gb} -sMODULARIZE=1 -sEXPORT_NAME=createEmber \
    -sEXPORTED_FUNCTIONS=_ember_load,_ember_chat,_ember_prepare,_ember_sample,_ember_finish,_malloc,_free -sEXPORTED_RUNTIME_METHODS=ccall,FS,HEAPF32 \
    -sFORCE_FILESYSTEM=1 -sENVIRONMENT=web,worker,node -sUSE_BOOST_HEADERS=1 -fexceptions \
    -o "$OUT/${EMBER_WASM_NAME:-ember}.js"
rm -rf "$INC"
ls -la "$OUT"
