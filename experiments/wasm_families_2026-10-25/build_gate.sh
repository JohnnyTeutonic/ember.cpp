#!/bin/bash
# Build tools/ir_wasm_gate twice from an ember source tree SRC (default ~/ember_wt/src, ext4):
# - native: linked with that tree's CPU build (BUILD, default ~/ember_wt/build_cpu: libtinyllama_core.a,
#   the library every CPU receipt runs), into OUT/ir_wasm_gate;
# - WebAssembly: deploy/wasm/build.sh's sources and compile flags (-O3 -msimd128 -std=c++17 -DNDEBUG,
#   -fexceptions, Boost from Emscripten's port) with tools/ir_wasm_gate.cpp in place of the page's
#   ember_wasm.cpp, linked for node with NODERAWFS (the model and the reference read from the real
#   filesystem) at two heap ceilings: OUT/ir_wasm_gate_1gb.js and OUT/ir_wasm_gate_4gb.js
#   (EMBER_WASM_MAX_MEMORY's 1gb default and 4gb). Objects are compiled once per run (JOBS em++
#   processes) into OUT/obj and linked twice.
# Usage (WSL): JOBS=2 bash build_gate.sh OUT   (needs emsdk at ~/emsdk; run under scripts/cpu_budget.py)
set -euo pipefail
OUT=${1:?out_dir}
SRC=${SRC:-$HOME/ember_wt/src}
BUILD=${BUILD:-$HOME/ember_wt/build_cpu}
JOBS=${JOBS:-1}
JSON_INC=$BUILD/_deps/nlohmann_json-src/include
mkdir -p "$OUT/obj"
cd "$SRC"
g++ -O3 -DNDEBUG -std=c++17 -I. -I"$JSON_INC" tools/ir_wasm_gate.cpp "$BUILD/libtinyllama_core.a" -fopenmp -lpthread \
  -o "$OUT/ir_wasm_gate"
echo "native rc=$?"
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
INC=$OUT/inc
mkdir -p "$INC"
ln -sfn "$JSON_INC/nlohmann" "$INC/nlohmann"
CORE="model.cpp model_config.cpp kv_cache.cpp weight_management.cpp cpu_attention.cpp model_utils.cpp
     gpu_initialization.cpp gpu_memory_optimization.cpp cpu_batch_processor.cpp utils.cpp safetensors_loader.cpp
     tokenizer.cpp logger.cpp api.cpp gguf_parser.cpp quantization.cpp graph_ir.cpp graph_ir_ops.cpp graph_ir_expr.cpp graph_ir_mamba.cpp graph_ir_validate.cpp parallel_for.cpp
     tools/ir_wasm_gate.cpp"
export SRC INC OUT
echo $CORE | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -c '
  f={}; o=$OUT/obj/$(echo $f | tr / _).o
  em++ -O3 -msimd128 -std=c++17 -DNDEBUG -I"$SRC" -I"$INC" -Wno-unknown-pragmas -sUSE_BOOST_HEADERS=1 -fexceptions -c "$SRC/$f" -o "$o"'
for mem in 1gb 4gb; do
  em++ -O3 -msimd128 "$OUT"/obj/*.o -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=$mem -sNODERAWFS=1 -sENVIRONMENT=node \
      -sEXIT_RUNTIME=1 -sUSE_BOOST_HEADERS=1 -fexceptions -o "$OUT/ir_wasm_gate_$mem.js"
  echo "wasm $mem rc=$?"
done
em++ --version | head -1
ls -la "$OUT"
