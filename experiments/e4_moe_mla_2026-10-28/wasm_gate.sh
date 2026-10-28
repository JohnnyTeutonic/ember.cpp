#!/bin/bash
# E4 (MoE + MLA) WebAssembly leg: tools/ir_wasm_gate from the e4b source tree, native (linked with the e4b CPU
# build's libtinyllama_core.a) and WebAssembly (deploy/wasm/build.sh's sources and flags: -O3 -msimd128
# -std=c++17 -DNDEBUG -fexceptions, Emscripten's Boost port; 1gb heap, the page's default), then per synthetic
# file: ref (native) and check (node) on ir_device_gate's 40 seeded prompts of 24 ids, 32 greedy tokens.
# Resumes (finished objects and prompts are kept).
#   SRC=~/ember_adapter_C/src BUILD=~/ember_adapter_C/build bash wasm_gate.sh native, then build, then run FAM ...
set -uo pipefail
SRC=${SRC:-$HOME/e4b/ember.cpp}
BUILD=${BUILD:-$HOME/e4b/build_cpu}
OUT=${OUT:-$HOME/e4b/wasm}
SYN=/mnt/c/ml_artifacts/thirdparty/e4_synth
JSON_INC=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
mkdir -p "$OUT/obj" "$OUT/inc" "$OUT/runs"
if [ "$1" = native ]; then
  cd "$SRC"
  g++ -O3 -DNDEBUG -std=c++17 -I. -I"$JSON_INC" tools/ir_wasm_gate.cpp "$BUILD/libtinyllama_core.a" -fopenmp -lpthread \
    -o "$OUT/ir_wasm_gate" && echo "native ok"
  sha256sum "$OUT/ir_wasm_gate"
  exit 0
fi
if [ "$1" = build ]; then
  cd "$SRC"
  source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
  ln -sfn "$JSON_INC/nlohmann" "$OUT/inc/nlohmann"
  CORE=$(sed -n '/^SRC="/,/"$/p' deploy/wasm/build.sh | tr -d '"' | sed 's/^SRC=//')
  for f in $CORE tools/ir_wasm_gate.cpp; do
    o=$OUT/obj/$(echo "$f" | tr / _).o
    if [ ! -f "$o" ] || [ "$SRC/$f" -nt "$o" ]; then
      em++ -O3 -msimd128 -std=c++17 -DNDEBUG -I"$SRC" -I"$OUT/inc" -Wno-unknown-pragmas -sUSE_BOOST_HEADERS=1 -fexceptions \
        -c "$SRC/$f" -o "$o" || { echo "em++ $f failed"; exit 1; }
    fi
  done
  em++ -O3 -msimd128 "$OUT"/obj/*.o -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sNODERAWFS=1 -sENVIRONMENT=node \
    -sEXIT_RUNTIME=1 -sUSE_BOOST_HEADERS=1 -fexceptions -o "$OUT/ir_wasm_gate_1gb.js" && echo "wasm ok"
  em++ --version | head -1
  sha256sum "$OUT/ir_wasm_gate_1gb.wasm"
  exit 0
fi
shift
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
for fam in "$@"; do
  d=$OUT/runs/$fam
  mkdir -p "$d"
  X=""
  [ "$fam" = mixtral ] && X=--build-ir  # a llama file: its IR built on request (EMBER_IR_BUILD=1); the page serves it on the llama path
  "$OUT/ir_wasm_gate" ref "$SYN/$fam.gguf" "$d" $X > "$d/ref.out" 2>> "$d/ref.log"
  echo "$fam ref rc=$?"
  (cd "$d" && "$NODE" "$OUT/ir_wasm_gate_1gb.js" check "$SYN/$fam.gguf" "$d" $X > "$d/summary.json" 2>> "$d/check.log")
  echo "$fam check rc=$? $(tr -d '\n ' < "$d/summary.json" | cut -c1-400)"
done
