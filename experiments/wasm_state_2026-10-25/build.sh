#!/bin/bash
# Build tools/state_wasm_gate and tools/ir_wasm_gate twice from an ember source tree SRC (ext4):
# - native: linked with that tree's CPU build (BUILD: libtinyllama_core.a) into OUT/;
# - WebAssembly: deploy/wasm/build.sh's sources and compile flags (-O3 -msimd128 -std=c++17 -DNDEBUG,
#   -fexceptions, Boost from Emscripten's port) with the tool in place of the page's ember_wasm.cpp,
#   linked for node with NODERAWFS at the page's 1gb heap ceiling: OUT/state_wasm_gate.js and
#   OUT/ir_wasm_gate_1gb.js. Objects are compiled once (JOBS em++ processes) into OUT/obj.
# Usage (WSL): SRC=... BUILD=... JOBS=2 bash build.sh OUT   (emsdk at ~/emsdk; under scripts/cpu_budget.py)
set -euo pipefail
OUT=${1:?out_dir}
SRC=${SRC:?source tree}
BUILD=${BUILD:?CPU build dir}
JOBS=${JOBS:-1}
JSON_INC=$BUILD/_deps/nlohmann_json-src/include
[ -d "$JSON_INC" ] || JSON_INC=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
mkdir -p "$OUT/obj"
cd "$SRC"
for t in state_wasm_gate ir_wasm_gate; do
  g++ -O3 -DNDEBUG -std=c++17 -I. -I"$JSON_INC" tools/$t.cpp "$BUILD/libtinyllama_core.a" -fopenmp -lpthread -o "$OUT/$t"
  echo "native $t rc=$?"
done
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
INC=$OUT/inc
mkdir -p "$INC"
ln -sfn "$JSON_INC/nlohmann" "$INC/nlohmann"
# deploy/wasm/build.sh's SRC list
CORE=$(sed -n '/^SRC="/,/"$/p' deploy/wasm/build.sh | tr -d '"' | sed 's/^SRC=//')
export SRC INC OUT
echo $CORE tools/state_wasm_gate.cpp tools/ir_wasm_gate.cpp | tr ' ' '\n' | grep -v '^$' | xargs -P "$JOBS" -I{} bash -c '
  f={}; o=$OUT/obj/$(echo $f | tr / _).o
  [ -f "$o" ] && [ "$o" -nt "$SRC/$f" ] && exit 0
  em++ -O3 -msimd128 -std=c++17 -DNDEBUG -I"$SRC" -I"$INC" -Wno-unknown-pragmas -sUSE_BOOST_HEADERS=1 -fexceptions -c "$SRC/$f" -o "$o"'
CORE_OBJ=$(ls "$OUT"/obj/*.o | grep -v -e tools_state_wasm_gate -e tools_ir_wasm_gate)
for t in state_wasm_gate ir_wasm_gate; do
  name=$t; [ $t = ir_wasm_gate ] && name=ir_wasm_gate_1gb
  em++ -O3 -msimd128 $CORE_OBJ "$OUT/obj/tools_$t.cpp.o" -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sNODERAWFS=1 \
      -sENVIRONMENT=node -sEXIT_RUNTIME=1 -sUSE_BOOST_HEADERS=1 -fexceptions -o "$OUT/$name.js"
  echo "wasm $t rc=$?"
done
em++ --version | head -1
ls -la "$OUT"
