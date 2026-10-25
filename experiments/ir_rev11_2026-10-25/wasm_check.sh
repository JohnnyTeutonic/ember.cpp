#!/bin/bash
# Revision 1.1 under WebAssembly: deploy/wasm/build.sh's sources and flags (-O3 -msimd128 -std=c++17 -DNDEBUG,
# -fexceptions, Boost headers from Emscripten's port; no FMA instruction exists in the build), compiled from SRC
# (an ext4 copy of ember) into OUT/obj, then linked for node (NODERAWFS) twice:
#   OUT/dequant_check.js  this folder's dequant_check.cpp: quant/blocks.json bit for bit (both dequant routes);
#   OUT/ir_wasm_gate.js   tools/ir_wasm_gate (check mode) against the native build's reference (ref mode) on the
#                         suite's pk_recurrence model (a recurrence op; its paperkiln.ir_sha256 checked at load).
# The native dequant_check and ir_wasm_gate link BUILD's libtinyllama_core.a. JOBS em++ processes.
# Usage (WSL, under scripts/cpu_budget.py): JOBS=2 SRC=... BUILD=... bash wasm_check.sh OUT
set -euo pipefail
OUT=${1:?out_dir}
SRC=${SRC:?ember sources (ext4)}
BUILD=${BUILD:?the CPU build with libtinyllama_core.a}
JOBS=${JOBS:-1}
H=$(cd "$(dirname "$0")" && pwd)
SUITE=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/spec/conformance
JSON_INC=$BUILD/_deps/nlohmann_json-src/include
[ -d "$JSON_INC" ] || JSON_INC=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
mkdir -p "$OUT/obj" "$OUT/gate"
cd "$SRC"
g++ -O3 -DNDEBUG -std=c++17 -I. -I"$JSON_INC" "$H/dequant_check.cpp" "$BUILD/libtinyllama_core.a" -fopenmp -lpthread \
  -o "$OUT/dequant_check"
g++ -O3 -DNDEBUG -std=c++17 -I. -I"$JSON_INC" tools/ir_wasm_gate.cpp "$BUILD/libtinyllama_core.a" -fopenmp -lpthread \
  -o "$OUT/ir_wasm_gate"
echo "native rc=$?"
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
INC=$OUT/inc
mkdir -p "$INC"
ln -sfn "$JSON_INC/nlohmann" "$INC/nlohmann"
CORE=$(sed -n 's/^SRC="//p;/^     /p' deploy/wasm/build.sh | tr -d '"' | tr ' ' '\n' | grep '\.cpp$')
export SRC INC OUT
echo "$CORE" | xargs -P "$JOBS" -I{} bash -c '
  f={}; o=$OUT/obj/$(echo $f | tr / _).o
  [ -f "$o" ] && [ "$o" -nt "$SRC/$f" ] && exit 0
  em++ -O3 -msimd128 -std=c++17 -DNDEBUG -I"$SRC" -I"$INC" -Wno-unknown-pragmas -sUSE_BOOST_HEADERS=1 -fexceptions -c "$SRC/$f" -o "$o"'
LINK="-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sNODERAWFS=1 -sENVIRONMENT=node -sEXIT_RUNTIME=1 -sUSE_BOOST_HEADERS=1 -fexceptions"
em++ -O3 -msimd128 -std=c++17 -DNDEBUG -I"$SRC" -I"$INC" -fexceptions -sUSE_BOOST_HEADERS=1 -c "$H/dequant_check.cpp" -o "$OUT/dequant_check.o"
em++ -O3 -msimd128 -std=c++17 -DNDEBUG -I"$SRC" -I"$INC" -fexceptions -sUSE_BOOST_HEADERS=1 -c tools/ir_wasm_gate.cpp -o "$OUT/ir_wasm_gate.o"
em++ -O3 -msimd128 "$OUT"/obj/*.o "$OUT/dequant_check.o" $LINK -o "$OUT/dequant_check.js"
em++ -O3 -msimd128 "$OUT"/obj/*.o "$OUT/ir_wasm_gate.o" $LINK -o "$OUT/ir_wasm_gate.js"
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
echo "== dequant native"; "$OUT/dequant_check" "$SUITE/quant/blocks.json"
echo "== dequant wasm"; "$NODE" "$OUT/dequant_check.js" "$SUITE/quant/blocks.json"
G=$SUITE/models/pk_recurrence/pk_recurrence.gguf
echo "== pk_recurrence native reference"; "$OUT/ir_wasm_gate" ref "$G" "$OUT/gate" --n 40 --len 8 --gen 6 --ctx 16 | tail -2
echo "== pk_recurrence wasm"; "$NODE" "$OUT/ir_wasm_gate.js" check "$G" "$OUT/gate" --ctx 16
em++ --version | head -1
