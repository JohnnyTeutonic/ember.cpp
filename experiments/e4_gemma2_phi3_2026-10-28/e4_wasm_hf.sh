#!/bin/bash
# The WebAssembly leg on HF's cases (WSL, under scripts/cpu_budget.py, 1 core), after e4_wasm.sh (its objects):
#   bash e4_wasm_hf.sh COMMIT OUT
# e4_device_gate.cpp built natively (against the pinned CPU build of COMMIT) and to WebAssembly (e4_wasm.sh's objects,
# the page's flags, node with NODERAWFS). On a CPU build its "device" is a second CPU interpreter, so each run gives
# ember's greedy against HF and the first token against HF's logits; both builds dump every row of the second
# interpreter (the prefill's last row, then each decode step: phi3_longrope's 24-id sequences cross original_ctx 32
# during the greedy run and recompute the slot), and e4_rows_cmp.py compares the WebAssembly rows with the native
# ones (bit-equal rows, worst |d| over the native row's range). OUT/wasm_hf_<file>.json.
set -u
C=${1:?commit}; OUT=${2:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
FULL=$(git -C $R/ember.cpp rev-parse "$C")
P=~/ember_adapter_${FULL:0:7}
W=~/e4a_work/wasm_${FULL:0:7}
E=$(cd "$(dirname "$0")" && pwd)
JSON_INC=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
mkdir -p "$OUT" $W/hf
g++ -O3 -DNDEBUG -std=c++17 -I$P/src -I$JSON_INC $P/src/experiments/e4_gemma2_phi3_2026-10-28/e4_device_gate.cpp \
  $P/build/libtinyllama_core.a -fopenmp -lpthread -o $W/hf/e4_device_gate
echo "native rc=$?"
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
CORE_OBJ=$(ls $W/bin/obj/*.o | grep -v -e tools_state_wasm_gate -e tools_ir_wasm_gate)
em++ -O3 -msimd128 -std=c++17 -DNDEBUG -I$P/src -I$W/bin/inc -Wno-unknown-pragmas -sUSE_BOOST_HEADERS=1 -fexceptions \
  -c $P/src/experiments/e4_gemma2_phi3_2026-10-28/e4_device_gate.cpp -o $W/hf/e4_device_gate.o
em++ -O3 -msimd128 $CORE_OBJ $W/hf/e4_device_gate.o -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sNODERAWFS=1 \
  -sENVIRONMENT=node -sEXIT_RUNTIME=1 -sUSE_BOOST_HEADERS=1 -fexceptions -o $W/hf/e4_device_gate.js
echo "wasm rc=$?"
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
S=/mnt/c/ml_artifacts/thirdparty/e4_synth
for f in gemma2 phi3_longrope phi3_swa; do
  OMP_NUM_THREADS=1 $W/hf/e4_device_gate $S/$f.gguf $S/$f.cases.json $W/hf/native_$f.json $W/hf/native_$f.bin > $W/hf/native_$f.out 2> /dev/null
  echo "native $f rc=$?: $(cat $W/hf/native_$f.out)"
  (cd $W/hf && $NODE $W/hf/e4_device_gate.js $S/$f.gguf $S/$f.cases.json $W/hf/wasm_$f.json $W/hf/wasm_$f.bin > $W/hf/wasm_$f.out 2> /dev/null)
  echo "wasm $f rc=$?: $(cat $W/hf/wasm_$f.out)"
  python3 $E/e4_rows_cmp.py $W/hf/native_$f.bin $W/hf/wasm_$f.bin 512 $W/hf/wasm_$f.json > $OUT/wasm_hf_$f.json
  cat $OUT/wasm_hf_$f.json
done
