#!/bin/bash
# WebAssembly leg of the X5 gate (WSL; run under the CPU budget): ../wasm_state_2026-10-25/build.sh builds
# tools/state_wasm_gate and tools/ir_wasm_gate natively (against ~/x5g/new/build_cpu) and to WebAssembly
# (deploy/wasm/build.sh's flags, 1gb heap) from the e115fd3 tree; then
#   (a) state_wasm_gate native and WebAssembly on ar_x5's 40 score_in sequences, compared by
#       ../wasm_state_2026-10-25/compare.py: OUT/wasm_ar_x5.json; the WebAssembly log-probs vs mtstudio (x5_wasm_logprobs.py);
#   (b) ir_wasm_gate on random ids, 40 x 8 ids, 8 greedy (ar3's sizes; the IR holds 16 positions): OUT/wasm_random_ar_x5.json.
#   bash x5_wasm.sh OUT
set -u
OUT=${1:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
WS=$R/ember.cpp/experiments/wasm_state_2026-10-25
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/attnres_x5_2026-10-25
G=/mnt/c/ml_artifacts/attnres_gate_x5/ar_x5/ar_x5.gguf
W=~/x5g/wasm
BIN=$W/bin
mkdir -p "$OUT" "$W/inputs"
cp $WS/build.sh $WS/compare.py $W/
SRC=~/x5g/new/src BUILD=~/x5g/new/build_cpu JOBS=2 bash $W/build.sh $BIN > $W/build.log 2>&1
echo "build rc=$? $(grep -E 'rc=|emcc' $W/build.log | tr '\n' ' ')"
python3 $S/x5_inputs.py $W/inputs
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
d=$W/ar_x5; mkdir -p $d/native $d/wasm
$BIN/state_wasm_gate $G $W/inputs/ar_x5_in.json $d/native 2> $d/native.log; echo "native rc=$?"
$NODE $BIN/state_wasm_gate.js $G $W/inputs/ar_x5_in.json $d/wasm 2> $d/wasm.log; echo "wasm rc=$?"
python3 $W/compare.py ar_x5 $W/inputs/ar_x5_in.json $d/native $d/wasm > $OUT/wasm_ar_x5.json
echo "compare rc=$?"
python3 $S/x5_wasm_logprobs.py $d/wasm > $OUT/wasm_ar_x5_logprobs.json; cat $OUT/wasm_ar_x5_logprobs.json
r=$W/random; mkdir -p $r
$BIN/ir_wasm_gate ref $G $r --len 8 --gen 8 > /dev/null 2> $r/ref.log; echo "random ref rc=$?"
$NODE $BIN/ir_wasm_gate_1gb.js check $G $r > $OUT/wasm_random_ar_x5.json 2> $r/check.log; echo "random check rc=$?"
tr -d '\n ' < $OUT/wasm_random_ar_x5.json | head -c 600; echo
grep -m1 "em++\|emcc" $W/build.log; ~/emsdk/upstream/emscripten/em++ --version 2>/dev/null | head -1
