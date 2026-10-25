#!/bin/bash
# WebAssembly leg (WSL, under the CPU budget): build state_wasm_gate native + WebAssembly from ~/x5s/src with
# ../../wasm_state_2026-10-25/build.sh (JOBS=2, timed; capped at 15 minutes), then x5s_wasm.py.
#   bash x5s_wasm.sh OUT_DIR
set -u
OUT=${1:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
WS=$R/ember.cpp/experiments/wasm_state_2026-10-25
S=$R/ember.cpp/experiments/attnres_x5_2026-10-25/smoke
W=~/x5s/wasm
mkdir -p $W "$OUT"
cp $WS/build.sh $WS/compare.py $W/
t0=$(date +%s)
SRC=~/x5s/src BUILD=~/x5s/build JOBS=2 timeout 900 bash $W/build.sh $W/bin > $W/build.log 2>&1
echo "build rc=$? in $(( $(date +%s) - t0 )) s: $(grep -E 'rc=' $W/build.log | tr '\n' ' ')"
grep -m1 "emcc\|clang version" $W/build.log
[ -f $W/bin/state_wasm_gate.js ] || exit 1
python3 $S/x5s_wasm.py $W/bin $W/compare.py "$OUT/x5s_wasm.json"
echo "wasm gate rc=$?"
