#!/bin/bash
# WebAssembly leg of the tensor_index gate (WSL; run under the CPU budget, tig_job.sh), the method of
# ../attnres_x5_2026-10-25/x5_wasm.sh: ../wasm_state_2026-10-25/build.sh builds tools/state_wasm_gate and
# tools/ir_wasm_gate natively (against ~/tig/ember/build_cpu) and to WebAssembly (deploy/wasm/build.sh's flags,
# 1gb heap) from the ember tree at ~/tig/ember/src; then per variant (ar_rec, ar_rec_q8_0):
#   (a) state_wasm_gate native and WebAssembly on the 40 score_in sequences, compared by
#       ../wasm_state_2026-10-25/compare.py: OUT/wasm_<v>.json; the WebAssembly prefill's log-probs vs mtstudio and
#       vs ember's CPU (tig_logprobs.py): OUT/wasm_<v>_logprobs.json;
#   (b) ir_wasm_gate on random ids, 40 x 8 ids, 8 greedy (the IR holds 16 positions): OUT/wasm_random_<v>.json.
#   bash tig_wasm.sh OUT
set -u
OUT=${1:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
WS=$R/ember.cpp/experiments/wasm_state_2026-10-25
S=$R/ember.cpp/experiments/tensor_index_gate_2026-10-27
D=/mnt/c/ml_artifacts/attnres_gate_rec/ar_rec
W=~/tig/wasm
BIN=$W/bin
mkdir -p "$OUT" "$W/inputs"
cp $WS/build.sh $WS/compare.py $W/
SRC=~/tig/ember/src BUILD=~/tig/ember/build_cpu JOBS=2 bash $W/build.sh $BIN > $W/build.log 2>&1
echo "build rc=$? $(grep -E 'rc=|emcc' $W/build.log | tr '\n' ' ')"
python3 $S/tig_inputs.py $W/inputs
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
for v in ar_rec ar_rec_q8_0; do
  G=$D/$v.gguf
  cv=f32; [ $v = ar_rec_q8_0 ] && cv=q8_0
  d=$W/$v; mkdir -p $d/native $d/wasm
  $BIN/state_wasm_gate $G $W/inputs/ar_rec_in.json $d/native 2> $d/native.log; echo "$v native rc=$?"
  $NODE $BIN/state_wasm_gate.js $G $W/inputs/ar_rec_in.json $d/wasm 2> $d/wasm.log; echo "$v wasm rc=$?"
  python3 $W/compare.py $v $W/inputs/ar_rec_in.json $d/native $d/wasm > $OUT/wasm_$v.json
  echo "$v compare rc=$?"
  python3 $S/tig_logprobs.py wasm $d/wasm ~/tig/work/cpu_${cv}_tagged.json > $OUT/wasm_${v}_logprobs.json
  cat $OUT/wasm_${v}_logprobs.json
  r=$W/random_$v; mkdir -p $r
  $BIN/ir_wasm_gate ref $G $r --len 8 --gen 8 > /dev/null 2> $r/ref.log; echo "$v random ref rc=$?"
  $NODE $BIN/ir_wasm_gate_1gb.js check $G $r > $OUT/wasm_random_$v.json 2> $r/check.log; echo "$v random check rc=$?"
  tr -d '\n ' < $OUT/wasm_random_$v.json | head -c 600; echo
done
# Does node hand EMBER_IR to the WebAssembly engine? The untagged document through EMBER_IR, F32 (reported).
d=$W/untagged_probe; mkdir -p $d/wasm
EMBER_IR=$HOME/tig/work/ar_rec_untagged.ir.json $NODE $BIN/state_wasm_gate.js $D/ar_rec.gguf $W/inputs/ar_rec_in.json $d/wasm 2> $d/wasm.log
echo "untagged probe rc=$?"
python3 $S/tig_logprobs.py wasm $d/wasm ~/tig/work/cpu_f32_untagged.json > $OUT/wasm_untagged_probe.json; cat $OUT/wasm_untagged_probe.json
~/emsdk/upstream/emscripten/em++ --version 2>/dev/null | head -1
