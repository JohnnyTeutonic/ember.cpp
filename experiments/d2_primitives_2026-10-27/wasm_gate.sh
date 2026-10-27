#!/bin/bash
# WebAssembly leg (WSL; under the CPU budget): ../wasm_state_2026-10-25/build.sh builds tools/ir_wasm_gate natively
# (against the pinned CPU build, ~/d2e/pin) and to WebAssembly (deploy/wasm/build.sh's sources and flags, 1gb heap)
# from the pinned sources; then per model (the suite's pk_flex_mdha and pk_flex_shaw at keramos v1.4.0):
#   native  ir_wasm_gate ref: 40 seeded prompts of 8 ids, 8 greedy (ctx 16), EMBER_IR_CTX 16 -> W/<m>/native
#   numpy   ref_numpy.py: the same prompts through keramos's numpy reference -> W/<m>/numpy
#   wasm    ir_wasm_gate check (node) against each of the two -> OUT/wasm_<m>_vs_native.json, _vs_numpy.json
#   bash wasm_gate.sh OUT
set -u
OUT=${1:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
X=$R/ember.cpp/experiments
H=$X/d2_primitives_2026-10-27
K=$HOME/d2e/keramos_v1.4.0
W=$HOME/d2e/wasm
BIN=$W/bin
mkdir -p "$OUT" "$W"
cp $X/wasm_state_2026-10-25/build.sh $W/
SRC=$HOME/d2e/pin/src BUILD=$HOME/d2e/pin/build JOBS=2 bash $W/build.sh $BIN > $W/build.log 2>&1
echo "build rc=$? $(grep -E 'rc=|emcc|em\+\+' $W/build.log | tr '\n' ' ')"
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
for m in pk_flex_mdha pk_flex_shaw; do
  G=$K/conformance/models/$m/$m.gguf
  d=$W/$m; mkdir -p $d/native $d/numpy $d/check_native $d/check_numpy
  $BIN/ir_wasm_gate ref $G $d/native --n 40 --len 8 --gen 8 --ctx 16 > $d/native.out 2> $d/native.log
  echo "$m native ref rc=$?"
  python3 $H/ref_numpy.py $K $K/conformance/models/$m $d/native $d/numpy > $d/numpy.out 2> $d/numpy.log
  echo "$m numpy ref rc=$? $(cat $d/numpy.out)"
  cp $d/native/ids.json $d/native/ref.jsonl $d/native/ref_*.bin $d/check_native/ 2>/dev/null
  cp $d/numpy/ids.json $d/numpy/ref.jsonl $d/numpy/ref_*.bin $d/check_numpy/ 2>/dev/null
  rm -f $d/check_native/check.jsonl $d/check_numpy/check.jsonl
  $NODE $BIN/ir_wasm_gate_1gb.js check $G $d/check_native --ctx 16 > $OUT/wasm_${m}_vs_native.json 2> $d/check_native.log
  echo "$m wasm vs native rc=$?"
  cp $d/check_native/check.jsonl $OUT/wasm_${m}_vs_native.check.jsonl
  $NODE $BIN/ir_wasm_gate_1gb.js check $G $d/check_numpy --ctx 16 --tol 1e-5 > $OUT/wasm_${m}_vs_numpy.json 2> $d/check_numpy.log
  echo "$m wasm vs numpy rc=$?"
  cp $d/check_numpy/check.jsonl $OUT/wasm_${m}_vs_numpy.check.jsonl
  cp $d/numpy.out $OUT/native_vs_numpy_${m}.json
  # the native and numpy greedy continuations
  python3 - $d/native/ref.jsonl $d/numpy/ref.jsonl <<'EOF'
import json, sys
a = {json.loads(l)["i"]: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
b = {json.loads(l)["i"]: json.loads(l) for l in open(sys.argv[2]) if l.strip()}
print("native greedy = numpy greedy:", sum(a[i]["greedy"] == b[i]["greedy"] for i in a), "/", len(a),
      "; argmax per position:", sum(x == y for i in a for x, y in zip(a[i]["top1"], b[i]["top1"])), "/",
      sum(len(a[i]["top1"]) for i in a))
EOF
done
~/emsdk/upstream/emscripten/em++ --version 2>/dev/null | head -1
