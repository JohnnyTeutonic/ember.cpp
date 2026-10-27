#!/bin/bash
# The arXiv chain's Shaw model on ember's other legs (WSL; under the CPU budget), wasm_gate.sh's method with the
# pinned build's tools (~/d2e/wasm/bin): 40 seeded prompts of 16 ids and 16 greedy tokens (EMBER_IR_CTX 64, the
# model's context), the references
#   native  ir_wasm_gate ref (ember's CPU interpreter)                   -> ~/d2e/wasm/chain/native
#   numpy   ref_numpy.py (keramos v1.4.0's numpy reference, a black box) -> ~/d2e/wasm/chain/numpy
# and the WebAssembly build's check against each -> OUT/wasm_chain_vs_native.json, _vs_numpy.json (--tol 1e-5).
#   bash chain_legs.sh OUT
set -u
OUT=${1:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/d2_primitives_2026-10-27
K=$HOME/d2e/keramos_v1.4.0
G0=/mnt/c/ml_artifacts/arxiv_chain_2026-10-27_d2/arxiv_1803_02155.gguf
MD=$HOME/d2e/chain/arxiv_1803_02155    # the runner's load_model reads <dir>/<dir name>.gguf
mkdir -p "$OUT" "$MD"
ln -sfn $G0 $MD/arxiv_1803_02155.gguf
G=$MD/arxiv_1803_02155.gguf
BIN=$HOME/d2e/wasm/bin
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
d=$HOME/d2e/wasm/chain; mkdir -p $d/native $d/numpy $d/check_native $d/check_numpy
$BIN/ir_wasm_gate ref $G $d/native --n 40 --len 16 --gen 16 --ctx 64 > $d/native.out 2> $d/native.log
echo "chain native ref rc=$?"
python3 $H/ref_numpy.py $K $MD $d/native $d/numpy > $d/numpy.out 2> $d/numpy.log
echo "chain numpy ref rc=$? $(cat $d/numpy.out)"
cp $d/numpy.out $OUT/native_vs_numpy_chain.json
python3 - $d/native/ref.jsonl $d/numpy/ref.jsonl <<'EOF' | tee $OUT/native_vs_numpy_chain_greedy.txt
import json, sys
a = {json.loads(l)["i"]: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
b = {json.loads(l)["i"]: json.loads(l) for l in open(sys.argv[2]) if l.strip()}
print("native greedy = numpy greedy:", sum(a[i]["greedy"] == b[i]["greedy"] for i in a), "/", len(a),
      "; argmax per position:", sum(x == y for i in a for x, y in zip(a[i]["top1"], b[i]["top1"])), "/",
      sum(len(a[i]["top1"]) for i in a))
EOF
for r in native numpy; do
  c=$d/check_$r
  cp $d/$r/ids.json $d/$r/ref.jsonl $d/$r/ref_*.bin $c/
  rm -f $c/check.jsonl
  tol=1e-4; [ $r = numpy ] && tol=1e-5
  $NODE $BIN/ir_wasm_gate_1gb.js check $G $c --ctx 64 --tol $tol > $OUT/wasm_chain_vs_$r.json 2> $c.log
  echo "chain wasm vs $r rc=$?"; tr -d '\n ' < $OUT/wasm_chain_vs_$r.json; echo
  cp $c/check.jsonl $OUT/wasm_chain_vs_$r.check.jsonl
done
