#!/bin/bash
# The WebAssembly leg of the E4 gates (WSL, under scripts/cpu_budget.py, 1 core):
#   bash e4_wasm.sh COMMIT OUT
# ../wasm_state_2026-10-25/build.sh builds tools/ir_wasm_gate natively (against the pinned CPU build of COMMIT,
# ~/ember_adapter_<hash>/build, from e4_cpu_gates.sh) and to WebAssembly with deploy/wasm/build.sh's sources and
# flags (-O3 -msimd128, 1gb heap, node with NODERAWFS) from the same archive. Per file: the native reference, then
# the WebAssembly check on the same seeded ids (prefill logits per position, token-by-token decode, greedy):
# gemma2, phi3_longrope and phi3_swa at 40 x 24 ids, 32 greedy (phi3_longrope's greedy run crosses
# original_ctx 32: the slot is recomputed in both builds), phi3_longrope also 40 x 40, 16 greedy (the prefill past
# original_ctx, the token-by-token decode crossing it); the suite's tp_gemma2 and tp_phi3 at 40 x 12, 16 greedy
# (vocab 24, ctx 64; tp_phi3's original_ctx is 6). OUT/wasm_<name>.json: each check's summary.
set -u
C=${1:?commit}; OUT=${2:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
FULL=$(git -C $R/ember.cpp rev-parse "$C")
P=~/ember_adapter_${FULL:0:7}
W=~/e4a_work/wasm_${FULL:0:7}
mkdir -p "$OUT" "$W"
tr -d '\r' < $R/ember.cpp/experiments/wasm_state_2026-10-25/build.sh > $W/build.sh
SRC=$P/src BUILD=$P/build JOBS=1 bash $W/build.sh $W/bin > $W/build.log 2>&1
echo "build rc=$? $(grep -E 'rc=|emcc' $W/build.log | tr '\n' ' ')" | tee "$OUT/wasm_build.txt"
(cd $W/bin && sha256sum ir_wasm_gate ir_wasm_gate_1gb.js ir_wasm_gate_1gb.wasm) | tee -a "$OUT/wasm_build.txt"
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
S=/mnt/c/ml_artifacts/thirdparty/e4_synth
K=~/keramos_v1.5.0/conformance/models
run() {  # run NAME GGUF ARGS...
  local name=$1 g=$2 d=$W/$1; shift 2
  mkdir -p $d
  $W/bin/ir_wasm_gate ref $g $d "$@" > $d/ref.out 2> $d/ref.log; echo "$name ref rc=$?"
  (cd $d && $NODE $W/bin/ir_wasm_gate_1gb.js check $g $d > $OUT/wasm_$name.json 2> $d/check.log); echo "$name check rc=$?"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], {k: d.get(k) for k in ('prompts','top1_agree','greedy_identical','worst_prefill_rel','worst_decode_rel','pass')})" "$OUT/wasm_$name.json" "$name" | tee -a "$OUT/wasm_summary.txt"
  rm -f $d/ref_*.bin
}
for f in gemma2 phi3_longrope phi3_swa; do run $f $S/$f.gguf --n 40 --len 24 --gen 32; done
run phi3_longrope_len40 $S/phi3_longrope.gguf --n 40 --len 40 --gen 16
for m in tp_gemma2 tp_phi3; do run $m $K/$m/$m.gguf --n 40 --len 12 --gen 16; done
