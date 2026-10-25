#!/bin/bash
# The WebAssembly legs of the state-token and llama_attnres gates (run in WSL, one core per job under the
# CPU budget). BIN holds build.sh's outputs; work files (inputs, logits) go to WORK on ext4 and are
# deleted once a model's comparison is written here.
#   (a) tools/state_wasm_gate, native and WebAssembly, on ../cuda_state_2026-10-25/make_inputs.py's inputs
#       (state4 with its session script, state4 Q8_0, ar3, the synthetic k = 4 model's script), compared
#       by compare.py: <model>.json beside this file;
#   (b) tools/ir_wasm_gate on random ids as the CUDA leg's ir_device_gate ran them (state4: 40 x 24 ids,
#       32 greedy; ar3: 40 x 8 ids, 8 greedy, ctx 16): random_<model>.json.
#   bash run.sh BIN WORK
set -u
BIN=${1:?bin}; WORK=${2:?work}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/wasm_state_2026-10-25
A=/mnt/c/ml_artifacts
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
bud() { local job=$1; shift; python3 $R/scripts/cpu_budget.py run --session ember --cores 1 --job "$job" --resume "rerun $H/run.sh $BIN $WORK" -- "$@"; }
mkdir -p "$WORK/inputs"
python3 $R/ember.cpp/experiments/cuda_state_2026-10-25/make_inputs.py "$WORK/inputs"
one() {  # name gguf input [manifest]
  local d=$WORK/$1
  mkdir -p "$d/native" "$d/wasm"
  bud "WASM state gate: $1 native" "$BIN/state_wasm_gate" "$2" "$3" "$d/native" 2> "$d/native.log"
  bud "WASM state gate: $1 WebAssembly" "$NODE" "$BIN/state_wasm_gate.js" "$2" "$3" "$d/wasm" 2> "$d/wasm.log"
  python3 $H/compare.py "$1" "$3" "$d/native" "$d/wasm" ${4:+--manifest "$4"} > "$H/$1.json"
  local rc=$?
  echo "$1 compare rc=$rc"
  [ $rc -eq 0 ] && rm -rf "$d"
}
one state4 $A/state_gate/state4/state4.gguf "$WORK/inputs/state4_in.json" $A/state_gate/state4/manifest.json
one state4_q8_0 $A/state_gate/state4/state4_q8_0.gguf "$WORK/inputs/state4q8_in.json"
one ar3 $A/attnres_gate/ar3/ar3.gguf "$WORK/inputs/ar3_in.json"
one synth_k4 ~/ember_wt/state_tokens/chat_state_k4.gguf "$WORK/inputs/synth_in.json"
rnd() {  # name gguf len gen
  local d=$WORK/random_$1
  mkdir -p "$d"
  bud "WASM state gate: $1 random ids, native" "$BIN/ir_wasm_gate" ref "$2" "$d" --len $3 --gen $4 > /dev/null 2> "$d/ref.log"
  bud "WASM state gate: $1 random ids, WebAssembly" \
    bash -c "'$NODE' '$BIN/ir_wasm_gate_1gb.js' check '$2' '$d' > '$H/random_$1.json' 2> '$d/check.log'"
  local rc=$?
  echo "random $1 rc=$rc: $(tr -d '\n ' < "$H/random_$1.json")"
  [ $rc -eq 0 ] && rm -rf "$d"
}
rnd state4 $A/state_gate/state4/state4.gguf 24 32
rnd ar3 $A/attnres_gate/ar3/ar3.gguf 8 8
rm -rf "$WORK/inputs"
echo "all done"
