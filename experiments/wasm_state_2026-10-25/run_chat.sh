#!/bin/bash
# state4 through the page's chat path (run in WSL, one core per job under the CPU budget): the 40 /chat
# requests of state4_gate.py's /chat leg (chat_state4.py) to ember's native CPU server (collect_native.py),
# then tools/wasm_chat_check.mjs on the page build (deploy/wasm/build.sh's ember.js, default heap and cap):
# ember_prepare's prompt ids and ember_chat's replies against the server's. Receipts beside this file:
# chat_state4_native.json (the server against score_in) and chat_state4.json (the page against the server).
#   bash run_chat.sh SRC BUILD PAGE_DIR WORK
set -u
SRC=${1:?src}; BUILD=${2:?build}; PAGE=${3:?page build}; WORK=${4:?work}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/wasm_state_2026-10-25
G=/mnt/c/ml_artifacts/state_gate/state4/state4.gguf
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
mkdir -p "$WORK"
bud() { local job=$1; shift; python3 $R/scripts/cpu_budget.py run --session ember --cores 1 --job "WASM state gate: $job" \
  --resume "rerun $H/run_chat.sh $SRC $BUILD $PAGE $WORK" -- "$@"; }
python3 $H/chat_state4.py requests "$WORK/state4_requests.jsonl"
bud "state4 /chat, native server" python3 $R/ember.cpp/experiments/wasm_ctx_2026-10-25/collect_native.py \
  $BUILD/tinyllama_server $G "$WORK/state4_requests.jsonl" "$WORK/state4_native.jsonl" > "$WORK/native.out" 2>&1
echo "native rc=$?: $(tail -1 "$WORK/native.out")"
python3 $H/chat_state4.py native "$WORK/state4_native.jsonl" > "$H/chat_state4_native.json"
cat "$H/chat_state4_native.json"
bud "state4 /chat, page build" "$NODE" $SRC/tools/wasm_chat_check.mjs "$PAGE" $G "$WORK/state4_native.jsonl" "$H/chat_state4.json" \
  > "$WORK/wasm.out" 2>&1
echo "wasm rc=$?: $(tail -1 "$WORK/wasm.out" | cut -c1-500)"
