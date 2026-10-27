#!/bin/bash
# E2.2 release gate runs (WSL, ext4 ~/e22rel/run), after e22_build.sh. Each step skips when its
# output exists, so a rerun resumes. Run under scripts/cpu_budget.py (2 cores).
#   bash e22_gate.sh [native|wasm|all]
set -uo pipefail
SP=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/e22rel
W=$HOME/e22rel
R=$W/run
M=/mnt/c/ml_artifacts/general_chat/sft
F32=$M/pk_sft_v1.gguf
Q8=$M/pk_sft_v1_q8_0_all.gguf
P=/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl
SRV=$W/build_cpu/tinyllama_server
HEAP=${HEAP:-1gb}
PORT=8121
WHAT=${1:-all}
mkdir -p $R
export OMP_NUM_THREADS=${CPU_BUDGET_CORES:-2}

serve() {   # serve MODEL TAG: greedy replies with ids from the native server
  local model=$1 tag=$2
  [ -f $R/${tag}_server.jsonl ] && { echo "$tag server: done"; return 0; }
  for p in $(pgrep -f "[t]inyllama_server .* $PORT "); do kill $p; done
  $SRV "$model" $PORT 127.0.0.1 $W/src/www 0 > $R/${tag}_server.log 2>&1 &
  local s=$!
  for i in $(seq 1 300); do curl -s 127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
  local t0=$(date +%s)
  python3 $SP/e22_chat_client.py $PORT $P 40 $R/${tag}_server.jsonl "$(basename $model)"
  echo "$tag server: $(( $(date +%s) - t0 )) s"
  kill $s; wait $s 2>/dev/null
}

if [ "$WHAT" = native ] || [ "$WHAT" = all ]; then
  serve $F32 f32
  serve $Q8 q8
  # teacher forcing on F32's greedy trajectories, both files
  for tag in f32 q8; do
    g=$F32; [ $tag = q8 ] && g=$Q8
    [ -f $R/tf_$tag.f32.json ] || { t0=$(date +%s); $W/gate/e22_page_gate tf $g $R/f32_server.jsonl $R/tf_$tag.f32 2> $R/tf_$tag.log; echo "tf $tag rc=$? $(( $(date +%s) - t0 )) s"; }
  done
  # the page's entry points natively, Q8 (logits of every sampled position)
  mkdir -p $R/native_q8
  [ -f $R/native_q8/greedy.jsonl ] && [ $(wc -l < $R/native_q8/greedy.jsonl) -eq 40 ] || {
    t0=$(date +%s); $W/gate/e22_page_gate greedy $Q8 $P 40 $R/native_q8 2> $R/native_q8/log.txt; echo "native page gate q8 rc=$? $(( $(date +%s) - t0 )) s"; }
fi

if [ "$WHAT" = wasm ] || [ "$WHAT" = all ]; then
  source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
  NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
  mkdir -p $R/wasm_q8
  [ -f $R/wasm_q8/greedy.jsonl ] && [ $(wc -l < $R/wasm_q8/greedy.jsonl) -eq 40 ] || {
    t0=$(date +%s); ( cd $R/wasm_q8 && $NODE $W/gate/e22_page_gate_$HEAP.js greedy $Q8 $P 40 $R/wasm_q8 2> $R/wasm_q8/log.txt ); echo "wasm page gate q8 rc=$? $(( $(date +%s) - t0 )) s"; }
  # the Space's own ember.js (deploy/wasm/build.sh) through tools/wasm_parity.mjs, against the native server's Q8 replies
  if [ ! -f $R/page_parity_q8.json ]; then
    python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; open(sys.argv[2],'w').writelines(json.dumps({'i':r['i'],'prompt':r['prompt'],'ember':r['reply'],'ember_tokens':r['tokens'],'ember_stop':r['stop_reason']})+chr(10) for r in rows)" $R/q8_server.jsonl $R/q8_ref.jsonl
    t0=$(date +%s)
    REF_FIELD=ember REF_TOKENS=ember_tokens REF_STOP=ember_stop $NODE $W/src/tools/wasm_parity.mjs $W/page $Q8 $R/q8_ref.jsonl ember.js $R/page_parity_q8.json > $R/page_parity_q8.log 2>&1
    echo "page ember.js parity rc=$? $(( $(date +%s) - t0 )) s"; tail -2 $R/page_parity_q8.log
  fi
fi
ls -la $R
