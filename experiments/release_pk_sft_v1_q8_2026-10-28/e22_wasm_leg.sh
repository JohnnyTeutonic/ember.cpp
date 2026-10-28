#!/bin/bash
# The WebAssembly leg for the chosen file (WSL, ext4 ~/e22rel/wasmleg/TAG), after e22_variants.sh:
#   1. e22_page_gate greedy natively and in node (the page's ember_wasm.cpp both sides, heap HEAP):
#      replies, ids, and the logits of every sampled position (TINYLLAMA_LOGITS_DUMP);
#   2. the Space's own ember.js (~/e22rel/page, deploy/wasm/build.sh) through tools/wasm_parity.mjs
#      against the native server's replies for this file;
#   3. prompts.json and greedy.json for tools/webgpu_parity/chat_test.mjs (the WebGPU leg, Windows).
# Each step skips when its output exists. Under scripts/cpu_budget.py, 1 core.
#   HEAP=2gb PAGE=~/e22rel/page2gb bash e22_wasm_leg.sh MODEL.gguf TAG SERVER_ROWS.jsonl
# (pk_sft_v1 F16 needs the 2gb heap: ember_load throws bad_alloc at 1gb)
set -uo pipefail
M=${1:?model}; TAG=${2:?tag}; ROWS=${3:?native server rows}
W=$HOME/e22rel
O=$W/wasmleg/$TAG
P=/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl
HEAP=${HEAP:-1gb}
export OMP_NUM_THREADS=${CPU_BUDGET_CORES:-1}
mkdir -p $O/native $O/wasm
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
[ "$(wc -l < $O/native/greedy.jsonl 2>/dev/null)" = 40 ] || {
  t0=$(date +%s); $W/gate/e22_page_gate greedy $M $P 40 $O/native 2> $O/native/log.txt; echo "native page gate rc=$? $(( $(date +%s) - t0 )) s"; }
[ "$(wc -l < $O/wasm/greedy.jsonl 2>/dev/null)" = 40 ] || {
  t0=$(date +%s); ( cd $O/wasm && $NODE $W/gate/e22_page_gate_$HEAP.js greedy $M $P 40 $O/wasm 2> $O/wasm/log.txt ); echo "wasm page gate rc=$? $(( $(date +%s) - t0 )) s"; }
if [ ! -f $O/page_parity.json ]; then
  python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; open(sys.argv[2],'w').writelines(json.dumps({'i':r['i'],'prompt':r['prompt'],'ember':r['reply'],'ember_tokens':r['tokens'],'ember_stop':r['stop_reason']})+chr(10) for r in rows)" $ROWS $O/ref.jsonl
  t0=$(date +%s)
  REF_FIELD=ember REF_TOKENS=ember_tokens REF_STOP=ember_stop $NODE $W/src/tools/wasm_parity.mjs ${PAGE:-$W/page} $M $O/ref.jsonl ember.js $O/page_parity.json > $O/page_parity.log 2>&1
  echo "page ember.js parity rc=$? $(( $(date +%s) - t0 )) s"; tail -1 $O/page_parity.log
fi
python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; json.dump([{'user_input': r['prompt']} for r in rows], open(sys.argv[2],'w')); json.dump({'prompts': [{'i': r['i'], 'reply': r['reply'], 'ids': r['ids']} for r in rows]}, open(sys.argv[3],'w'))" $ROWS $O/prompts.json $O/greedy.json
ls -la $O
