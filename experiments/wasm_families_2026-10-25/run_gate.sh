#!/bin/bash
# The WebAssembly gate per third-party family: tools/ir_wasm_gate built by build_gate.sh (BIN, default
# ~/wasmfam/bin). Per family, the reference from the native CPU build, then the WebAssembly build in
# node, on the first PROMPTS of ir_device_gate's 40 prompts; two queues run side by side, one core each
# (the two 0.5B models share a queue). Work directories (the reference logits, deleted once the
# family passes) under WORK (default ~/wasmfam/runs); the rows and the summary are copied here.
# Every job runs under the CPU budget and resumes: rerun this script after an interruption, or with a
# larger PROMPTS to extend a family (the prompts already done are kept).
#   bash run_gate.sh [FAMILY ...]        (default: all; WSL)
set -u
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/wasm_families_2026-10-25
BIN=${BIN:-$HOME/wasmfam/bin}
WORK=${WORK:-$HOME/wasmfam/runs}
T=/mnt/c/ml_artifacts/thirdparty
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
BUD="python3 $R/scripts/cpu_budget.py run --session ember --cores 1"
# family | GGUF | WebAssembly heap | prompts | queue
FAMILIES="
qwen2.5-0.5b-instruct|$T/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf|4gb|${PROMPTS_QWEN:-10}|1
qwen3-0.6b|$T/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf|4gb|${PROMPTS_QWEN:-10}|1
gemma-3-270m-it|$T/gemma-3-270m-it/gemma-3-270m-it-f32.gguf|4gb|40|2
pythia-160m|$T/pythia-160m/pythia-160m-f32.gguf|4gb|40|2
mamba-130m|$T/mamba-130m/mamba-130m-f32.gguf|4gb|40|2
synthetic-mistral-swa|$HOME/ember_wt/bcheck_mistral_swa/mistral_swa.gguf|1gb|40|2
synthetic-gpt2|$HOME/ember_wt/bcheck_gpt2/gpt2.gguf|1gb|40|2
"
sel=" $* "
family() {
  local fam=$1 gguf=$2 mem=$3 k=$4 d=$WORK/$1
  mkdir -p "$d"
  $BUD --job "WASM family gate: $fam reference (native, $k prompts)" --resume "rerun $H/run_gate.sh $fam" -- \
    "$BIN/ir_wasm_gate" ref "$gguf" "$d" --first "$k" > "$d/ref.out" 2>> "$d/ref.log"
  echo "$fam reference rc=$? ($(wc -l < "$d/ref.jsonl") prompts)"
  $BUD --job "WASM family gate: $fam check (WebAssembly $mem, $k prompts)" --resume "rerun $H/run_gate.sh $fam" -- \
    bash -c "cd '$d' && '$NODE' '$BIN/ir_wasm_gate_$mem.js' check '$gguf' '$d' --first $k > '$d/summary.json' 2>> '$d/check.log'"
  local rc=$?
  echo "$fam check rc=$rc: $(grep -E '"(prompts|top1_agree|greedy_identical|worst_prefill_rel|worst_decode_rel|pass)"' "$d/summary.json" | tr -d '\n ')"
  mkdir -p "$H/$fam"
  cp "$d/ids.json" "$d/ref.jsonl" "$d/check.jsonl" "$d/summary.json" "$H/$fam/"
  [ $rc -eq 0 ] && rm -f "$d"/ref_*.bin
  return 0
}
queue() {
  echo "$FAMILIES" | grep -v '^$' | while IFS='|' read -r fam gguf mem k q; do
    [ "$q" = "$1" ] || continue
    [ "$sel" = "  " ] || [[ "$sel" == *" $fam "* ]] || continue
    family "$fam" "$gguf" "$mem" "$k"
  done
}
queue 1 &
queue 2 &
wait
echo "all done"
