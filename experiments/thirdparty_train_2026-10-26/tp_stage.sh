#!/bin/bash
# Stage the VM inputs on ext4 (WSL): tp_ember.tgz, COMMITS, tp_vm.sh, small.tgz (the five model directories,
# per-model input JSON for tp_session_logits / state_device_gate, tp_session_logits.cpp, SHA256SUMS).
#   bash tp_stage.sh STAGE_DIR COMMIT
set -e
ST=${1:?stage}; COMMIT=${2:?commit}
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/thirdparty_train_2026-10-26
D=/mnt/c/ml_artifacts/thirdparty_train_gate
rm -rf "$ST"; mkdir -p "$ST/small/models"
cp $S/tp_ember.tgz $S/tp_vm.sh "$ST/"
sed -i 's/\r$//' "$ST/tp_vm.sh"
echo "$COMMIT" > "$ST/COMMITS"
cp $S/tp_session_logits.cpp "$ST/small/"
for n in tp_gqa tp_qwen tp_mqa_partial tp_neox_partial tp_mamba1; do
  mkdir -p "$ST/small/models/$n"
  cp $D/$n/*.gguf "$ST/small/models/$n/"
  rm -f "$ST/small/models/$n/v.gguf"
  python3 $S/tp_inputs.py $D/$n/score_in.jsonl "$ST/small/in_$n.json"
done
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
ls -la "$ST"
