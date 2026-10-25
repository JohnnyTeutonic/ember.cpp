#!/bin/bash
# Stage the /score CUDA gate's VM inputs on ext4 (run in WSL): this work's tree (~/ember_score/src), small.tgz
# (sc_client.py and the request sets) and the two paperkiln GGUFs in parts of 36 MB.
#   bash sc_stage.sh <stage dir>
set -e
ST=${1:?stage}
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
A=/mnt/c/ml_artifacts
rm -rf "$ST"; mkdir -p "$ST/small/req"
tar czf "$ST/ember_new.tgz" -C ~/ember_score/src .
cp $S/sc_vm.sh "$ST/vm.sh"; sed -i 's/\r$//' "$ST/vm.sh"
cp $S/sc_client.py "$ST/small/"; cp $S/req/*.jsonl "$ST/small/req/"
tar czf "$ST/small.tgz" -C "$ST" small && rm -rf "$ST/small"
cd "$ST"
split -b 36M -d $A/tiny_chat/e1/pick_e1_L16/model.gguf pick.part
split -b 36M -d $A/next_base/base/pk_base_v1.gguf pkbase.part
{
  echo "$(sha256sum < $A/tiny_chat/e1/pick_e1_L16/model.gguf | cut -d' ' -f1)  pick_L16.gguf"
  echo "$(sha256sum < $A/next_base/base/pk_base_v1.gguf | cut -d' ' -f1)  pk_base_v1.gguf"
} > BIG_SHA256SUMS
ls -la "$ST"; cat BIG_SHA256SUMS
