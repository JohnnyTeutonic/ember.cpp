#!/bin/bash
# Stage the second (run-to-run control) session on ext4: archives, COMMITS, x5_vm2.sh, small.tgz (the collector,
# comparison and dump scripts, 40 tinychat prompts, Carry-5 items) and the L16 control's SFT GGUF in 36 MB parts.
#   bash x5_stage2.sh STAGE_DIR      (a new directory)
set -e
ST=${1:?stage}
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/attnres_x5_2026-10-25
A=/mnt/c/ml_artifacts
PICK=$A/tiny_chat/e1/pick_e1_L16/model.gguf
mkdir "$ST" && mkdir "$ST/small"
cp $S/ember_new.tgz $S/ember_old.tgz $S/COMMITS $S/x5_vm2.sh "$ST/"
sed -i 's/\r$//' "$ST/x5_vm2.sh"
cp $S/x5_dump9.py $S/x5_plain_collect.py $S/x5_plain_compare.py "$ST/small/"
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > "$ST/small/prompts.jsonl"
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/eval/carry5_v1.jsonl "$ST/small/"
(cd "$ST/small" && sha256sum * | grep -v SHA256SUMS > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
cd "$ST"
split -b 36M -d $PICK pick.part
(cd $(dirname $PICK) && sha256sum model.gguf) > BIG_SHA256SUMS
grep -q 96f6afdcbbcf1a4cf53ad6c0e451e75301dc4187f726f3c2218600fbe72a8cc9 BIG_SHA256SUMS
ls -la "$ST"
