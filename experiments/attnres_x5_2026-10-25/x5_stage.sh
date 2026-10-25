#!/bin/bash
# Stage the VM inputs of the X5 gate on ext4 (WSL): the two ember archives, ce41ffb's diff, COMMITS, x5_vm.sh,
# small.tgz (the scripts and tool source, ar_x5_q8_0.gguf, ar_x5_in.json, 40 tinychat prompts, Carry-5 items,
# SHA256SUMS) and, in 36 MB parts, ar_x5.gguf and the L16 control's SFT GGUF (pick_e1_L16/model.gguf).
#   bash x5_stage.sh STAGE_DIR
set -e
ST=${1:?stage}
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/attnres_x5_2026-10-25
A=/mnt/c/ml_artifacts
X=$A/attnres_gate_x5/ar_x5
PICK=$A/tiny_chat/e1/pick_e1_L16/model.gguf
rm -rf "$ST"; mkdir -p "$ST/small"
cp $S/ember_new.tgz $S/ember_old.tgz $S/ce41ffb.patch $S/COMMITS $S/x5_vm.sh "$ST/"
sed -i 's/\r$//' "$ST/x5_vm.sh"
cp $S/x5_dump9.py $S/x5_plain_collect.py $S/x5_plain_compare.py $S/x5_device_logits.cpp $X/ar_x5_q8_0.gguf "$ST/small/"
python3 $S/x5_inputs.py "$ST/small"
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > "$ST/small/prompts.jsonl"
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/eval/carry5_v1.jsonl "$ST/small/"
(cd "$ST/small" && sha256sum * | grep -v SHA256SUMS > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
cd "$ST"
split -b 36M -d $X/ar_x5.gguf arx5.part
split -b 36M -d $PICK pick.part
(cd $X && sha256sum ar_x5.gguf; cd $(dirname $PICK) && sha256sum model.gguf) > BIG_SHA256SUMS
cat BIG_SHA256SUMS
grep -q 226fe90badf633c372343d77d72deed7cceaecdb539884440b95274a90686c8e BIG_SHA256SUMS
grep -q 96f6afdcbbcf1a4cf53ad6c0e451e75301dc4187f726f3c2218600fbe72a8cc9 BIG_SHA256SUMS
ls -la "$ST"
