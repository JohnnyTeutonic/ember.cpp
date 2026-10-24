#!/bin/bash
# Stage the VM inputs on ext4: scripts, prompts, the fix patch, models split in 36 MB parts.
set -e
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/rmsnorm_2026-10-24
W=~/ember_wt/shape_parity
ST=~/sp_stage
mkdir -p $ST && cd $ST
rm -f sft.part* l16.part*
cp $S/vm.sh $S/dumpprec.py $S/sp_compare.py $S/fix.patch $ST/
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/tools/rmsnorm_batch_test.cu $ST/
head -40 /mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl > prompts.jsonl
split -b 36M -d $W/sft/tc_sft_L32_r1.gguf sft.part
split -b 36M -d $W/sp_L16/sp_L16.gguf l16.part
(cd $W/sft && sha256sum tc_sft_L32_r1.gguf; cd $W/sp_L16 && sha256sum sp_L16.gguf) > SHA256SUMS
ls -la $ST; cat SHA256SUMS
