#!/bin/bash
# Stage the VM inputs of the deterministic-mode gate on ext4 (run in WSL): the two ember trees (git archives made
# with Windows git), COMMITS, vm.sh, small.tgz (the collector and comparison scripts of ../attnres_x5_2026-10-25,
# bench.py, sass_compare.py, 40 tinychat prompts, Carry-5 items, the interpreter device-gate models, the long-prompt
# ids of the third-party device gate) and the three large models in parts of 36 MB (pick_e1_L16's model.gguf,
# chat_sft_bpe.gguf, ar_x5.gguf).
#   bash stage.sh <dir holding ember_new.tgz, ember_head.tgz and COMMITS> <stage dir>
set -e
IN=${1:?archives}; ST=${2:?stage}
H=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/determinism_2026-10-26
X=$H/../attnres_x5_2026-10-25
A=/mnt/c/ml_artifacts
rm -rf "$ST"; mkdir -p "$ST/small/models"
cp "$IN/ember_new.tgz" "$IN/ember_head.tgz" "$IN/COMMITS" "$ST/"
cp $H/vm.sh "$ST/" && sed -i 's/\r$//' "$ST/vm.sh"
S=$ST/small
cp $X/x5_plain_collect.py $X/x5_plain_compare.py $H/bench.py $H/sass_compare.py $S/
sed -i 's/\r$//' $S/*.py
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > $S/prompts.jsonl
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/eval/carry5_v1.jsonl $S/
M=$S/models
mkdir -p $M/mor $M/state4 $M/ar3 $M/synth $M/gpt2_tiny $M/recursion
for m in mor3 mor2p0; do cp $A/mor_gate/$m/$m.gguf $A/mor_gate/$m/${m}_q8_0.gguf $M/mor/; done
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do cp $A/recursion_gate/$m/r.gguf $M/recursion/$m.gguf; done
cp $A/state_gate/state4/state4.gguf $A/state_gate/state4/state4_q8_0.gguf $M/state4/
cp $A/attnres_gate/ar3/ar3.gguf $M/ar3/
cp $A/attnres_gate_x5/ar_x5/ar_x5_q8_0.gguf $M/ar3/
for f in mistral_swa gptneox gpt2 mamba gemma3; do cp ~/ember_wt/bcheck_$f/$f.gguf $M/synth/; done
cp ~/ember_wt/bcheck_gemma3/gemma3.per_layer_freq_scale_0.25.ir.json ~/ember_wt/bcheck_gemma3/gemma3.weight_offset_0.25.ir.json $M/synth/
cp $A/gpt2_tiny/gpt2_tiny.gguf $M/gpt2_tiny/
cp $A/devgate/stage-ember-gate-1022d/ids.tgz $S/
(cd $S && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$S"
cd "$ST"
split -b 36M -d $A/tiny_chat/e1/pick_e1_L16/model.gguf pick.part
split -b 36M -d $A/chat_base/sft/chat_sft_bpe.gguf sft.part
split -b 36M -d $A/attnres_gate_x5/ar_x5/ar_x5.gguf arx5.part
{
  echo "$(sha256sum < $A/tiny_chat/e1/pick_e1_L16/model.gguf | cut -d' ' -f1)  pick_L16.gguf"
  echo "$(sha256sum < $A/chat_base/sft/chat_sft_bpe.gguf | cut -d' ' -f1)  chat_sft_bpe.gguf"
  echo "$(sha256sum < $A/attnres_gate_x5/ar_x5/ar_x5.gguf | cut -d' ' -f1)  ar_x5.gguf"
} > BIG_SHA256SUMS
grep -q 96f6afdcbbcf1a4cf53ad6c0e451e75301dc4187f726f3c2218600fbe72a8cc9 BIG_SHA256SUMS
grep -q 226fe90b BIG_SHA256SUMS
ls -la "$ST"; cat BIG_SHA256SUMS
