#!/bin/bash
# Stage the VM inputs of verify_2026-10-27 on ext4 (run in WSL):
#   bash stage.sh <dir holding ember_head.tgz (git archive 70ed063)> <new source tree> <stage dir>
# ember_new.tgz is the new source tree (this work's files, tools/vf_bench.cpp and its CMake target, as
# vf_build.sh lays it out); small.tgz the scripts, request sets and the interpreter's local gate models;
# pick_e1_L16's and pk_base_v1's GGUFs go in parts of 36 MB.
set -e
IN=${1:?archives}; SRC=${2:?new tree}; ST=${3:?stage}
H=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments
X=$H/attnres_x5_2026-10-25
V=$H/verify_2026-10-27
A=/mnt/c/ml_artifacts
rm -rf "$ST"; mkdir -p "$ST/small/models"
cp "$IN/ember_head.tgz" "$ST/"
tar czf "$ST/ember_new.tgz" -C "$(dirname $SRC)" --transform "s,^$(basename $SRC),ember," "$(basename $SRC)"
echo "70ed063+verify 70ed063" > "$ST/COMMITS"
cp $V/vm.sh "$ST/" && sed -i 's/\r$//' "$ST/vm.sh"
S=$ST/small
cp $X/x5_plain_collect.py $X/x5_plain_compare.py $H/determinism_2026-10-26/sass_compare.py $V/vf_e2e.py $V/vf_summary.py $S/
python3 $V/vf_prompts.py $S
sed -i 's/\r$//' $S/*.py
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > $S/prompts.jsonl
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/eval/carry5_v1.jsonl $S/
M=$S/models
mkdir -p $M/mor $M/state4 $M/ar3 $M/synth $M/gpt2_tiny $M/recursion
for m in mor3 mor2p0; do cp $A/mor_gate/$m/$m.gguf $A/mor_gate/$m/${m}_q8_0.gguf $M/mor/; done
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do cp $A/recursion_gate/$m/r.gguf $M/recursion/$m.gguf; done
cp $A/state_gate/state4/state4.gguf $A/state_gate/state4/state4_q8_0.gguf $M/state4/
cp $A/attnres_gate/ar3/ar3.gguf $A/attnres_gate_x5/ar_x5/ar_x5_q8_0.gguf $M/ar3/
for f in mistral_swa gptneox gpt2 mamba gemma3; do cp ~/ember_wt/bcheck_$f/$f.gguf $M/synth/; done
cp $A/gpt2_tiny/gpt2_tiny.gguf $M/gpt2_tiny/
(cd $S && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$S"
cd "$ST"
split -b 36M -d $A/tiny_chat/e1/pick_e1_L16/model.gguf pick.part
split -b 36M -d $A/next_base/base/pk_base_v1.gguf pk.part
{
  echo "$(sha256sum < $A/tiny_chat/e1/pick_e1_L16/model.gguf | cut -d' ' -f1)  pick_L16.gguf"
  echo "$(sha256sum < $A/next_base/base/pk_base_v1.gguf | cut -d' ' -f1)  pk_base_v1.gguf"
} > BIG_SHA256SUMS
grep -q 510d288cf10ead4d3902247e3d898b6fa39e1d31d7465d3a8b6820761d6c9178 BIG_SHA256SUMS
ls -la "$ST"; cat BIG_SHA256SUMS
