#!/bin/bash
# Stage the VM inputs of decode_occupancy_2026-10-27 on ext4 (run in WSL):
#   bash stage.sh <dir holding ember_head.tgz (git archive of ember main)> <head commit> <stage dir>
# ember_new.tgz is ~/occ/src (build.sh tree: the working tree plus tools/occ_step.cpp and its CMake target) with
# tools/occ_bench.cu; small.tgz the scripts, request sets and the interpreter's local gate models; pick_e1_L16's and
# pick_e1_60m_L32's GGUFs go in parts of 36 MB.
set -e
IN=${1:?archives}; HEADC=${2:?head commit}; ST=${3:?stage}
H=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments
O=$H/decode_occupancy_2026-10-27
A=/mnt/c/ml_artifacts
SRC=~/occ/src
bash $O/build.sh tree
cp $O/occ_bench.cu $SRC/tools/occ_bench.cu
rm -rf "$ST"; mkdir -p "$ST/small/models"
cp "$IN/ember_head.tgz" "$ST/"
tar czf "$ST/ember_new.tgz" -C "$(dirname $SRC)" --transform "s,^$(basename $SRC),ember," "$(basename $SRC)"
echo "$HEADC+occupancy $HEADC" > "$ST/COMMITS"
cp $O/vm.sh "$ST/" && sed -i 's/\r$//' "$ST/vm.sh"
S=$ST/small
cp $H/attnres_x5_2026-10-25/x5_plain_collect.py $H/attnres_x5_2026-10-25/x5_plain_compare.py \
   $H/determinism_2026-10-26/sass_compare.py $H/verify_det_2026-10-27/vd_collect.py $H/verify_det_2026-10-27/vd_compare.py \
   $H/verify_2026-10-27/vf_e2e.py $O/dump_compare.py $S/
python3 $O/occ_prompts.py $S
cat $S/all60.jsonl $S/c12_40.jsonl $S/long40.jsonl > $S/all140.jsonl
sed -i 's/\r$//' $S/*.py
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > $S/prompts.jsonl
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/eval/carry5_v1.jsonl $S/
M=$S/models
mkdir -p $M/mor $M/state4 $M/ar3 $M/synth $M/gpt2_tiny $M/recursion
for m in mor3 mor2p0; do cp $A/mor_gate/$m/$m.gguf $A/mor_gate/$m/${m}_q8_0.gguf $M/mor/; done
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do cp $A/recursion_gate/$m/r.gguf $M/recursion/$m.gguf; done
cp $A/state_gate/state4/state4.gguf $A/state_gate/state4/state4_q8_0.gguf $M/state4/
cp $A/attnres_gate/ar3/ar3.gguf $A/attnres_gate_x5/ar_x5/ar_x5_q8_0.gguf $M/ar3/
for f in mistral_swa gptneox gpt2 mamba gemma3; do
  if [ -s ~/ember_wt/bcheck_$f/$f.gguf ]; then cp ~/ember_wt/bcheck_$f/$f.gguf $M/synth/; else echo "no synthetic $f"; fi
done
cp $A/gpt2_tiny/gpt2_tiny.gguf $M/gpt2_tiny/
(cd $S && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$S"
cd "$ST"
split -b 36M -d $A/tiny_chat/e1/pick_e1_L16/model.gguf pick.part
split -b 36M -d $A/tiny_chat/e1_60m/pick_e1_60m_L32/model.gguf m60.part
{
  echo "$(sha256sum < $A/tiny_chat/e1/pick_e1_L16/model.gguf | cut -d' ' -f1)  pick_L16.gguf"
  echo "$(sha256sum < $A/tiny_chat/e1_60m/pick_e1_60m_L32/model.gguf | cut -d' ' -f1)  pick_60m_L32.gguf"
} > BIG_SHA256SUMS
ls -la "$ST"; cat BIG_SHA256SUMS
