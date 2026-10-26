#!/bin/bash
# Stage the VM inputs of draft_2026-10-27 on ext4 (run in WSL):
#   bash stage.sh <stage dir> <tables dir> <pick table> <pk table> <theta>
# ember_new.tgz is the ember working tree (no .git, experiments, build directories, deploy, docs); small.tgz the
# scripts, the request sets (dr_prompts.py), the two draft tables and config.sh; pick_e1_L16's and pk_base_v1's
# GGUFs go in parts of 36 MB.
set -e
ST=${1:?stage}; TD=${2:?tables dir}; TPICK=${3:?pick table}; TPK=${4:?pk table}; THETA=${5:?theta}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp
D=$E/experiments/draft_2026-10-27
A=/mnt/c/ml_artifacts
rm -rf "$ST"; mkdir -p "$ST/src" "$ST/small"
rsync -a --exclude .git --exclude experiments --exclude 'build*' --exclude deploy --exclude docs $E/ "$ST/src/ember/"
tar czf "$ST/ember_new.tgz" -C "$ST/src" ember
rm -rf "$ST/src"
echo "${BASE_COMMIT:-08ead73}+draft" > "$ST/COMMITS"
cp $D/vm.sh "$ST/" && sed -i 's/\r$//' "$ST/vm.sh"
S=$ST/small
cp $D/dr_e2e.py $S/
python3 $D/dr_prompts.py $S
cp $TD/$TPICK $TD/$TPK $S/ 2>/dev/null || cp $TD/$TPICK $S/
printf 'TABLE_pick=%s\nTABLE_pk=%s\nTHETA=%s\n' "$TPICK" "$TPK" "$THETA" > $S/config.sh
sed -i 's/\r$//' $S/*.py $S/config.sh
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
