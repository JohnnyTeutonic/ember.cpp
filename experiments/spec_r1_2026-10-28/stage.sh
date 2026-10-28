#!/bin/bash
# Stage the VM inputs of spec_r1_2026-10-28 on ext4 (run in WSL):
#   bash stage.sh <ember_head.tgz> <head commit> <stage dir> [vm script, default vm.sh]
# ember_new.tgz is ~/r1/src (build.sh's tree: the working tree plus tools/r1_bench.cpp and its CMake target);
# small.tgz the scripts, the request sets (dr_prompts.py), the pools (r1_pool.py), the two draft tables and
# config.sh, plus any extra files listed in $EXTRA (space-separated paths); pick_e1_L16's and pk_base_v1's GGUFs go
# in parts of 36 MB.
set -e
HT=${1:?ember_head.tgz}; HEADC=${2:?head commit}; ST=${3:?stage}; VMS=${4:-vm.sh}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp
X=$E/experiments/spec_r1_2026-10-28
DR=$E/experiments/draft_2026-10-27
A=/mnt/c/ml_artifacts
TD=$A/ember_draft/tables
[ -d ~/r1/src ] || { echo "run build.sh first (it writes ~/r1/src)"; exit 1; }
rm -rf "$ST"; mkdir -p "$ST/small"
cp "$HT" "$ST/ember_head.tgz"
tar czf "$ST/ember_new.tgz" -C ~/r1 --transform "s,^src,ember," src
echo "$HEADC+r1 $HEADC" > "$ST/COMMITS"
cp $X/$VMS "$ST/vm.sh" && sed -i 's/\r$//' "$ST/vm.sh"
S=$ST/small
cp $DR/dr_e2e.py $E/experiments/determinism_2026-10-26/sass_compare.py $S/
python3 $DR/dr_prompts.py $S
python3 $X/r1_pool.py $S 4000
rm -rf $S/_eval
cp $TD/h_c8_p10.bin $TD/s_c8_p10.bin $S/
printf 'TABLE_pick=%s\nTABLE_pk=%s\n' h_c8_p10.bin s_c8_p10.bin > $S/config.sh
for f in ${EXTRA:-}; do cp "$f" $S/; done
if [ "$VMS" = vm2.sh ]; then  # session 2: the larger pools and the table tools in their folders
  mkdir -p $S/p20 $S/r1/spec_r1_2026-10-28 $S/r1/draft_2026-10-27
  python3 $X/r1_pool.py $S/p20 20000 8000
  for m in pick pk; do mv $S/p20/${m}_pool.jsonl $S/${m}_pool20.jsonl; done
  rm -rf $S/p20
  cp $X/r1_sim.py $X/r1_rows.py $S/r1/spec_r1_2026-10-28/
  cp $DR/dr_sim.py $DR/dr_table.py $S/r1/draft_2026-10-27/
  sed -i 's/\r$//' $S/r1/*/*.py
fi
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
