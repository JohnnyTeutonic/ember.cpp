#!/bin/bash
# CPU legs of decode_occupancy_2026-10-27 (run in WSL; every job under the CPU budget, session ember):
#   (1) the CPU build is this change's no-op: HEAD's tree and this work's tree built in turn in one directory
#       (HAS_CUDA=OFF, Release), and tinyllama_server, ir_logits and ir_adapter compared byte for byte;
#   (2) the CPU references of the end-to-end requests: vf_e2e.py over all140.jsonl (all60 + c12_40 + long40) on
#       pick_e1_L16 and pick_e1_60m_L32 at gpu_layers 0, for the CUDA runs' ids to be compared with.
#   bash cpu_legs.sh <ember_head.tgz> <work dir>
set -u
HT=${1:?head archive}; W=${2:?work}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
X=$R/ember.cpp/experiments/decode_occupancy_2026-10-27
A=/mnt/c/ml_artifacts
mkdir -p $W/src $W/build $W/e2e
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "occupancy CPU: $1" --cores ${CORES:-4} \
  --resume "rerun ember.cpp/experiments/decode_occupancy_2026-10-27/cpu_legs.sh $HT $W (finished parts are skipped)" -- "${@:2}"; }
T="tinyllama_server ir_logits ir_adapter"
if [ ! -s $W/cpu_identity.txt ]; then
  rm -rf $W/src $W/build && mkdir -p $W/src $W/build
  tar xzf $HT -C $W/src --strip-components=1
  (cd $W/build && cmake $W/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF > cmake.log 2>&1)
  budget "build HEAD" make -C $W/build -j${CORES:-4} $T > $W/build_head.log 2>&1 || { tail -20 $W/build_head.log; exit 1; }
  mkdir -p $W/head_bins && for b in $T; do cp $W/build/$b $W/head_bins/; done
  bash $X/build.sh tree
  # content-checked, times not kept: a file whose content changed is newer than its objects, the rest untouched
  rsync -rlpc --delete --exclude 'build*' ~/occ/src/ $W/src/
  budget "build new" make -C $W/build -j${CORES:-4} $T > $W/build_new.log 2>&1 || { tail -20 $W/build_new.log; exit 1; }
  {
    echo "HEAD tree, then this work's tree, built in $W/build (HAS_CUDA=OFF, Release)"
    for b in $T; do
      h=$(sha256sum < $W/head_bins/$b | cut -c1-64); n=$(sha256sum < $W/build/$b | cut -c1-64)
      echo "$b head $h new $n $([ $h = $n ] && echo identical || echo DIFFERENT)"
    done
    echo "objects rebuilt for the new tree: $(grep -c 'Building' $W/build_new.log)"
    grep 'Building' $W/build_new.log | sed 's/.*Building/Building/'
  } > $W/cpu_identity.txt
fi
cat $W/cpu_identity.txt
mkdir -p $W/prompts && python3 $X/occ_prompts.py $W/prompts > /dev/null
cat $W/prompts/all60.jsonl $W/prompts/c12_40.jsonl $W/prompts/long40.jsonl > $W/prompts/all140.jsonl
port=8680
for m in pick:$A/tiny_chat/e1/pick_e1_L16/model.gguf m60:$A/tiny_chat/e1_60m/pick_e1_60m_L32/model.gguf; do
  n=${m%%:*}; g=${m#*:}
  port=$((port + 1))
  if [ -s $W/e2e/${n}_cpu.jsonl ]; then echo "e2e ${n}_cpu: already done"; continue; fi
  OMP_NUM_THREADS=${CORES:-4} budget "e2e $n" python3 $R/ember.cpp/experiments/verify_2026-10-27/vf_e2e.py run \
    $W/build/tinyllama_server $g 0 $port $W/prompts/all140.jsonl $W/e2e/${n}_cpu.jsonl EMBER_NOTHING=1 > $W/e2e/${n}_cpu.txt 2>&1
  echo "e2e ${n}_cpu rc=$? $(tail -1 $W/e2e/${n}_cpu.txt | cut -c1-240)"
  rm -rf $W/e2e/${n}_cpu.jsonl.run
done
echo "cpu legs done"
