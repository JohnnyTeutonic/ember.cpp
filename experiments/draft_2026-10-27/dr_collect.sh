#!/bin/bash
# CPU collection for the drafting simulator (run in WSL; every job under the CPU budget, session ember): plain
# greedy decode of the evaluation and development sets on both models, each request once, prompt ids and ids kept.
#   bash dr_collect.sh <CPU build> <work dir>
# Finished runs are skipped (resume = rerun).
B=${1:?build}; W=${2:?work}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
D=$R/ember.cpp/experiments/draft_2026-10-27
A=/mnt/c/ml_artifacts
mkdir -p $W/collect
python3 $D/dr_prompts.py $W/prompts
port=8740
for m in pick:$A/tiny_chat/e1/pick_e1_L16/model.gguf pk:$A/next_base/base/pk_base_v1.gguf; do
  n=${m%%:*}; g=${m#*:}
  for s in chat40 copy20 carry40 chatdev carrydev; do
    port=$((port + 1))
    o=$W/collect/${n}_${s}.jsonl
    if [ -s $o ]; then echo "collect ${n}_${s}: already done"; continue; fi
    OMP_NUM_THREADS=4 python3 $R/scripts/cpu_budget.py run --session ember --job "draft: CPU collect $n $s" --cores 4 \
      --resume "rerun ember.cpp/experiments/draft_2026-10-27/dr_collect.sh $B $W (finished sets are skipped)" -- \
      python3 $D/dr_e2e.py run $B/tinyllama_server $g 0 $port $W/prompts/$s.jsonl $o --once EMBER_NOTHING=1 \
      > $W/collect/${n}_${s}.txt 2>&1
    echo "collect ${n}_${s} rc=$? $(tail -1 $W/collect/${n}_${s}.txt | cut -c1-200)"
    rm -rf $o.run
  done
done
echo "collect done"
