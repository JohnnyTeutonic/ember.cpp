#!/bin/bash
# CPU legs of draft_2026-10-27 (run in WSL; every job under the CPU budget, session ember):
#   (1) speculation off, unchanged: ../cuda_state_2026-10-25/cpu_regress.sh with this work's CPU build against the
#       committed receipts (REF: `git show HEAD:<path>` copies), the adapter self-test included; regress_bits' old
#       side is the CPU build of HEAD (08ead73) the simulator's records came from;
#   (2) EMBER_DRAFT=table on the CPU: dr_e2e.py over eval100.jsonl (chat40, copy20, carry40) on both models at
#       gpu_layers 0, each request once: plain decode and EMBER_SPEC=4 and 8 with the table; ids against plain
#       decode, and plain decode against HEAD's (the simulator's records).
#   bash cpu_legs.sh <CPU build of this work> <CPU build of HEAD> <work dir> <REF dir> <tables dir> <pick table> <pk table>
B=${1:?build}; BH=${2:?head build}; W=${3:?work}; REF=${4:?ref}; TD=${5:?tables}; TPICK=${6:?}; TPK=${7:?}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp/experiments
D=$E/draft_2026-10-27
A=/mnt/c/ml_artifacts
mkdir -p $W/cpu_e2e
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "draft CPU: $1" --cores ${CORES:-4} \
  --resume "rerun ember.cpp/experiments/draft_2026-10-27/cpu_legs.sh $B $BH $W $REF $TD $TPICK $TPK (finished runs are skipped)" -- "${@:2}"; }
python3 $D/dr_prompts.py $W/prompts
if [ "${SKIP_REGRESS:-0}" != 1 ]; then
  mkdir -p ~/ember_69d7ca2/build
  ln -sfn $BH/ir_logits ~/ember_69d7ca2/build/ir_logits
  echo "regress_bits old side: $(readlink -f ~/ember_69d7ca2/build/ir_logits)"
  bash $E/cuda_state_2026-10-25/cpu_regress.sh $B $W/cpu_regress $REF recursion_gate mor_gate mor_refusals attnres_gate \
    state_tokens_gate state4_gate regress_bits regress_parity g5_thirdparty adapter
fi
port=8840
for m in pick:$A/tiny_chat/e1/pick_e1_L16/model.gguf:$TPICK pk:$A/next_base/base/pk_base_v1.gguf:$TPK; do
  n=${m%%:*}; rest=${m#*:}; g=${rest%:*}; t=${rest##*:}
  for cfg in off:EMBER_NOTHING=1 table_k4:EMBER_SPEC=4 table_k8:EMBER_SPEC=8; do
    tag=${cfg%%:*}; env=${cfg#*:}
    port=$((port + 1))
    o=$W/cpu_e2e/${n}_${tag}.jsonl
    if [ -s $o ]; then echo "e2e ${n}_${tag}: already done"; continue; fi
    extra=""
    [ $tag != off ] && extra="EMBER_DRAFT=table EMBER_DRAFT_TABLE=$TD/$t"
    OMP_NUM_THREADS=${CORES:-4} budget "e2e $n $tag" python3 $D/dr_e2e.py run $B/tinyllama_server $g 0 $port \
      $W/prompts/eval100.jsonl $o --once $env $extra > $W/cpu_e2e/${n}_${tag}.txt 2>&1
    echo "e2e ${n}_${tag} rc=$? $(tail -1 $W/cpu_e2e/${n}_${tag}.txt | cut -c1-240)"
    rm -rf $o.run
  done
  for tag in table_k4 table_k8; do
    echo "compare $n off vs $tag: $(python3 $D/dr_e2e.py compare $W/cpu_e2e/${n}_off.jsonl $W/cpu_e2e/${n}_$tag.jsonl | cut -c1-400)"
  done
  cat $W/collect/${n}_chat40.jsonl $W/collect/${n}_copy20.jsonl $W/collect/${n}_carry40.jsonl > $W/cpu_e2e/${n}_head_off.jsonl
  echo "compare $n HEAD off vs this build off: $(python3 $D/dr_e2e.py compare $W/cpu_e2e/${n}_head_off.jsonl $W/cpu_e2e/${n}_off.jsonl | cut -c1-300)"
done
echo "cpu legs done"
