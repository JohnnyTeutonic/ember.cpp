#!/bin/bash
# CPU legs of verify_2026-10-27 (run in WSL; every job under the CPU budget, session ember):
#   (1) the CPU regression: ../cuda_state_2026-10-25/cpu_regress.sh with this work's CPU build against the
#       committed receipts (REF: `git show HEAD:<path>` copies); regress_bits' old side is 70ed063's ir_logits
#       (its own old build, 69d7ca2's, is gone, as in ../determinism_2026-10-26);
#   (2) speculation on the CPU: vf_e2e.py over all60.jsonl (chat40 + copy20) on pick_e1_L16 and pk_base_v1 at
#       gpu_layers 0, EMBER_SPEC off, 2, 4 and 8; replies, ids and stop ids against plain decode.
#   bash cpu_legs.sh <CPU build of this work> <CPU build of 70ed063> <work dir> <REF dir>
B=${1:?build}; BH=${2:?head build}; W=${3:?work}; REF=${4:?ref}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp/experiments
V=$E/verify_2026-10-27
A=/mnt/c/ml_artifacts
mkdir -p $W/e2e
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "verify CPU: $1" --cores ${CORES:-2} \
  --resume "rerun ember.cpp/experiments/verify_2026-10-27/cpu_legs.sh $B $BH $W $REF (finished runs are skipped)" -- "${@:2}"; }
python3 $V/vf_prompts.py $W/prompts
if [ "${SKIP_REGRESS:-0}" != 1 ]; then
  mkdir -p ~/ember_69d7ca2/build
  ln -sfn $BH/ir_logits ~/ember_69d7ca2/build/ir_logits
  echo "regress_bits old side: $(readlink -f ~/ember_69d7ca2/build/ir_logits)"
  bash $E/cuda_state_2026-10-25/cpu_regress.sh $B $W/cpu_regress $REF recursion_gate mor_gate mor_refusals attnres_gate \
    state_tokens_gate state4_gate regress_bits regress_parity g5_thirdparty adapter
fi
port=8640
for m in pick:$A/tiny_chat/e1/pick_e1_L16/model.gguf pk:$A/next_base/base/pk_base_v1.gguf; do
  n=${m%%:*}; g=${m#*:}
  for cfg in off:EMBER_NOTHING=1 k2:EMBER_SPEC=2 k4:EMBER_SPEC=4 k8:EMBER_SPEC=8; do
    tag=${cfg%%:*}; env=${cfg#*:}
    port=$((port + 1))
    if [ -s $W/e2e/${n}_${tag}.jsonl ]; then echo "e2e ${n}_${tag}: already done"; continue; fi
    OMP_NUM_THREADS=${CORES:-2} budget "e2e $n $tag" python3 $V/vf_e2e.py run $B/tinyllama_server $g 0 $port \
      $W/prompts/all60.jsonl $W/e2e/${n}_${tag}.jsonl $env > $W/e2e/${n}_${tag}.txt 2>&1
    echo "e2e ${n}_${tag} rc=$? $(tail -1 $W/e2e/${n}_${tag}.txt | cut -c1-240)"
    rm -rf $W/e2e/${n}_${tag}.jsonl.run
  done
done
python3 $V/vf_summary.py e2e $W
echo "cpu legs done"
