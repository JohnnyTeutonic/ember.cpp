#!/bin/bash
# CPU legs of the deterministic-mode gate (run in WSL; every job under the CPU budget at 2 cores, session ember):
#   (1) the CPU references of the 360-request set: x5_plain_collect.py through this work's CPU tinyllama_server at
#       gpu_layers 0, on pick_e1_L16 and chat_sft_bpe (the VM's on-mode replies are compared with them);
#   (2) the CPU regression: ../cuda_state_2026-10-25/cpu_regress.sh with this work's CPU build against the
#       committed receipts (REF: `git show HEAD:<path>` copies).
#   bash cpu_legs.sh <CPU build dir> <work dir> <REF dir>
B=${1:?build}; W=${2:?work}; REF=${3:?ref}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp/experiments
X=$E/attnres_x5_2026-10-25
A=/mnt/c/ml_artifacts
mkdir -p $W/inputs $W/plain
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > $W/inputs/prompts.jsonl
cp $R/microtorch/corpus/eval/carry5_v1.jsonl $W/inputs/
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "determinism gate CPU: $1" --cores 2 \
  --resume "rerun ember.cpp/experiments/determinism_2026-10-26/cpu_legs.sh $B $W $REF" -- "${@:2}"; }
for m in pick:$A/tiny_chat/e1/pick_e1_L16/model.gguf sft:$A/chat_base/sft/chat_sft_bpe.gguf; do
  n=${m%%:*}; g=${m#*:}
  if [ -s $W/plain/${n}_cpu/records.jsonl ]; then echo "collect ${n}_cpu: already done"; continue; fi
  budget "collect $n" python3 $X/x5_plain_collect.py $B/tinyllama_server $g 0 $((8460 + ${#n})) $W/inputs/prompts.jsonl \
    $W/inputs/carry5_v1.jsonl $W/plain/${n}_cpu > $W/collect_${n}_cpu.txt 2>&1
  echo "collect ${n}_cpu rc=$? $(tail -1 $W/collect_${n}_cpu.txt)"
done
bash $E/cuda_state_2026-10-25/cpu_regress.sh $B $W/cpu_regress $REF recursion_gate mor_gate mor_refusals attnres_gate \
  state_tokens_gate state4_gate regress_bits regress_parity g5_thirdparty adapter
echo "cpu legs done"
