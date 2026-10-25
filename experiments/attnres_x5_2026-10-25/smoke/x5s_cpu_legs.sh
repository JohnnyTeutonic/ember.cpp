#!/bin/bash
# CPU legs (WSL, under the CPU budget): x5s_gate_cpu.py (ir_logits) then x5s_chat.py (tinyllama_server /chat).
#   bash x5s_cpu_legs.sh OUT_DIR
set -u
OUT=${1:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
S=$R/ember.cpp/experiments/attnres_x5_2026-10-25/smoke
B=~/x5s/build
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
t0=$(date +%s)
python3 $S/x5s_gate_cpu.py $B "$OUT/x5s_gate_cpu.json"
echo "gate rc=$? in $(( $(date +%s) - t0 )) s"
python3 $S/x5s_chat.py $B/tinyllama_server 8431 $R/microtorch/corpus/eval/carry5_v1.jsonl "$OUT/x5s_chat_cpu.json"
echo "chat rc=$? in $(( $(date +%s) - t0 )) s"
