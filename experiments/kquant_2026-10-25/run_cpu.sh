#!/bin/bash
# CPU half of the k-quant gate (WSL, ~/kq): every job resumes from its own output, so a rerun continues.
# Run under the budget, e.g. python3 scripts/cpu_budget.py run --session ember --cores 2 ... -- bash run_cpu.sh
set -euo pipefail
G="python3 $(dirname "$(readlink -f "$0")")/gate.py"
B=/mnt/c/ml_artifacts/next_base/base
W=$HOME/kq/work
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
# first-token logits: the llama path, then the IR interpreter (EMBER_INTERPRETER=1; K-quant rows stay in
# their blocks for a model this size, dequantised row by row)
$G probe f32 $B/pk_base_v1.gguf
for v in q4_k_m q4_k_s; do
  $G probe $v $B/pk_base_v1_$v.gguf
  $G probe ${v}_ref $W/pk_base_v1_${v}_ref.gguf
  $G probe ir_$v $B/pk_base_v1_$v.gguf EMBER_INTERPRETER=1
  $G probe ir_${v}_ref $W/pk_base_v1_${v}_ref.gguf EMBER_INTERPRETER=1
done
$G probe ir_f32 $B/pk_base_v1.gguf EMBER_INTERPRETER=1
# greedy replies (60 tokens), llama path
$G replies f32 $B/pk_base_v1.gguf
for v in q4_k_m q4_k_s; do
  $G replies $v $B/pk_base_v1_$v.gguf
  $G replies ${v}_ref $W/pk_base_v1_${v}_ref.gguf
done
$G summary
echo CPU_GATE_DONE
