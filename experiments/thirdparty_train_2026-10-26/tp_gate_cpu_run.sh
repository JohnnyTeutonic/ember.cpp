#!/bin/bash
# The CPU gate on ~/tptrain/build_cpu (run under cpu_budget, 2 cores).
export OMP_NUM_THREADS=2
python3 /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/thirdparty_train_2026-10-26/tp_gate_cpu.py ~/tptrain/build_cpu ~/tptrain/gate_cpu.json
echo "gate_cpu rc=$?"
