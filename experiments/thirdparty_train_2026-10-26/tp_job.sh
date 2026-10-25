#!/bin/bash
# tp_job.sh NAME SCRIPT [ARGS...]: run SCRIPT under the CPU budget (session ember, 2 cores), log to ~/tptrain/logs/NAME.log
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
N=$1; shift
mkdir -p ~/tptrain/logs
python3 $R/scripts/cpu_budget.py run --session ember --job "third-party-ops train gate: $N" --cores 2 \
  --resume "rerun tp_job.sh $N $* (/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/thirdparty_train_2026-10-26)" -- bash "$@" > ~/tptrain/logs/$N.log 2>&1
echo "$N rc=$?"
tail -8 ~/tptrain/logs/$N.log
