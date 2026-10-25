#!/bin/bash
# x5s_job.sh NAME SCRIPT [ARGS...]: run SCRIPT under the CPU budget (session ember, 2 cores), log to ~/x5s/logs/NAME.log
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
N=$1; shift
mkdir -p ~/x5s/logs
python3 $R/scripts/cpu_budget.py run --session ember --job "X5 smoke gate: $N" --cores 2 \
  --resume "rerun x5s_job.sh $N $*" -- bash "$@" > ~/x5s/logs/$N.log 2>&1
echo "$N rc=$?"
tail -25 ~/x5s/logs/$N.log
