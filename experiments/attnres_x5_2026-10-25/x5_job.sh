#!/bin/bash
# x5_job.sh NAME SCRIPT [ARGS...]: run SCRIPT under the CPU budget (session ember, 2 cores), log to ~/x5g/logs/NAME.log
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
N=$1; shift
mkdir -p ~/x5g/logs
python3 $R/scripts/cpu_budget.py run --session ember --job "X5 attnres gate: $N" --cores 2 \
  --resume "rerun x5_job.sh $N $*" -- bash "$@" > ~/x5g/logs/$N.log 2>&1
echo "$N rc=$?"
tail -5 ~/x5g/logs/$N.log
