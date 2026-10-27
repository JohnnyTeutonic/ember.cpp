#!/bin/bash
# tig_job.sh NAME COMMAND [ARGS...]: run COMMAND under the CPU budget (session ember, 2 cores), log to ~/tig/logs/NAME.log
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
N=$1; shift
mkdir -p ~/tig/logs
python3 $R/scripts/cpu_budget.py run --session ember --job "tensor_index gate: $N" --cores 2 \
  --resume "rerun tig_job.sh $N $*" -- "$@" > ~/tig/logs/$N.log 2>&1
echo "$N rc=$?"
tail -5 ~/tig/logs/$N.log
