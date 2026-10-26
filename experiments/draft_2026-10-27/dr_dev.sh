#!/bin/bash
# The drafter's settings, chosen on the development sets only (run in WSL, under the CPU budget):
#   bash dr_dev.sh <work dir> <out.jsonl>
# dr_sim.py over chatdev and carrydev (dr_collect.sh's CPU records) for both models: prompt lookup alone at k = 4, 8,
# and prompt lookup then each candidate table (dr_tables.sh, dr_table.py build) at k = 8 over theta 0, 0.02, 0.05,
# 0.1. One JSON line per (model, set, table, k, theta).
W=${1:?work}; OUT=${2:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
D=$R/ember.cpp/experiments/draft_2026-10-27
: > $OUT
run() {  # model set table args...
  local m=$1 s=$2 t=$3; shift 3
  local tp=-; [ "$t" != - ] && tp=$W/tables/$t.bin
  python3 $D/dr_sim.py sweep $W/collect/${m}_$s.jsonl $tp --model $m "$@" | python3 -c "
import json, sys
for l in sys.stdin:
    d = json.loads(l); d.update({'model': sys.argv[1], 'set': sys.argv[2], 'table': sys.argv[3]}); print(json.dumps(d))
" $m $s $t >> $OUT
}
job() {
  for m in pick pk; do
    for s in chatdev carrydev; do
      run $m $s - --k 4,8 --modes pld
      for t in h_c8_p10 h_c4_p10 h_c4_p05 h_o3_c4_p10 h_c8_p25 h_c4_p25 h_c2_p10 s_c8_p10 s_c4_p10; do
        [ -s $W/tables/$t.bin ] && run $m $s $t --k 8 --modes pld+table --theta 0,0.02,0.05,0.1
      done
    done
  done
}
if [ "${DR_DEV_INNER:-0}" = 1 ]; then job; exit; fi
DR_DEV_INNER=1 python3 $R/scripts/cpu_budget.py run --session ember --job "draft: dev sweep" --cores 1 \
  --resume "rerun ember.cpp/experiments/draft_2026-10-27/dr_dev.sh $W $OUT (from the start)" -- bash $0 $W $OUT
wc -l $OUT
