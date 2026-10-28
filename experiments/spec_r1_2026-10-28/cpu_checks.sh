#!/bin/bash
# The CPU-side checks of spec_r1_2026-10-28 (run in WSL; each a few seconds):
#   bash cpu_checks.sh
# fixed_order_check.json: fixed_linear's statement order replayed on the host (r1_fixed_order_check.cpp).
# table_check.jsonl: draft_table.h's drafter (r1_table_check.cpp) against r1_sim.py at every decode position of the
#   CPU records, on the merged self-output tables (orders 6 and 4) and the two corpus tables.
# ../sim/tables.sha256: every table and token stream used.
set -e
E=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
X=$E/experiments/spec_r1_2026-10-28
W=~/r1/work
C=~/dr/work/collect
T=/mnt/c/ml_artifacts/ember_draft/tables
O=$X/receipts/cpu
mkdir -p $O
g++ -O2 -std=c++17 -ffp-contract=off -o $W/r1_fixed_order_check $X/r1_fixed_order_check.cpp
$W/r1_fixed_order_check > $O/fixed_order_check.json
g++ -O2 -std=c++17 -I $E -o $W/r1_table_check $X/r1_table_check.cpp
{
  for t in $W/tables/pick_merge_o6.bin $W/tables/pick_merge_o4.bin; do
    python3 $X/r1_table_check.py $W/r1_table_check $t $C/pick_chatdev.jsonl,$C/pick_carrydev.jsonl
  done
  python3 $X/r1_table_check.py $W/r1_table_check $T/h_c8_p10.bin $C/pick_chat40.jsonl,$C/pick_carry40.jsonl
  python3 $X/r1_table_check.py $W/r1_table_check $T/s_c8_p10.bin $C/pk_chat40.jsonl,$C/pk_carry40.jsonl
} > $O/table_check.jsonl
{
  for f in $W/tables/*.npz $W/tables/*.bin $T/h_c8_p10.bin $T/s_c8_p10.bin $W/corpus/human.u16 $W/corpus/soda.u16; do
    echo "$(sha256sum < $f | cut -c1-64) $(stat -c %s $f) $(basename $f)"
  done
} > $X/receipts/sim/tables.sha256
cat $O/fixed_order_check.json
cut -c1-220 $O/table_check.jsonl
