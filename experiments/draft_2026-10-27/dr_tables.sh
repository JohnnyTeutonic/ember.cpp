#!/bin/bash
# Tokenise paperkiln's chat text and build the candidate draft tables (run in WSL, under the CPU budget).
#   bash dr_tables.sh <work dir>
# human.u16: the 13 human-written dialogue sources of the tiny-chat SFT data (tiny_chat_data.py's list; text.txt in
#   the chat format), pick_e1_L16's SFT corpus. soda.u16: lever2's SODA and OASST text (pk_base_v1's dialogue
#   sources), SODA cut to its first 150 MB. Both through pick_e1_L16's GGUF vocabulary (both models carry the same
#   16k Llama 3-scheme vocabulary; dr_check_vocab in dr_sim.py checks it).
W=${1:?work}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
D=$R/ember.cpp/experiments/draft_2026-10-27
A=/mnt/c/ml_artifacts
H=$A/human_dialogue
mkdir -p $W/corpus
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "draft: $1" --cores 4 \
  --resume "rerun ember.cpp/experiments/draft_2026-10-27/dr_tables.sh $W (finished files are skipped)" -- "${@:2}"; }
V=$A/tiny_chat/e1/pick_e1_L16/model.gguf
HT=""
for s in oasst convai2 wow bst taskmaster multiwoz self_dialogue; do HT="$HT $H/hd_default/human_${s}_v1/text.txt"; done
for s in topical_chat dolly quac empathetic dailydialog no_robots; do HT="$HT $H/hd_optin/human_${s}_v1/text.txt"; done
[ -s $W/corpus/human.u16 ] || { budget "tokenise human" python3 $D/dr_table.py tokenize $V $W/corpus/human.u16.tmp $HT \
  > $W/corpus/human.log 2>&1 && mv $W/corpus/human.u16.tmp $W/corpus/human.u16; }
tail -1 $W/corpus/human.log
[ -s $W/corpus/soda.u16 ] || { budget "tokenise soda" python3 $D/dr_table.py tokenize $V $W/corpus/soda.u16.tmp \
  --max-bytes 150000000 $A/capacity/lever2/dialogue_soda_v1/text.txt $A/capacity/lever2/dialogue_oasst_v1/text.txt \
  > $W/corpus/soda.log 2>&1 && mv $W/corpus/soda.u16.tmp $W/corpus/soda.u16; }
tail -1 $W/corpus/soda.log
# The two tables measured (chosen on the development sets, receipts/sim/dev_sweep.jsonl): orders 1-4, contexts seen
# at least 8 times, top share at least 0.10, redundant back-off entries pruned. The other candidates differ only in
# --min-count, --min-conf and --orders (their names: h|s, c<min-count>, p<min-conf x 100>, o3 for orders 3).
mkdir -p $W/tables
for t in h_c8_p10:human.u16 s_c8_p10:soda.u16; do
  n=${t%%:*}; c=${t#*:}
  [ -s $W/tables/$n.bin ] || budget "table $n" python3 $D/dr_table.py build $W/corpus/$c $W/tables/$n.bin --orders 4 \
    --min-count 8 --min-conf 0.10 --prune-redundant 1 | tail -1
done
echo "tables: done"
