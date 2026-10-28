#!/bin/bash
# The two token streams of draft_2026-10-27 again (its dr_tables.sh commands, one core each; run in WSL):
#   bash r1_corpus.sh <work dir>
# human.u16 (pick_e1_L16's 13 human dialogue sources) and soda.u16 (pk_base_v1's SODA, first 150 MB, and OASST),
# through pick_e1_L16's GGUF vocabulary. Finished files are skipped (resume = rerun). Their sha256 must equal
# draft_2026-10-27's receipts/tables.json streams.
W=${1:?work}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
D=$R/ember.cpp/experiments/draft_2026-10-27
A=/mnt/c/ml_artifacts
H=$A/human_dialogue
mkdir -p $W/corpus
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "spec R1: $1" --cores 1 \
  --resume "rerun ember.cpp/experiments/spec_r1_2026-10-28/r1_corpus.sh $W (finished files are skipped)" -- "${@:2}"; }
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
sha256sum $W/corpus/human.u16 $W/corpus/soda.u16
