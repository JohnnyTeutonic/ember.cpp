#!/bin/bash
# Float64 arbitration of the largest ember-vs-mtstudio differences, both paperkiln models (budget, 1 core).
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
O=~/ember_score/work/out
for m in pkbase:/mnt/c/ml_artifacts/next_base/base/pk_base_v1.gguf pick:/mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf; do
  n=${m%%:*}; g=${m#*:}
  [ -s $O/f64_$n.json ] && { echo "f64 $n: done"; continue; }
  python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: float64 check $n" --cores 1 \
    --resume "bash $S/sc_f64.sh" -- python3 $S/sc_f64.py $g $S/req/pk_requests.jsonl $O/mts_$n.jsonl \
    $O/${n}_def.pk_requests.jsonl 10 $O/f64_$n.json 2>&1 | tail -12
done
