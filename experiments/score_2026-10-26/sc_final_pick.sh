#!/bin/bash
# pick_e1_L16's gate set again with the final build (the tied-head fallback added after its first run): records
# must be byte-identical to the first run's.
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
W=~/ember_score/work
python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: pick on the final build" --cores 2 \
  --resume "bash $S/sc_final_pick.sh" -- python3 $S/sc_client.py serve ~/ember_score/build_cpu/tinyllama_server \
  /mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf 8495 $W/out pick_final $S/req/pk_requests.jsonl 2>&1 | grep -v "^\[" | tail -2
cmp $W/out/pick_final.pk_requests.jsonl $W/out/pick_def.pk_requests.jsonl && echo "pick final build: byte-identical to the first run"
sha256sum ~/ember_score/build_cpu/tinyllama_server
