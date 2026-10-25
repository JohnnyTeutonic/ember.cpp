#!/bin/bash
# ember HEAD's own CPU build (the old side of regress_bits and state_tokens_gate), Release, every target.
set -e
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
W=~/ember_score
[ -d $W/head ] || { mkdir -p $W/head && tar xzf $S/sc_head.tgz -C $W/head; }
mkdir -p $W/build_head && cd $W/build_head
[ -f Makefile ] || cmake -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF $W/head > cmake.log 2>&1
python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score: HEAD CPU build (regression old side)" --cores 2 \
  --resume "bash $S/sc_build_head.sh (incremental)" -- make -j2 > make.log 2>&1 || { grep -n "error" make.log | head; exit 1; }
ls -la tinyllama_server ir_logits
