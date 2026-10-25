#!/bin/bash
# Build this work's CPU tree on ext4: ember HEAD (sc_head.tgz) + sc_mine.patch, Release, every target.
# Only files whose content changed are copied into src (rsync -c), so rebuilds stay incremental.
#   bash sc_build.sh
set -e
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
W=~/ember_score
if [ ! -d $W/head ]; then mkdir -p $W/head && tar xzf $S/sc_head.tgz -C $W/head; fi
rm -rf $W/stage && cp -a $W/head $W/stage
(cd $W/stage && git apply $S/sc_mine.patch)
mkdir -p $W/src && rsync -rc --delete $W/stage/ $W/src/ && rm -rf $W/stage
mkdir -p $W/build_cpu && cd $W/build_cpu
[ -f Makefile ] || cmake -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF $W/src > cmake.log 2>&1
python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score: CPU build" --cores 2 \
  --resume "bash $S/sc_build.sh (incremental)" -- make -j2 > make.log 2>&1 || { grep -n "error" make.log | head -30; tail -20 make.log; exit 1; }
tail -2 make.log
ls -la tinyllama_server ir_adapter ir_logits
