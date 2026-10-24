#!/bin/bash
# Sync ember's sources into the ext4 build tree (~/ember_wt/src) and build the conformance adapter with
# the CPU tools and server the regression gates use, under the CPU budget (the pattern of
# ../state_tokens_2026-10-24/build_cpu.sh, plus ir_adapter).
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
SRC=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
cd "$SRC"
for f in *.cpp *.h *.cu CMakeLists.txt tools/*.cpp; do
  [ -f "$f" ] || continue
  cmp -s "$f" ~/ember_wt/src/"$f" || { cp "$f" ~/ember_wt/src/"$f"; echo "synced $f"; }
done
cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
python3 scripts/cpu_budget.py run --session ember --job "ember CPU build: IR conformance adapter" --cores 4 \
  --resume "rerun ember.cpp/experiments/ir_adapter_2026-10-25/build_cpu.sh (incremental build)" -- \
  bash -c 'cd ~/ember_wt/build_cpu && cmake . > cmake_ad.log 2>&1 && make -j4 ir_adapter ir_logits state_tokens_test tinyllama_server chat_template_check > ~/ember_wt/build_cpu/build_ad.log 2>&1; echo rc=$?'
tail -3 ~/ember_wt/build_cpu/build_ad.log
grep -E "error" ~/ember_wt/build_cpu/build_ad.log | head -30 || true
