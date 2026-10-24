#!/bin/bash
# Sync ember's sources into the ext4 build tree (~/ember_wt/src) and build the CPU tools this series' gates use,
# under the CPU budget (the pattern of ../ir_adapter_2026-10-25/build_cpu.sh, plus ir_doc_check,
# ir_numerics_test and ir_path_check).
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
SRC=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
cd "$SRC"
for f in *.cpp *.h *.cu CMakeLists.txt tools/*.cpp; do
  [ -f "$f" ] || continue
  cmp -s "$f" ~/ember_wt/src/"$f" || { cp "$f" ~/ember_wt/src/"$f"; echo "synced $f"; }
done
T="ir_adapter ir_doc_check ir_logits state_tokens_test tinyllama_server chat_template_check"
for t in ir_numerics_test ir_path_check; do
  grep -q "add_executable($t " CMakeLists.txt && T="$T $t"
done
cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
python3 scripts/cpu_budget.py run --session ember --job "ember CPU build: IR_SPEC v1 series" --cores 4 \
  --resume "rerun ember.cpp/experiments/ir_spec_2026-10-25/build_cpu.sh (incremental build)" -- \
  bash -c "cd ~/ember_wt/build_cpu && cmake . > cmake_spec.log 2>&1 && make -j4 $T > ~/ember_wt/build_cpu/build_spec.log 2>&1; echo rc=\$?"
tail -3 ~/ember_wt/build_cpu/build_spec.log
grep -E "error|warning: unused" ~/ember_wt/build_cpu/build_spec.log | head -30 || true
