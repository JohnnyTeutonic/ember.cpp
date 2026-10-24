#!/bin/bash
# Compile this series' changed sources in the CUDA build tree (HAS_CUDA on), under the CPU budget. Compile only:
# no GPU use (the pattern of ../state_tokens_2026-10-24/build_cuda_check.sh, plus graph_ir_validate.cpp, which
# needs the CMake tree regenerated).
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
SRC=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
cd "$SRC"
for f in *.cpp *.h *.cu CMakeLists.txt tools/*.cpp; do
  [ -f "$f" ] || continue
  cmp -s "$f" ~/ember_wt/src_cuda/$f || { cp "$f" ~/ember_wt/src_cuda/$f; echo "synced $f"; }
done
cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
python3 scripts/cpu_budget.py run --session ember --job "ember CUDA compile check: IR_SPEC series" --cores 2 \
  --resume "rerun ember.cpp/experiments/ir_spec_2026-10-25/build_cuda_check.sh" -- \
  bash -c 'cd ~/ember_wt/build_cuda && cmake . > cmake_spec.log 2>&1 && make -j2 graph_ir.o graph_ir_validate.o api.o model.o > ~/ember_wt/build_cuda/build_spec.log 2>&1; echo rc=$?'
tail -4 ~/ember_wt/build_cuda/build_spec.log
grep -E "error" ~/ember_wt/build_cuda/build_spec.log | head -10 || true
