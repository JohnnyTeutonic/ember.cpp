#!/bin/bash
# Compile the changed sources in the CUDA build tree (HAS_CUDA on), under the CPU budget. Compile only:
# no GPU use. Usage: wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
SRC=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
for f in graph_ir.cpp graph_ir.h cpu_attention.cpp cpu_attention.h api.cpp api.h model.cpp state_tokens.h; do
  cmp -s "$SRC/$f" ~/ember_wt/src_cuda/$f || { cp "$SRC/$f" ~/ember_wt/src_cuda/$f; echo "synced $f"; }
done
cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
python3 scripts/cpu_budget.py run --session ember --job "ember CUDA compile check: state tokens" --cores 2 \
  --resume "rerun ember.cpp/experiments/state_tokens_2026-10-24/build_cuda_check.sh" -- \
  bash -c 'cd ~/ember_wt/build_cuda && make graph_ir.o cpu_attention.o api.o model.o > ~/ember_wt/build_cuda/build_st.log 2>&1; echo rc=$?'
tail -4 ~/ember_wt/build_cuda/build_st.log
grep -E "error" ~/ember_wt/build_cuda/build_st.log | head -10 || true
