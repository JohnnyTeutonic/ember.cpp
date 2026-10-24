#!/bin/bash
# The pre-change engine for G4 and the regression receipts: ember main 69d7ca2 exported with git
# archive into ~/ember_69d7ca2/src (ext4) and built for the CPU (-DHAS_CUDA=OFF, Release) in
# ~/ember_69d7ca2/build under the CPU budget. The fetched dependencies come from the main build
# tree's _deps (the same pinned tarballs), so nothing is downloaded.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
SRC=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
B=~/ember_69d7ca2
if [ ! -f "$B/src/CMakeLists.txt" ]; then
  mkdir -p "$B/src"
  git -C "$SRC" archive 69d7ca2 | tar -x -C "$B/src"
fi
mkdir -p "$B/build"
D=~/ember_wt/build_cpu/_deps
cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
python3 scripts/cpu_budget.py run --session ember --job "ember CPU build: 69d7ca2 baseline (chat template gates)" --cores 4 \
  --resume "rerun ember.cpp/experiments/chat_template_2026-10-24/build_base.sh (incremental build)" -- \
  bash -c "cd $B/build && cmake ../src -DHAS_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$D/nlohmann_json-src -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB=$D/cpp-httplib-src -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$D/pybind11-src > cmake.log 2>&1 && make -j4 ir_logits state_tokens_test tinyllama_server > build.log 2>&1; echo rc=\$?"
tail -3 "$B/build/build.log"
grep -E "error" "$B/build/build.log" | head -20 || true
ls -la "$B/build/tinyllama_server" "$B/build/ir_logits"
