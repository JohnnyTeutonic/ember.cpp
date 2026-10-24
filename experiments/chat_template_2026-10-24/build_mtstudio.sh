#!/bin/bash
# mtstudio for G5: paperkiln's HEAD (the commit is printed and recorded in the receipts) exported with
# git archive into ~/pk_cpu_ct/src (ext4) and built for the CPU (CUDA off, Release) in ~/pk_cpu_ct/build
# under the CPU budget, as ~/pk_cpu_1024 was (pk_base_v1_2026-10-24/README.md). nlohmann/json comes
# from that tree's _deps, so nothing is downloaded.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
B=~/pk_cpu_ct
REV=$(git -C "$PK" rev-parse HEAD)
echo "paperkiln HEAD $REV"
if [ ! -f "$B/src/CMakeLists.txt" ] || [ "$(cat $B/REV 2>/dev/null)" != "$REV" ]; then
  rm -rf "$B/src"
  mkdir -p "$B/src"
  git -C "$PK" archive "$REV" | tar -x -C "$B/src"
  echo "$REV" > "$B/REV"
fi
mkdir -p "$B/build"
cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
python3 scripts/cpu_budget.py run --session ember --job "mtstudio CPU build at paperkiln HEAD (ember chat template G5)" --cores 4 \
  --resume "rerun ember.cpp/experiments/chat_template_2026-10-24/build_mtstudio.sh (incremental build)" -- \
  bash -c "cd $B/build && cmake ../src -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=OFF -DMICROTORCH_CUDA=OFF -DFETCHCONTENT_SOURCE_DIR_JSON=$HOME/pk_cpu_1024/build/_deps/json-src > cmake.log 2>&1 && make -j4 mtstudio > make.log 2>&1; echo rc=\$?"
tail -3 "$B/build/make.log"
grep -E "error" "$B/build/make.log" | head -20 || true
ls -la "$B/build/mtstudio"
