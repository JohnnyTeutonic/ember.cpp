#!/bin/bash
# A separate CPU build tree (~/gpt2_gate/src, ~/gpt2_gate/build): ember HEAD (git archive) plus the working
# tree's FILES (default graph_ir.cpp, the ir_version fix before it was committed), so ~/ember_wt, which
# other gates sync, is untouched. Reuses ~/ember_wt/build_cpu's fetched dependencies.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$ROOT/ember.cpp
W=~/gpt2_gate
D=~/ember_wt/build_cpu/_deps
rm -rf $W/src
mkdir -p $W/src $W/build
git -C $E archive --format=tar HEAD | tar -x -C $W/src
git -C $E rev-parse --short HEAD > $W/src/HEAD_COMMIT
for f in ${FILES:-graph_ir.cpp}; do tr -d '\r' < $E/$f > $W/src/$f; echo "from the working tree: $f"; done
python3 $ROOT/scripts/cpu_budget.py run --session ember --cores 2 --job "ember GPT-2 gate: CPU build (ir_version fix)" \
  --resume "rerun ember.cpp/experiments/gpt2_2026-10-25/build.sh" -- \
  bash -c "cd $W/build && cmake $W/src -DHAS_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$D/nlohmann_json-src -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB=$D/cpp-httplib-src \
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$D/pybind11-src > cmake.log 2>&1 && make -j2 ir_logits ir_adapter > build.log 2>&1; echo build rc=\$?"
tail -3 $W/build/build.log
grep -E "error" $W/build/build.log | head -20 || true
