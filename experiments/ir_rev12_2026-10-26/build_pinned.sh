#!/bin/bash
# A clean pinned CPU build of ember's conformance adapter (and, with targets named, the regression gates' tools) from
# a commit: git archive, not the working tree, into ~/ember_adapter_<short-hash>/ on ext4 (src/, build/, COMMIT).
# The FetchContent sources come from an existing build's _deps (DEPS, default ~/ember_wt/build_cpu/_deps).
# Usage (WSL, under scripts/cpu_budget.py with 2 cores): bash build_pinned.sh COMMIT [targets...]
#   default targets: ir_adapter ir_doc_check; the gates add ir_logits state_tokens_test tinyllama_server
#   chat_template_check
set -euo pipefail
C=${1:?commit}
shift
REPO=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
FULL=$(git -C "$REPO" rev-parse "$C")
D=~/ember_adapter_${FULL:0:7}
DEPS=${DEPS:-$HOME/ember_wt/build_cpu/_deps}
mkdir -p "$D/src" "$D/build"
[ -f "$D/COMMIT" ] || { git -C "$REPO" archive "$FULL" | tar -x -C "$D/src"; echo "$FULL" > "$D/COMMIT"; }
[ "$(cat "$D/COMMIT")" = "$FULL" ] || { echo "$D holds another commit"; exit 1; }
cd "$D/build"
[ -f Makefile ] || cmake "$D/src" -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF -DUSE_CUDA=OFF \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$DEPS/nlohmann_json-src" \
  -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB="$DEPS/cpp-httplib-src" \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$DEPS/pybind11-src" > cmake.log 2>&1
make -j2 ${*:-ir_adapter ir_doc_check} > build.log 2>&1
tail -2 build.log
grep -c "HAS_CUDA:BOOL=OFF" CMakeCache.txt
sha256sum "$D/build/ir_adapter"
echo "$D/build/ir_adapter at $FULL"
