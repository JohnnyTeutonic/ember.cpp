#!/bin/bash
# CPU Release build (HAS_CUDA off) of ember e115fd3 from a git archive extracted at ~/x5s/src (ext4), targets
# ir_logits and tinyllama_server (and libtinyllama_core.a). FetchContent sources from ~/ember_wt/build_cpu/_deps.
# Run under scripts/cpu_budget.py (session ember, 2 cores).
set -euo pipefail
W=~/x5s
DEPS=$HOME/ember_wt/build_cpu/_deps
mkdir -p $W/build
cd $W/build
[ -f Makefile ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF -DUSE_CUDA=OFF \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$DEPS/nlohmann_json-src" \
  -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB="$DEPS/cpp-httplib-src" \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$DEPS/pybind11-src" > cmake.log 2>&1
t0=$(date +%s)
make -j2 ir_logits tinyllama_server > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built in $(( $(date +%s) - t0 )) s: $(ls ir_logits tinyllama_server libtinyllama_core.a | tr '\n' ' ')"
grep -m1 "CMAKE_BUILD_TYPE" CMakeCache.txt
