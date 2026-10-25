#!/bin/bash
# CPU build (Release, HAS_CUDA off) of ember 5ad59b0 (git archive origin/main) in ~/tptrain on ext4.
# FetchContent sources from ~/ember_wt/build_cpu/_deps (nothing downloaded). Run under cpu_budget (2 cores).
set -euo pipefail
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/thirdparty_train_2026-10-26
W=~/tptrain
DEPS=$HOME/ember_wt/build_cpu/_deps
mkdir -p $W
if [ ! -d $W/src ]; then
  mkdir -p $W/src
  tar xzf $S/tp_ember.tgz -C $W/src --strip-components=1
  printf '\nadd_executable(tp_session_logits tools/tp_session_logits.cpp)\ntarget_link_libraries(tp_session_logits PRIVATE tinyllama_core)\n' >> $W/src/CMakeLists.txt
fi
cp $S/tp_session_logits.cpp $W/src/tools/
mkdir -p $W/build_cpu
cd $W/build_cpu
[ -f Makefile ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF -DUSE_CUDA=OFF \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$DEPS/nlohmann_json-src" \
  -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB="$DEPS/cpp-httplib-src" \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$DEPS/pybind11-src" > cmake.log 2>&1
make -j2 ir_logits ir_path_check tinyllama_server tp_session_logits > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built: $(ls ir_logits ir_path_check tinyllama_server tp_session_logits | tr '\n' ' ')"
