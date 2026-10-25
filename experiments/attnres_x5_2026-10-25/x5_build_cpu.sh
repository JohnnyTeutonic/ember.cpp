#!/bin/bash
# CPU builds (Release, HAS_CUDA off) of the three trees in ~/x5g on ext4, from the scratch archives:
#   new    e115fd3 (git archive)                      ir_logits tinyllama_server
#   old    cbdcb915 (git archive)                     ir_logits tinyllama_server
#   oldce  cbdcb915 + ce41ffb's diff (graph_ir.cpp)   ir_logits tinyllama_server
# FetchContent sources from ~/ember_wt/build_cpu/_deps (nothing downloaded). Run under cpu_budget (2 cores).
set -euo pipefail
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/attnres_x5_2026-10-25
W=~/x5g
DEPS=$HOME/ember_wt/build_cpu/_deps
mkdir -p $W
for t in new old oldce; do
  if [ ! -d $W/$t/src ]; then
    mkdir -p $W/$t/src
    a=new; [ $t != new ] && a=old
    tar xzf $S/ember_$a.tgz -C $W/$t/src --strip-components=1
    if [ $t = oldce ]; then (cd $W/$t/src && patch -p1 < $S/ce41ffb.patch); fi
  fi
  mkdir -p $W/$t/build_cpu
  cd $W/$t/build_cpu
  [ -f Makefile ] || cmake $W/$t/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF -DUSE_CUDA=OFF \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$DEPS/nlohmann_json-src" \
    -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB="$DEPS/cpp-httplib-src" \
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$DEPS/pybind11-src" > cmake.log 2>&1
  make -j2 ir_logits tinyllama_server > build.log 2>&1 || { tail -30 build.log; exit 1; }
  echo "$t built: $(ls ir_logits tinyllama_server | tr '\n' ' ')"
done
cp $S/COMMITS $W/
grep -n "ember reads" $W/*/src/graph_ir.cpp
