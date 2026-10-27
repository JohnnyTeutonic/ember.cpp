#!/bin/bash
# E2.2 release gate builds, all on ext4 under ~/e22rel (ember 9d993f4, git archive):
#   build_cpu/   native CPU build (tinyllama_server, libtinyllama_core.a), -DHAS_CUDA=OFF, Release
#   gate/        e22_page_gate native + WebAssembly (the page's em++ flags, NODERAWFS, node)
#   page/        deploy/wasm/build.sh's ember.js + ember.wasm (the Space's own build)
# Resumable: each step skips when its output exists.
set -euo pipefail
SP=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/e22rel
W=$HOME/e22rel
SRC=$W/src
DEPS=$HOME/ember_ttft/build_cpu/_deps
JOBS=${JOBS:-2}
HEAP=${HEAP:-1gb}
mkdir -p $W
if [ ! -f $SRC/api.cpp ]; then mkdir -p $SRC && tar -xf $SP/e22_src_9d993f4.tar -C $SRC; fi
cp $SP/e22_page_gate.cpp $SRC/e22_page_gate.cpp
# native
if [ ! -x $W/build_cpu/tinyllama_server ]; then
  cmake -S $SRC -B $W/build_cpu -DHAS_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$DEPS/nlohmann_json-src \
    -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB=$DEPS/cpp-httplib-src \
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$DEPS/pybind11-src > $W/cmake.log 2>&1
  make -C $W/build_cpu -j$JOBS tinyllama_server > $W/make.log 2>&1
fi
JSON_INC=$DEPS/nlohmann_json-src/single_include
mkdir -p $W/gate/stub/emscripten $W/gate/obj $W/gate/inc
printf '#pragma once\n#define EMSCRIPTEN_KEEPALIVE\n' > $W/gate/stub/emscripten/emscripten.h
if [ ! -x $W/gate/e22_page_gate ]; then
  ( cd $SRC && g++ -O3 -DNDEBUG -std=c++17 -I. -I$W/gate/stub -I$JSON_INC -DEMBER_WASM_CTX=2048 e22_page_gate.cpp \
      $W/build_cpu/libtinyllama_core.a -fopenmp -lpthread -o $W/gate/e22_page_gate )
fi
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
em++ --version | head -1 > $W/emcc_version.txt
ln -sfn $JSON_INC/nlohmann $W/gate/inc/nlohmann
# the page's sources, as deploy/wasm/build.sh lists them, with e22_page_gate.cpp in place of ember_wasm.cpp (plus score.cpp, which the tf mode's score_rows needs; the page does not link it)
CORE=$(sed -n '/^SRC="/,/"$/p' $SRC/deploy/wasm/build.sh | tr -d '"' | sed 's/^SRC=//' | tr -s ' \n' ' ')
echo "core sources: $CORE" > $W/gate/sources.txt
export SRC W
for f in $CORE score.cpp e22_page_gate.cpp; do echo $f; done | xargs -P $JOBS -I{} bash -c '
  f={}; o=$W/gate/obj/$(echo $f | tr / _).o
  [ -f "$o" ] || em++ -O3 -msimd128 -DEMBER_WASM_CTX=2048 -std=c++17 -DNDEBUG -I"$SRC" -I"$W/gate/inc" -Wno-unknown-pragmas -sUSE_BOOST_HEADERS=1 -fexceptions -c "$SRC/$f" -o "$o"'
if [ ! -f $W/gate/e22_page_gate_$HEAP.js ]; then
  em++ -O3 -msimd128 $W/gate/obj/*.o -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=$HEAP -sNODERAWFS=1 -sENVIRONMENT=node \
      -sEXIT_RUNTIME=1 -sUSE_BOOST_HEADERS=1 -fexceptions -o $W/gate/e22_page_gate_$HEAP.js
fi
# the page's own build
if [ ! -f $W/page/ember.wasm ]; then
  ( cd $SRC && JSON_INC=$JSON_INC EMBER_WASM_MAX_MEMORY=$HEAP bash deploy/wasm/build.sh $W/page ) > $W/page_build.log 2>&1 || true
fi
ls -la $W/build_cpu/tinyllama_server $W/gate $W/page
