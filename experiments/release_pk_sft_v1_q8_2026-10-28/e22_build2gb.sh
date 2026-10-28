#!/bin/bash
# 2gb heap builds (F16 pk_sft_v1 is dequantised to f32 at load: bad_alloc at 1gb): the gate linked
# from e22_build.sh's objects, and the page's own build (deploy/wasm/build.sh, EMBER_WASM_MAX_MEMORY=2gb).
set -euo pipefail
W=$HOME/e22rel
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
[ -f $W/gate/e22_page_gate_2gb.js ] || em++ -O3 -msimd128 $W/gate/obj/*.o -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2gb -sNODERAWFS=1 \
  -sENVIRONMENT=node -sEXIT_RUNTIME=1 -sUSE_BOOST_HEADERS=1 -fexceptions -o $W/gate/e22_page_gate_2gb.js
[ -f $W/page2gb/ember.wasm ] || ( cd $W/src && JSON_INC=$HOME/ember_ttft/build_cpu/_deps/nlohmann_json-src/single_include \
  EMBER_WASM_MAX_MEMORY=2gb bash deploy/wasm/build.sh $W/page2gb ) > $W/page2gb_build.log 2>&1
sha256sum $W/page2gb/ember.wasm $W/page2gb/ember.js
