#!/bin/bash
set -euo pipefail
ROOT=/home/jonat/ember_e10_integration_20261031
OUT=/home/jonat/ember_e10_integration_checks_20261031
BUILD=/home/jonat/ember_e10_integration_build_20261031
DEPS=/home/jonat/ember_publication_build_20261030/_deps
cmake -S "$ROOT" -B "$BUILD" -DHAS_CUDA=OFF -DCMAKE_BUILD_TYPE=Release  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$DEPS/nlohmann_json-src"  -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB="$DEPS/cpp-httplib-src"  -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$DEPS/pybind11-src" > "$OUT/cmake_cpu.log" 2>&1
cmake --build "$BUILD" --target tinyllama_server ir_adapter ir_numerics_test state_tokens_test cpu_dot_test cpu_dot_portable_test ir_expr_header_portability_test -j1 > "$OUT/build_cpu.log" 2>&1
ctest --test-dir "$BUILD" --output-on-failure > "$OUT/ctest.log" 2>&1
cat "$OUT/ctest.log"
"$BUILD/ir_numerics_test" > "$OUT/ir_numerics.json" 2>&1
"$BUILD/state_tokens_test" template > "$OUT/state_templates.log" 2>&1
python3 /home/jonat/keramos_publication_20261030/tools/ir_conformance.py --adapter "cmd:$BUILD/ir_adapter" --json "$OUT/conformance.json" > "$OUT/conformance.log" 2>&1
cat "$OUT/conformance.log" | tail -3
