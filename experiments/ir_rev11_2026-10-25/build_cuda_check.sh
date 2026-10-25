#!/bin/bash
# Compile check of the revision 1.1 sources in a CUDA build tree (HAS_CUDA on; no GPU used): the translation units
# that changed or include graph_ir.h, from SRC (an ext4 copy) into OUT, for sm_75 only (CMAKE_CUDA_ARCHITECTURES).
# Usage (WSL, under scripts/cpu_budget.py with 2 cores): SRC=... bash build_cuda_check.sh OUT
set -e
SRC=${SRC:?ember sources (ext4)}
OUT=${1:?build dir}
DEPS=${DEPS:-$HOME/ember_wt/build_cpu/_deps}
mkdir -p "$OUT"
cd "$OUT"
if [ ! -f Makefile ]; then
  CMAKE_CUDA_ARCHITECTURES=75 cmake "$SRC" -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=ON \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$DEPS/nlohmann_json-src" \
    -DFETCHCONTENT_SOURCE_DIR_CPP-HTTPLIB="$DEPS/cpp-httplib-src" \
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$DEPS/pybind11-src" > cmake.log 2>&1
fi
grep -E "HAS_CUDA:BOOL|CMAKE_CUDA_ARCHITECTURES:" CMakeCache.txt
make -j2 graph_ir.o graph_ir_cuda.o graph_ir_recurrence.o graph_ir_validate.o ir_digest.o api.o weight_management.o \
  quantization.o > build.log 2>&1 && rc=0 || rc=$?
grep -E "error" build.log | head -20 || true
tail -3 build.log
echo "cuda compile rc=$rc"
