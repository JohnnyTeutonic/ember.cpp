#!/bin/bash
set -euo pipefail
ROOT=/home/jonat/ember_e10_integration_20261031
BASE=/home/jonat/ember_publication_20261030
OUT=/home/jonat/ember_e10_integration_checks_20261031
JSON=/home/jonat/ember_publication_build_20261030/_deps/nlohmann_json-src/include
mkdir -p "$OUT"
cd "$ROOT"
python3 -m pytest -q tools/row_pick_profile/test_integration.py tools/row_pick_profile/test_profile.py tools/row_pick_profile/test_audit.py experiments/spec_r1_2026-10-28/test_decode_pick_gate.py experiments/spec_r1_2026-10-28/test_decode_pick_fallback.py > "$OUT/pytest.log" 2>&1
cat "$OUT/pytest.log"
python3 experiments/spec_r1_2026-10-28/test_decode_pick.py > "$OUT/sampler.log" 2>&1
cat "$OUT/sampler.log"
for source in model.cpp api.cpp; do
 g++ -std=c++17 -fsyntax-only -DHAS_CUDA -I"$ROOT" -I/usr/local/cuda/include -I"$JSON" "$ROOT/$source" > "$OUT/cuda_host_$source.log" 2>&1
 echo "CUDA host syntax: $source PASS"
 g++ -std=c++17 -fsyntax-only -I"$ROOT" -I"$JSON" "$ROOT/$source" > "$OUT/cpu_host_$source.log" 2>&1
 echo "CPU host syntax: $source PASS"
done
for item in baseline candidate; do
 if [ "$item" = baseline ]; then SOURCE="$BASE"; else SOURCE="$ROOT"; fi
 /usr/local/cuda/bin/nvcc -allow-unsupported-compiler -O3 -DNDEBUG -std=c++17 -DHAS_CUDA -arch=sm_89 -I"$SOURCE" -c "$SOURCE/verify_kernels.cu" -o "$OUT/verify_$item.o" > "$OUT/nvcc_$item.log" 2>&1
 /usr/local/cuda/bin/cuobjdump --dump-sass "$OUT/verify_$item.o" > "$OUT/sass_$item.txt"
 echo "CUDA compile-only: $item PASS"
done
