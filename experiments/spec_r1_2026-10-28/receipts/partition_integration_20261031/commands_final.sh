#!/bin/bash
set -euo pipefail
ROOT=/home/jonat/ember_e10_integration_20261031
OUT=/home/jonat/ember_e10_integration_checks_20261031
BUILD=/home/jonat/ember_e10_integration_build_20261031
KERAMOS=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos
JSON=/home/jonat/ember_publication_build_20261030/_deps/nlohmann_json-src/include
cd "$ROOT"
cmake --build "$BUILD" --target tinyllama_server ir_adapter ir_numerics_test state_tokens_test -j1 > "$OUT/build_cpu_final.log" 2>&1
python3 -m pytest -q tools/row_pick_profile/test_integration.py tools/row_pick_profile/test_profile.py tools/row_pick_profile/test_audit.py experiments/spec_r1_2026-10-28/test_decode_pick_gate.py experiments/spec_r1_2026-10-28/test_decode_pick_fallback.py --junitxml="$OUT/pytest_final.xml" > "$OUT/pytest_final.log" 2>&1
cat "$OUT/pytest_final.log"
python3 -m ruff check tools/row_pick_profile/test_integration.py tools/row_pick_profile/check_integration_sass.py experiments/spec_r1_2026-10-28/cpu_pick_integration_check.py > "$OUT/ruff_final.log" 2>&1
ctest --test-dir "$BUILD" --output-on-failure > "$OUT/ctest_final.log" 2>&1
"$BUILD/ir_numerics_test" > "$OUT/ir_numerics_final.json" 2>&1
"$BUILD/state_tokens_test" template > "$OUT/state_templates_final.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 python3 "$KERAMOS/tools/ir_conformance.py" --adapter "cmd:$BUILD/ir_adapter" --json "$OUT/conformance_final.json" > "$OUT/conformance_final.log" 2>&1
python3 "$ROOT/experiments/spec_r1_2026-10-28/cpu_pick_integration_check.py"  /home/jonat/ember_publication_build_20261030/tinyllama_server "$BUILD/tinyllama_server"  /mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf "$OUT/cpu_http_final" > "$OUT/cpu_http_final.log" 2>&1
for source in model.cpp api.cpp; do
 g++ -std=c++17 -fsyntax-only -DHAS_CUDA -I"$ROOT" -I/usr/local/cuda/include -I"$JSON" "$ROOT/$source" > "$OUT/cuda_host_final_$source.log" 2>&1
 g++ -std=c++17 -fsyntax-only -I"$ROOT" -I"$JSON" "$ROOT/$source" > "$OUT/cpu_host_final_$source.log" 2>&1
done
cat "$OUT/ctest_final.log"
tail -1 "$OUT/conformance_final.log"
cat "$OUT/cpu_http_final.log"
