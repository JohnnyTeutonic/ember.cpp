#!/bin/bash
set -euo pipefail
ROOT=/home/jonat/ember_e10_integration_20261031
OUT=/home/jonat/ember_e10_integration_checks_20261031
BUILD=/home/jonat/ember_e10_integration_build_20261031
KERAMOS=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos
mv "$OUT/conformance.log" "$OUT/conformance_invocation_path_failure.log"
PYTHONDONTWRITEBYTECODE=1 python3 "$KERAMOS/tools/ir_conformance.py" --adapter "cmd:$BUILD/ir_adapter" --json "$OUT/conformance.json" > "$OUT/conformance.log" 2>&1
cat "$OUT/conformance.log" | tail -3
cmake --build /home/jonat/ember_publication_build_20261030 --target tinyllama_server -j1 > "$OUT/build_baseline_cpu.log" 2>&1
python3 "$ROOT/experiments/spec_r1_2026-10-28/cpu_pick_integration_check.py"  /home/jonat/ember_publication_build_20261030/tinyllama_server "$BUILD/tinyllama_server"  /mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf "$OUT/cpu_http" > "$OUT/cpu_http.log" 2>&1
cat "$OUT/cpu_http.log"
