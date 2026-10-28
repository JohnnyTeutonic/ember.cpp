#!/bin/bash
# The synthetic-file gate (synth_gate.py) against a pinned CPU build B (default ~/ember_adapter_<short>/build of the
# commit given): ember's builder against keramos v1.6.1's gguf_ir.py, its interpreter's greedy and first-token logits
# against HF's on each file's 40 sequences, and Mixtral's tuned llama path (llama_path.cpp, linked with B's
# libtinyllama_core.a). Under scripts/cpu_budget.py, 1 core.   bash synth_gate.sh COMMIT OUT_DIR
set -u
C=${1:?commit}; O=${2:?out dir}
B=${B:-$HOME/ember_adapter_${C:0:7}/build}
SRC=$(dirname "$B")/src
H=$(cd "$(dirname "$0")" && pwd)
J=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
mkdir -p "$O"
g++ -O3 -std=c++17 -fopenmp -I"$SRC" -I"$J" "$H/llama_path.cpp" "$B/libtinyllama_core.a" -lboost_regex -lpthread \
  -o "$O/llama_path" || exit 1
python3 "$H/synth_gate.py" --adapter "$B/ir_adapter" --ir-logits "$B/ir_logits" --keramos ${KERAMOS:-$HOME/keramos_v1.6.1} \
  --dir /mnt/c/ml_artifacts/thirdparty/e4_synth --out "$O/synth_gate.json" --save-ir "$O/ir" --llama-path "$O/llama_path" \
  ${FAMS:-mixtral qwen2moe qwen3moe deepseek2 deepseek2_qlora deepseek2_dense_yarn deepseek3}
