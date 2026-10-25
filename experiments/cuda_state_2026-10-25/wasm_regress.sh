#!/bin/bash
# The WebAssembly leg of the CPU regression (run in WSL): ../chat_template_2026-10-24/wasm_gate.sh's build and
# parity runs, on this work's source tree instead of the shared ~/ember_wt/src (nothing is synced from the
# OneDrive working tree), into OUT; receipts compared with the committed ones field by field, ms_per_reply
# (a timing) excepted. One core under the CPU budget.
#   bash wasm_regress.sh <source tree> <out dir> <reference dir>
set -e
SRC=${1:?source}; O=${2:?out}; REF=${3:?reference receipts}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/chat_template_2026-10-24
mkdir -p "$O"
sed -i 's/\r$//' "$SRC/deploy/wasm/build.sh"  # a Windows git archive writes CRLF
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
export JSON_INC=~/ember_wt/build_cpu/_deps/nlohmann_json-src/include
python3 $R/scripts/cpu_budget.py run --session ember --job "ember CUDA items: WASM build (regression)" --cores 1 \
  --resume "rerun ember.cpp/experiments/cuda_state_2026-10-25/wasm_regress.sh $SRC $O $REF" -- \
  bash -c "bash $SRC/deploy/wasm/build.sh $O/wasm > $O/wasm_build.log 2>&1; echo build rc=\$?"
python3 - "$H" "$O" <<'EOF'
import json, sys
h, o = sys.argv[1], sys.argv[2]
rows = [json.loads(x) for x in open(h + "/g5_mtstudio_sftfinal_v3_rows.jsonl", encoding="utf-8")]
for s in ("frozen40", "system_set_v1"):
    with open(o + "/wasm_ref_" + s + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["set"] == s:
                f.write(json.dumps({"i": r["i"], "request": r["request"], "mtstudio": r["mtstudio"],
                                    "mts_tokens": r["mts_tokens"], "mts_stop": r["mts_stop"]}, ensure_ascii=False) + "\n")
EOF
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
for s in frozen40 system_set_v1; do
  python3 $R/scripts/cpu_budget.py run --session ember --job "ember CUDA items: WASM parity $s (regression)" --cores 1 \
    --resume "rerun ember.cpp/experiments/cuda_state_2026-10-25/wasm_regress.sh $SRC $O $REF" -- \
    bash -c "cd $O && $NODE $SRC/tools/wasm_parity.mjs $O/wasm /mnt/c/ml_artifacts/chat_base/sftfinal/chat_sftfinal_v3.gguf $O/wasm_ref_$s.jsonl ember.js $O/wasm_$s.json | tail -3"
  python3 - "$O/wasm_$s.json" "$REF/wasm_$s.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for d in (a, b):
    d.pop("ms_per_reply", None)
print(sys.argv[1].split("/")[-1], "equal to the committed receipt (ms_per_reply aside):", a == b)
EOF
done
