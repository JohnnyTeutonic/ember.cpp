#!/bin/bash
# WebAssembly: build this change with deploy/wasm/build.sh on ext4 (~/ember_wt/src, one em++ process:
# one core under the budget) into ~/ct24/wasm, then tools/wasm_parity.mjs in emsdk's node on
# chat_sftfinal_v3 against mtstudio's replies from g5_mtstudio_sftfinal_v3_rows.jsonl: the 40 frozen
# prompts and the 60 system-set requests (system field, leading system entry, histories), greedy, 60
# new tokens. Receipts wasm_frozen40.json and wasm_system_set_v1.json beside this file.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
SRC=$R/ember.cpp
H=$SRC/experiments/chat_template_2026-10-24
cd "$SRC"
mkdir -p ~/ember_wt/src/deploy/wasm ~/ember_wt/src/tools
for f in *.cpp *.h deploy/wasm/build.sh deploy/wasm/ember_wasm.cpp tools/wasm_parity.mjs; do
  cmp -s "$f" ~/ember_wt/src/"$f" || { cp "$f" ~/ember_wt/src/"$f"; echo "synced $f"; }
done
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
export JSON_INC=~/ember_wt/build_cpu/_deps/nlohmann_json-src/include
python3 $R/scripts/cpu_budget.py run --session ember --job "ember WASM build (chat templates)" --cores 1 \
  --resume "rerun ember.cpp/experiments/chat_template_2026-10-24/wasm_gate.sh" -- \
  bash -c 'bash ~/ember_wt/src/deploy/wasm/build.sh ~/ct24/wasm > ~/ct24/wasm_build.log 2>&1; echo build rc=$?'
grep -E " error" ~/ct24/wasm_build.log | head || true
ls -la ~/ct24/wasm
python3 - "$H" <<'EOF'
import json, sys
h = sys.argv[1]
rows = [json.loads(x) for x in open(h + "/g5_mtstudio_sftfinal_v3_rows.jsonl", encoding="utf-8")]
for s in ("frozen40", "system_set_v1"):
    with open("/home/jonat/ct24/wasm_ref_" + s + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["set"] == s:
                f.write(json.dumps({"i": r["i"], "request": r["request"], "mtstudio": r["mtstudio"],
                                    "mts_tokens": r["mts_tokens"], "mts_stop": r["mts_stop"]}, ensure_ascii=False) + "\n")
EOF
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
for s in frozen40 system_set_v1; do
  python3 $R/scripts/cpu_budget.py run --session ember --job "ember WASM parity $s" --cores 1 \
    --resume "rerun ember.cpp/experiments/chat_template_2026-10-24/wasm_gate.sh" -- \
    bash -c "cd ~/ct24 && $NODE ~/ember_wt/src/tools/wasm_parity.mjs ~/ct24/wasm /mnt/c/ml_artifacts/chat_base/sftfinal/chat_sftfinal_v3.gguf ~/ct24/wasm_ref_$s.jsonl ember.js $H/wasm_$s.json | tail -3"
done
