#!/bin/bash
# The Space page's worker on WebGPU and WASM (page_test.html), from a fresh build. Run in WSL in a
# CPU turn (em++ build under .cpu_lock); the browser part runs on Windows (Edge via Playwright).
#   bash tools/webgpu_parity/page_test.sh MODEL.gguf GREEDY.json OUT_DIR
# GREEDY.json: make_greedy_fixture.sh's output for the same model.
set -euo pipefail
M=${1:?model}; G=${2:?greedy fixture}; OUT=${3:?out dir}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/.cpu_lock
[ -e "$LOCK" ] && { echo "lock held: $(cat "$LOCK")"; exit 1; }
LINE="ember: page_test em++ build, started $(date -Iseconds)"
echo "$LINE" > "$LOCK"
trap '[ "$(cat "$LOCK" 2>/dev/null)" = "$LINE" ] && rm -f "$LOCK"' EXIT
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
JSON_INC=${JSON_INC:-$HOME/ember_ttft/build_cpu/_deps/nlohmann_json-src/single_include}
rm -rf "$OUT" && mkdir -p "$OUT/wasm"
JSON_INC=$JSON_INC bash "$REPO/deploy/wasm/build.sh" "$OUT/wasm" > "$OUT/wasm_build.log" 2>&1
bash "$REPO/tools/build_wasm_space.sh" "$M" "$OUT/wasm" "$OUT/space"
cp "$G" "$OUT/space/greedy.json"
python3 -c "import json,sys; p=[json.loads(l) for l in open('/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl', encoding='utf-8')][:40]; json.dump([{'user_input': r['prompt']} for r in p], open(sys.argv[1], 'w'))" "$OUT/space/prompts.json"
cp "$REPO/tools/webgpu_parity/page_test.html" "$REPO/tools/webgpu_parity/run.py" "$OUT/space/"
echo "built; now on Windows: PAGE=page_test.html python run.py PORT (from $OUT/space)"
