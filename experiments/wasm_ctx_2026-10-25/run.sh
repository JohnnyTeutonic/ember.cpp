#!/bin/bash
# The page's context cap, gated (run in WSL; every job under the CPU budget, one core each, two at a time).
#   bash run.sh SRC BUILD WORK [step ...]   (steps: build qwen3 qwen25 uncapped paperkiln; default all)
# SRC: an ember source tree on ext4 with this change; BUILD: its native CPU build (tinyllama_server);
# WORK: ext4 scratch (the two page builds, requests, the native references), removed by hand afterwards.
#   build      deploy/wasm/build.sh twice: the page's default (1gb heap, cap 2048) into WORK/wasm1g and
#              EMBER_WASM_MAX_MEMORY=4gb into WORK/wasm4g;
#   qwen3, qwen25
#              make_requests.py's 10 requests: the native server with EMBER_IR_CTX=2048 (collect_native.py),
#              then tools/wasm_chat_check.mjs on the 4gb page build at its default cap (no ember_set_ctx);
#              the 3 long requests at a cap of 256 on both sides (EMBER_IR_CTX=256; ember_set_ctx(256), the
#              page's ?ctx=256): <model>.json and <model>_long.json beside this file;
#   uncapped   Qwen3-0.6B in the 4gb page build with ember_set_ctx(0) (the file's 40,960 positions):
#              uncapped_qwen3.txt, the load's outcome;
#   paperkiln  ../chat_template_2026-10-24's WebAssembly gates on the default page build: chat_sftfinal_v3
#              against mtstudio on the 40 frozen prompts and the 60 system-set requests (tools/wasm_parity.mjs),
#              each receipt compared with the committed one, ms_per_reply aside: paperkiln_wasm.txt.
set -u
SRC=${1:?src}; BUILD=${2:?build}; WORK=${3:?work}; shift 3
STEPS=${*:-build qwen3 qwen25 uncapped paperkiln}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/wasm_ctx_2026-10-25
CT=$R/ember.cpp/experiments/chat_template_2026-10-24
T=/mnt/c/ml_artifacts/thirdparty
NODE=$(ls -d ~/emsdk/node/*/bin/node | head -1)
mkdir -p "$WORK"
bud() { local job=$1; shift; python3 $R/scripts/cpu_budget.py run --session ember --cores 1 --job "WASM context cap: $job" \
  --resume "rerun $H/run.sh $SRC $BUILD $WORK" -- "$@"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
if has build; then
  sed -i 's/\r$//' "$SRC/deploy/wasm/build.sh"
  source ~/emsdk/emsdk_env.sh > /dev/null 2>&1
  export JSON_INC=$BUILD/_deps/nlohmann_json-src/include
  [ -d "$JSON_INC" ] || JSON_INC=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
  bud "page build, default (1gb)" bash -c "bash $SRC/deploy/wasm/build.sh $WORK/wasm1g > $WORK/wasm1g.log 2>&1; echo 1gb build rc=\$?" &
  bud "page build, 4gb" bash -c "EMBER_WASM_MAX_MEMORY=4gb bash $SRC/deploy/wasm/build.sh $WORK/wasm4g > $WORK/wasm4g.log 2>&1; echo 4gb build rc=\$?" &
  wait
  sha256sum $WORK/wasm1g/ember.wasm $WORK/wasm4g/ember.wasm
fi
python3 $H/make_requests.py "$WORK/req"
model() {  # name gguf
  local n=$1 g=$2
  bud "$n native, cap 2048" python3 $H/collect_native.py $BUILD/tinyllama_server "$g" "$WORK/req/requests.jsonl" \
    "$WORK/${n}_native.jsonl" EMBER_IR_CTX=2048 > "$WORK/${n}_native.out" 2>&1
  echo "$n native rc=$?: $(tail -1 $WORK/${n}_native.out)"
  bud "$n WebAssembly, default cap" "$NODE" $SRC/tools/wasm_chat_check.mjs $WORK/wasm4g "$g" "$WORK/${n}_native.jsonl" "$H/$n.json" \
    > "$WORK/${n}_wasm.out" 2>&1
  echo "$n wasm rc=$?: $(tail -1 $WORK/${n}_wasm.out | cut -c1-400)"
  bud "$n native, cap 256 (long)" python3 $H/collect_native.py $BUILD/tinyllama_server "$g" "$WORK/req/long_requests.jsonl" \
    "$WORK/${n}_native_long.jsonl" EMBER_IR_CTX=256 > "$WORK/${n}_native_long.out" 2>&1
  echo "$n native long rc=$?: $(tail -1 $WORK/${n}_native_long.out)"
  bud "$n WebAssembly, cap 256 (long)" "$NODE" $SRC/tools/wasm_chat_check.mjs $WORK/wasm4g "$g" "$WORK/${n}_native_long.jsonl" \
    "$H/${n}_long.json" --ctx 256 > "$WORK/${n}_wasm_long.out" 2>&1
  echo "$n wasm long rc=$?: $(tail -1 $WORK/${n}_wasm_long.out | cut -c1-400)"
}
# One model at a time: a 4gb WebAssembly heap each, on WSL's 8 GB.
has qwen3 && model qwen3 $T/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
has qwen25 && model qwen25 $T/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf
if has uncapped; then
  head -1 "$WORK/req/requests.jsonl" > "$WORK/one.jsonl"
  bud "Qwen3 uncapped load (4gb)" bash -c "'$NODE' $SRC/tools/wasm_chat_check.mjs $WORK/wasm4g $T/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf \
    $WORK/one.jsonl $WORK/uncapped.json --ctx 0 > $WORK/uncapped.out 2>&1; echo exit \$? >> $WORK/uncapped.out"
  { echo "Qwen3-0.6B-Q8_0, 4gb page build, ember_set_ctx(0) (the file's own context): node's output"; grep -v '^\s*at ' $WORK/uncapped.out; } > $H/uncapped_qwen3.txt
  cat $H/uncapped_qwen3.txt
fi
if has paperkiln; then
  python3 - "$CT" "$WORK" <<'EOF'
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
  : > $H/paperkiln_wasm.txt
  for s in frozen40 system_set_v1; do
    bud "paperkiln parity $s" bash -c "cd $WORK && $NODE $SRC/tools/wasm_parity.mjs $WORK/wasm1g /mnt/c/ml_artifacts/chat_base/sftfinal/chat_sftfinal_v3.gguf $WORK/wasm_ref_$s.jsonl ember.js $WORK/wasm_$s.json | tail -1" >> $H/paperkiln_wasm.txt
    python3 - "$WORK/wasm_$s.json" "$CT/wasm_$s.json" >> $H/paperkiln_wasm.txt <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for d in (a, b):
    d.pop("ms_per_reply", None)
print("wasm_" + sys.argv[1].split("wasm_")[-1], "equal to the committed receipt (ms_per_reply aside):", a == b)
EOF
  done
  cat $H/paperkiln_wasm.txt
fi
echo "run.sh done: $STEPS"
