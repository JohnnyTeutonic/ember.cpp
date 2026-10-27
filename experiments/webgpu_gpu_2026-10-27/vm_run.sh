#!/usr/bin/env bash
# E6 on the GPU VM, after vm_probe.sh (Chrome has a hardware adapter) and vm_setup.sh (builds, checkpoints):
#   E6_W=work E6_OUT=out E6_STAGE=uploaded stage bash vm_run.sh [PART ...]
# PART (default all, in this order): refs gates_official pages_main bench bench_official gates_ir pages_ir bench_mt
# - refs: per gated family the document ember's CPU runs (ir_logits with EMBER_IR_PRINT: find_ir's document)
#   and tools/ir_wasm_gate's CPU reference (40 seeded prompts of 24 ids and 32 greedy tokens, 8 + 8 for the
#   families whose IR declares fewer than 64 positions; EMBER_IR_CTX 1024), as ../webgpu_ir_2026-10-26/ref.sh.
# - gates_*: tools/webgpu_parity/gate_page.html in Chrome on the hardware adapter against that reference
#   (ir_gate.mjs's measures), ir_webgpu.js on the printed document.
# - pages_*: ember CPU's /chat replies (vm_fixture.py, tinyllama_server of the same tree), then the page's
#   own worker on WebGPU and ?backend=wasm (page_test.html) in Chrome, against them.
# - bench / bench_official / bench_mt: bench_page.html, WebGPU against the single-threaded SIMD build (the
#   Space's ember.js; ember_4gb.js for the official checkpoints), and the 8-thread build (ember-mt.js, cross-origin isolated)
#   on WebAssembly alone.
# E6_EXTRA_FAMILIES adds families to the IR gates; with E6_ONLY_EXTRA=1 refs and gates_ir take only those.
# Every result goes to $E6_OUT at once (the driver copies it home every poll); a part reruns from its rows.
set -uo pipefail
W=${E6_W:?}; SRC=$W/ember; OUT=${E6_OUT:?}; ST=${E6_STAGE:?}
B=$W/build_cpu; H=$SRC/experiments/webgpu_gpu_2026-10-27; RC=$SRC/tools/webgpu_parity/remote_chrome.py
WWW=$W/www
export EMBER_WEBGPU_REMOTE=1 OMP_NUM_THREADS=8
mkdir -p "$W/models" "$W/refs" "$W/docs" "$OUT/gates" "$OUT/pages" "$OUT/bench" "$OUT/fixtures" "$OUT/docs" "$WWW"
log() { echo "$(date -u +%T) $*"; }
for f in "$ST"/m/*.gguf; do ln -sf "$f" "$W/models/"; done
ln -sf "$W/tp/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf" "$W/models/qwen3-0.6b.gguf"
ln -sf "$W/tp/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf" "$W/models/qwen2.5-0.5b-instruct.gguf"
for m in gemma-3-270m-it pythia-160m mamba-130m; do ln -sf "$W/tp/$m/$m-f32.gguf" "$W/models/$m.gguf"; done
# The served root: the page sources as committed, the WebAssembly builds, models, references, documents.
for f in e6_common.js gate_page.html bench_page.html page_test.html gpu_probe.html; do ln -sf "$SRC/tools/webgpu_parity/$f" "$WWW/"; done
ln -sfn "$SRC/deploy/wasm/web/webgpu" "$WWW/webgpu"
ln -sf "$SRC/deploy/wasm/web/worker.js" "$WWW/"
for f in "$W"/wasm/*.js "$W"/wasm/*.wasm; do ln -sf "$f" "$WWW/"; done
ln -sfn "$W/models" "$WWW/m"; ln -sfn "$W/refs" "$WWW/refs"; ln -sfn "$W/docs" "$WWW/docs"; ln -sfn "$OUT/fixtures" "$WWW/fx"

OFFICIAL="gemma-3-270m-it qwen2.5-0.5b-instruct qwen3-0.6b"
IRFAM="kimi_tiny srd_tiny attnres_tiny flex_highway_tiny mamba_tiny gpt2_tiny swa_tiny moe_tiny_v3 rec_b2x4 rec_p1_b2x3_c1
       rec_p1_b2x3_c1_inject mor3 mor2p0 state4 state4_q8_0 ar3 ar3_q8_0 synthetic-gpt2 synthetic-mistral-swa gpt2_tiny_expr
       gpt2_tiny_exprnorm pythia-160m mamba-130m ${E6_EXTRA_FAMILIES:-}"
CHATS_MAIN="chat_sftfinal_v3 state4 tc_e1_mor_MoR3_r1 mamba_tiny"
CHATS_IR="srd_tiny attnres_tiny flex_highway_tiny gpt2_tiny kimi_tiny swa_tiny"
if [ "${E6_ONLY_EXTRA:-0}" = 1 ]; then OFFICIAL=""; IRFAM="${E6_EXTRA_FAMILIES:-}"; fi
# A supplied document (EMBER_IR for the reference; the printed document, the same text, is gate_page.html's doc=):
# the expr rewrites of gpt2_tiny, keramos's tp_mamba1 recurrence rewrite on its model, Mamba-130m's recurrence
# rewrite (../webgpu_recurrence_2026-10-27/docs) on the converted checkpoint.
model_of() { case $1 in gpt2_tiny_expr*) echo "$W/models/gpt2_tiny.gguf" ;; tp_mamba1_rec) echo "$W/models/tp_mamba1.gguf" ;;
                        mamba-130m_rec) echo "$W/models/mamba-130m.gguf" ;; *) echo "$W/models/$1.gguf" ;; esac; }
irenv_of() { case $1 in gpt2_tiny_expr*|tp_mamba1_rec|mamba-130m_rec) echo "EMBER_IR=$ST/in/$1.ir.json" ;; *) echo "E6_NONE=1" ;; esac; }

refs() {
  for fam in $OFFICIAL $IRFAM; do
    d=$W/refs/$fam; g=$(model_of $fam); e=$(irenv_of $fam)
    [ -f "$g" ] || { log "ref $fam: no model $g"; continue; }
    mkdir -p "$d" "$OUT/gates/$fam"
    [ -s "$W/docs/$fam.json" ] || echo '[]' | env $e EMBER_IR_PRINT="$W/docs/$fam.json" "$B/ir_logits" "$g" > /dev/null 2> "$d/doc.log"
    cp "$W/docs/$fam.json" "$OUT/docs/" 2>/dev/null
    extra=""
    ctx=$(python3 "$SRC/experiments/webgpu_ir_2026-10-26/ir_ctx.py" "$g")
    [ "$ctx" -lt 64 ] && extra="--len 8 --gen 8"
    env $e "$B/ir_wasm_gate" ref "$g" "$d" --ctx 1024 --first 40 $extra > "$d/ref.out" 2> "$d/ref.log"
    log "ref $fam rc=$? ($(wc -l < "$d/ref.jsonl" 2>/dev/null) prompts) $(tail -c 200 "$d/ref.out")"
    cp "$d/ids.json" "$d/ref.jsonl" "$d/ref.out" "$OUT/gates/$fam/" 2>/dev/null
  done
}

gate() {   # gate FAM
  local fam=$1 g; g=$(basename "$(model_of $1)")
  [ -f "$W/refs/$fam/ref.jsonl" ] || { log "gate $fam: no reference"; return; }
  python3 "$RC" --root "$WWW" --out "$OUT/gates/$fam.json" --rows "$OUT/gates/$fam.rows.jsonl" --timeout 5400 \
    --page "gate_page.html?model=m/$g&ref=refs/$fam&doc=docs/$fam.json&path=ir&first=40&ctx=1024" > /dev/null
  log "gate $fam rc=$? $(python3 -c "
import json,sys; r=json.load(open(sys.argv[1])); s=r.get('result') or {}
print({k: s.get(k) for k in ('greedy_identical','top1_agree','worst_rel','median_rel','pass','ms_per_forward_median')}, r.get('error') or s.get('error','')[:300])" "$OUT/gates/$fam.json")"
}

page() {   # page MODEL [EMBER_JS]: fixture, then the worker in Chrome on both backends
  local m=$1 js=${2:-ember} p=$ST/in/tinychat_hf_prompts_v1.jsonl sp=$W/space/$1
  [ "$m" = state4 ] && p=$ST/in/state4_requests.jsonl
  [ -s "$OUT/fixtures/$m.json" ] || python3 "$H/vm_fixture.py" "$B/tinyllama_server" "$SRC/www" "$W/models/$m.gguf" "$p" \
    "$OUT/fixtures/$m.json" --port 8099 > "$OUT/fixtures/$m.out" 2>&1
  log "fixture $m: $(tail -1 "$OUT/fixtures/$m.out")"
  rm -rf "$sp"; mkdir -p "$sp"
  for f in page_test.html e6_common.js worker.js webgpu; do ln -sfn "$WWW/$f" "$sp/$f"; done
  ln -sf "$W/wasm/$js.js" "$sp/ember.js"; ln -sf "$W/wasm/$js.wasm" "$sp/$js.wasm"
  ln -sf "$W/models/$m.gguf" "$sp/chat.gguf"
  ln -sf "$OUT/fixtures/$m.json" "$sp/greedy.json"; ln -sf "$OUT/fixtures/$m.requests.jsonl" "$sp/requests.jsonl"
  python3 "$RC" --root "$sp" --out "$OUT/pages/$m.json" --timeout 5400 --page "page_test.html?requests=requests.jsonl" > /dev/null
  log "page $m rc=$? $(python3 -c "
import json,sys; r=json.load(open(sys.argv[1])); s=r.get('result') or {}
print({k: s.get(k) for k in ('backends','n','webgpu_vs_wasm','webgpu_vs_ember_cpu','wasm_vs_ember_cpu','tokens','ms_per_reply','errors','pass')}, r.get('error') or s.get('error','')[:300])" "$OUT/pages/$m.json")"
}

bench() {   # bench NAME MODEL PROMPTS QUERY [REMOTE_CHROME FLAGS]
  local name=$1 m=$2 p=$3 qs=$4 iso=${5:-}
  python3 "$RC" --root "$WWW" --out "$OUT/bench/$name.json" --timeout 7200 $iso \
    --page "bench_page.html?model=m/$m.gguf&prompts=$p&$qs" > /dev/null
  log "bench $name rc=$? $(python3 -c "
import json,sys; r=json.load(open(sys.argv[1])); s=r.get('result') or {}
print(s.get('gpu_path'), s.get('webgpu_vs_wasm_same'), json.dumps(s.get('median')), r.get('error') or s.get('error','')[:300])" "$OUT/bench/$name.json")"
}

PARTS=${*:-refs gates_official pages_main bench bench_official gates_ir pages_ir bench_mt}
for part in $PARTS; do
  log "=== $part"
  case $part in
    refs) refs ;;
    gates_official) for f in $OFFICIAL; do gate $f; done ;;
    gates_ir) for f in $IRFAM; do gate $f; done ;;
    pages_main) for m in $CHATS_MAIN; do page $m; done ;;
    pages_ir) for m in $CHATS_IR; do page $m; done ;;
    bench)
      for m in chat_sftfinal_v3 tc_e1_mor_MoR3_r1 mamba_tiny; do bench $m $m fx/$m.requests.jsonl "n=40&runs=3&warm=2"; done
      bench state4 state4 fx/state4.requests.jsonl "n=40&runs=3&warm=2" ;;
    bench_official)
      for m in $OFFICIAL; do bench $m $m fx/chat_sftfinal_v3.prompts.json "ember=ember_4gb.js&n=6&runs=2&warm=1&max_new=32"; done ;;
    bench_mt)
      bench chat_sftfinal_v3_mt8 chat_sftfinal_v3 fx/chat_sftfinal_v3.requests.jsonl "ember=ember-mt.js&backends=wasm&n=40&runs=3&warm=2" "--isolate --wasm-only"
      bench qwen3-0.6b_mt8 qwen3-0.6b fx/chat_sftfinal_v3.prompts.json "ember=ember-mt.js&backends=wasm&n=6&runs=2&warm=1&max_new=32" "--isolate --wasm-only" ;;
    *) log "unknown part $part" ;;
  esac
done
log "vm_run done: $PARTS"
