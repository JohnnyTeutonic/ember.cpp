#!/bin/bash
# E2.2 variants between Q8_0-all (118 MB) and F32 (441 MB), each through the same native gate as
# e22_gate.sh: greedy /chat replies with ids (the F32 gate's request) and teacher-forced logits on F32's
# greedy ids, compared by e22_variant_cmp.py. Files and outputs on ext4 under ~/e22rel/variants.
#   q8emb_f32  ember tools/quantize_gguf.py (default): blk.* matrices Q8_0; token_embd and output F32
#              (= paperkiln's gguf_q8_0, byte-identical per LOCAL_LLM_STACK 20 Oct)
#   q8_head    tools/quantize_gguf.py --embed-q8: blk.* and token_embd Q8_0; output (LM head) F32
#   f16        llama.cpp b6000 (4762ad7) llama-quantize ... F16: every 2-D tensor F16, norms F32
# Each step skips when its output exists (rerun resumes). Run under scripts/cpu_budget.py, 1 core.
#   bash e22_variants.sh [TAG ...]
set -uo pipefail
SP=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/e22rel
W=$HOME/e22rel
V=$W/variants
R=$W/run
F32=/mnt/c/ml_artifacts/general_chat/sft/pk_sft_v1.gguf
P=/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl
Q=$W/src/tools/quantize_gguf.py
LQ=$HOME/llamacpp_b6000_build/build/bin/llama-quantize
SRV=$W/build_cpu/tinyllama_server
PORT=8122
export OMP_NUM_THREADS=${CPU_BUDGET_CORES:-1}
TAGS=${*:-q8_head q8emb_f32 f16}
mkdir -p $V
# the F32 reference rows (the first run's binaries were deleted after recording)
[ -f $R/tf_f32.f32 ] || { t0=$(date +%s); $W/gate/e22_page_gate tf $F32 $R/f32_server.jsonl $R/tf_f32.f32 2> $R/tf_f32.log; echo "tf f32 rc=$? $(( $(date +%s) - t0 )) s"; }
for tag in $TAGS; do
  g=$V/pk_sft_v1_$tag.gguf
  if [ ! -f $g ]; then
    case $tag in
      q8emb_f32) python3 $Q $F32 $g > $V/$tag.make.log 2>&1 ;;
      q8_head) python3 $Q $F32 $g --embed-q8 > $V/$tag.make.log 2>&1 ;;
      f16) $LQ $F32 $g F16 1 > $V/$tag.make.log 2>&1 ;;
    esac
    echo "$tag made rc=$? $(stat -c %s $g) bytes"
  fi
  [ -f $V/$tag.sha256 ] || sha256sum $g > $V/$tag.sha256
  if [ ! -f $V/${tag}_server.jsonl ]; then
    for p in $(pgrep -f "[t]inyllama_server .* $PORT "); do kill $p; done
    $SRV "$g" $PORT 127.0.0.1 $W/src/www 0 > $V/${tag}_server.log 2>&1 &
    s=$!
    for i in $(seq 1 300); do curl -s 127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
    t0=$(date +%s)
    python3 $SP/e22_chat_client.py $PORT $P 40 $V/${tag}_server.jsonl "$(basename $g)" > /dev/null
    echo "$tag server rc=$? $(( $(date +%s) - t0 )) s"
    kill $s; wait $s 2>/dev/null
  fi
  [ -f $V/tf_$tag.f32.json ] || { t0=$(date +%s); $W/gate/e22_page_gate tf $g $R/f32_server.jsonl $V/tf_$tag.f32 2> $V/tf_$tag.log; echo "tf $tag rc=$? $(( $(date +%s) - t0 )) s"; }
  python3 $SP/e22_variant_cmp.py $R $V $tag > $V/$tag.summary.json && rm -f $V/tf_$tag.f32
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], json.dumps({k: d[k] for k in ('bytes','greedy_ids','flip_rows','max_abs_over_row_range','mean_abs_dlogit','top1_agree')}))" $V/$tag.summary.json $tag
done
