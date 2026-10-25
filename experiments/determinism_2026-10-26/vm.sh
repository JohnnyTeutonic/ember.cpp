#!/usr/bin/env bash
# VM side of the deterministic-mode gate (ember.cpp, 26 Oct 2026). Launched detached by drive.sh.
#   bash vm.sh
# Inputs in /content: ember_new.tgz (this work's commit), ember_head.tgz (d74e052, ember main before it), COMMITS,
# small.tgz, pick.part* sft.part* arx5.part*, BIG_SHA256SUMS.
# Both trees are built as paperkiln's ensure_ember builds them: Release, -DCMAKE_CUDA_ARCHITECTURES=89 on an L4
# (sm_89), the GPU's own architecture on any other GPU.
# Legs:
#   (a) SASS: cuobjdump -sass of both tinyllama_server binaries, compared function by function;
#   (b) the tuned llama path through tinyllama_server (x5_plain_collect.py: 360 greedy requests, every sampled
#       position's logits, gpu_layers -1, EMBER_GEMM_TF32=0), each run its own server process:
#       pick_e1_L16: head, new off, new on, new off, new on; chat_sft_bpe: head, new off, new on, new on;
#   (c) cost: bench.py, new off and on alternating (pick twice each, chat_sft_bpe once each);
#   (d) the interpreter: ir_device_dump of head, new off, new on, new on on the device-gate set (+ ar_x5, its Q8_0
#       file, mor3, mor2p0 and the third-party checkpoints), compared bit for bit.
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; M=$SM/models; BIG=/content/big
mkdir -p "$OUT" "$BIG" /content/plain /content/dumps "$OUT/dumps"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r NEWC HEADC < /content/COMMITS
echo "=== vm $(date -u) new=$NEWC head=$HEADC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ARCH=$CC   # 89 on the L4, as ensure_ember's -DCMAKE_CUDA_ARCHITECTURES=89; native elsewhere
echo "gpu $GPU, compute capability $CC, building for $ARCH, nproc $(nproc)"
echo "$GPU $CC $ARCH" > $OUT/gpu.txt
tar xzf small.tgz && tar xzf small/ids.tgz -C /content || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cat pick.part* > $BIG/pick_L16.gguf && cat sft.part* > $BIG/chat_sft_bpe.gguf && cat arx5.part* > $BIG/ar_x5.gguf || fail "reassemble"
(cd $BIG && sha256sum -c /content/BIG_SHA256SUMS) > $OUT/big.sha256 2>&1 || fail "big sha256"
rm -f pick.part* sft.part* arx5.part*

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
echo "--- third-party checkpoints in the background $(date -u +%T)"
( pip install -q gguf sentencepiece "transformers==4.57.0" > /dev/null 2>&1
  git clone -q --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp /content/llamacpp_b6000
  python3 - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", local_dir="/content/tp/qwen3")
hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf", local_dir="/content/tp/qwen25")
snapshot_download("EleutherAI/pythia-160m", revision="50f5173d932e8e61f858120bcb800b97af589f46",
                  local_dir="/content/tp/pythia-160m", allow_patterns=["*.json", "*.safetensors", "*.txt"])
snapshot_download("unsloth/gemma-3-270m-it", local_dir="/content/tp/gemma-3-270m-it", allow_patterns=["*.json", "*.safetensors", "*.model"])
print("downloads done")
PY
  for m in pythia-160m gemma-3-270m-it; do
    python3 /content/llamacpp_b6000/convert_hf_to_gguf.py /content/tp/$m --outtype f32 --outfile /content/tp/$m/$m-f32.gguf > $OUT/convert_$m.log 2>&1 \
      || rm -f /content/tp/$m/$m-f32.gguf
  done
  touch /content/TP_DONE ) > $OUT/thirdparty_fetch.log 2>&1 &

echo "--- builds $(date -u +%T)"
mkdir -p /content/new /content/head
tar xzf ember_new.tgz -C /content/new --strip-components=1 && tar xzf ember_head.tgz -C /content/head --strip-components=1 || fail "untar ember"
for t in new head; do
  t0=$(date +%s)
  cmake -S /content/$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH > $OUT/cmake_$t.log 2>&1 || { tail -20 $OUT/cmake_$t.log; fail "cmake $t"; }
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
  make -C /content/b_$t -j"$(nproc)" tinyllama_server ir_device_dump > $OUT/make_$t.log 2>&1 || { tail -30 $OUT/make_$t.log; fail "build $t"; }
  echo "built $t in $(( $(date +%s) - t0 )) s"
done
sha256sum /content/b_*/tinyllama_server /content/b_*/ir_device_dump > $OUT/bins.sha256

echo "--- (a) SASS $(date -u +%T)"
for t in new head; do cuobjdump -sass /content/b_$t/tinyllama_server > /content/sass_$t.txt 2> $OUT/cuobjdump_$t.log; done
python3 $SM/sass_compare.py /content/sass_head.txt /content/sass_new.txt > $OUT/sass.json
echo "sass: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('arches','head_functions','new_functions','identical','differ','only_in_head','only_in_new')}))" $OUT/sass.json)"

echo "--- (b) tuned llama path, tinyllama_server $(date -u +%T)"
P=$SM/prompts.jsonl; C5=$SM/carry5_v1.jsonl
export X5_KEEP_ALL=1
port=8500
collect() {  # run gguf build off|on
  local run=$1 g=$2 b=$3 mode=$4 e=EMBER_NOTHING=1
  [ "$mode" = on ] && e=EMBER_DETERMINISTIC=1
  port=$((port + 1))
  env $e python3 $SM/x5_plain_collect.py /content/b_$b/tinyllama_server $g -1 $port $P $C5 /content/plain/$run > $OUT/collect_$run.txt 2>&1
  echo "collect $run ($b, $mode) rc=$? $(tail -1 $OUT/collect_$run.txt) /health deterministic: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['health'].get('deterministic'))" /content/plain/$run/meta.json 2>/dev/null)"
}
cmp_pair() {  # a b
  python3 $SM/x5_plain_compare.py /content/plain/$1 /content/plain/$2 > $OUT/cmp_$1_vs_$2.json
  echo "compare $1 vs $2 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['bit_identical'], json.dumps({k: [v[x] for x in ('replies_identical','ids_identical','first_token_logits_bit_equal','all_positions_logits_bit_equal')] + [v['every_position']['bit_equal_rows'], v['every_position']['rows'], v['every_position']['worst_abs']] for k, v in d['sets'].items()}))" $OUT/cmp_$1_vs_$2.json)"
}
G=$BIG/pick_L16.gguf
collect pick_head_a $G head off
collect pick_off_a $G new off
collect pick_on_a $G new on
collect pick_off_b $G new off
collect pick_on_b $G new on
G=$BIG/chat_sft_bpe.gguf
collect sft_head_a $G head off
collect sft_off_a $G new off
collect sft_on_a $G new on
collect sft_on_b $G new on
for pair in pick_on_a:pick_on_b sft_on_a:sft_on_b pick_off_a:pick_head_a pick_off_b:pick_head_a pick_off_a:pick_off_b \
            sft_off_a:sft_head_a pick_on_a:pick_off_a pick_on_a:pick_head_a sft_on_a:sft_off_a; do
  cmp_pair ${pair%:*} ${pair#*:}
done
for run in pick_head_a pick_off_a pick_on_a pick_off_b pick_on_b sft_head_a sft_off_a sft_on_a sft_on_b; do
  mkdir -p $OUT/plain/$run
  cp /content/plain/$run/records.jsonl /content/plain/$run/meta.json $OUT/plain/$run/
done
cp /content/plain/pick_on_a/first_logits.f32 $OUT/plain/pick_on_a/
cp /content/plain/sft_on_a/first_logits.f32 $OUT/plain/sft_on_a/
rm -f /content/plain/*/all_logits.f32

echo "--- (c) cost $(date -u +%T)"
bench() {  # tag gguf off|on
  local tag=$1 g=$2 mode=$3 e=EMBER_NOTHING=1
  [ "$mode" = on ] && e=EMBER_DETERMINISTIC=1
  port=$((port + 1))
  env $e python3 $SM/bench.py /content/b_new/tinyllama_server $g $port $P $C5 $OUT/bench_$tag.json > $OUT/bench_$tag.txt 2>&1
  echo "bench $tag rc=$? $(tail -1 $OUT/bench_$tag.txt)"
}
bench pick_off_1 $BIG/pick_L16.gguf off
bench pick_on_1 $BIG/pick_L16.gguf on
bench pick_off_2 $BIG/pick_L16.gguf off
bench pick_on_2 $BIG/pick_L16.gguf on
bench sft_off_1 $BIG/chat_sft_bpe.gguf off
bench sft_on_1 $BIG/chat_sft_bpe.gguf on

echo "--- (d) interpreter: ir_device_dump head, new off, new on twice $(date -u +%T)"
DD_ENV=EMBER_NOTHING=1
same() { cmp -s "$1" "$2" && echo identical || echo DIFFERS; }
dd() {  # name gguf [args]
  local name=$1 g=$2; shift 2
  [ -s "$g" ] || { echo "dump $name skipped: no $g"; return; }
  local v b e
  for v in head off on_a on_b; do
    b=new; e=$DD_ENV
    [ $v = head ] && b=head
    case $v in on_*) e="$DD_ENV EMBER_DETERMINISTIC=1";; esac
    env $e /content/b_$b/ir_device_dump $g /content/dumps/${name}_$v.bin "$@" > /dev/null 2> $OUT/dumps/${name}_$v.log \
      || echo "dump $name $v failed: $(tail -1 $OUT/dumps/${name}_$v.log)"
  done
  local d=/content/dumps/$name
  local r="$name head_vs_off=$(same ${d}_head.bin ${d}_off.bin) on_a_vs_on_b=$(same ${d}_on_a.bin ${d}_on_b.bin) off_vs_on=$(same ${d}_off.bin ${d}_on_a.bin) bytes=$(stat -c %s ${d}_off.bin 2>/dev/null)"
  echo "dump $r"; echo "$r" >> $OUT/dumps.txt
  sha256sum ${d}_*.bin >> $OUT/dumps.sha256 2>/dev/null
  rm -f ${d}_*.bin
}
dd ar_x5 $BIG/ar_x5.gguf --len 8 --gen 8   # ctx 16
dd ar_x5_q8_0 $M/ar3/ar_x5_q8_0.gguf --len 8 --gen 8
dd state4 $M/state4/state4.gguf
dd state4_q8_0 $M/state4/state4_q8_0.gguf
dd mor3 $M/mor/mor3.gguf --len 8 --gen 8
dd mor3_q8_0 $M/mor/mor3_q8_0.gguf --len 8 --gen 8
dd mor2p0 $M/mor/mor2p0.gguf --len 8 --gen 8
dd ar3 $M/ar3/ar3.gguf --len 8 --gen 8
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do dd rec_$m $M/recursion/$m.gguf --len 8 --gen 8; done
dd syn_mistral_swa $M/synth/mistral_swa.gguf
dd syn_gptneox $M/synth/gptneox.gguf
dd syn_gpt2 $M/synth/gpt2.gguf
dd syn_mamba $M/synth/mamba.gguf
dd syn_gemma3 $M/synth/gemma3.gguf
DD_ENV=EMBER_IR=$M/synth/gemma3.per_layer_freq_scale_0.25.ir.json dd syn_gemma3_freq_scale $M/synth/gemma3.gguf
DD_ENV=EMBER_IR=$M/synth/gemma3.weight_offset_0.25.ir.json dd syn_gemma3_weight_offset $M/synth/gemma3.gguf
DD_ENV=EMBER_NOTHING=1
dd gpt2_tiny $M/gpt2_tiny/gpt2_tiny.gguf
# Everything but the third-party dumps, packed now (the first session's VM was lost during them).
(cd /content && rm -f out1.zip && python3 -c "import shutil; shutil.make_archive('/content/out1', 'zip', '/content', 'out')" && touch /content/OUT1_READY && echo "packed out1 $(du -sh out1.zip | cut -f1)")
for i in $(seq 1 120); do [ -e /content/TP_DONE ] && break; sleep 10; done
echo "third-party fetch: $(tail -2 $OUT/thirdparty_fetch.log | tr '\n' ' ')"
TP=/content/tp
sha256sum $TP/qwen3/Qwen3-0.6B-Q8_0.gguf $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf $TP/pythia-160m/pythia-160m-f32.gguf $TP/gemma-3-270m-it/gemma-3-270m-it-f32.gguf > $OUT/thirdparty.sha256 2>/dev/null
dd qwen3_06b $TP/qwen3/Qwen3-0.6B-Q8_0.gguf
dd pythia_160m $TP/pythia-160m/pythia-160m-f32.gguf
dd gemma3_270m $TP/gemma-3-270m-it/gemma-3-270m-it-f32.gguf
dd qwen25_05b $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf
dd gemma3_270m_long $TP/gemma-3-270m-it/gemma-3-270m-it-f32.gguf --ids /content/ids/gemma3_270m_pastwin.ids.json
dd qwen25_long $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf --ids /content/ids/qwen25_long.ids.json
dd qwen3_long $TP/qwen3/Qwen3-0.6B-Q8_0.gguf --ids /content/ids/qwen3_long.ids.json

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
