#!/usr/bin/env bash
# VM side of the Mixture-of-Recursions CUDA session (ember.cpp, 25 Oct 2026). Launched detached by drive.sh.
#   bash vm.sh
# Inputs in /content: ember_new.tgz (this work's commit), ember_head.tgz (ember main before it), COMMITS,
# small.tgz (gate models, mor_device_gate inputs, ids.tgz, SHA256SUMS), tc.part* (the tiny-chat MoR3 model),
# BIG_SHA256SUMS.
# Legs: (a) tools/mor_kernels_test.cu; (b) mor_device_gate on mor3, mor2p0 (F32 and Q8_0) and the tiny-chat MoR3
# model (with timings), ir_device_gate on the three; (c) ir_device_gate (new) on the synthetic families,
# gpt2_tiny's expr IR, state4, ar3 and the third-party checkpoints, and ir_device_dump (new and old build) on
# those for a bit comparison.
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; M=/content/small/models; I=/content/small/inputs
mkdir -p "$OUT"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r NEWC HEADC < /content/COMMITS
echo "=== vm $(date -u) new=$NEWC head=$HEADC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC"
tar xzf small.tgz && tar xzf small/ids.tgz -C /content || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
mkdir -p $M/tc && cat tc.part* > $M/tc/tc_e1_mor_MoR3_r1.gguf || fail "reassemble"
(cd $M/tc && sha256sum tc_e1_mor_MoR3_r1.gguf) > $OUT/big.sha256
diff <(awk '{print $1}' $OUT/big.sha256) <(awk '{print $1}' BIG_SHA256SUMS) || fail "big sha256"
rm -f tc.part*

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
echo "--- checkpoints in the background $(date -u +%T)"
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
  cmake -S /content/$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$t.log 2>&1 || { tail -20 $OUT/cmake_$t.log; fail "cmake $t"; }
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
  if [ $t = new ]; then targets="ir_device_gate mor_device_gate ir_device_dump"; else targets="ir_device_dump"; fi
  make -C /content/b_$t -j"$(nproc)" $targets > $OUT/make_$t.log 2>&1 || { tail -30 $OUT/make_$t.log; fail "build $t"; }
  echo "built $t in $(( $(date +%s) - t0 )) s"
done
BN=/content/b_new; BH=/content/b_head

echo "--- (a) kernel test $(date -u +%T)"
(cd /content/new && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. -Xcompiler -ffp-contract=off tools/mor_kernels_test.cu graph_ir_cuda.cu graph_ir_expr.cpp graph_ir_ops.cpp -o /content/mkt > $OUT/nvcc_mkt.log 2>&1) \
  && /content/mkt > $OUT/mor_kernels_test.txt 2>&1
echo "mor_kernels_test: $(tail -1 $OUT/mor_kernels_test.txt 2>/dev/null || tail -2 $OUT/nvcc_mkt.log)"

echo "--- (b) Mixture-of-Recursions, device vs CPU $(date -u +%T)"
mg() {  # name gguf input [args]
  local name=$1 g=$2 in=$3; shift 3
  $BN/mor_device_gate $g "$@" < $in > $OUT/mdg_$name.json 2> $OUT/mdg_$name.log
  echo "mor_device_gate $name rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['summary']), json.dumps(d.get('bench')), 'pass', d['pass'])" $OUT/mdg_$name.json 2>/dev/null || tail -2 $OUT/mdg_$name.log)"
}
mg mor3 $M/mor/mor3.gguf $I/mor3_in.json
mg mor3_q8_0 $M/mor/mor3_q8_0.gguf $I/mor3_in.json
mg mor2p0 $M/mor/mor2p0.gguf $I/mor2p0_in.json
mg mor2p0_q8_0 $M/mor/mor2p0_q8_0.gguf $I/mor2p0_in.json
mg tc_mor3 $M/tc/tc_e1_mor_MoR3_r1.gguf $I/tc_mor3_in.json --bench 64
dg() {  # name gguf len gen [env assignment] [n]: ir_device_gate (new), n prompts of len ids, gen greedy
  local name=$1 g=$2 len=$3 gen=$4 e=${5:-EMBER_NOTHING=1} n=${6:-40}
  [ -s "$g" ] || { echo "ir_device_gate $name skipped: no $g"; return; }
  env "$e" $BN/ir_device_gate $g --n $n --len $len --gen $gen > $OUT/dg_$name.json 2> $OUT/dg_$name.log
  echo "ir_device_gate $name rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('rows',None); print(d['greedy_identical'], d['top1_agree'], d['worst_prefill_rel'], d['worst_decode_rel'], 'pass', d['pass'])" $OUT/dg_$name.json 2>/dev/null || tail -2 $OUT/dg_$name.log)"
}
dg mor3 $M/mor/mor3.gguf 8 8  # ctx 16
dg mor2p0 $M/mor/mor2p0.gguf 8 8
dg tc_mor3 $M/tc/tc_e1_mor_MoR3_r1.gguf 24 32 EMBER_NOTHING=1 10

echo "--- (c) the device-gate set: ir_device_gate (new) $(date -u +%T)"
dg syn_mistral_swa $M/synth/mistral_swa.gguf 24 32
dg syn_gptneox $M/synth/gptneox.gguf 24 32
dg syn_gpt2 $M/synth/gpt2.gguf 24 32
dg syn_mamba $M/synth/mamba.gguf 24 32
dg syn_gemma3 $M/synth/gemma3.gguf 24 32
dg syn_gemma3_freq_scale $M/synth/gemma3.gguf 24 32 EMBER_IR=$M/synth/gemma3.per_layer_freq_scale_0.25.ir.json
dg syn_gemma3_weight_offset $M/synth/gemma3.gguf 24 32 EMBER_IR=$M/synth/gemma3.weight_offset_0.25.ir.json
dg gpt2_tiny_expr $M/gpt2_tiny/gpt2_tiny.gguf 24 32 EMBER_IR=$M/gpt2_tiny/gpt2_tiny_expr.ir.json
dg state4 $M/state4/state4.gguf 24 32
dg ar3 $M/ar3/ar3.gguf 8 8  # ctx 16
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do dg rec_$m $M/recursion/$m.gguf 8 8; done  # weight_index, ctx 16

echo "--- (c) device results new vs old build, bit for bit $(date -u +%T)"
DD_ENV=EMBER_NOTHING=1
dd() {  # name gguf args... (an EMBER_IR side file through DD_ENV)
  local name=$1 g=$2; shift 2
  [ -s "$g" ] || { echo "dump $name skipped: no $g"; return; }
  for t in new head; do
    env "$DD_ENV" /content/b_$t/ir_device_dump $g $OUT/dump_${name}_$t.bin "$@" > /dev/null 2> $OUT/dump_${name}_$t.log || echo "dump $name $t failed: $(tail -1 $OUT/dump_${name}_$t.log)"
  done
  if cmp -s $OUT/dump_${name}_new.bin $OUT/dump_${name}_head.bin; then echo "dump $name: identical ($(stat -c %s $OUT/dump_${name}_new.bin) bytes)"; echo "$name identical" >> $OUT/dumps.txt
  else echo "dump $name: DIFFERS"; echo "$name differs" >> $OUT/dumps.txt; fi
  sha256sum $OUT/dump_${name}_new.bin $OUT/dump_${name}_head.bin >> $OUT/dumps.sha256 2>/dev/null
  rm -f $OUT/dump_${name}_new.bin $OUT/dump_${name}_head.bin
}
dd syn_mistral_swa $M/synth/mistral_swa.gguf
dd syn_gptneox $M/synth/gptneox.gguf
dd syn_gpt2 $M/synth/gpt2.gguf
dd syn_mamba $M/synth/mamba.gguf
dd syn_gemma3 $M/synth/gemma3.gguf
DD_ENV=EMBER_IR=$M/synth/gemma3.per_layer_freq_scale_0.25.ir.json dd syn_gemma3_freq_scale $M/synth/gemma3.gguf
DD_ENV=EMBER_IR=$M/synth/gemma3.weight_offset_0.25.ir.json dd syn_gemma3_weight_offset $M/synth/gemma3.gguf
DD_ENV=EMBER_IR=$M/gpt2_tiny/gpt2_tiny_expr.ir.json dd gpt2_tiny_expr $M/gpt2_tiny/gpt2_tiny.gguf
DD_ENV=EMBER_NOTHING=1
dd state4 $M/state4/state4.gguf
dd state4_q8_0 $M/state4/state4_q8_0.gguf
dd ar3 $M/ar3/ar3.gguf --len 8 --gen 8
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do dd rec_$m $M/recursion/$m.gguf --len 8 --gen 8; done
for i in $(seq 1 120); do [ -e /content/TP_DONE ] && break; sleep 10; done
echo "third-party fetch: $(tail -2 $OUT/thirdparty_fetch.log | tr '\n' ' ')"
TP=/content/tp
sha256sum $TP/qwen3/Qwen3-0.6B-Q8_0.gguf $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf $TP/pythia-160m/pythia-160m-f32.gguf $TP/gemma-3-270m-it/gemma-3-270m-it-f32.gguf > $OUT/thirdparty.sha256 2>/dev/null
dg qwen3_06b $TP/qwen3/Qwen3-0.6B-Q8_0.gguf 24 24 EMBER_NOTHING=1 10
dg pythia_160m $TP/pythia-160m/pythia-160m-f32.gguf 24 24 EMBER_NOTHING=1 10
dg gemma3_270m $TP/gemma-3-270m-it/gemma-3-270m-it-f32.gguf 24 24 EMBER_NOTHING=1 10
dg qwen25_05b $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf 24 24 EMBER_NOTHING=1 10
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
