#!/usr/bin/env bash
# VM side of the CUDA state-token / head_dim session (ember.cpp, 25 Oct 2026). Launched detached by drive.sh.
#   bash vm.sh
# Inputs in /content: ember_new.tgz (this work's commit), ember_head.tgz (ember main before it), COMMITS,
# small.tgz (gate models, state_device_gate inputs, ids.tgz, prompts.jsonl, sp_compare.py, SHA256SUMS),
# k4.part* (chat_state_k4.gguf), l32.part* (tc_sft_L32_r1.gguf), BIG_SHA256SUMS.
# Legs: (a) kernel tests (new; the prefill and rmsnorm tests also against the old cuda_kernels.cu, outputs
# dumped for a bit comparison); (b) state_device_gate on state4, state4 Q8_0, ar3 and the synthetic k=4 model,
# ir_device_gate on state4 and ar3; (c) ir_device_gate (new) on the synthetic families and gpt2_tiny's expr IR,
# and ir_device_dump (new and old build) on those and the third-party checkpoints for a bit comparison;
# (d) SFT L32 (d 192) through the new tinyllama_server on CUDA, 40 tinychat prompts.
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
cat k4.part* > $M/synth/chat_state_k4.gguf && cat l32.part* > /content/tc_sft_L32_r1.gguf || fail "reassemble"
(cd $M/synth && sha256sum chat_state_k4.gguf; cd /content && sha256sum tc_sft_L32_r1.gguf) > $OUT/big.sha256
diff <(awk '{print $1}' $OUT/big.sha256) <(awk '{print $1}' BIG_SHA256SUMS) || fail "big sha256"
rm -f k4.part* l32.part*

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
# The old tree gets the dump tool (public interpreter API only) so the two builds dump the same way.
cp /content/new/tools/ir_device_dump.cpp /content/head/tools/
printf '\nadd_executable(ir_device_dump tools/ir_device_dump.cpp)\ntarget_link_libraries(ir_device_dump PRIVATE tinyllama_core)\n' >> /content/head/CMakeLists.txt
for t in new head; do
  t0=$(date +%s)
  cmake -S /content/$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$t.log 2>&1 || { tail -20 $OUT/cmake_$t.log; fail "cmake $t"; }
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
  if [ $t = new ]; then targets="ir_device_gate state_device_gate ir_device_dump tinyllama_server"; else targets="ir_device_dump"; fi
  make -C /content/b_$t -j"$(nproc)" $targets > $OUT/make_$t.log 2>&1 || { tail -30 $OUT/make_$t.log; fail "build $t"; }
  echo "built $t in $(( $(date +%s) - t0 )) s"
done
BN=/content/b_new; BH=/content/b_head

echo "--- (a) kernel tests $(date -u +%T)"
for t in new head; do
  (cd /content/$t && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. /content/new/tools/prefill_attention_test.cu cuda_kernels.cu logger.cpp -lcublas -o /content/pa_$t > $OUT/nvcc_pa_$t.log 2>&1) \
    && /content/pa_$t $OUT/prefill_out_$t.f32 > $OUT/prefill_attention_test_$t.txt 2>&1
  echo "prefill_attention_test $t: $(tail -1 $OUT/prefill_attention_test_$t.txt 2>/dev/null || tail -2 $OUT/nvcc_pa_$t.log)"
  (cd /content/$t && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. tools/rmsnorm_batch_test.cu cuda_kernels.cu logger.cpp -lcublas -o /content/rt_$t > $OUT/nvcc_rt_$t.log 2>&1) \
    && /content/rt_$t $OUT/rmsnorm_out_$t.f32 > $OUT/rmsnorm_batch_test_$t.txt 2>&1
  echo "rmsnorm_batch_test $t: $(tail -1 $OUT/rmsnorm_batch_test_$t.txt 2>/dev/null || tail -2 $OUT/nvcc_rt_$t.log)"
done
cmp $OUT/rmsnorm_out_new.f32 $OUT/rmsnorm_out_head.f32 && echo "rmsnorm outputs: new = head bit for bit"
(cd /content/new && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. tools/turn_state_attention_test.cu graph_ir_cuda.cu graph_ir_expr.cpp -o /content/tsat > $OUT/nvcc_tsat.log 2>&1) \
  && /content/tsat > $OUT/turn_state_attention_test.txt 2>&1
echo "turn_state_attention_test: $(tail -1 $OUT/turn_state_attention_test.txt 2>/dev/null || tail -2 $OUT/nvcc_tsat.log)"
(cd /content/new && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. tools/decode_attention_test.cu cuda_kernels.cu logger.cpp -lcublas -o /content/dat > $OUT/nvcc_dat.log 2>&1) \
  && /content/dat > $OUT/decode_attention_test.txt 2>&1
echo "decode_attention_test (new, unchanged kernels): $(tail -1 $OUT/decode_attention_test.txt 2>/dev/null)"

echo "--- (b) state tokens and llama_attnres, device vs CPU $(date -u +%T)"
sg() {  # name gguf input
  $BN/state_device_gate $2 < $3 > $OUT/sdg_$1.json 2> $OUT/sdg_$1.log
  echo "state_device_gate $1 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['summary']), 'pass', d['pass'])" $OUT/sdg_$1.json 2>/dev/null || tail -2 $OUT/sdg_$1.log)"
}
sg state4 $M/state4/state4.gguf $I/state4_in.json
sg state4_q8_0 $M/state4/state4_q8_0.gguf $I/state4q8_in.json
sg ar3 $M/ar3/ar3.gguf $I/ar3_in.json
sg synth_k4 $M/synth/chat_state_k4.gguf $I/synth_in.json
dg() {  # name gguf len gen [env assignment]: ir_device_gate (new), 40 prompts of len ids, gen greedy
  local name=$1 g=$2 len=$3 gen=$4 e=${5:-EMBER_NOTHING=1}
  env "$e" $BN/ir_device_gate $g --n 40 --len $len --gen $gen > $OUT/dg_$name.json 2> $OUT/dg_$name.log
  echo "ir_device_gate $name rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('rows',None); print(d['greedy_identical'], d['top1_agree'], d['worst_prefill_rel'], d['worst_decode_rel'], 'pass', d['pass'])" $OUT/dg_$name.json 2>/dev/null || tail -2 $OUT/dg_$name.log)"
}
dg state4 $M/state4/state4.gguf 24 32
dg ar3 $M/ar3/ar3.gguf 8 8  # ctx 16

echo "--- (c) the device-gate set: ir_device_gate (new) on the synthetic families $(date -u +%T)"
dg syn_mistral_swa $M/synth/mistral_swa.gguf 24 32
dg syn_gptneox $M/synth/gptneox.gguf 24 32
dg syn_gpt2 $M/synth/gpt2.gguf 24 32
dg syn_mamba $M/synth/mamba.gguf 24 32
dg syn_gemma3 $M/synth/gemma3.gguf 24 32
dg syn_gemma3_freq_scale $M/synth/gemma3.gguf 24 32 EMBER_IR=$M/synth/gemma3.per_layer_freq_scale_0.25.ir.json
dg syn_gemma3_weight_offset $M/synth/gemma3.gguf 24 32 EMBER_IR=$M/synth/gemma3.weight_offset_0.25.ir.json
dg gpt2_tiny_expr $M/gpt2_tiny/gpt2_tiny.gguf 24 32 EMBER_IR=$M/gpt2_tiny/gpt2_tiny_expr.ir.json

echo "--- (d) SFT L32 (d 192) on CUDA through tinyllama_server, 40 tinychat prompts $(date -u +%T)"
mkdir -p /content/run/www; cd /content/run; ln -sf /dev/null debugging.log
EMBER_GEMM_TF32=0 $BN/tinyllama_server /content/tc_sft_L32_r1.gguf 8411 127.0.0.1 /content/run/www -1 > /dev/null 2> $OUT/l32_server.log &
SP=$!
for i in $(seq 1 180); do curl -s -m 2 127.0.0.1:8411/health > /dev/null && break; sleep 1; done
for V in spot p1; do
  python3 /content/small/sp_compare.py collect http://127.0.0.1:8411 /content/small/prompts.jsonl 40 48 $V $OUT/l32_new_cuda_$V.jsonl || echo "collect $V failed"
done
kill $SP; wait $SP 2>/dev/null
cd /content
echo "l32 collected: $(wc -l < $OUT/l32_new_cuda_spot.jsonl 2>/dev/null) + $(wc -l < $OUT/l32_new_cuda_p1.jsonl 2>/dev/null)"

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
dd ar3 $M/ar3/ar3.gguf --len 8 --gen 8
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
