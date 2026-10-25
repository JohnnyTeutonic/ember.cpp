#!/usr/bin/env bash
# VM side of the k-quant CUDA gate (ember.cpp, 25 Oct 2026). Launched detached by drive.sh.
# Inputs in /content: ember_kq.tgz (the gate's ember tree), COMMITS, small.tgz (gate.py, the 40 prompts,
# the CPU receipts' reply files, the mtstudio receipt, SHA256SUMS), qm.part* / qs.part* (the two k-quant
# GGUFs), BIG_SHA256SUMS. One CUDA build serves both columns: GPU_LAYERS=0 is the CPU path, -1 the device.
# Legs: kernel test (kquant_test on random blocks, and every tensor of both files against gguf-py b6000's
# dequantised F32); first-token logits on the CPU and on CUDA (llama path, and the IR interpreter on CUDA)
# for both files and their gguf-py references; greedy replies on CUDA for all four.
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; W=/content/work
mkdir -p "$OUT" "$W"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
echo "=== vm $(date -u) $(cat /content/COMMITS) ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cat qm.part* > $W/pk_base_v1_q4_k_m.gguf && cat qs.part* > $W/pk_base_v1_q4_k_s.gguf || fail "reassemble"
(cd $W && sha256sum pk_base_v1_q4_k_m.gguf pk_base_v1_q4_k_s.gguf) > $OUT/big.sha256
diff <(awk '{print $1}' $OUT/big.sha256) <(awk '{print $1}' BIG_SHA256SUMS) || fail "big sha256"
rm -f qm.part* qs.part*

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
git clone -q --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp /content/llamacpp_b6000 || fail "clone b6000"
GP=/content/llamacpp_b6000/gguf-py

echo "--- build $(date -u +%T)"
mkdir -p /content/ember && tar xzf ember_kq.tgz -C /content/ember --strip-components=1 || fail "untar ember"
find /content/ember -name "*.sh" -exec sed -i "s/\r$//" {} +
t0=$(date +%s)
cmake -S /content/ember -B /content/b -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || { tail -20 $OUT/cmake.log; fail "cmake"; }
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log || fail "configured without CUDA"
make -C /content/b -j"$(nproc)" tinyllama_server kquant_test > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built in $(( $(date +%s) - t0 )) s"

echo "--- kernel test and references $(date -u +%T)"
PYTHONPATH=$GP python3 /content/ember/tools/kquant_blocks.py $W/blocks.bin 4096 20261025 > $OUT/kquant_blocks.txt 2>&1 || fail "blocks"
/content/b/kquant_test blocks $W/blocks.bin > $OUT/kquant_test_blocks.txt 2>&1; tail -1 $OUT/kquant_test_blocks.txt
for v in q4_k_m q4_k_s; do
  PYTHONPATH=$GP python3 /content/ember/tools/quantize_gguf.py $W/pk_base_v1_$v.gguf $W/pk_base_v1_${v}_ref.gguf --dequantize >> $OUT/refs.txt 2>&1 || fail "ref $v"
  (cd $W && sha256sum pk_base_v1_${v}_ref.gguf) >> $OUT/refs.txt
  /content/b/kquant_test gguf $W/pk_base_v1_$v.gguf $W/pk_base_v1_${v}_ref.gguf > $OUT/kquant_test_$v.txt 2>&1; tail -1 $OUT/kquant_test_$v.txt
done

echo "--- logits $(date -u +%T)"
export EMBER_BIN=/content/b/tinyllama_server KQ_WORK=$W KQ_OUT=$OUT KQ_PROMPTS=/content/small/prompts.jsonl
G="python3 /content/small/kquant_2026-10-25/gate.py"
for v in q4_k_m q4_k_s; do
  GPU_LAYERS=0 $G probe cpu_$v $W/pk_base_v1_$v.gguf || fail "probe cpu $v"
  GPU_LAYERS=-1 $G probe cuda_$v $W/pk_base_v1_$v.gguf || fail "probe cuda $v"
  GPU_LAYERS=-1 $G probe cuda_${v}_ref $W/pk_base_v1_${v}_ref.gguf || fail "probe cuda ref $v"
  GPU_LAYERS=-1 $G probe cuda_ir_$v $W/pk_base_v1_$v.gguf EMBER_INTERPRETER=1 || fail "probe cuda ir $v"
  GPU_LAYERS=-1 $G probe cuda_ir_${v}_ref $W/pk_base_v1_${v}_ref.gguf EMBER_INTERPRETER=1 || fail "probe cuda ir ref $v"
done
grep -h -iE "interpreter|device|gpu layers" $W/cuda_ir_q4_k_m_server.log | head -5 > $OUT/cuda_ir_path.txt
echo "--- replies $(date -u +%T)"
for v in q4_k_m q4_k_s; do
  GPU_LAYERS=-1 $G replies cuda_$v $W/pk_base_v1_$v.gguf || fail "replies cuda $v"
  GPU_LAYERS=-1 $G replies cuda_${v}_ref $W/pk_base_v1_${v}_ref.gguf || fail "replies cuda ref $v"
done
cp /content/small/receipts/*_replies.jsonl $OUT/ 2>/dev/null
$G summary > $OUT/summary.txt 2>&1 || fail "summary"
cat $OUT/summary.txt
cp $W/*_server.log $OUT/ 2>/dev/null
gzip -c $W/cpu_q4_k_m.logits > $OUT/cpu_q4_k_m.logits.gz; gzip -c $W/cpu_q4_k_s.logits > $OUT/cpu_q4_k_s.logits.gz
(cd /content && zip -qr out.zip out) || fail "zip"
echo "=== done $(date -u) ==="
touch /content/VM_DONE
