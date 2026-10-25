#!/usr/bin/env bash
# VM side of the /score CUDA gate (ember.cpp, 26 Oct 2026). Launched detached by sc_drive.sh.
# Inputs in /content: ember_new.tgz (this work's tree), small.tgz (sc_client.py, the request sets), pick.part*,
# pkbase.part*, BIG_SHA256SUMS. Pythia-160m and Qwen2.5-0.5B-Instruct come from the hub (Pythia converted to an F32
# GGUF with llama.cpp b6000, as ../determinism_2026-10-26/vm.sh does; their sha256 is checked against the local files).
# Per model, each its own server process at gpu_layers -1: default, EMBER_DETERMINISTIC=1 twice; the gate set and,
# where there is one, the 256-request batch. Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED).
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; BIG=/content/big; TP=/content/tp
mkdir -p "$OUT" "$BIG"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
echo "=== vm $(date -u) ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "gpu $GPU, compute capability $CC, nproc $(nproc)"; echo "$GPU $CC" > $OUT/gpu.txt
tar xzf small.tgz || fail "untar small"
cat pick.part* > $BIG/pick_L16.gguf && cat pkbase.part* > $BIG/pk_base_v1.gguf || fail "reassemble"
(cd $BIG && sha256sum -c /content/BIG_SHA256SUMS) > $OUT/big.sha256 2>&1 || fail "big sha256"
rm -f pick.part* pkbase.part*
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
( pip install -q gguf sentencepiece "transformers==4.57.0" > /dev/null 2>&1
  git clone -q --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp /content/llamacpp_b6000
  python3 - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf", local_dir="/content/tp/qwen25")
snapshot_download("EleutherAI/pythia-160m", revision="50f5173d932e8e61f858120bcb800b97af589f46",
                  local_dir="/content/tp/pythia-160m", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("downloads done")
PY
  python3 /content/llamacpp_b6000/convert_hf_to_gguf.py $TP/pythia-160m --outtype f32 --outfile $TP/pythia-160m/pythia-160m-f32.gguf \
    > $OUT/convert_pythia.log 2>&1 || rm -f $TP/pythia-160m/pythia-160m-f32.gguf
  sha256sum $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf $TP/pythia-160m/pythia-160m-f32.gguf > $OUT/thirdparty.sha256
  touch /content/TP_DONE ) > $OUT/thirdparty_fetch.log 2>&1 &
mkdir -p /content/new && tar xzf ember_new.tgz -C /content/new || fail "untar ember"
t0=$(date +%s)
cmake -S /content/new -B /content/b_new -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || fail "cmake"
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log || fail "configured without CUDA"
make -C /content/b_new -j"$(nproc)" tinyllama_server > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built in $(( $(date +%s) - t0 )) s"; sha256sum /content/b_new/tinyllama_server > $OUT/bin.sha256
BIN=/content/b_new/tinyllama_server
export SC_GPU_LAYERS=-1
port=8600
run() {  # tag gguf reqs... (env from the caller)
  local tag=$1 g=$2; shift 2
  port=$((port + 1))
  timeout 3000 python3 $SM/sc_client.py serve $BIN $g $port $OUT $tag "$@" 2>&1 | grep -v "^\[" | tail -6
}
leg() {  # name gguf reqset
  local n=$1 g=$2 r=$3 extra=""
  [ -f $SM/req/${r}_bench256.jsonl ] && extra=$SM/req/${r}_bench256.jsonl
  run ${n}_cuda_def $g $SM/req/${r}_requests.jsonl $extra --errors --chat
  EMBER_DETERMINISTIC=1 run ${n}_cuda_det1 $g $SM/req/${r}_requests.jsonl $extra
  EMBER_DETERMINISTIC=1 run ${n}_cuda_det2 $g $SM/req/${r}_requests.jsonl $extra
}
echo "--- tuned llama path $(date -u +%T)"
leg pick $BIG/pick_L16.gguf pk
leg pkbase $BIG/pk_base_v1.gguf pk
echo "--- interpreter $(date -u +%T)"
for i in $(seq 1 90); do [ -f /content/TP_DONE ] && break; sleep 20; done
cat $OUT/thirdparty.sha256
[ -s $TP/pythia-160m/pythia-160m-f32.gguf ] && leg pythia $TP/pythia-160m/pythia-160m-f32.gguf pythia
[ -s $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf ] && leg qwen $TP/qwen25/qwen2.5-0.5b-instruct-fp16.gguf qwen
rm -f $OUT/*.server.log.gz; for f in $OUT/*.server.log; do tail -200 "$f" > "$f.tail" && rm -f "$f"; done
(cd /content && zip -qr out.zip out) && touch /content/VM_DONE
echo "=== done $(date -u) ==="
