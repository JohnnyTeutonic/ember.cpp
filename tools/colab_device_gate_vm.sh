#!/usr/bin/env bash
# VM side of a device-gate session: ember's IR interpreter on CUDA against its CPU interpreter for the
# third-party families, on a Colab L4 (local GPU work is off since the WSL service faults of 22 Oct
# 2026). Launched detached by tools/colab_device_gate_drive.sh; never run locally.
#
#   bash colab_device_gate_vm.sh <ember_commit> <paperkiln_commit> [only]
#
# only = pythia: Pythia-160m's conversion, short gate and float64 tie check alone.
#
# Inputs in /content: ember.tgz, paperkiln.tgz (its tools/), ids/*.ids.json (tools/chat_prompt_ids.py:
# chat prompts past a 512-token window), this script. The checkpoints are fetched here, never
# uploaded: Qwen3-0.6B Q8_0 and Qwen2.5-0.5B-Instruct f16 GGUFs from their Qwen repositories,
# Pythia-160m and Gemma-3-270m-it converted to f32 by llama.cpp b6000's convert_hf_to_gguf.py, the
# way paperkiln converted the local copies; each file's sha256 is recorded beside the local one.
# Outputs: /content/gate_out.zip, then /content/GATE_DONE (or /content/GATE_FAILED with the reason).
# Log: /content/gate_vm.log.
set -uo pipefail
cd /content
exec > /content/gate_vm.log 2>&1
EMBER_COMMIT=${1:?ember commit}; PK_COMMIT=${2:?paperkiln commit}; ONLY=${3:-}
OUT=/content/out; M=/content/models
mkdir -p "$OUT" "$M" /content/src
fail() { echo "FAILED: $*"; echo "$*" > /content/GATE_FAILED; exit 1; }
echo "=== device gate vm $(date -u) ember=$EMBER_COMMIT paperkiln=$PK_COMMIT ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1

echo "--- deps + build $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
# transformers as paperkiln converted the local copies (4.57.0): newer releases rename GPT-NeoX's
# rotary_pct as they load the config, which b6000's convert reads (KeyError on 22 Oct 2026).
pip install -q gguf sentencepiece "transformers==4.57.0" > /dev/null 2>&1 || fail "pip gguf, transformers"
tar xzf ember.tgz -C /content/src && tar xzf paperkiln.tgz -C /content/src || fail "untar"
git clone -q --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp /content/llamacpp_b6000 || fail "llama.cpp b6000"
mkdir -p /content/build && cd /content/build
CMAKE_CUDA_ARCHITECTURES=89 cmake /content/src/ember -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=ON > cmake.log 2>&1 || fail "cmake"
grep -q "CUDA toolkit and cuBLAS found" cmake.log || fail "ember configured without CUDA"
make ir_device_gate ir_logits -j"$(nproc)" > make.log 2>&1 || { tail -30 make.log; fail "ember build"; }
cd /content
GATE=/content/build/ir_device_gate

echo "--- checkpoints $(date -u +%T)"
python3 - "$ONLY" <<'PY' || fail "downloads"
from huggingface_hub import hf_hub_download, snapshot_download
import sys
only = sys.argv[1] if len(sys.argv) > 1 else ""
if not only:
    hf_hub_download("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", local_dir="/content/models/qwen3")
    hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf", local_dir="/content/models/qwen25")
# The revision the local copy came from (transformers is pinned above for the same reason).
snapshot_download("EleutherAI/pythia-160m", revision="50f5173d932e8e61f858120bcb800b97af589f46",
                  local_dir="/content/models/pythia-160m", allow_patterns=["*.json", "*.safetensors", "*.txt"])
# Google's repository needs a licence click; unsloth's copy of the same release does not.
if not only:
    snapshot_download("unsloth/gemma-3-270m-it", local_dir="/content/models/gemma-3-270m-it", allow_patterns=["*.json", "*.safetensors", "*.model"])
print("downloads done")
PY
# The local copies' folder and file names, so the files can come out byte-identical.
MODELS="pythia-160m gemma-3-270m-it"; [ "$ONLY" = pythia ] && MODELS=pythia-160m
for m in $MODELS; do
  # A failed conversion drops that model's gates only; the others still run.
  python3 /content/llamacpp_b6000/convert_hf_to_gguf.py $M/$m --outtype f32 --outfile $M/$m/$m-f32.gguf > $OUT/convert_$m.log 2>&1 \
    || { echo "convert $m failed: $(tail -1 $OUT/convert_$m.log)"; rm -f $M/$m/$m-f32.gguf; }
done
QWEN3=$M/qwen3/Qwen3-0.6B-Q8_0.gguf; QWEN25=$M/qwen25/qwen2.5-0.5b-instruct-fp16.gguf
PYTHIA=$M/pythia-160m/pythia-160m-f32.gguf; GEMMA=$M/gemma-3-270m-it/gemma-3-270m-it-f32.gguf
sha256sum $QWEN3 $QWEN25 $PYTHIA $GEMMA > $OUT/models.sha256 2>/dev/null
cat >> $OUT/models.sha256 <<'EOF'
# local copies (C:/ml_artifacts/thirdparty) for comparison:
# 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031  Qwen3-0.6B-Q8_0.gguf
# 8e0ae260...  qwen2.5-0.5b-instruct-fp16.gguf (prefix as recorded)
# 97522db7c8b426c8ae91ee23697a7ea9c56fa51bf05736b62824476daabf08f3  pythia-160m-f32.gguf
# 0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa  gemma-3-270m-it-f32.gguf
EOF
cat $OUT/models.sha256

run() {  # name gguf args...: one gate, its receipt in $OUT/name.json
  local name=$1 g=$2; shift 2
  [ -s "$g" ] || { echo "--- $name skipped: no $g"; return; }
  echo "--- $name $(date -u +%T)"
  $GATE "$g" "$@" > $OUT/$name.json 2> $OUT/$name.log
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('rows',None); print(json.dumps(d))" $OUT/$name.json \
    || echo "$name: no receipt ($(tail -2 $OUT/$name.log))"
}
[ -z "$ONLY" ] && run qwen3_06b $QWEN3 --n 40 --len 24 --gen 32
run pythia_160m_short $PYTHIA --n 40 --len 24 --gen 32
[ -z "$ONLY" ] && run gemma3_270m_long $GEMMA --ids /content/ids/gemma3_270m_pastwin.ids.json --gen 32
[ -z "$ONLY" ] && run qwen25_long $QWEN25 --ids /content/ids/qwen25_long.ids.json --gen 32
[ -z "$ONLY" ] && run qwen3_long $QWEN3 --ids /content/ids/qwen3_long.ids.json --gen 32

echo "--- Pythia flips against float64 $(date -u +%T)"
[ -s $OUT/pythia_160m_short.json ] && python3 /content/src/ember/tools/ir_float64_tie.py --gguf $PYTHIA --paperkiln /content/src/microtorch \
  --gate $OUT/pythia_160m_short.json > $OUT/pythia_flips_float64.json 2> $OUT/pythia_flips_float64.log
cat $OUT/pythia_flips_float64.json

echo "--- Gemma past the window: ember CPU vs numpy and HF $(date -u +%T)"
cd /content/src/ember
[ -z "$ONLY" ] && [ -s $GEMMA ] && python3 tools/thirdparty_ember_check.py --gguf $GEMMA --paperkiln /content/src/microtorch --bin /content/build/ir_logits \
  --work $OUT/gemma3_270m_pastwin_cpu --ids /content/ids/gemma3_270m_pastwin.ids.json 2>&1 | grep -vE '^\{"i": ' | tail -16
cd /content

echo "--- package $(date -u +%T)"
python3 -c "import shutil; shutil.make_archive('/content/gate_out', 'zip', '/content/out')"
echo "=== done $(date -u) ==="
touch /content/GATE_DONE
