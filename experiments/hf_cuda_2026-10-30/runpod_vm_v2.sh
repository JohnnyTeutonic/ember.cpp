#!/bin/bash
# Separate Smol-only measurement; canonical byte hash is mandatory before forwards.
set -euo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SRC=/content/src; B=/content/b
mkdir -p "$OUT" "$SRC" /content/converter
pack() { flock /content/hf_cuda_pack.lock python3 -c "import os,shutil; p=shutil.make_archive('/content/hf_cuda_tmp_'+str(os.getpid()),'zip','/content','out'); os.replace(p,'/content/out.zip')"; }
packer=""
finish() { rc=$?; trap - EXIT; if [ -n "$packer" ]; then kill "$packer" 2>/dev/null || true; wait "$packer" 2>/dev/null || true; fi; cp /content/vm.log "$OUT/vm.log"; pack || true; exit "$rc"; }
trap finish EXIT
trap 'exit 130' INT TERM
sha256sum -c INPUT_SHA256SUMS > "$OUT/input_verification.txt"
mkdir -p "$OUT/inputs"
cp INPUT_SHA256SUMS COMMITS ember.tgz converter.tgz smol_manifest.json "$OUT/inputs/"
{ date -u; nvidia-smi; nvcc --version; python3 --version; } > "$OUT/hardware.txt" 2>&1
tar --no-same-owner --no-same-permissions -xzf ember.tgz -C "$SRC" --strip-components=1
tar --no-same-owner --no-same-permissions -xzf converter.tgz -C /content/converter --strip-components=1
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1
# Keep the image's CUDA PyTorch; pin the text-only loader/converter dependencies.
python3 -m pip install 'numpy==1.26.4' 'transformers==4.57.0' 'tokenizers==0.22.1' 'gguf==0.19.0' 'safetensors==0.6.2' 'huggingface-hub==0.35.3' 'sentencepiece==0.2.2' 'accelerate==1.13.0' > "$OUT/pip.log" 2>&1
python3 -m pip list --format=json > "$OUT/packages.json"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
unset EMBER_IR EMBER_KV EMBER_DETERMINISTIC EMBER_DECODE_LINEAR EMBER_DECODE_PICK
(while sleep 120; do pack || true; done) &
packer=$!
python3 - <<'PY'
import hashlib, json, pathlib, torch
from huggingface_hub import snapshot_download
m=json.load(open('/content/smol_manifest.json'))
assert torch.__version__ == '2.4.1+cu124', torch.__version__
provenance=json.load(open('/content/chat/provenance.json'))
assert provenance['image_digest'] == 'sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5'
import importlib.metadata
expected={'numpy':'1.26.4','transformers':'4.57.0','tokenizers':'0.22.1','gguf':'0.19.0','safetensors':'0.6.2','huggingface-hub':'0.35.3','sentencepiece':'0.2.2','accelerate':'1.13.0'}
versions={k:importlib.metadata.version(k) for k in expected}
assert versions == expected, (versions,expected)
pathlib.Path('/content/out/runtime_environment_v2.json').write_text(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,'versions':versions,'image':provenance,'revision':'operational-v2-actual-image-patch-version'},indent=2)+'\n')
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
script=pathlib.Path('/content/converter/convert_hf_to_gguf.py')
assert hashlib.sha256(script.read_bytes()).hexdigest() == m['converter_script_sha256']
folder=pathlib.Path('/content/hf_smollm2')
snapshot_download(m['repo'], revision=m['revision'], local_dir=str(folder), allow_patterns=list(m['files']), token=False)
report={}
for name, expected in m['files'].items():
 p=folder/name; h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''): h.update(block)
 actual={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 assert actual == expected, (name,actual,expected)
 report[name]=actual
pathlib.Path('/content/out/hf_input_verification.json').write_text(json.dumps(report,indent=1)+'\n')
PY
python3 /content/converter/convert_hf_to_gguf.py /content/hf_smollm2 --outtype f32 --outfile /content/SmolLM2-135M-f32.gguf > "$OUT/conversion.log" 2>&1
python3 - <<'PY'
import hashlib,json,pathlib
m=json.load(open('/content/smol_manifest.json'));p=pathlib.Path('/content/SmolLM2-135M-f32.gguf');h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(4<<20),b''):h.update(b)
r={'bytes':p.stat().st_size,'sha256':h.hexdigest(),'expected_sha256':m['gguf_sha256']}
pathlib.Path('/content/out/gguf_reconstruction.json').write_text(json.dumps(r,indent=1)+'\n')
assert r['sha256']==m['gguf_sha256'],'canonical GGUF mismatch; no forward permitted'
PY
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CC" > "$OUT/cmake.log" 2>&1
grep -q 'CUDA toolkit and cuBLAS found' "$OUT/cmake.log"
J=$(nproc); [ "$J" -le 8 ] || J=8
make -C "$B" -j"$J" ir_hf_cuda_bench > "$OUT/make.log" 2>&1
mkdir -p "$OUT/binaries"
cp "$B/ir_hf_cuda_bench" "$OUT/binaries/"
P="$SRC/experiments/hf_cuda_2026-10-30/benchmark.py"
cp "$P" "$OUT/controller.py"
rc=0
python3 "$P" --gguf /content/SmolLM2-135M-f32.gguf --binary "$B/ir_hf_cuda_bench" --out "$OUT/smollm2-135m" --lengths 16 128 512 --steps 32 --cycles 2 || rc=$?
if [ "$rc" -eq 0 ]; then
 python3 "$P" --out "$OUT/smollm2-135m" --analyze > "$OUT/gate_summary.json"
else
 python3 - "$rc" <<'PY'
import json,pathlib,sys
p=pathlib.Path('/content/out/smollm2-135m/HALTED')
pathlib.Path('/content/out/gate_summary.json').write_text(json.dumps({'status':'HALTED','complete':False,'exit_code':int(sys.argv[1]),'reason':p.read_text() if p.exists() else 'controller stopped before HALTED receipt'},indent=1)+'\n')
PY
fi
# A scientific halt is final evidence, never an invitation to rerun for a pass.
echo done > /content/VM_DONE
