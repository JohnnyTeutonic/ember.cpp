#!/usr/bin/env bash
# Fixed diagnostic campaign; any scientific negative is retained, never retried for pass.
set -euo pipefail
cd /content
DEST=${1:?job output directory}; OUT="$DEST/raw"
SRC=/content/e9_diag_src; NORMAL=/content/e9_diag_normal; DIAG=/content/e9_diag_build
mkdir -p "$OUT" "$SRC"
exec > "$DEST/vm.log" 2>&1
pack() {
 flock /content/e9_diag_pack.lock python3 - "$OUT" "$DEST/out.zip" <<'PY'
from pathlib import Path
import os,sys,zipfile
root=Path(sys.argv[1]);target=Path(sys.argv[2]);tmp=target.with_suffix('.partial')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,strict_timestamps=False) as archive:
 for p in sorted(root.rglob('*')):
  if p.is_file():archive.write(p,p.relative_to(root).as_posix())
os.replace(tmp,target)
PY
}
packer=''
finish() {
 rc=$?;trap - EXIT
 if [ -n "$packer" ]; then kill "$packer" 2>/dev/null || true;wait "$packer" 2>/dev/null || true;fi
 cp "$DEST/vm.log" "$OUT/vm.log"
 if ! pack; then rc=1; fi
 exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT TERM
sha256sum -c INPUT_SHA256SUMS > "$OUT/input_verification.txt"
cp INPUT_SHA256SUMS COMMITS "$OUT/"
tar --no-same-owner --no-same-permissions -xzf ember.tgz -C "$SRC" --strip-components=1
EXP="$SRC/experiments/kv_quant_r2_2026-10-28"
{ date -u;nvidia-smi;nvcc --version;python3 --version; } > "$OUT/hardware.txt"
apt-get update -qq > "$OUT/apt.log" 2>&1
apt-get install -y -q libboost-regex-dev >> "$OUT/apt.log" 2>&1
python3 -m pip install 'numpy==1.26.4' > "$OUT/pip.log" 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 EMBER_IR_BUILD=1 EMBER_IR_CTX=8448
unset EMBER_IR EMBER_IR_LAST EMBER_KV EMBER_DETERMINISTIC EMBER_KV_TRACE_DIR EMBER_DECODE_LINEAR EMBER_DECODE_OPT EMBER_DECODE_PICK
python3 - "$OUT" <<'PY'
import importlib.metadata,json,pathlib,sys,torch
assert torch.__version__=='2.4.1+cu124'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert importlib.metadata.version('numpy')=='1.26.4'
p=json.load(open('/content/chat/provenance.json'))
assert p['image_digest']=='sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5'
pathlib.Path(sys.argv[1],'runtime.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':importlib.metadata.version('numpy'),'provenance':p},indent=1)+'\n')
PY
# No credentials; immutable repository revision plus exact canonical byte hash.
python3 - "$OUT" <<'PY'
import hashlib,json,pathlib,sys,urllib.request
p=pathlib.Path('/content/e9_qwen3.gguf')
url='https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf'
expected='9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031'
def sha():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
if not p.exists():
 tmp=p.with_suffix('.partial');urllib.request.urlretrieve(url,tmp);tmp.replace(p)
actual=sha()
pathlib.Path(sys.argv[1],'model_verification.json').write_text(json.dumps({'url':url,'bytes':p.stat().st_size,'sha256':actual},indent=1)+'\n')
assert actual==expected,'canonical GGUF mismatch; no forward allowed'
PY
(while sleep 180;do pack || true;done) &
packer=$!
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
J=$(nproc); [ "$J" -le 8 ] || J=8
cmake -S "$SRC" -B "$NORMAL" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CC" -DEMBER_KV_DIAGNOSTICS=OFF > "$OUT/cmake_normal.log" 2>&1
grep -q 'CUDA toolkit and cuBLAS found' "$OUT/cmake_normal.log"
cmake --build "$NORMAL" -j"$J" --target ir_device_gate > "$OUT/build_normal.log" 2>&1
# Compile original and patched graph_ir.cpp at the identical pathname with identical flags.
# finally restores frozen bytes even when the compile or comparison fails.
python3 "$EXP/f16_normal_object_proof.py" --source "$SRC" --build "$NORMAL" --original /content/graph_ir_original.cpp --output "$OUT/normal_object_proof.json"
cmake -S "$SRC" -B "$DIAG" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CC" -DEMBER_KV_DIAGNOSTICS=ON > "$OUT/cmake_diagnostic.log" 2>&1
grep -q 'CUDA toolkit and cuBLAS found' "$OUT/cmake_diagnostic.log"
cmake --build "$DIAG" -j"$J" --target kv_f16_trace kv_f16_component_probe kv_f16_seeded_ids > "$OUT/build_diagnostic.log" 2>&1
mkdir -p "$OUT/binaries"
cp "$NORMAL/ir_device_gate" "$DIAG/kv_f16_trace" "$DIAG/kv_f16_component_probe" "$DIAG/kv_f16_seeded_ids" "$OUT/binaries/"
cp "$EXP/receipts/e9_f16_diagnosis_20261031/ids_seed0.json" "$OUT/ids_seed0.json"
cp "$EXP/run_f16_diagnosis.py" "$EXP/f16_trace_analysis.py" "$OUT/"
python3 "$EXP/run_f16_diagnosis.py" --model /content/e9_qwen3.gguf --normal "$OUT/binaries/ir_device_gate" --trace "$OUT/binaries/kv_f16_trace" --component "$OUT/binaries/kv_f16_component_probe" --seeded-ids "$OUT/binaries/kv_f16_seeded_ids" --ids "$OUT/ids_seed0.json" --out "$OUT/diagnosis" > "$OUT/controller.log" 2>&1
cp "$OUT/diagnosis/diagnosis_report.json" "$DEST/diagnosis_report.json"
echo done > "$DEST/VM_DONE"