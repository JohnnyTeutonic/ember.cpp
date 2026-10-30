#!/bin/bash
# E10.1: both original GGUFs, complete CPU/cublas/fixed rows, no engine default change.
set -euo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; B=/content/b; SRC=/content/src
mkdir -p "$OUT" "$SRC"
pack() { flock /content/fixed_pack.lock python3 -c "import os,shutil; p=shutil.make_archive('/content/fixed_tmp_'+str(os.getpid()),'zip','/content','out'); os.replace(p,'/content/out.zip')"; }
packer=""
finish() { rc=$?; trap - EXIT; if [ -n "$packer" ]; then kill "$packer" 2>/dev/null || true; wait "$packer" 2>/dev/null || true; fi; cp /content/vm.log "$OUT/vm.log"; pack || true; exit "$rc"; }
trap finish EXIT
trap 'exit 130' INT TERM
sha256sum -c INPUT_SHA256SUMS > "$OUT/input_verification.txt"
cp INPUT_SHA256SUMS COMMITS "$OUT/"
{ date -u; nvidia-smi; nvcc --version; python3 --version; } >> "$OUT/hardware.txt" 2>&1
tar --no-same-owner --no-same-permissions -xzf ember.tgz -C "$SRC" --strip-components=1
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
(while sleep 120; do pack || true; done) &
packer=$!
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CC" > "$OUT/cmake.log" 2>&1
grep -q 'CUDA toolkit and cuBLAS found' "$OUT/cmake.log"
J=$(nproc); [ "$J" -le 8 ] || J=8
make -C "$B" -j"$J" tinyllama_server > "$OUT/make.log" 2>&1
mkdir -p "$OUT/binaries"
cp "$B/tinyllama_server" "$OUT/binaries/tinyllama_server"
sha256sum "$B/tinyllama_server" > "$OUT/binaries.sha256"
# Bound remote CPU reference threads, independently of the laptop's CPU budget.
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
P="$SRC/experiments/spec_r1_2026-10-28/fixed_reference_gate.py"
if [ ! -e "$OUT/reference/plan.json" ]; then
 python3 "$P" plan "$B/tinyllama_server" /content/pick.gguf /content/pk.gguf /content/eval100.jsonl "$OUT/reference"
fi
python3 "$P" run "$OUT/reference"
rc=0
python3 "$P" analyze "$OUT/reference" > "$OUT/gate_summary.json" || rc=$?
[ "$rc" -le 1 ]
python3 - "$OUT/gate_summary.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result['status'] in ('PASS', 'FAIL'), result
PY
echo done > /content/VM_DONE
