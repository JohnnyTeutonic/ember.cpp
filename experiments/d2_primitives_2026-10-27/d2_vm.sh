#!/bin/bash
# VM side of the D2 primitives' CUDA leg (ember.cpp, 27 Oct 2026). Launched detached by d2_drive.sh.
# Inputs in /content: COMMITS (the ember commit), ember.tgz (its git archive), small.tgz (d2_stage.sh: keramos
# v1.4.0's git archive and SHA256SUMS).
# Build (HAS_CUDA on, Release, CMAKE_CUDA_ARCHITECTURES = the device's): ir_device_gate, ir_adapter.
# Legs:
#   ir_device_gate (the interpreter on the device against the same interpreter on the CPU, one process; 40 seeded
#     prompts of 8 ids, 8 greedy) on pk_flex_mdha (conv1d_causal beside the kv slot: device kernels) and
#     pk_flex_shaw (attention.relative_kv: device_supported false, so CUDA builds serve it on the CPU);
#   ir_device_gate on pk_srd and pk_llama (kv + recurrent, plain kv: the existing device paths, unchanged);
#   the conformance suite at v1.4.0 against the CUDA build's ir_adapter (the CPU interpreter in a CUDA build).
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small
mkdir -p "$OUT"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r C < /content/COMMITS
echo "=== vm $(date -u) commit=$C ==="
{ nvidia-smi; nvcc --version; nproc; free -g; python3 --version; python3 -c "import numpy; print('numpy', numpy.__version__)"; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC, nproc $(nproc)"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cp $SM/SHA256SUMS $OUT/small.sha256
K=$SM/keramos

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"

echo "--- build $(date -u +%T)"
mkdir -p /content/src
tar xzf ember.tgz -C /content/src --strip-components=1 || fail "untar ember"
find /content/src -name "*.sh" -exec sed -i 's/\r$//' {} +
J=$(nproc)
B=/content/b
t0=$(date +%s)
cmake -S /content/src -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || { tail -20 $OUT/cmake.log; fail "cmake"; }
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log && echo "CUDA toolkit and cuBLAS found" || fail "configured without CUDA"
grep -E "HAS_CUDA" $B/CMakeCache.txt | head -3
make -C $B -j$J ir_device_gate ir_adapter > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built (sm_$CC) in $(( $(date +%s) - t0 )) s"
sha256sum $B/ir_device_gate $B/ir_adapter > $OUT/bins.sha256

for m in pk_flex_mdha pk_flex_shaw pk_srd pk_llama; do
  echo "--- $m $(date -u +%T)"
  G=$K/conformance/models/$m/$m.gguf
  $B/ir_device_gate $G --n 40 --len 8 --gen 8 > $OUT/dg_$m.json 2> $OUT/dg_$m.log
  echo "ir_device_gate $m rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('rows',None); print(json.dumps({k: d.get(k) for k in ('device_supported','greedy_identical','top1_agree','worst_prefill_rel','worst_decode_rel','pass')}))" $OUT/dg_$m.json 2>/dev/null || tail -2 $OUT/dg_$m.log)"
done

echo "--- suite $(date -u +%T)"
(cd $K && python3 tools/ir_conformance.py --adapter "cmd:$B/ir_adapter --conformance" --json $OUT/suite_v1.4.0_cuda_build.json > $OUT/suite.log 2>&1; echo "suite rc=$?")
tail -1 $OUT/suite.log

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
