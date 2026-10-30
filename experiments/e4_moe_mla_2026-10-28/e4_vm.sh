#!/bin/bash
# E4 device gate, 30 Oct 2026. Fail closed over all ten synthetic files.
# Eight supported device families; dense/YaRN and V3 documents remain CPU fallbacks.
set -euo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; B=/content/b
mkdir -p "$OUT"
pack() { flock /content/out_pack.lock python3 -c "import os,shutil; p=shutil.make_archive('/content/out_tmp_'+str(os.getpid()),'zip','/content','out'); os.replace(p,'/content/out.zip')"; }
packer=""
finish() {
 rc=$?
 trap - EXIT
 if [ -n "$packer" ]; then kill "$packer" 2>/dev/null || true; wait "$packer" 2>/dev/null || true; fi
 if [ "$rc" -ne 0 ]; then echo "failed rc=$rc" > /content/VM_FAILED; cp /content/vm.log "$OUT/vm.log"; fi
 pack || true
 exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT TERM
read -r C < /content/COMMITS
printf '%s\n' "$C" > "$OUT/COMMIT"
{ nvidia-smi; nvcc --version; nproc; free -g; python3 --version; python3 -c "import numpy; print('numpy',numpy.__version__)"; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
tar --no-same-owner --no-same-permissions -xzf small.tgz
(cd "$SM" && sha256sum -c --quiet SHA256SUMS)
cp "$SM/SHA256SUMS" "$OUT/small.sha256"
K=$SM/keramos
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1
mkdir -p /content/src
tar --no-same-owner --no-same-permissions -xzf ember.tgz -C /content/src --strip-components=1
find /content/src -name '*.sh' -exec sed -i 's/\r$//' {} +
(while sleep 120; do pack || true; done) &
packer=$!
cmake -S /content/src -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CC" > "$OUT/cmake.log" 2>&1
grep -q 'CUDA toolkit and cuBLAS found' "$OUT/cmake.log"
J=$(nproc); [ "$J" -le 8 ] || J=8
make -C "$B" -j"$J" e4_device_gate ir_adapter > "$OUT/make.log" 2>&1
sha256sum "$B/e4_device_gate" "$B/ir_adapter" > "$OUT/bins.sha256"
for family in mixtral qwen2moe qwen3moe deepseek2 deepseek2_qlora gemma2 phi3_longrope phi3_swa deepseek2_dense_yarn deepseek3; do
 for mode in default det1 det2; do
  det=0; [ "$mode" = default ] || det=1
  echo "--- $family $mode $(date -u +%T)"
  env EMBER_DETERMINISTIC="$det" EMBER_GEMM_TF32=0 "$B/e4_device_gate" \
   "$SM/e4_synth/$family.gguf" "$SM/e4_synth/$family.cases.json" \
   "$OUT/${family}_${mode}.json" "$OUT/${family}_${mode}.bin" \
   > "$OUT/${family}_${mode}.stdout" 2> "$OUT/${family}_${mode}.log" || echo "nonzero gate: $family $mode"
 done
done
(cd "$K" && python3 tools/ir_conformance.py --adapter "cmd:$B/ir_adapter --conformance" \
 --json "$OUT/suite_v1.6.1_cuda_build.json" > "$OUT/suite.log" 2>&1) || echo 'suite returned nonzero'
python3 /content/src/experiments/e4_moe_mla_2026-10-28/e4_check.py "$OUT"
echo done > /content/VM_DONE
echo "=== E4 gate complete $(date -u) ==="
