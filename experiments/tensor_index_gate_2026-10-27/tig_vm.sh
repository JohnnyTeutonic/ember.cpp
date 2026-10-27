#!/usr/bin/env bash
# VM side of the tensor_index gate's CUDA leg (ember.cpp, 27 Oct 2026). Launched detached by tig_drive.sh.
# Inputs in /content: COMMITS (the ember commit), ember.tgz (its git archive), small.tgz (tig_stage.sh).
# Build: the tree + x5_device_logits as a target, Release, CMAKE_CUDA_ARCHITECTURES = the device's;
#   targets x5_device_logits, state_device_gate, ir_device_gate.
# Legs, per variant (ar_rec, ar_rec_q8_0):
#   x5_device_logits (prefill, token-by-token decode through the captured graph, free-running greedy) on the
#     embedded (tagged) document: default mode twice, EMBER_DETERMINISTIC=1 twice (cmp of the outputs);
#   the untagged document through EMBER_IR; the rec_app.{4,5} NaN copy, tagged and untagged;
#   state_device_gate (device against the CPU in one process); ir_device_gate on random ids (40 x 8, 8 greedy).
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; M=$SM/models
mkdir -p "$OUT"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r C < /content/COMMITS
echo "=== vm $(date -u) commit=$C ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC, nproc $(nproc)"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cp $SM/SHA256SUMS $OUT/small.sha256

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"

echo "--- build $(date -u +%T)"
mkdir -p /content/src
tar xzf ember.tgz -C /content/src --strip-components=1 || fail "untar ember"
find /content/src -name "*.sh" -exec sed -i 's/\r$//' {} +
cp $SM/x5_device_logits.cpp /content/src/tools/
printf '\nadd_executable(x5_device_logits tools/x5_device_logits.cpp)\ntarget_link_libraries(x5_device_logits PRIVATE tinyllama_core)\n' >> /content/src/CMakeLists.txt
J=$(nproc)
B=/content/b
t0=$(date +%s)
cmake -S /content/src -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || { tail -20 $OUT/cmake.log; fail "cmake"; }
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log && echo "CUDA toolkit and cuBLAS found" || fail "configured without CUDA"
make -C $B -j$J x5_device_logits state_device_gate ir_device_gate > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built (sm_$CC) in $(( $(date +%s) - t0 )) s"
sha256sum $B/x5_device_logits $B/state_device_gate $B/ir_device_gate > $OUT/bins.sha256

IN=$SM/ar_rec_in.json
UT=$SM/ar_rec_untagged.ir.json
for v in ar_rec ar_rec_q8_0; do
  echo "--- $v $(date -u +%T)"
  G=$M/$v.gguf
  for r in a b; do
    $B/x5_device_logits $G < $IN > $OUT/dev_${v}_$r.json 2> $OUT/dev_${v}_$r.log; echo "x5_device_logits $v $r rc=$?"
    EMBER_DETERMINISTIC=1 $B/x5_device_logits $G < $IN > $OUT/dev_${v}_det_$r.json 2> $OUT/dev_${v}_det_$r.log; echo "x5_device_logits $v det $r rc=$?"
  done
  cmp -s $OUT/dev_${v}_a.json $OUT/dev_${v}_b.json && echo "$v default mode a = b" || echo "$v default mode a != b"
  cmp -s $OUT/dev_${v}_det_a.json $OUT/dev_${v}_det_b.json && echo "$v EMBER_DETERMINISTIC=1 a = b" || echo "$v EMBER_DETERMINISTIC=1 a != b"
  EMBER_IR=$UT $B/x5_device_logits $G < $IN > $OUT/dev_${v}_untagged.json 2> $OUT/dev_${v}_untagged.log; echo "untagged $v rc=$?"
  $B/x5_device_logits $M/${v}_nan_app45.gguf < $IN > $OUT/dev_${v}_nan45.json 2> $OUT/dev_${v}_nan45.log; echo "nan45 tagged $v rc=$?"
  EMBER_IR=$UT $B/x5_device_logits $M/${v}_nan_app45.gguf < $IN > $OUT/dev_${v}_nan45_untagged.json 2> $OUT/dev_${v}_nan45_untagged.log; echo "nan45 untagged $v rc=$?"
  cmp -s $OUT/dev_${v}_nan45_untagged.json $OUT/dev_${v}_untagged.json && echo "$v nan45 untagged = untagged (bits)" || echo "$v nan45 untagged != untagged"
  $B/state_device_gate $G < $IN > $OUT/sdg_$v.json 2> $OUT/sdg_$v.log
  echo "state_device_gate $v rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['summary']), 'pass', d['pass'])" $OUT/sdg_$v.json 2>/dev/null || tail -2 $OUT/sdg_$v.log)"
  $B/ir_device_gate $G --n 40 --len 8 --gen 8 > $OUT/dg_$v.json 2> $OUT/dg_$v.log
  echo "ir_device_gate $v rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('rows',None); print(d['greedy_identical'], d['top1_agree'], d['worst_prefill_rel'], d['worst_decode_rel'], 'pass', d['pass'])" $OUT/dg_$v.json 2>/dev/null || tail -2 $OUT/dg_$v.log)"
done

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
