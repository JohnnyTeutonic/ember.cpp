#!/usr/bin/env bash
# VM side of the third-party-ops training gate (ember.cpp, 26 Oct 2026). Launched detached by tp_drive.sh.
# Inputs in /content: COMMITS (the ember commit, full hash), tp_ember.tgz (its git archive), small.tgz (the five
# models' directories, tp_session_logits.cpp, SHA256SUMS).
# Build: the archive + tp_session_logits as a target, Release, at the device's arch: tp_session_logits,
#   state_device_gate, ir_device_gate.
# Legs, every GGUF (F32 and Q8_0), EMBER_GEMM_TF32=0 (the parity setting):
#   dev_session_TAG.json  tp_session_logits at gpu_layers -1: the path and device the server would take, its prefill,
#                         decode and free-running greedy on the 40 score_in sequences;
#   sdg_TAG.json          tools/state_device_gate (interpreter: device vs CPU in one process);
#   dg_TAG.json           tools/ir_device_gate, 40 seeded random prompts of 8 ids, 8 greedy (interpreter);
#   tp_gqa only: dev_session_interp_TAG.json (EMBER_INTERPRETER=1: the interpreter on the device) and
#                dev_session_tf32_TAG.json (the server's default TF32 GEMMs; reported, not gated).
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small
mkdir -p "$OUT"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r COMMIT < /content/COMMITS
echo "=== vm $(date -u) ember=$COMMIT ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC, nproc $(nproc)"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
echo "small sha256 ok"

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
pip install -q numpy > /dev/null 2>&1

echo "--- build $(date -u +%T)"
mkdir -p /content/src
tar xzf tp_ember.tgz -C /content/src --strip-components=1 || fail "untar ember"
cp $SM/tp_session_logits.cpp /content/src/tools/
printf '\nadd_executable(tp_session_logits tools/tp_session_logits.cpp)\ntarget_link_libraries(tp_session_logits PRIVATE tinyllama_core)\n' >> /content/src/CMakeLists.txt
t0=$(date +%s)
cmake -S /content/src -B /content/b -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || { tail -20 $OUT/cmake.log; fail "cmake"; }
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log && echo "CUDA toolkit and cuBLAS found" || fail "configured without CUDA"
make -C /content/b -j$(nproc) tp_session_logits state_device_gate ir_device_gate > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built (sm_$CC) in $(( $(date +%s) - t0 )) s"
sha256sum /content/b/tp_session_logits /content/b/state_device_gate /content/b/ir_device_gate > $OUT/bins.sha256

echo "--- legs $(date -u +%T)"
export EMBER_GEMM_TF32=0
for name in tp_gqa tp_qwen tp_mqa_partial tp_neox_partial tp_mamba1; do
  d=$SM/models/$name
  IN=$SM/in_$name.json
  for g in $d/$name.gguf $d/${name}_q8_0.gguf; do
    [ -f $g ] || continue
    tag=$(basename $g .gguf)
    /content/b/tp_session_logits $g -1 < $IN > $OUT/dev_session_$tag.json 2> $OUT/dev_session_$tag.log
    echo "session $tag rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['path'], d['device'], d['heads'], d['kv_heads'])" $OUT/dev_session_$tag.json 2>/dev/null || tail -2 $OUT/dev_session_$tag.log)"
    grep -E "IR\]|IR interpreter" $OUT/dev_session_$tag.log | head -3
    /content/b/state_device_gate $g < $IN > $OUT/sdg_$tag.json 2> $OUT/sdg_$tag.log
    echo "state_device_gate $tag rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['summary']), 'pass', d['pass'])" $OUT/sdg_$tag.json 2>/dev/null || tail -2 $OUT/sdg_$tag.log)"
    /content/b/ir_device_gate $g --n 40 --len 8 --gen 8 > $OUT/dg_$tag.json 2> $OUT/dg_$tag.log
    echo "ir_device_gate $tag rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['greedy_identical'], d['top1_agree'], d['worst_prefill_rel'], d['worst_decode_rel'], 'pass', d['pass'])" $OUT/dg_$tag.json 2>/dev/null || tail -2 $OUT/dg_$tag.log)"
    if [ $name = tp_gqa ]; then
      EMBER_INTERPRETER=1 /content/b/tp_session_logits $g -1 < $IN > $OUT/dev_session_interp_$tag.json 2> $OUT/dev_session_interp_$tag.log
      echo "session (EMBER_INTERPRETER=1) $tag rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['path'], d['device'])" $OUT/dev_session_interp_$tag.json 2>/dev/null)"
      env -u EMBER_GEMM_TF32 /content/b/tp_session_logits $g -1 < $IN > $OUT/dev_session_tf32_$tag.json 2> $OUT/dev_session_tf32_$tag.log
      echo "session (default TF32) $tag rc=$?"
    fi
  done
done
for f in $OUT/*.log; do [ $(stat -c %s $f) -gt 20000 ] && { tail -c 20000 $f > $f.t && mv $f.t $f; }; done

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
