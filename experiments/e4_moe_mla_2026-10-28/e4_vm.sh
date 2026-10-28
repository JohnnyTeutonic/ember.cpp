#!/bin/bash
# VM side of E4's CUDA leg (third-party MoE and DeepSeek-V2 MLA; ember.cpp, 28 Oct 2026). Launched detached by
# e4_drive.sh. Inputs in /content: COMMITS (the ember commit), ember.tgz (its git archive), small.tgz (e4_stage.sh:
# the seven synthetic files and their cases, keramos v1.6.1's suite models, SHA256SUMS).
# Build (HAS_CUDA on, Release, CMAKE_CUDA_ARCHITECTURES = the device's): ir_device_gate, ir_adapter.
# Legs, each file through ir_device_gate (the interpreter on the device against the same interpreter on the CPU,
# one process; Mixtral with EMBER_IR_BUILD=1, its IR interpreter rather than the llama path):
#   rand:  40 seeded prompts of 24 ids, every prefill position, the device's decode, 32 greedy tokens;
#   hf:    the 40 sequences of <fam>.cases.json, the last row of prefill and decode, 16 greedy tokens (HF's count);
#   det1, det2: rand under EMBER_DETERMINISTIC=1, twice; the two receipts compared byte for byte;
#   the suite's tp_qwen2moe, tp_qwen3moe, tp_deepseek2 and pk_moe (40 prompts of 8 ids, 8 greedy);
#   the conformance suite at v1.6.1 against the CUDA build's ir_adapter (the CPU interpreter in a CUDA build).
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
make -C $B -j$J ir_device_gate ir_adapter > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built (sm_$CC) in $(( $(date +%s) - t0 )) s"
sha256sum $B/ir_device_gate $B/ir_adapter > $OUT/bins.sha256

summ() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d.get(k) for k in ('device_supported','on_gpu','greedy_identical','top1_agree','worst_prefill_rel','worst_decode_rel','pass')}))" "$1" 2>/dev/null || echo "no receipt"; }
for fam in mixtral qwen2moe qwen3moe deepseek2 deepseek2_qlora deepseek2_dense_yarn deepseek3; do
  G=$SM/e4_synth/$fam.gguf
  python3 -c "import json,sys; json.dump([c['ids'] for c in json.load(open(sys.argv[1]))['cases']], open(sys.argv[2], 'w'))" \
    $SM/e4_synth/$fam.cases.json $OUT/$fam.ids.json
  E=""; [ $fam = mixtral ] && E="EMBER_IR_BUILD=1"
  echo "--- $fam $(date -u +%T)"
  env $E $B/ir_device_gate $G --n 40 --len 24 --gen 32 > $OUT/${fam}_rand.json 2> $OUT/${fam}_rand.log
  echo "$fam rand rc=$?: $(summ $OUT/${fam}_rand.json)"
  env $E $B/ir_device_gate $G --ids $OUT/$fam.ids.json --gen 16 > $OUT/${fam}_hf.json 2> $OUT/${fam}_hf.log
  echo "$fam hf rc=$?: $(summ $OUT/${fam}_hf.json)"
  for k in 1 2; do
    env $E EMBER_DETERMINISTIC=1 $B/ir_device_gate $G --n 40 --len 24 --gen 32 > $OUT/${fam}_det$k.json 2> $OUT/${fam}_det$k.log
  done
  cmp -s $OUT/${fam}_det1.json $OUT/${fam}_det2.json && echo "$fam det: identical run to run ($(summ $OUT/${fam}_det1.json))" \
    || echo "$fam det: the two EMBER_DETERMINISTIC=1 receipts differ"
  grep -h "mla\|CUDA" $OUT/${fam}_rand.log | sort -u | head -4
done
for m in tp_qwen2moe tp_qwen3moe tp_deepseek2 pk_moe; do
  G=$K/conformance/models/$m/$m.gguf
  $B/ir_device_gate $G --n 40 --len 8 --gen 8 > $OUT/dg_$m.json 2> $OUT/dg_$m.log
  echo "$m rc=$?: $(summ $OUT/dg_$m.json)"
done

echo "--- suite $(date -u +%T)"
(cd $K && python3 tools/ir_conformance.py --adapter "cmd:$B/ir_adapter --conformance" --json $OUT/suite_v1.6.1_cuda_build.json > $OUT/suite.log 2>&1; echo "suite rc=$?")
tail -1 $OUT/suite.log

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
