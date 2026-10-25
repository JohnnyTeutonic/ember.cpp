#!/usr/bin/env bash
# VM side of the X5 gate's CUDA run-to-run control (second session). The first session found e115fd3 and cbdcb915
# giving the same 360 replies and ids through tinyllama_server on CUDA, the same first-token logits on every fresh
# prompt, but logits differing in rounding at decode positions. This run asks whether the same build differs from
# itself: e115fd3 and cbdcb915 (+ the nine-digit dump) tinyllama_server at the device's arch, each run twice on the
# plain-llama set (x5_plain_collect.py, gpu_layers -1, EMBER_GEMM_TF32=0, X5_KEEP_ALL=1 so every position's
# logits are kept), interleaved new, old, new, old; then x5_plain_compare.py on every pair.
# Inputs: COMMITS, ember_new.tgz, ember_old.tgz, small.tgz, pick.part*, BIG_SHA256SUMS. Outputs: out.zip, VM_DONE.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; M=/content/models
mkdir -p "$OUT" "$M" /content/bin
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r NEWC OLDC CEC < /content/COMMITS
echo "=== vm2 $(date -u) new=$NEWC old=$OLDC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC, nproc $(nproc)"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cat pick.part* > $M/pick_L16.gguf || fail "reassemble"
(cd $M && sha256sum pick_L16.gguf) > $OUT/big.sha256
grep -q "$(awk '{print $1}' $OUT/big.sha256)" BIG_SHA256SUMS || fail "big sha256"
rm -f pick.part*
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
mkdir -p /content/src_new /content/src_old
tar xzf ember_new.tgz -C /content/src_new --strip-components=1 && tar xzf ember_old.tgz -C /content/src_old --strip-components=1 || fail "untar ember"
python3 $SM/x5_dump9.py /content/src_old || fail "dump9"
J=$(nproc)
for t in new old; do
  t0=$(date +%s)
  cmake -S /content/src_$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$t.log 2>&1 || fail "cmake $t"
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
  make -C /content/b_$t -j$J tinyllama_server > $OUT/make_$t.log 2>&1 || { tail -30 $OUT/make_$t.log; fail "build $t"; }
  cp /content/b_$t/tinyllama_server /content/bin/${t}_server
  echo "built $t in $(( $(date +%s) - t0 )) s"
done
sha256sum /content/bin/* > $OUT/bins.sha256
P=$SM/prompts.jsonl; C5=$SM/carry5_v1.jsonl
export X5_KEEP_ALL=1
port=8430
for run in new_a old_a new_b old_b; do
  b=${run%_*}
  port=$((port + 1))
  python3 $SM/x5_plain_collect.py /content/bin/${b}_server $M/pick_L16.gguf -1 $port $P $C5 /content/plain/$run > $OUT/collect_$run.txt 2>&1
  echo "collect $run rc=$? $(tail -1 $OUT/collect_$run.txt)"
done
for pair in new_a:new_b old_a:old_b new_a:old_a new_b:old_b new_a:old_b new_b:old_a; do
  a=${pair%:*}; b=${pair#*:}
  python3 $SM/x5_plain_compare.py /content/plain/$a /content/plain/$b > $OUT/ctl_${a}_vs_${b}.json
  echo "compare $a vs $b rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: [v[x] for x in ('replies_identical','ids_identical','first_token_logits_bit_equal','all_positions_logits_bit_equal')] + [v['every_position']['bit_equal_rows'], v['every_position']['rows'], v['every_position']['worst_abs']] for k, v in d['sets'].items()}))" $OUT/ctl_${a}_vs_${b}.json)"
done
for run in new_a old_a new_b old_b; do
  mkdir -p $OUT/plain/$run
  cp /content/plain/$run/records.jsonl /content/plain/$run/meta.json $OUT/plain/$run/
done
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
