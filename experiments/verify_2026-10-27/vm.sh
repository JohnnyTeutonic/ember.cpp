#!/usr/bin/env bash
# VM side of the verification-pass work (ember.cpp, 27 Oct 2026). Launched detached by drive.sh while the large
# models are still uploading.
#   bash vm.sh
# Inputs in /content: ember_new.tgz (this work: ember main 70ed063 plus the change, tools/vf_bench.cpp and its
# CMake target), ember_head.tgz (70ed063), COMMITS, small.tgz; later pick.part*, pk.part*, BIG_SHA256SUMS and the
# marker UPLOADS_DONE. Both trees are built Release for the GPU's own architecture (sm_89 on an L4).
# Legs (outputs under /content/out, packed as out1.zip .. out3.zip as they finish, then out.zip):
#   (a) SASS of both tinyllama_server binaries, kernel by kernel (sass_compare.py);
#   (b) vf_bench on pick_e1_L16 and pk_base_v1, default mode and EMBER_DETERMINISTIC=1;
#   (c) end to end: vf_e2e.py over chat40 + copy20 (all60.jsonl) per model, speculation off and on (k = 2, 4, 8,
#       each linear kernel at k = 4, the legacy pass), and under EMBER_DETERMINISTIC=1 off and on;
#   (d) speculation off, this build against 70ed063: x5_plain_collect (360 greedy requests, every sampled
#       position's logits) on pick_e1_L16, both modes, and ir_device_dump on the interpreter's local gate models;
#   (e) nsys, when the VM has it: kernel time per NVTX range of a short vf_bench run.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; M=$SM/models; BIG=/content/big
mkdir -p "$OUT" "$BIG" "$OUT/e2e" "$OUT/plain" "$OUT/dumps" /content/plain /content/dumps
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
pack() {  # name
  (cd /content && rm -f $1.zip && python3 -c "import shutil; shutil.make_archive('/content/$1', 'zip', '/content', 'out')" \
     && touch /content/${1^^}_READY && echo "packed $1 $(du -sh $1.zip | cut -f1)")
}
read -r NEWC HEADC < /content/COMMITS
echo "=== vm $(date -u) new=$NEWC head=$HEADC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "gpu $GPU, compute capability $CC, nproc $(nproc)"
echo "$GPU $CC" > $OUT/gpu.txt
NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1 || true)
echo "nsys: ${NSYS:-none}"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"

echo "--- builds $(date -u +%T)"
mkdir -p /content/new /content/head
tar xzf ember_new.tgz -C /content/new --strip-components=1 && tar xzf ember_head.tgz -C /content/head --strip-components=1 || fail "untar ember"
for t in new head; do
  t0=$(date +%s)
  tg="tinyllama_server ir_device_dump"
  [ $t = new ] && tg="$tg vf_bench"
  cmake -S /content/$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$t.log 2>&1 || { tail -20 $OUT/cmake_$t.log; fail "cmake $t"; }
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
  make -C /content/b_$t -j"$(nproc)" $tg > $OUT/make_$t.log 2>&1 || { tail -30 $OUT/make_$t.log; fail "build $t"; }
  echo "built $t in $(( $(date +%s) - t0 )) s"
done
sha256sum /content/b_*/tinyllama_server /content/b_*/ir_device_dump /content/b_new/vf_bench > $OUT/bins.sha256

echo "--- (a) SASS $(date -u +%T)"
for t in new head; do cuobjdump -sass /content/b_$t/tinyllama_server > /content/sass_$t.txt 2> $OUT/cuobjdump_$t.log; done
python3 $SM/sass_compare.py /content/sass_head.txt /content/sass_new.txt > $OUT/sass.json
echo "sass: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('arches','head_functions','new_functions','identical','differ','only_in_head','only_in_new')}))" $OUT/sass.json)"
rm -f /content/sass_*.txt

echo "--- waiting for the models $(date -u +%T)"
for i in $(seq 1 360); do [ -e /content/UPLOADS_DONE ] && break; sleep 10; done
[ -e /content/UPLOADS_DONE ] || fail "models never arrived"
cat pick.part* > $BIG/pick_L16.gguf && cat pk.part* > $BIG/pk_base_v1.gguf || fail "reassemble"
(cd $BIG && sha256sum -c /content/BIG_SHA256SUMS) > $OUT/big.sha256 2>&1 || fail "big sha256"
rm -f pick.part* pk.part*
PICK=$BIG/pick_L16.gguf; PK=$BIG/pk_base_v1.gguf

echo "--- (b) vf_bench $(date -u +%T)"
mkdir -p /content/benchrun && cd /content/benchrun && ln -sf /dev/null debugging.log
for m in pick:$PICK pk:$PK; do
  n=${m%%:*}; g=${m#*:}
  for mode in off det; do
    e=EMBER_NOTHING=1; [ $mode = det ] && e=EMBER_DETERMINISTIC=1
    t0=$(date +%s)
    env $e /content/b_new/vf_bench $g $OUT/bench_${n}_${mode}.json 200 32,384 > $OUT/bench_${n}_${mode}.log 2>&1
    echo "bench $n $mode rc=$? in $(( $(date +%s) - t0 )) s"
  done
done
cd /content
python3 $SM/vf_summary.py bench $OUT > $OUT/bench_summary.txt 2>&1; cat $OUT/bench_summary.txt
pack out1

echo "--- (c) end to end $(date -u +%T)"
port=8600
e2e() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/vf_e2e.py run /content/b_new/tinyllama_server $g -1 $port $SM/all60.jsonl $OUT/e2e/${n}_${tag}.jsonl "$@" \
    > $OUT/e2e/${n}_${tag}.txt 2>&1
  echo "e2e $n $tag rc=$? $(tail -1 $OUT/e2e/${n}_${tag}.txt | cut -c1-260)"
  rm -rf $OUT/e2e/${n}_${tag}.jsonl.run
}
for m in pick:$PICK pk:$PK; do
  n=${m%%:*}; g=${m#*:}
  e2e $n $g off EMBER_NOTHING=1
  e2e $n $g k2 EMBER_SPEC=2
  e2e $n $g k4 EMBER_SPEC=4
  e2e $n $g k8 EMBER_SPEC=8
  e2e $n $g k4_gemvb EMBER_SPEC=4 EMBER_VERIFY_LINEAR=gemvb
  e2e $n $g k4_gemm EMBER_SPEC=4 EMBER_VERIFY_LINEAR=gemm
  e2e $n $g k4_rows EMBER_SPEC=4 EMBER_VERIFY_LINEAR=rows
  e2e $n $g k4_legacy EMBER_SPEC=4 EMBER_VERIFY=legacy
  e2e $n $g k8_legacy EMBER_SPEC=8 EMBER_VERIFY=legacy
  e2e $n $g det_off EMBER_DETERMINISTIC=1
  e2e $n $g det_k2 EMBER_DETERMINISTIC=1 EMBER_SPEC=2
  e2e $n $g det_k4 EMBER_DETERMINISTIC=1 EMBER_SPEC=4
  e2e $n $g det_k8 EMBER_DETERMINISTIC=1 EMBER_SPEC=8
  e2e $n $g det_k4_rows EMBER_DETERMINISTIC=1 EMBER_SPEC=4 EMBER_VERIFY_LINEAR=rows
  e2e $n $g det_k8_rows EMBER_DETERMINISTIC=1 EMBER_SPEC=8 EMBER_VERIFY_LINEAR=rows
  e2e $n $g off2 EMBER_NOTHING=1
done
python3 $SM/vf_summary.py e2e $OUT > $OUT/e2e_summary.txt 2>&1; cat $OUT/e2e_summary.txt
pack out2

echo "--- (d) speculation off: this build against 70ed063 $(date -u +%T)"
P=$SM/prompts.jsonl; C5=$SM/carry5_v1.jsonl
export X5_KEEP_ALL=0
collect() {  # run build off|det
  local run=$1 b=$2 mode=$3 e=EMBER_NOTHING=1
  [ "$mode" = det ] && e=EMBER_DETERMINISTIC=1
  port=$((port + 1))
  env $e python3 $SM/x5_plain_collect.py /content/b_$b/tinyllama_server $PICK -1 $port $P $C5 /content/plain/$run > $OUT/collect_$run.txt 2>&1
  echo "collect $run rc=$? $(tail -1 $OUT/collect_$run.txt)"
  mkdir -p $OUT/plain/$run && cp /content/plain/$run/records.jsonl /content/plain/$run/meta.json $OUT/plain/$run/
}
cmp_pair() {
  python3 $SM/x5_plain_compare.py /content/plain/$1 /content/plain/$2 > $OUT/cmp_$1_vs_$2.json
  echo "compare $1 vs $2 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['bit_identical'], json.dumps({k: [v[x] for x in ('replies_identical','ids_identical','first_token_logits_bit_equal','all_positions_logits_bit_equal')] for k, v in d['sets'].items()}))" $OUT/cmp_$1_vs_$2.json)"
}
collect head_det head det
collect new_det new det
collect head_off head off
collect new_off new off
cmp_pair head_det new_det
cmp_pair head_off new_off
same() { cmp -s "$1" "$2" && echo identical || echo DIFFERS; }
dd() {  # name gguf [args]
  local name=$1 g=$2; shift 2
  [ -s "$g" ] || { echo "dump $name skipped: no $g"; return; }
  for b in head new; do
    /content/b_$b/ir_device_dump $g /content/dumps/${name}_$b.bin "$@" > /dev/null 2> $OUT/dumps/${name}_$b.log \
      || echo "dump $name $b failed: $(tail -1 $OUT/dumps/${name}_$b.log)"
  done
  local r="$name head_vs_new=$(same /content/dumps/${name}_head.bin /content/dumps/${name}_new.bin) bytes=$(stat -c %s /content/dumps/${name}_new.bin 2>/dev/null)"
  echo "dump $r"; echo "$r" >> $OUT/dumps.txt
  sha256sum /content/dumps/${name}_*.bin >> $OUT/dumps.sha256 2>/dev/null
  rm -f /content/dumps/${name}_*.bin
}
dd ar_x5_q8_0 $M/ar3/ar_x5_q8_0.gguf --len 8 --gen 8
dd state4 $M/state4/state4.gguf
dd state4_q8_0 $M/state4/state4_q8_0.gguf
dd mor3 $M/mor/mor3.gguf --len 8 --gen 8
dd mor3_q8_0 $M/mor/mor3_q8_0.gguf --len 8 --gen 8
dd mor2p0 $M/mor/mor2p0.gguf --len 8 --gen 8
dd ar3 $M/ar3/ar3.gguf --len 8 --gen 8
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do dd rec_$m $M/recursion/$m.gguf --len 8 --gen 8; done
dd syn_mistral_swa $M/synth/mistral_swa.gguf
dd syn_gptneox $M/synth/gptneox.gguf
dd syn_gpt2 $M/synth/gpt2.gguf
dd syn_mamba $M/synth/mamba.gguf
dd syn_gemma3 $M/synth/gemma3.gguf
dd gpt2_tiny $M/gpt2_tiny/gpt2_tiny.gguf
pack out3

if [ -n "${NSYS:-}" ]; then
  echo "--- (e) nsys $(date -u +%T)"
  cd /content/benchrun
  for m in pick:$PICK pk:$PK; do
    n=${m%%:*}; g=${m#*:}
    timeout 900 $NSYS profile -t cuda,nvtx -o /content/nsys_$n --force-overwrite true /content/b_new/vf_bench $g /content/nsys_bench_$n.json 30 384 \
      > $OUT/nsys_$n.log 2>&1
    timeout 900 $NSYS stats -r nvtx_kern_sum,cuda_gpu_kern_sum -f csv -o $OUT/nsys_$n /content/nsys_$n.nsys-rep >> $OUT/nsys_$n.log 2>&1
    echo "nsys $n rc=$? $(ls $OUT/nsys_${n}* 2>/dev/null | tr '\n' ' ')"
    rm -f /content/nsys_$n.nsys-rep /content/nsys_$n.sqlite
  done
  cd /content
fi

echo "--- pack $(date -u +%T)"
pack out
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
