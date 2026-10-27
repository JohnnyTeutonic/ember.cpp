#!/usr/bin/env bash
# VM side of decode_occupancy_2026-10-27 (ember.cpp). Launched detached by drive.sh while the two models upload.
#   bash vm.sh
# Inputs in /content: ember_new.tgz (ember main plus this change, tools/occ_step.cpp + its CMake target,
# tools/occ_bench.cu), ember_head.tgz (ember main), COMMITS, small.tgz; later pick.part*, m60.part*,
# BIG_SHA256SUMS and the marker UPLOADS_DONE. Both trees are built Release for the GPU's own architecture.
# Legs (outputs under /content/out; out1.zip .. out4.zip as they finish, then out.zip):
#   (a) tools/decode_attention_test: the tiled kernel (default), split, legacy, and the default under
#       EMBER_DETERMINISTIC=1;
#   (b) occ_bench (synthetic caches: the old split kernel, its smaller-chunk forms, the tiled kernel's chunk grid),
#       once plain (graph times) and once under nsys (each kernel's own time, hot and cold L2);
#   (c) SASS of both tinyllama_server binaries, kernel by kernel;
#   (d) occ_step on pick_e1_L16 and pick_e1_60m_L32: the decode step at 32..500 cached tokens, split against tiled
#       (and two other tilings, and both under EMBER_DETERMINISTIC=1), plus nsys of split and tiled;
#   (e) x5_plain_collect (360 greedy requests, every sampled position's logits), HEAD against this build: default
#       mode, the mode, the mode twice, the mode with EMBER_DECODE_ATTN=split;
#   (f) speculative verification under the mode (vd_collect / vd_compare): plain against k = 2, 4, 8;
#   (g) end to end (vf_e2e.py): all60 + c12_40 + long40, split against tiled, alternating, and under the mode;
#   (h) ir_device_dump on the interpreter's local gate models, HEAD against this build (dump_compare.py).
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; M=$SM/models; BIG=/content/big
mkdir -p "$OUT"/{bench,step,plain,vd,e2e,dumps} "$BIG" /content/plain /content/vdruns /content/dumps /content/run
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
  [ $t = new ] && tg="$tg occ_step"
  cmake -S /content/$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$t.log 2>&1 || { tail -20 $OUT/cmake_$t.log; fail "cmake $t"; }
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
  make -C /content/b_$t -j"$(nproc)" $tg > $OUT/make_$t.log 2>&1 || { tail -30 $OUT/make_$t.log; fail "build $t"; }
  echo "built $t in $(( $(date +%s) - t0 )) s"
done
(cd /content/new && nvcc -O3 -std=c++17 -arch=sm_$CC -I. tools/occ_bench.cu -o /content/b_new/occ_bench > $OUT/nvcc_bench.log 2>&1 \
  && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. tools/decode_attention_test.cu cuda_kernels.cu logger.cpp -lcublas \
       -o /content/b_new/decode_attention_test > $OUT/nvcc_dat.log 2>&1) || fail "nvcc bench/test"
sha256sum /content/b_*/tinyllama_server /content/b_*/ir_device_dump /content/b_new/occ_step /content/b_new/occ_bench \
  /content/b_new/decode_attention_test > $OUT/bins.sha256

echo "--- (a) decode_attention_test $(date -u +%T)"
for cfg in tiled:EMBER_NOTHING=1 split:EMBER_DECODE_ATTN=split legacy:EMBER_DECODE_ATTN=legacy det:EMBER_DETERMINISTIC=1; do
  tag=${cfg%%:*}
  env ${cfg#*:} /content/b_new/decode_attention_test > $OUT/dat_$tag.txt 2>&1
  echo "dat $tag rc=$? $(tail -1 $OUT/dat_$tag.txt)"
done

echo "--- (b) occ_bench $(date -u +%T)"
/content/b_new/occ_bench $OUT/bench/occ_bench.jsonl > $OUT/bench/occ_bench.txt 2>&1
echo "occ_bench rc=$? $(tail -1 $OUT/bench/occ_bench.txt)"
if [ -n "${NSYS:-}" ]; then
  timeout 1200 $NSYS profile -t cuda,nvtx -o /content/nsys_bench --force-overwrite true \
    /content/b_new/occ_bench /content/occ_bench_nsys.jsonl > $OUT/bench/nsys_bench.log 2>&1
  timeout 1200 $NSYS stats -r nvtx_kern_sum,cuda_gpu_kern_sum -f csv -o $OUT/bench/nsys_bench /content/nsys_bench.nsys-rep \
    >> $OUT/bench/nsys_bench.log 2>&1
  echo "nsys bench rc=$? $(ls $OUT/bench/nsys_bench* | tr '\n' ' ')"
  rm -f /content/nsys_bench.nsys-rep /content/nsys_bench.sqlite
fi

echo "--- (c) SASS $(date -u +%T)"
for t in new head; do cuobjdump -sass /content/b_$t/tinyllama_server > /content/sass_$t.txt 2> $OUT/cuobjdump_$t.log; done
python3 $SM/sass_compare.py /content/sass_head.txt /content/sass_new.txt > $OUT/sass.json
echo "sass: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('arches','head_functions','new_functions','identical','differ','only_in_head','only_in_new')}))" $OUT/sass.json)"
rm -f /content/sass_*.txt
pack out1

echo "--- waiting for the models $(date -u +%T)"
for i in $(seq 1 540); do [ -e /content/UPLOADS_DONE ] && break; sleep 10; done
[ -e /content/UPLOADS_DONE ] || fail "models never arrived"
cat pick.part* > $BIG/pick_L16.gguf && cat m60.part* > $BIG/pick_60m_L32.gguf || fail "reassemble"
(cd $BIG && sha256sum -c /content/BIG_SHA256SUMS) > $OUT/big.sha256 2>&1 || fail "big sha256"
rm -f pick.part* m60.part*
MODELS="pick:$BIG/pick_L16.gguf m60:$BIG/pick_60m_L32.gguf"

echo "--- (d) occ_step $(date -u +%T)"
cd /content/run && ln -sf /dev/null debugging.log
for m in $MODELS; do
  n=${m%%:*}; g=${m#*:}
  for cfg in split:EMBER_DECODE_ATTN=split tiled:EMBER_NOTHING=1 t16:EMBER_DECODE_ATTN_TILE=16,64 t64:EMBER_DECODE_ATTN_TILE=64,64 \
             det_split:EMBER_DETERMINISTIC=1,EMBER_DECODE_ATTN=split det_tiled:EMBER_DETERMINISTIC=1 split2:EMBER_DECODE_ATTN=split tiled2:EMBER_NOTHING=1; do
    tag=${cfg%%:*}; envs=$(echo ${cfg#*:} | tr , ' ')
    env $envs /content/b_new/occ_step $g $OUT/step/${n}_$tag.json 300 32,128,256,384,500 > $OUT/step/${n}_$tag.txt 2>&1
    echo "step $n $tag rc=$? $(tr '\n' ' ' < $OUT/step/${n}_$tag.txt | cut -c1-300)"
  done
  if [ -n "${NSYS:-}" ]; then
    for cfg in split:EMBER_DECODE_ATTN=split tiled:EMBER_NOTHING=1; do
      tag=${cfg%%:*}
      env ${cfg#*:} timeout 900 $NSYS profile -t cuda,nvtx -o /content/nsys_$n --force-overwrite true \
        /content/b_new/occ_step $g /content/nsys_step.json 100 32,128,256,384,500 > $OUT/step/nsys_${n}_$tag.log 2>&1
      timeout 900 $NSYS stats -r nvtx_kern_sum,cuda_gpu_kern_sum -f csv -o $OUT/step/nsys_${n}_$tag /content/nsys_$n.nsys-rep \
        >> $OUT/step/nsys_${n}_$tag.log 2>&1
      echo "nsys step $n $tag rc=$?"
      rm -f /content/nsys_$n.nsys-rep /content/nsys_$n.sqlite
    done
  fi
done
cd /content
pack out2

echo "--- (e) x5_plain_collect: HEAD against this build $(date -u +%T)"
P=$SM/prompts.jsonl; C5=$SM/carry5_v1.jsonl
export X5_KEEP_ALL=1
port=8700
collect() {  # model gguf run build env...
  local n=$1 g=$2 run=$3 b=$4; shift 4
  port=$((port + 1))
  env "$@" python3 $SM/x5_plain_collect.py /content/b_$b/tinyllama_server $g -1 $port $P $C5 /content/plain/${n}_$run > $OUT/plain/collect_${n}_$run.txt 2>&1
  echo "collect $n $run rc=$? $(tail -1 $OUT/plain/collect_${n}_$run.txt | cut -c1-200)"
  mkdir -p $OUT/plain/${n}_$run && cp /content/plain/${n}_$run/records.jsonl /content/plain/${n}_$run/meta.json $OUT/plain/${n}_$run/
}
cmp_pair() {  # model a b
  python3 $SM/x5_plain_compare.py /content/plain/${1}_$2 /content/plain/${1}_$3 > $OUT/plain/cmp_${1}_${2}_vs_$3.json
  echo "compare $1 $2 vs $3 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['bit_identical'], json.dumps({k: [v.get(x) for x in ('replies_identical','ids_identical','first_token_logits_bit_equal','all_positions_logits_bit_equal')] + [v.get('every_position', {}).get(y) for y in ('bit_equal_rows','rows','worst_abs','worst_rel_to_range')] for k, v in d['sets'].items()}))" $OUT/plain/cmp_${1}_${2}_vs_$3.json)"
}
for m in $MODELS; do
  n=${m%%:*}; g=${m#*:}
  collect $n $g head_off head EMBER_NOTHING=1
  collect $n $g new_off new EMBER_NOTHING=1
  collect $n $g head_det head EMBER_DETERMINISTIC=1
  collect $n $g new_det new EMBER_DETERMINISTIC=1
  collect $n $g new_det2 new EMBER_DETERMINISTIC=1
  collect $n $g new_det_split new EMBER_DETERMINISTIC=1 EMBER_DECODE_ATTN=split
  [ $n = pick ] && collect $n $g head_off2 head EMBER_NOTHING=1
  cmp_pair $n head_off new_off
  cmp_pair $n head_det new_det
  cmp_pair $n new_det new_det2
  cmp_pair $n head_det new_det_split
  [ $n = pick ] && cmp_pair $n head_off head_off2
  rm -f /content/plain/${n}_*/all_logits.f32
done
pack out3

echo "--- (f) speculative verification under the mode $(date -u +%T)"
col() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/vd_collect.py /content/b_new/tinyllama_server $g -1 $port $SM/all60.jsonl /content/vdruns/${n}_$tag "$@" > $OUT/vd/${n}_$tag.txt 2>&1
  echo "vd $n $tag rc=$? $(tail -1 $OUT/vd/${n}_$tag.txt | cut -c1-200)"
  mkdir -p $OUT/vd/${n}_$tag
  cp /content/vdruns/${n}_$tag/records.jsonl /content/vdruns/${n}_$tag/meta.json $OUT/vd/${n}_$tag/ 2>/dev/null
}
vcmp() {  # model base other
  python3 $SM/vd_compare.py /content/vdruns/${1}_$2 /content/vdruns/${1}_$3 > $OUT/vd/cmp_${1}_${2}_vs_$3.json 2> $OUT/vd/cmp_${1}_${2}_vs_$3.err
  echo "vd compare $1 $2 vs $3 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ('bit_identical','requests','ids_identical','replies_identical','rows_compared','rows_bit_identical','max_abs_diff','other_spec')})" $OUT/vd/cmp_${1}_${2}_vs_$3.json 2>&1)"
}
for m in $MODELS; do
  n=${m%%:*}; g=${m#*:}
  D=EMBER_DETERMINISTIC=1
  col $n $g det_plain $D
  col $n $g det_k2 $D EMBER_SPEC=2
  col $n $g det_k4 $D EMBER_SPEC=4
  col $n $g det_k8 $D EMBER_SPEC=8
  col $n $g det_k4_rows $D EMBER_SPEC=4 EMBER_VERIFY_LINEAR=rows
  col $n $g off_plain EMBER_NOTHING=1
  col $n $g off_k4 EMBER_SPEC=4
  for o in det_k2 det_k4 det_k8 det_k4_rows; do vcmp $n det_plain $o; done
  vcmp $n off_plain off_k4
  rm -f /content/vdruns/${n}_*/rows.f32
done
pack out4

echo "--- (g) end to end $(date -u +%T)"
e2e() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/vf_e2e.py run /content/b_new/tinyllama_server $g -1 $port $SM/all140.jsonl $OUT/e2e/${n}_$tag.jsonl "$@" \
    > $OUT/e2e/${n}_$tag.txt 2>&1
  echo "e2e $n $tag rc=$? $(tail -1 $OUT/e2e/${n}_$tag.txt | cut -c1-260)"
  rm -rf $OUT/e2e/${n}_$tag.jsonl.run
}
for m in $MODELS; do
  n=${m%%:*}; g=${m#*:}
  e2e $n $g split EMBER_DECODE_ATTN=split
  e2e $n $g tiled EMBER_NOTHING=1
  e2e $n $g split2 EMBER_DECODE_ATTN=split
  e2e $n $g tiled2 EMBER_NOTHING=1
  e2e $n $g det_split EMBER_DETERMINISTIC=1 EMBER_DECODE_ATTN=split
  e2e $n $g det_tiled EMBER_DETERMINISTIC=1
done

echo "--- (h) interpreter dumps $(date -u +%T)"
dd() {  # name gguf n len gen
  local name=$1 g=$2 nn=$3 ln=$4 gen=$5
  [ -s "$g" ] || { echo "dump $name skipped: no $g"; return; }
  for b in head new; do
    /content/b_$b/ir_device_dump $g /content/dumps/${name}_$b.bin --n $nn --len $ln --gen $gen > $OUT/dumps/${name}_$b.log 2>&1 \
      || echo "dump $name $b failed: $(tail -1 $OUT/dumps/${name}_$b.log)"
  done
  if [ -s /content/dumps/${name}_head.bin ] && [ -s /content/dumps/${name}_new.bin ]; then
    python3 $SM/dump_compare.py /content/dumps/${name}_head.bin /content/dumps/${name}_new.bin $nn $ln $gen > $OUT/dumps/$name.json 2>&1
    echo "dump $name $(cut -c1-400 $OUT/dumps/$name.json)"
  fi
  sha256sum /content/dumps/${name}_*.bin >> $OUT/dumps.sha256 2>/dev/null
  rm -f /content/dumps/${name}_*.bin
}
dd ar_x5_q8_0 $M/ar3/ar_x5_q8_0.gguf 40 8 8
dd state4 $M/state4/state4.gguf 40 24 32
dd state4_q8_0 $M/state4/state4_q8_0.gguf 40 24 32
dd mor3 $M/mor/mor3.gguf 40 8 8
dd mor3_q8_0 $M/mor/mor3_q8_0.gguf 40 8 8
dd mor2p0 $M/mor/mor2p0.gguf 40 8 8
dd ar3 $M/ar3/ar3.gguf 40 8 8
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do dd rec_$m $M/recursion/$m.gguf 40 8 8; done
for f in mistral_swa gptneox gpt2 mamba gemma3; do dd syn_$f $M/synth/$f.gguf 40 24 32; done
dd gpt2_tiny $M/gpt2_tiny/gpt2_tiny.gguf 40 24 32

echo "--- pack $(date -u +%T)"
pack out
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
