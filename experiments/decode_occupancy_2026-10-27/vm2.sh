#!/usr/bin/env bash
# VM side of decode_occupancy_2026-10-27's second session (ember.cpp): the legs the first session could not give.
#   bash vm.sh      (staged under that name by stage2.sh)
# Inputs in /content as for vm.sh (ember_new.tgz is the same tree plus the decode_attention_test change). The first
# session's VM was reclaimed during its end-to-end leg, so its legs (g) and (h) run here again. Legs (outputs under
# /content/out; out1.zip .. out4.zip as they finish, then out.zip):
#   (a) tools/decode_attention_test: tiled, split, legacy, legacy under EMBER_DETERMINISTIC=1, the default under
#       the mode, and EMBER_DECODE_ATTN_TILE=16,64;
#   (h) ir_device_dump on the interpreter's local gate models, HEAD against this build (dump_compare.py);
#   (g) end to end (vf_e2e.py): all140, split against tiled, alternating, and both under EMBER_DETERMINISTIC=1;
#   (b) occ_step under EMBER_DECODE_ATTN_TILE=16,64 and 64,64 (the first session's legs split the value on the
#       comma and did not run);
#   (c) the decode attention kernels in situ: occ_step at one prefix per run (33, 129, 257, 385, 501 keys) under
#       nsys with --cuda-graph-trace=node, split against tiled, each kernel's time inside the captured decode graph.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; BIG=/content/big; SM=/content/small; M=$SM/models
mkdir -p "$OUT"/{dat,step,insitu,e2e,dumps} "$BIG" /content/run /content/dumps
pack() {  # name
  (cd /content && rm -f $1.zip && python3 -c "import shutil; shutil.make_archive('/content/$1', 'zip', '/content', 'out')"      && touch /content/${1^^}_READY && echo "packed $1 $(du -sh $1.zip | cut -f1)")
}
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r NEWC HEADC < /content/COMMITS
echo "=== vm2 $(date -u) new=$NEWC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute capability $CC, nproc $(nproc)"
NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1 || true)
echo "nsys: ${NSYS:-none}"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
mkdir -p /content/new /content/head
tar xzf ember_new.tgz -C /content/new --strip-components=1 && tar xzf ember_head.tgz -C /content/head --strip-components=1 || fail "untar ember"
t0=$(date +%s)
for t in new head; do
  tg="tinyllama_server ir_device_dump"
  [ $t = new ] && tg="$tg occ_step"
  cmake -S /content/$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$t.log 2>&1 || fail "cmake $t"
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
  make -C /content/b_$t -j"$(nproc)" $tg > $OUT/make_$t.log 2>&1 || { tail -30 $OUT/make_$t.log; fail "build $t"; }
done
(cd /content/new && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. tools/decode_attention_test.cu cuda_kernels.cu logger.cpp -lcublas \
   -o /content/b_new/decode_attention_test > $OUT/nvcc_dat.log 2>&1) || fail "nvcc test"
echo "built in $(( $(date +%s) - t0 )) s"
sha256sum /content/b_*/tinyllama_server /content/b_*/ir_device_dump /content/b_new/occ_step /content/b_new/decode_attention_test > $OUT/bins.sha256

echo "--- (a) decode_attention_test $(date -u +%T)"
dat() {  # tag env...
  local tag=$1; shift
  env "$@" /content/b_new/decode_attention_test > $OUT/dat/dat_$tag.txt 2>&1
  echo "dat $tag rc=$? $(head -1 $OUT/dat/dat_$tag.txt | cut -c1-100) | $(tail -1 $OUT/dat/dat_$tag.txt)"
}
dat tiled EMBER_NOTHING=1
dat split EMBER_DECODE_ATTN=split
dat legacy EMBER_DECODE_ATTN=legacy
dat legacy_det EMBER_DECODE_ATTN=legacy EMBER_DETERMINISTIC=1
dat det EMBER_DETERMINISTIC=1
dat tile16 EMBER_DECODE_ATTN_TILE=16,64

echo "--- (h) interpreter dumps $(date -u +%T)"
dd() {  # name gguf n len gen
  local name=$1 g=$2 nn=$3 ln=$4 gen=$5
  [ -s "$g" ] || { echo "dump $name skipped: no $g"; return; }
  for b in head new; do
    /content/b_$b/ir_device_dump $g /content/dumps/${name}_$b.bin --n $nn --len $ln --gen $gen > $OUT/dumps/${name}_$b.log 2>&1       || echo "dump $name $b failed: $(tail -1 $OUT/dumps/${name}_$b.log)"
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
pack out1

echo "--- waiting for the models $(date -u +%T)"
for i in $(seq 1 540); do [ -e /content/UPLOADS_DONE ] && break; sleep 10; done
[ -e /content/UPLOADS_DONE ] || fail "models never arrived"
cat pick.part* > $BIG/pick_L16.gguf && cat m60.part* > $BIG/pick_60m_L32.gguf || fail "reassemble"
(cd $BIG && sha256sum -c /content/BIG_SHA256SUMS) > $OUT/big.sha256 2>&1 || fail "big sha256"
rm -f pick.part* m60.part*
MODELS="pick:$BIG/pick_L16.gguf m60:$BIG/pick_60m_L32.gguf"

echo "--- (g) end to end $(date -u +%T)"
port=8700
e2e() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/vf_e2e.py run /content/b_new/tinyllama_server $g -1 $port $SM/all140.jsonl $OUT/e2e/${n}_$tag.jsonl "$@"     > $OUT/e2e/${n}_$tag.txt 2>&1
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
pack out2

echo "--- (b) occ_step, other tilings $(date -u +%T)"
cd /content/run && ln -sf /dev/null debugging.log
for m in $MODELS; do
  n=${m%%:*}; g=${m#*:}
  for t in t16:16,64 t64:64,64 tiled:32,64; do
    tag=${t%%:*}
    EMBER_DECODE_ATTN_TILE=${t#*:} /content/b_new/occ_step $g $OUT/step/${n}_$tag.json 300 32,128,256,384,500 > $OUT/step/${n}_$tag.txt 2>&1
    echo "step $n $tag rc=$? $(tr '\n' ' ' < $OUT/step/${n}_$tag.txt | cut -c1-300)"
  done
done

pack out3

echo "--- (c) in situ under nsys $(date -u +%T)"
if [ -n "${NSYS:-}" ]; then
  for m in $MODELS; do
    n=${m%%:*}; g=${m#*:}
    for P in 32 128 256 384 500; do
      for k in split tiled; do
        e=EMBER_NOTHING=1; [ $k = split ] && e=EMBER_DECODE_ATTN=split
        r=/content/ns_${n}_${k}_$P
        env $e timeout 600 $NSYS profile -t cuda,nvtx --cuda-graph-trace=node -o $r --force-overwrite true \
          /content/b_new/occ_step $g $r.json 100 $P > $OUT/insitu/${n}_${k}_$P.log 2>&1
        timeout 600 $NSYS stats -r cuda_gpu_kern_sum -f csv -o $OUT/insitu/${n}_${k}_$P $r.nsys-rep >> $OUT/insitu/${n}_${k}_$P.log 2>&1
        echo "insitu $n $k $P rc=$? $(grep -E 'decode_attn|cublas|gemv' $OUT/insitu/${n}_${k}_${P}_cuda_gpu_kern_sum.csv 2>/dev/null | head -2 | cut -c1-160 | tr '\n' ' ')"
        rm -f $r.nsys-rep $r.sqlite
      done
    done
  done
fi
cd /content
echo "--- pack $(date -u +%T)"
pack out4
pack out
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
