#!/usr/bin/env bash
# VM side of the Carry-12 CUDA gates (ember.cpp, 27 Oct 2026). Launched detached by c12_drive.sh.
# Inputs in /content: COMMITS ("<pin> <head>"), ember_pin.tgz and ember_head.tgz (git archives), small.tgz and
# x5.partNN + BIG_SHA256SUMS (c12_stage.sh).
# Build, per tree: Release, CMAKE_CUDA_ARCHITECTURES = the device's; targets x5_device_logits (the X5 gate's tool,
#   added as a target), state_device_gate, ir_device_gate.
# The pin (ec3201b), per file (state4, state4_q8_0, ar3, ar3_q8_0, ar_x5, ar_x5_q8_0), in each of four
# configurations, {default mode, EMBER_DETERMINISTIC=1} x {decode attention default (tiled), EMBER_DECODE_ATTN=split}:
#   x5_device_logits twice (a, b: prefill, token-by-token decode through the captured graph, free-running greedy);
#   state_device_gate (device against the CPU in one process, with the session script on state4);
#   ir_device_gate on random ids (state4 40 x 24, 32 greedy; the others 40 x 8, 8 greedy).
#   EMBER_IR control (ar3, ar_x5 and their Q8_0 files, the mode): the unchanged document through EMBER_IR.
#   Long leg (ar3, ar_x5 and their Q8_0 files): the ctx-1024 document through EMBER_IR, long_in.json (8 x 600 ids,
#   300 free-running) in the four configurations (a and b under the mode), ir_device_gate 4 x 600 ids, 64 greedy
#   (mode tiled, default tiled, mode split).
#   nsys (--cuda-graph-trace=node) of x5_device_logits on state4, ar3 and ar_x5 (mode tiled, mode split, default
#   tiled) and of the long leg on ar_x5 (mode tiled): every kernel launched, by name.
# The head: per file, the mode with the default kernel: x5_device_logits and state_device_gate; the long leg on ar3
#   and ar_x5. Compared byte for byte with the pin's.
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; M=$SM/models; I=$SM/inputs
mkdir -p "$OUT/pin" "$OUT/head"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r PIN HEADC < /content/COMMITS
echo "=== vm $(date -u) pin=$PIN head=$HEADC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute capability $CC, nproc $(nproc)"
NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1 || true)
echo "nsys: ${NSYS:-none}"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cat /content/x5.part* > $M/ar_x5.gguf
(cd $M && sha256sum -c --quiet /content/BIG_SHA256SUMS) || fail "ar_x5 sha256"
cp $SM/SHA256SUMS $OUT/small.sha256; cp /content/BIG_SHA256SUMS $OUT/big.sha256

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"

build() {  # build TAG ARCHIVE: /content/b_TAG
  local tag=$1 tgz=$2 src=/content/src_$1 b=/content/b_$1 t0
  mkdir -p $src
  tar xzf $tgz -C $src --strip-components=1 || fail "untar $tag"
  find $src -name "*.sh" -exec sed -i 's/\r$//' {} +
  cp $SM/x5_device_logits.cpp $src/tools/
  printf '\nadd_executable(x5_device_logits tools/x5_device_logits.cpp)\ntarget_link_libraries(x5_device_logits PRIVATE tinyllama_core)\n' >> $src/CMakeLists.txt
  t0=$(date +%s)
  cmake -S $src -B $b -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$tag.log 2>&1 || { tail -20 $OUT/cmake_$tag.log; fail "cmake $tag"; }
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$tag.log || fail "$tag configured without CUDA"
  make -C $b -j$(nproc) x5_device_logits state_device_gate ir_device_gate > $OUT/make_$tag.log 2>&1 || { tail -30 $OUT/make_$tag.log; fail "build $tag"; }
  echo "built $tag (sm_$CC) in $(( $(date +%s) - t0 )) s"
  sha256sum $b/x5_device_logits $b/state_device_gate $b/ir_device_gate > $OUT/bins_$tag.sha256
}

cfg() {  # cfg MODE KERN COMMAND...: the configuration's environment
  local mode=$1 kern=$2; shift 2
  local e=()
  [ "$mode" = det ] && e+=(EMBER_DETERMINISTIC=1)
  [ "$kern" = split ] && e+=(EMBER_DECODE_ATTN=split)
  env "${e[@]}" "$@"
}
same() { cmp -s "$1" "$2" && echo "equal" || echo "DIFFER"; }
infile() { case $1 in state4) echo $I/state4_in.json;; state4_q8_0) echo $I/state4q8_in.json;;
  ar3|ar3_q8_0) echo $I/ar3_in.json;; ar_x5|ar_x5_q8_0) echo $I/ar_x5_in.json;; esac; }
dgargs() { case $1 in state4*) echo "--n 40 --len 24 --gen 32";; *) echo "--n 40 --len 8 --gen 8";; esac; }
sdg_line() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['summary']; print('pass', d['pass'], s.get('top1_agree'), s.get('free_greedy'), s.get('worst_prefill_on_rel'), s.get('worst_decode_rel'))" "$1" 2>/dev/null || echo "no report"; }
dg_line() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('pass', d['pass'], d['greedy_identical'], d['top1_agree'], d['worst_prefill_rel'], d['worst_decode_rel'])" "$1" 2>/dev/null || echo "no report"; }

FILES="state4 state4_q8_0 ar3 ar3_q8_0 ar_x5 ar_x5_q8_0"
LONG="ar3 ar3_q8_0 ar_x5 ar_x5_q8_0"
NSYS_SHORT="state4 ar3 ar_x5"
# /content/LEGS = "long" (c12_stage.sh's third argument): the long leg alone (its nsys run and the head's long runs)
LEGS=all; [ -s /content/LEGS ] && read -r LEGS < /content/LEGS
echo "legs: $LEGS"
[ "$LEGS" = long ] && { FILES=""; NSYS_SHORT=""; }
echo "--- build pin $(date -u +%T)"
build pin /content/ember_pin.tgz
B=/content/b_pin; P=$OUT/pin
for v in $FILES; do
  G=$M/$v.gguf; IN=$(infile $v)
  echo "--- pin $v $(date -u +%T)"
  for mode in def det; do for kern in tiled split; do
    k=${v}_${mode}_${kern}
    for r in a b; do
      cfg $mode $kern timeout 900 $B/x5_device_logits $G < $IN > $P/dev_${k}_$r.json 2> $P/dev_${k}_$r.log || echo "x5_device_logits $k $r rc=$?"
    done
    echo "$k a vs b: $(same $P/dev_${k}_a.json $P/dev_${k}_b.json)"
    cfg $mode $kern timeout 900 $B/state_device_gate $G < $IN > $P/sdg_$k.json 2> $P/sdg_$k.log
    echo "state_device_gate $k: $(sdg_line $P/sdg_$k.json)"
    cfg $mode $kern timeout 900 $B/ir_device_gate $G $(dgargs $v) > $P/dg_$k.json 2> $P/dg_$k.log
    echo "ir_device_gate $k: $(dg_line $P/dg_$k.json)"
  done; done
  echo "$v mode tiled vs mode split: $(same $P/dev_${v}_det_tiled_a.json $P/dev_${v}_det_split_a.json)"
  case $v in ar*)
    base=${v%_q8_0}
    EMBER_DETERMINISTIC=1 EMBER_IR=$I/$base.ir.json timeout 900 $B/x5_device_logits $G < $IN > $P/dev_${v}_det_tiled_irdoc.json 2> $P/dev_${v}_det_tiled_irdoc.log
    echo "$v EMBER_IR unchanged document vs embedded (mode): $(same $P/dev_${v}_det_tiled_irdoc.json $P/dev_${v}_det_tiled_a.json)";;
  esac
done
for v in $LONG; do
  G=$M/$v.gguf; base=${v%_q8_0}; D=$I/${base}_ctx1024.ir.json
  echo "--- pin long $v $(date -u +%T)"
  for mode in def det; do for kern in tiled split; do
    k=${v}_${mode}_${kern}
    rs=a; [ $mode = det ] && rs="a b"
    for r in $rs; do
      cfg $mode $kern env EMBER_IR=$D timeout 1200 $B/x5_device_logits $G < $I/long_in.json > $P/long_${k}_$r.json 2> $P/long_${k}_$r.log || echo "long x5_device_logits $k $r rc=$?"
    done
    [ $mode = det ] && echo "long $k a vs b: $(same $P/long_${k}_a.json $P/long_${k}_b.json)"
  done; done
  for mk in det_tiled def_tiled det_split; do
    k=${v}_$mk
    cfg ${mk%_*} ${mk#*_} env EMBER_IR=$D timeout 1800 $B/ir_device_gate $G --n 4 --len 600 --gen 64 > $P/dgl_$k.json 2> $P/dgl_$k.log
    echo "long ir_device_gate $k: $(dg_line $P/dgl_$k.json)"
  done
done
if [ -n "$NSYS" ]; then
  mkdir -p $OUT/nsys
  for v in $NSYS_SHORT; do
    for mk in det_tiled det_split def_tiled; do
      k=${v}_$mk
      cfg ${mk%_*} ${mk#*_} timeout 900 $NSYS profile -t cuda --cuda-graph-trace=node -o /content/ns_$k --force-overwrite true \
        bash -c "exec $B/x5_device_logits $M/$v.gguf < $(infile $v) > $OUT/nsys/$k.json" > $OUT/nsys/$k.log 2>&1
      timeout 900 $NSYS stats -r cuda_gpu_kern_sum -f csv -o $OUT/nsys/$k /content/ns_$k.nsys-rep >> $OUT/nsys/$k.log 2>&1
      echo "nsys $k rc=$? attention kernels: $(grep -o '[a-z_]*attn[a-z_]*kernel\|[a-z_]*attention[a-z_]*kernel' $OUT/nsys/${k}_cuda_gpu_kern_sum.csv 2>/dev/null | sort -u | tr '\n' ' ')"
      rm -f /content/ns_$k.nsys-rep /content/ns_$k.sqlite
    done
  done
  k=long_ar_x5_det_tiled
  EMBER_DETERMINISTIC=1 EMBER_IR=$I/ar_x5_ctx1024.ir.json timeout 1200 $NSYS profile -t cuda --cuda-graph-trace=node -o /content/ns_$k \
    --force-overwrite true bash -c "exec $B/x5_device_logits $M/ar_x5.gguf < $I/long_in.json > /dev/null" > $OUT/nsys/$k.log 2>&1
  timeout 900 $NSYS stats -r cuda_gpu_kern_sum -f csv -o $OUT/nsys/$k /content/ns_$k.nsys-rep >> $OUT/nsys/$k.log 2>&1
  echo "nsys $k rc=$? attention kernels: $(grep -o '[a-z_]*attn[a-z_]*kernel\|[a-z_]*attention[a-z_]*kernel' $OUT/nsys/${k}_cuda_gpu_kern_sum.csv 2>/dev/null | sort -u | tr '\n' ' ')"
  rm -f /content/ns_$k.nsys-rep /content/ns_$k.sqlite
fi

echo "--- build head $(date -u +%T)"
build head /content/ember_head.tgz
B=/content/b_head; Hd=$OUT/head
for v in $FILES; do
  G=$M/$v.gguf; IN=$(infile $v); k=${v}_det_tiled
  EMBER_DETERMINISTIC=1 timeout 900 $B/x5_device_logits $G < $IN > $Hd/dev_${k}_a.json 2> $Hd/dev_${k}_a.log
  echo "head vs pin $k: $(same $Hd/dev_${k}_a.json $P/dev_${k}_a.json)"
  EMBER_DETERMINISTIC=1 timeout 900 $B/state_device_gate $G < $IN > $Hd/sdg_$k.json 2> $Hd/sdg_$k.log
  echo "head state_device_gate $k: $(sdg_line $Hd/sdg_$k.json)"
done
for v in ar3 ar_x5; do
  k=${v}_det_tiled
  EMBER_DETERMINISTIC=1 EMBER_IR=$I/${v}_ctx1024.ir.json timeout 1200 $B/x5_device_logits $M/$v.gguf < $I/long_in.json > $Hd/long_${k}_a.json 2> $Hd/long_${k}_a.log
  echo "head vs pin long $k: $(same $Hd/long_${k}_a.json $P/long_${k}_a.json)"
done

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
