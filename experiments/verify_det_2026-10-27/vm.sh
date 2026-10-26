#!/usr/bin/env bash
# VM side of verify_det_2026-10-27 (ember.cpp). Launched detached by drive.sh while the two models upload.
#   bash vm.sh
# Inputs in /content: ember_new.tgz (ember main 345d7f1 plus this change), COMMITS, small.tgz (scripts, all60.jsonl);
# later pick.part*, pk.part*, BIG_SHA256SUMS and the marker UPLOADS_DONE. The tree is built Release for the GPU's own
# architecture. Per model (pick_e1_L16, pk_base_v1), over the 60 requests of all60.jsonl (chat40 + copy20):
#   (a) vd_collect.py, every sampled row's logits: EMBER_DETERMINISTIC=1 plain decode twice, k = 2, 4, 8 with
#       EMBER_VERIFY_LINEAR unset (the mode's default, rows), k = 4 with rows and with smallm set explicitly; default
#       mode plain and k = 4. Each compared with vd_compare.py against the plain run of its mode.
#   (b) vf_e2e.py without the dump (ids and speed): the mode's plain decode and k = 2, 4, 8.
# Outputs under /content/out, packed as out.zip.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; BIG=/content/big; RUNS=/content/runs
mkdir -p "$OUT/collect" "$OUT/cmp" "$OUT/e2e" "$BIG" "$RUNS"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r NEWC < /content/COMMITS
echo "=== vm $(date -u) new=$NEWC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "gpu $GPU, compute capability $CC, nproc $(nproc)"
echo "$GPU $CC" > $OUT/gpu.txt
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"

echo "--- build $(date -u +%T)"
mkdir -p /content/new
tar xzf ember_new.tgz -C /content/new --strip-components=1 || fail "untar ember"
t0=$(date +%s)
cmake -S /content/new -B /content/b_new -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || { tail -20 $OUT/cmake.log; fail "cmake"; }
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log || fail "configured without CUDA"
make -C /content/b_new -j"$(nproc)" tinyllama_server > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built in $(( $(date +%s) - t0 )) s"
BIN=/content/b_new/tinyllama_server
sha256sum $BIN > $OUT/bins.sha256

echo "--- waiting for the models $(date -u +%T)"
for i in $(seq 1 540); do [ -e /content/UPLOADS_DONE ] && break; sleep 10; done
[ -e /content/UPLOADS_DONE ] || fail "models never arrived"
cat pick.part* > $BIG/pick_L16.gguf && cat pk.part* > $BIG/pk_base_v1.gguf || fail "reassemble"
(cd $BIG && sha256sum -c /content/BIG_SHA256SUMS) > $OUT/big.sha256 2>&1 || fail "big sha256"
rm -f pick.part* pk.part*

port=8700
col() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/vd_collect.py $BIN $g -1 $port $SM/all60.jsonl $RUNS/${n}_$tag "$@" > $OUT/collect/${n}_$tag.txt 2>&1
  echo "collect $n $tag rc=$? $(tail -1 $OUT/collect/${n}_$tag.txt | cut -c1-240)"
  mkdir -p $OUT/collect/${n}_$tag
  cp $RUNS/${n}_$tag/records.jsonl $RUNS/${n}_$tag/meta.json $RUNS/${n}_$tag/server.log $OUT/collect/${n}_$tag/ 2>/dev/null
  sha256sum $RUNS/${n}_$tag/rows.f32 >> $OUT/rows.sha256 2>/dev/null
}
cmpr() {  # model base other
  python3 $SM/vd_compare.py $RUNS/${1}_$2 $RUNS/${1}_$3 > $OUT/cmp/${1}_${2}_vs_$3.json 2> $OUT/cmp/${1}_${2}_vs_$3.err
  echo "compare $1 $2 vs $3 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ('bit_identical','requests','ids_identical','replies_identical','requests_all_rows_bit_identical','rows_compared','rows_bit_identical','max_abs_diff','argmax_differs','other_spec')})" $OUT/cmp/${1}_${2}_vs_$3.json 2>&1)"
}
e2e() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/vf_e2e.py run $BIN $g -1 $port $SM/all60.jsonl $OUT/e2e/${n}_$tag.jsonl "$@" > $OUT/e2e/${n}_$tag.txt 2>&1
  echo "e2e $n $tag rc=$? $(tail -1 $OUT/e2e/${n}_$tag.txt | cut -c1-240)"
  rm -rf $OUT/e2e/${n}_$tag.jsonl.run
}
for m in pick:$BIG/pick_L16.gguf pk:$BIG/pk_base_v1.gguf; do
  n=${m%%:*}; g=${m#*:}
  echo "--- (a) $n dumps $(date -u +%T)"
  D=EMBER_DETERMINISTIC=1
  col $n $g det_plain $D
  col $n $g det_plain2 $D
  col $n $g det_k2 $D EMBER_SPEC=2
  col $n $g det_k4 $D EMBER_SPEC=4
  col $n $g det_k8 $D EMBER_SPEC=8
  col $n $g det_k4_rows $D EMBER_SPEC=4 EMBER_VERIFY_LINEAR=rows
  col $n $g det_k4_smallm $D EMBER_SPEC=4 EMBER_VERIFY_LINEAR=smallm
  col $n $g off_plain EMBER_NOTHING=1
  col $n $g off_k4 EMBER_SPEC=4
  for o in det_plain2 det_k2 det_k4 det_k8 det_k4_rows det_k4_smallm; do cmpr $n det_plain $o; done
  cmpr $n off_plain off_k4
  rm -rf $RUNS/${n}_*
  echo "--- (b) $n ids and speed without the dump $(date -u +%T)"
  e2e $n $g det_off EMBER_DETERMINISTIC=1
  e2e $n $g det_k2 EMBER_DETERMINISTIC=1 EMBER_SPEC=2
  e2e $n $g det_k4 EMBER_DETERMINISTIC=1 EMBER_SPEC=4
  e2e $n $g det_k8 EMBER_DETERMINISTIC=1 EMBER_SPEC=8
  for k in 2 4 8; do
    echo "e2e compare $n det_off vs det_k$k: $(python3 $SM/vf_e2e.py compare $OUT/e2e/${n}_det_off.jsonl $OUT/e2e/${n}_det_k$k.jsonl | tee $OUT/e2e/${n}_det_off_vs_det_k$k.json | cut -c1-400)"
  done
done

echo "--- pack $(date -u +%T)"
(cd /content && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')")
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
