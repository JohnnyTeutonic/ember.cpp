#!/usr/bin/env bash
# VM side of draft_2026-10-27 (ember.cpp). Launched detached by drive.sh while the two models upload.
#   bash vm.sh
# Inputs in /content: ember_new.tgz (ember main plus EMBER_DRAFT=table), COMMITS, small.tgz (scripts, request sets,
# the draft tables, config.sh: TABLE_pick, TABLE_pk, THETA); later pick.part*, pk.part*, BIG_SHA256SUMS and the
# marker UPLOADS_DONE. The tree is built Release for the GPU's own architecture. Per model (pick_e1_L16, then
# pk_base_v1), over eval100.jsonl (chat40, copy20, carry40: 300 requests):
#   (a) dr_e2e.py without the dump (ids, speed, drafter time): plain decode; prompt lookup (EMBER_SPEC=k, the
#       default drafter) and EMBER_DRAFT=table at k = 4, 8, 15; the table at k = 8 with theta 0.1; plain decode
#       again; the same under
#       EMBER_DETERMINISTIC=1 at k = 4, 8.
#   (b) dr_e2e.py --dump (every sampled row's logits) under EMBER_DETERMINISTIC=1: plain decode, EMBER_DRAFT=table at
#       k = 4 and 8, compared bit for bit (dr_e2e.py bits).
# Outputs under /content/out, packed as out1.zip (after pick_e1_L16), out2.zip (after pk_base_v1), then out.zip.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; BIG=/content/big; RUNS=/content/runs
mkdir -p "$OUT/e2e" "$OUT/bits" "$BIG" "$RUNS"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
pack() {  # name
  (cd /content && rm -f $1.zip && python3 -c "import shutil; shutil.make_archive('/content/$1', 'zip', '/content', 'out')" \
     && touch /content/${1^^}_READY && echo "packed $1 $(du -sh $1.zip | cut -f1)")
}
read -r NEWC < /content/COMMITS
echo "=== vm $(date -u) new=$NEWC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "gpu $GPU, compute capability $CC, nproc $(nproc)"
echo "$GPU $CC" > $OUT/gpu.txt
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
. $SM/config.sh
echo "tables: pick $TABLE_pick, pk $TABLE_pk; theta $THETA"

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

port=8800
e2e() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/dr_e2e.py run $BIN $g -1 $port $SM/eval100.jsonl $OUT/e2e/${n}_$tag.jsonl "$@" > $OUT/e2e/${n}_$tag.txt 2>&1
  echo "e2e $n $tag rc=$? $(tail -1 $OUT/e2e/${n}_$tag.txt | cut -c1-300)"
  rm -rf $OUT/e2e/${n}_$tag.jsonl.run
}
dump() {  # model gguf tag env...
  local n=$1 g=$2 tag=$3; shift 3
  port=$((port + 1))
  python3 $SM/dr_e2e.py run $BIN $g -1 $port $SM/eval100.jsonl $RUNS/${n}_$tag.jsonl --dump "$@" > $OUT/bits/${n}_$tag.txt 2>&1
  echo "dump $n $tag rc=$? $(tail -1 $OUT/bits/${n}_$tag.txt | cut -c1-300)"
  cp $RUNS/${n}_$tag.jsonl $RUNS/${n}_$tag.jsonl.meta.json $OUT/bits/ 2>/dev/null
  sha256sum $RUNS/${n}_$tag.jsonl.rows.f32 >> $OUT/rows.sha256 2>/dev/null
  rm -rf $RUNS/${n}_$tag.jsonl.run
}
bits() {  # model base other
  python3 $SM/dr_e2e.py bits $RUNS/${1}_$2.jsonl $RUNS/${1}_$3.jsonl > $OUT/bits/${1}_${2}_vs_$3.json 2> $OUT/bits/${1}_${2}_vs_$3.err
  echo "bits $1 $2 vs $3 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ('bit_identical','requests','ids_identical','requests_all_rows_bit_identical','rows_compared','rows_bit_identical','max_abs_diff','other_spec')})" $OUT/bits/${1}_${2}_vs_$3.json 2>&1)"
}
pk_n=0
for m in pick:$BIG/pick_L16.gguf pk:$BIG/pk_base_v1.gguf; do
  n=${m%%:*}; g=${m#*:}
  pk_n=$((pk_n + 1))
  TV=TABLE_$n; T=$SM/${!TV}
  TAB="EMBER_DRAFT=table EMBER_DRAFT_TABLE=$T EMBER_DRAFT_THETA=$THETA"
  echo "--- (a) $n end to end $(date -u +%T)"
  e2e $n $g off EMBER_NOTHING=1
  for k in 4 8 15; do
    e2e $n $g pld_k$k EMBER_SPEC=$k
    e2e $n $g table_k$k EMBER_SPEC=$k $TAB
  done
  e2e $n $g table_k8_t10 EMBER_SPEC=8 EMBER_DRAFT=table EMBER_DRAFT_TABLE=$T EMBER_DRAFT_THETA=0.1
  e2e $n $g off2 EMBER_NOTHING=1
  D=EMBER_DETERMINISTIC=1
  e2e $n $g det_off $D
  for k in 4 8; do
    e2e $n $g det_pld_k$k $D EMBER_SPEC=$k
    e2e $n $g det_table_k$k $D EMBER_SPEC=$k $TAB
  done
  for o in pld_k4 pld_k8 pld_k15 table_k4 table_k8 table_k15 table_k8_t10 off2; do
    echo "compare $n off vs $o: $(python3 $SM/dr_e2e.py compare $OUT/e2e/${n}_off.jsonl $OUT/e2e/${n}_$o.jsonl | tee $OUT/e2e/${n}_off_vs_$o.json | cut -c1-300)"
  done
  for o in det_pld_k4 det_pld_k8 det_table_k4 det_table_k8; do
    echo "compare $n det_off vs $o: $(python3 $SM/dr_e2e.py compare $OUT/e2e/${n}_det_off.jsonl $OUT/e2e/${n}_$o.jsonl | tee $OUT/e2e/${n}_det_off_vs_$o.json | cut -c1-300)"
  done
  echo "--- (b) $n deterministic rows $(date -u +%T)"
  dump $n $g det_plain $D
  for k in 4 8; do
    dump $n $g det_table_k$k $D EMBER_SPEC=$k $TAB
    bits $n det_plain det_table_k$k
    rm -f $RUNS/${n}_det_table_k$k.jsonl.rows.f32
  done
  rm -f $RUNS/${n}_*
  pack out$pk_n
done

echo "--- pack $(date -u +%T)"
pack out
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
