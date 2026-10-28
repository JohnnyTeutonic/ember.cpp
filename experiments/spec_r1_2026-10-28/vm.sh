#!/usr/bin/env bash
# VM side of spec_r1_2026-10-28 session 1 (ember.cpp). Launched detached by drive.sh while the two models upload.
#   bash vm.sh
# Inputs in /content: ember_head.tgz (git archive of ember main), ember_new.tgz (main plus EMBER_DECODE_LINEAR and
# tools/r1_bench.cpp), COMMITS, small.tgz (scripts, request sets, the two draft tables, the two pools); later
# pick.part*, pk.part*, BIG_SHA256SUMS and the marker UPLOADS_DONE. Both trees built Release for the GPU's own
# architecture. Per model (pick_e1_L16, then pk_base_v1):
#   (a) r1_bench in the default mode and under EMBER_DETERMINISTIC=1 (decode step by linear, pass by linear and n,
#       bits of every pass row against the decode step);
#   (b) nsys --cuda-graph-trace=node under EMBER_DETERMINISTIC=1: decode steps (cublas, fixed) and passes (rows,
#       fixed, smallm) at P 384 and P 32, each kernel's time inside the captured graphs;
#   (c) end to end over eval100 (chat40, copy20, carry40; 260 records), dr_e2e.py: main's binary in the default mode
#       (plain, prompt lookup k 8, table k 8) and under the mode (plain, prompt lookup k 4 and 8, table k 4 and 8: the
#       re-measure of the `rows` pass); the new binary under the mode (the same five, on fixed_linear) and in the
#       default mode (plain, table k 8; unchanged);
#   (d) dumps under the mode (every sampled row): new plain twice (run to run), new prompt lookup k 8 and table k 4
#       and 8 against new plain; main plain against new plain with EMBER_DECODE_LINEAR=cublas (the old mode kept);
#       main plain against new plain (ids, distance);
#   (e) the self-output pool: the new binary, default mode, EMBER_SPEC=8 with the table (ids are plain decode's),
#       every pool request once, ids kept.
# Outputs under /content/out, packed as out1.zip (after pick_e1_L16), out2.zip (after pk_base_v1), then out.zip.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; BIG=/content/big; RUNS=/content/runs
mkdir -p "$OUT/e2e" "$OUT/bits" "$OUT/bench" "$OUT/nsys" "$OUT/pool" "$BIG" "$RUNS"
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
. $SM/config.sh
echo "tables: pick $TABLE_pick, pk $TABLE_pk"

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"

echo "--- build $(date -u +%T)"
mkdir -p /content/new /content/head
tar xzf ember_new.tgz -C /content/new --strip-components=1 || fail "untar new"
tar xzf ember_head.tgz -C /content/head || fail "untar head"
t0=$(date +%s)
for t in new head; do
  cmake -S /content/$t -B /content/b_$t -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_$t.log 2>&1 \
    || { tail -20 $OUT/cmake_$t.log; fail "cmake $t"; }
  grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_$t.log || fail "$t configured without CUDA"
done
J=$(( $(nproc) / 2 ))
make -C /content/b_new -j$J tinyllama_server r1_bench > $OUT/make_new.log 2>&1 &
pn=$!
make -C /content/b_head -j$J tinyllama_server > $OUT/make_head.log 2>&1 || { tail -30 $OUT/make_head.log; fail "build head"; }
wait $pn || { tail -30 $OUT/make_new.log; fail "build new"; }
echo "built in $(( $(date +%s) - t0 )) s"
BN=/content/b_new/tinyllama_server; BH=/content/b_head/tinyllama_server; BENCH=/content/b_new/r1_bench
sha256sum $BN $BH $BENCH > $OUT/bins.sha256
for t in new head; do cuobjdump -sass /content/b_$t/tinyllama_server > /content/sass_$t.txt 2> $OUT/cuobjdump_$t.log; done
python3 $SM/sass_compare.py /content/sass_head.txt /content/sass_new.txt > $OUT/sass.json 2> $OUT/sass.err
echo "sass: $(python3 -c "import json; d=json.load(open('$OUT/sass.json')); print({k: v for k, v in d.items() if not isinstance(v, (list, dict))})" 2>&1 | cut -c1-300)"
rm -f /content/sass_*.txt

echo "--- waiting for the models $(date -u +%T)"
for i in $(seq 1 540); do [ -e /content/UPLOADS_DONE ] && break; sleep 10; done
[ -e /content/UPLOADS_DONE ] || fail "models never arrived"
cat pick.part* > $BIG/pick_L16.gguf && cat pk.part* > $BIG/pk_base_v1.gguf || fail "reassemble"
(cd $BIG && sha256sum -c /content/BIG_SHA256SUMS) > $OUT/big.sha256 2>&1 || fail "big sha256"
rm -f pick.part* pk.part*

port=8800
e2e() {  # binary model gguf tag env...
  local b=$1 n=$2 g=$3 tag=$4; shift 4
  port=$((port + 1))
  python3 $SM/dr_e2e.py run $b $g -1 $port $SM/eval100.jsonl $OUT/e2e/${n}_$tag.jsonl "$@" > $OUT/e2e/${n}_$tag.txt 2>&1
  echo "e2e $n $tag rc=$? $(tail -1 $OUT/e2e/${n}_$tag.txt | cut -c1-260)"
  rm -rf $OUT/e2e/${n}_$tag.jsonl.run
}
dump() {  # binary model gguf tag env...
  local b=$1 n=$2 g=$3 tag=$4; shift 4
  port=$((port + 1))
  python3 $SM/dr_e2e.py run $b $g -1 $port $SM/eval100.jsonl $RUNS/${n}_$tag.jsonl --dump "$@" > $OUT/bits/${n}_$tag.txt 2>&1
  echo "dump $n $tag rc=$? $(tail -1 $OUT/bits/${n}_$tag.txt | cut -c1-200)"
  cp $RUNS/${n}_$tag.jsonl $RUNS/${n}_$tag.jsonl.meta.json $OUT/bits/ 2>/dev/null
  sha256sum $RUNS/${n}_$tag.jsonl.rows.f32 >> $OUT/rows.sha256 2>/dev/null
  rm -rf $RUNS/${n}_$tag.jsonl.run
}
bits() {  # model base other
  python3 $SM/dr_e2e.py bits $RUNS/${1}_$2.jsonl $RUNS/${1}_$3.jsonl > $OUT/bits/${1}_${2}_vs_$3.json 2> $OUT/bits/${1}_${2}_vs_$3.err
  echo "bits $1 $2 vs $3 rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ('bit_identical','requests','ids_identical','requests_all_rows_bit_identical','rows_compared','rows_bit_identical','max_abs_diff','other_spec')})" $OUT/bits/${1}_${2}_vs_$3.json 2>&1)"
}
cmp() {  # model base other
  echo "compare $1 $2 vs $3: $(python3 $SM/dr_e2e.py compare $OUT/e2e/${1}_$2.jsonl $OUT/e2e/${1}_$3.jsonl | tee $OUT/e2e/${1}_${2}_vs_$3.json | cut -c1-260)"
}
D=EMBER_DETERMINISTIC=1
# the new binary's deterministic runs on fixed_linear (opt-in since the sessions; it was then the mode's default)
DN="EMBER_DETERMINISTIC=1 EMBER_DECODE_LINEAR=fixed"
pk_n=0
for m in pick:$BIG/pick_L16.gguf pk:$BIG/pk_base_v1.gguf; do
  n=${m%%:*}; g=${m#*:}
  pk_n=$((pk_n + 1))
  TV=TABLE_$n; T=$SM/${!TV}
  TAB="EMBER_DRAFT=table EMBER_DRAFT_TABLE=$T EMBER_DRAFT_THETA=0.02"

  echo "--- (a) $n bench $(date -u +%T)"
  (cd /content && timeout 900 $BENCH $g $OUT/bench/${n}_default.json 200 32,384 > $OUT/bench/${n}_default.log 2>&1)
  echo "bench $n default rc=$?"
  (cd /content && env $D timeout 900 $BENCH $g $OUT/bench/${n}_det.json 200 32,384 > $OUT/bench/${n}_det.log 2>&1)
  echo "bench $n det rc=$?"

  echo "--- (b) $n nsys $(date -u +%T)"
  if [ -n "${NSYS:-}" ]; then
    for P in 384 32; do
      for cfg in "decode cublas" "decode fixed" "pass rows 5" "pass rows 9" "pass fixed 5" "pass fixed 9" "pass smallm 9"; do
        tag=$(echo "$cfg" | tr ' ' '_')_$P
        r=/content/ns_${n}_$tag
        (cd /content && env $D timeout 600 $NSYS profile -t cuda,nvtx --cuda-graph-trace=node -o $r --force-overwrite true \
          $BENCH $g $OUT/nsys/${n}_$tag.json 100 $P profile $cfg > $OUT/nsys/${n}_$tag.log 2>&1)
        timeout 600 $NSYS stats -r cuda_gpu_kern_sum -f csv -o $OUT/nsys/${n}_$tag $r.nsys-rep >> $OUT/nsys/${n}_$tag.log 2>&1
        echo "nsys $n $tag rc=$?"
        rm -f $r.nsys-rep $r.sqlite
      done
    done
  fi

  echo "--- (c) $n end to end $(date -u +%T)"
  e2e $BH $n $g h_off EMBER_NOTHING=1
  e2e $BH $n $g h_pld_k8 EMBER_SPEC=8
  e2e $BH $n $g h_table_k8 EMBER_SPEC=8 $TAB
  e2e $BH $n $g h_det_off $D
  for k in 4 8; do
    e2e $BH $n $g h_det_pld_k$k $D EMBER_SPEC=$k
    e2e $BH $n $g h_det_table_k$k $D EMBER_SPEC=$k $TAB
  done
  e2e $BN $n $g n_det_off $DN
  for k in 4 8; do
    e2e $BN $n $g n_det_pld_k$k $DN EMBER_SPEC=$k
    e2e $BN $n $g n_det_table_k$k $DN EMBER_SPEC=$k $TAB
  done
  e2e $BN $n $g n_off EMBER_NOTHING=1
  e2e $BN $n $g n_table_k8 EMBER_SPEC=8 $TAB
  e2e $BH $n $g h_off2 EMBER_NOTHING=1
  e2e $BN $n $g n_det_off2 $DN
  for o in h_pld_k8 h_table_k8 n_off n_table_k8 h_off2; do cmp $n h_off $o; done
  for o in h_det_pld_k4 h_det_pld_k8 h_det_table_k4 h_det_table_k8 n_det_off; do cmp $n h_det_off $o; done
  for o in n_det_pld_k4 n_det_pld_k8 n_det_table_k4 n_det_table_k8 n_det_off2; do cmp $n n_det_off $o; done
  cmp $n h_off n_det_off

  echo "--- (d) $n deterministic rows $(date -u +%T)"
  dump $BN $n $g n_det_plain $DN
  dump $BN $n $g n_det_plain2 $DN
  bits $n n_det_plain n_det_plain2
  rm -f $RUNS/${n}_n_det_plain2.jsonl.rows.f32
  for c in "pld_k8 EMBER_SPEC=8" "table_k4 EMBER_SPEC=4 $TAB" "table_k8 EMBER_SPEC=8 $TAB"; do
    tag=${c%% *}; envs=${c#* }
    dump $BN $n $g n_det_$tag $DN $envs
    bits $n n_det_plain n_det_$tag
    rm -f $RUNS/${n}_n_det_$tag.jsonl.rows.f32
  done
  dump $BH $n $g h_det_plain $D
  dump $BN $n $g n_det_plain_cublas $D EMBER_DECODE_LINEAR=cublas
  bits $n h_det_plain n_det_plain_cublas
  bits $n h_det_plain n_det_plain
  rm -f $RUNS/${n}_*.rows.f32

  echo "--- (e) $n self-output pool $(date -u +%T)"
  port=$((port + 1))
  python3 $SM/dr_e2e.py run $BN $g -1 $port $SM/${n}_pool.jsonl $OUT/pool/${n}_pool.jsonl --once EMBER_SPEC=8 $TAB \
    > $OUT/pool/${n}_pool.txt 2>&1
  echo "pool $n rc=$? $(tail -1 $OUT/pool/${n}_pool.txt | cut -c1-200)"
  rm -rf $OUT/pool/${n}_pool.jsonl.run
  gzip -f $OUT/pool/${n}_pool.jsonl
  rm -f $RUNS/${n}_*
  pack out$pk_n
done

echo "--- pack $(date -u +%T)"
pack out
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
