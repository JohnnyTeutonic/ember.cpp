#!/usr/bin/env bash
# VM side of spec_r1_2026-10-28 session 2 (ember.cpp). Launched detached by drive.sh while the two models upload.
#   bash vm2.sh            (staged as vm.sh)
# Session 1 (vm.sh) finished pick_e1_L16's legs and was reclaimed during pk_base_v1's. Inputs as vm.sh's, the trees at
# ember main 2fbf454 (head) and 2fbf454 plus this work (new: EMBER_DECODE_LINEAR, draft_table.h's orders 5-8,
# tools/r1_bench.cpp); small.tgz adds r1/ (r1_sim.py, r1_rows.py and draft_2026-10-27's dr_sim.py and dr_table.py
# in their folders) and the larger pools (pick 20,000 requests, pk 8,000). In order, each leg packed when done:
#   out1  pk_base_v1: vm.sh's (a) bench, (b) nsys, (c) end to end, (d) dumps, plus the accuracy leg below;
#   out2  pick_e1_L16: (c) and (d) again at 2fbf454 plus the accuracy leg (the bench and nsys of session 1 stand:
#         no CUDA source differs between 934393f and 2fbf454);
#   accuracy: CPU plain decode (the same binary, no GPU layers) dumped as the reference; main's plain decode in
#         the default mode, main's and the new binary's under EMBER_DETERMINISTIC=1, each against it (r1_rows.py);
#   out3, out4  the self-output tables (pick_e1_L16, pk_base_v1): each model's larger pool served once (default mode, EMBER_SPEC=8 with the corpus
#         table: plain decode's ids), a table of the model's own replies built from it (orders 1-6, contexts seen at
#         least twice, share at least 0.10; r1_sim.py build) and merged behind the corpus table (r1_sim.py merge
#         --orders 6: the corpus table's entry where both hold the context), the recipe chosen on pick_e1_L16's
#         development sets with session 1's 4,000-request pool; then eval100 end to end with the merged table
#         against the corpus table, default mode and under EMBER_DETERMINISTIC=1 (k 8), ids against plain, and the
#         deterministic rows bit for bit (pick_e1_L16).
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; BIG=/content/big; RUNS=/content/runs
mkdir -p "$OUT/e2e" "$OUT/bits" "$OUT/bench" "$OUT/nsys" "$OUT/pool" "$OUT/rows" "$OUT/tables" "$BIG" "$RUNS"
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
dump() {  # binary model gguf tag env... (GPUL=0 for the CPU)
  local b=$1 n=$2 g=$3 tag=$4; shift 4
  port=$((port + 1))
  python3 $SM/dr_e2e.py run $b $g ${GPUL:--1} $port $SM/eval100.jsonl $RUNS/${n}_$tag.jsonl --dump "$@" > $OUT/bits/${n}_$tag.txt 2>&1
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
gates() {  # model gguf: vm.sh's (c) and (d), then the accuracy leg
  local n=$1 g=$2
  local TV=TABLE_$n; local T=$SM/${!TV}
  local TAB="EMBER_DRAFT=table EMBER_DRAFT_TABLE=$T EMBER_DRAFT_THETA=0.02"
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
    local tag=${c%% *} envs=${c#* }
    dump $BN $n $g n_det_$tag $DN $envs
    bits $n n_det_plain n_det_$tag
    rm -f $RUNS/${n}_n_det_$tag.jsonl.rows.f32
  done
  dump $BH $n $g h_det_plain $D
  dump $BN $n $g n_det_plain_cublas $D EMBER_DECODE_LINEAR=cublas
  bits $n h_det_plain n_det_plain_cublas
  bits $n h_det_plain n_det_plain
  rm -f $RUNS/${n}_n_det_plain_cublas.jsonl.rows.f32
  echo "--- accuracy $n $(date -u +%T)"
  dump $BH $n $g h_plain EMBER_NOTHING=1
  GPUL=0 dump $BN $n $g cpu_plain EMBER_NOTHING=1
  python3 $SM/r1/spec_r1_2026-10-28/r1_rows.py $RUNS/${n}_cpu_plain.jsonl $RUNS/${n}_h_plain.jsonl $RUNS/${n}_h_det_plain.jsonl \
    $RUNS/${n}_n_det_plain.jsonl > $OUT/rows/${n}_vs_cpu.jsonl 2> $OUT/rows/${n}_vs_cpu.err
  python3 $SM/r1/spec_r1_2026-10-28/r1_rows.py $RUNS/${n}_h_det_plain.jsonl $RUNS/${n}_n_det_plain.jsonl $RUNS/${n}_h_plain.jsonl \
    > $OUT/rows/${n}_vs_h_det.jsonl 2> $OUT/rows/${n}_vs_h_det.err
  echo "rows $n rc=$?: $(cut -c1-400 $OUT/rows/${n}_vs_cpu.jsonl | tr '\n' ' ')"
  rm -f $RUNS/${n}_*.rows.f32
}

echo "--- (a) pk bench $(date -u +%T)"
g=$BIG/pk_base_v1.gguf
(cd /content && timeout 900 $BENCH $g $OUT/bench/pk_default.json 200 32,384 > $OUT/bench/pk_default.log 2>&1)
echo "bench pk default rc=$?"
(cd /content && env $D timeout 900 $BENCH $g $OUT/bench/pk_det.json 200 32,384 > $OUT/bench/pk_det.log 2>&1)
echo "bench pk det rc=$?"
echo "--- (b) pk nsys $(date -u +%T)"
if [ -n "${NSYS:-}" ]; then
  for P in 384 32; do
    for cfg in "decode cublas" "decode fixed" "pass rows 5" "pass rows 9" "pass fixed 5" "pass fixed 9" "pass smallm 9"; do
      tag=$(echo "$cfg" | tr ' ' '_')_$P
      r=/content/ns_pk_$tag
      (cd /content && env $D timeout 600 $NSYS profile -t cuda,nvtx --cuda-graph-trace=node -o $r --force-overwrite true \
        $BENCH $g $OUT/nsys/pk_$tag.json 100 $P profile $cfg > $OUT/nsys/pk_$tag.log 2>&1)
      timeout 600 $NSYS stats -r cuda_gpu_kern_sum -f csv -o $OUT/nsys/pk_$tag $r.nsys-rep >> $OUT/nsys/pk_$tag.log 2>&1
      echo "nsys pk $tag rc=$?"
      rm -f $r.nsys-rep $r.sqlite
    done
  done
fi
gates pk $BIG/pk_base_v1.gguf
pack out1
gates pick $BIG/pick_L16.gguf
pack out2

echo "--- (e) self-output tables $(date -u +%T)"
R1=$SM/r1/spec_r1_2026-10-28
for m in pick:$BIG/pick_L16.gguf pk:$BIG/pk_base_v1.gguf; do
  pn3=$(( ${pn3:-2} + 1 ))
  n=${m%%:*}; g=${m#*:}
  TV=TABLE_$n; T=$SM/${!TV}
  TAB="EMBER_DRAFT=table EMBER_DRAFT_TABLE=$T EMBER_DRAFT_THETA=0.02"
  port=$((port + 1))
  python3 $SM/dr_e2e.py run $BN $g -1 $port $SM/${n}_pool20.jsonl $OUT/pool/${n}_pool20.jsonl --once EMBER_SPEC=8 $TAB \
    > $OUT/pool/${n}_pool20.txt 2>&1
  echo "pool $n rc=$? $(tail -1 $OUT/pool/${n}_pool20.txt | cut -c1-200)"
  rm -rf $OUT/pool/${n}_pool20.jsonl.run
  python3 $R1/r1_sim.py build /content/${n}_self.npz --pool $OUT/pool/${n}_pool20.jsonl --orders 6 --min-count 2 --min-conf 0.10 \
    > $OUT/tables/${n}_self.txt 2>&1
  python3 $R1/r1_sim.py merge $OUT/tables/${n}_merge_o6.bin $T,/content/${n}_self.npz --orders 6 > $OUT/tables/${n}_merge_o6.txt 2>&1
  echo "table $n: $(tail -1 $OUT/tables/${n}_self.txt | cut -c1-160) / $(tail -1 $OUT/tables/${n}_merge_o6.txt | cut -c1-200)"
  gzip -f $OUT/pool/${n}_pool20.jsonl
  M="EMBER_DRAFT=table EMBER_DRAFT_TABLE=$OUT/tables/${n}_merge_o6.bin EMBER_DRAFT_THETA=0.02"
  e2e $BN $n $g s_off EMBER_NOTHING=1
  e2e $BN $n $g s_table_k8 EMBER_SPEC=8 $TAB
  e2e $BN $n $g s_merge_k8 EMBER_SPEC=8 $M
  e2e $BN $n $g s_det_off $DN
  e2e $BN $n $g s_det_table_k8 $DN EMBER_SPEC=8 $TAB
  e2e $BN $n $g s_det_merge_k8 $DN EMBER_SPEC=8 $M
  e2e $BN $n $g s_off2 EMBER_NOTHING=1
  for o in s_table_k8 s_merge_k8 s_off2; do cmp $n s_off $o; done
  for o in s_det_table_k8 s_det_merge_k8; do cmp $n s_det_off $o; done
  if [ $n = pick ]; then
    dump $BN $n $g s_det_plain $DN
    dump $BN $n $g s_det_merge_k8 $DN EMBER_SPEC=8 $M
    bits $n s_det_plain s_det_merge_k8
    rm -f $RUNS/${n}_*.rows.f32
  fi
  pack out$pn3
done

echo "--- pack $(date -u +%T)"
pack out
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
