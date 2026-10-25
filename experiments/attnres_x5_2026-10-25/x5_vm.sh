#!/usr/bin/env bash
# VM side of the X5 attnres gate (ember.cpp, 25 Oct 2026). Launched detached by x5_drive.sh.
# Inputs in /content: COMMITS (e115fd3 cbdcb915 ce41ffb, full hashes), ember_new.tgz, ember_old.tgz (git archives),
# ce41ffb.patch, small.tgz (scripts, ar_x5_q8_0.gguf, ar_x5_in.json, prompts, Carry-5 items, SHA256SUMS),
# arx5.part* (ar_x5.gguf), pick.part* (pick_e1_L16/model.gguf), BIG_SHA256SUMS.
# Builds:
#   pin    e115fd3 exactly as paperkiln's tiny_chat_run.ensure_ember: cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
#          -DCMAKE_CUDA_ARCHITECTURES=89; make -C build -j<nproc> tinyllama_server (on a T4: compile only);
#   tools  e115fd3 + the gate tools as targets, at the device's arch (the pin's own tree when the device is sm_89);
#   old    cbdcb915 + the nine-digit logits dump (x5_dump9.py), at the device's arch: tinyllama_server; then
#   oldce  the same tree with ce41ffb's diff: tinyllama_server, x5_device_logits, ir_device_gate.
# Legs: (a) ar_x5 / ar_x5_q8_0 on the device (x5_device_logits, state_device_gate, ir_device_gate), new and oldce;
#       (b) plain llama (pick_e1_L16, the L16 control's SFT GGUF): x5_plain_collect.py through the new and the
#       old tinyllama_server, gpu_layers -1, EMBER_GEMM_TF32=0; x5_plain_compare.py.
# Outputs: /content/out.zip, /content/VM_DONE (or VM_FAILED). Log: /content/vm.log.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small; M=/content/models
mkdir -p "$OUT" "$M" /content/bin
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r NEWC OLDC CEC < /content/COMMITS
echo "=== vm $(date -u) new=$NEWC old=$OLDC ce=$CEC ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC, nproc $(nproc)"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cat arx5.part* > $M/ar_x5.gguf && cat pick.part* > $M/pick_L16.gguf || fail "reassemble"
cp $SM/ar_x5_q8_0.gguf $M/
(cd $M && sha256sum ar_x5.gguf pick_L16.gguf) > $OUT/big.sha256
diff <(awk '{print $1}' $OUT/big.sha256) <(awk '{print $1}' BIG_SHA256SUMS) || fail "big sha256"
rm -f arx5.part* pick.part*

echo "--- deps $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
pip install -q numpy > /dev/null 2>&1

echo "--- builds $(date -u +%T)"
mkdir -p /content/src_new /content/src_old
tar xzf ember_new.tgz -C /content/src_new --strip-components=1 && tar xzf ember_old.tgz -C /content/src_old --strip-components=1 || fail "untar ember"
J=$(nproc)
# (1) the pin's build, exactly as ensure_ember
t0=$(date +%s)
cmake -S /content/src_new -B /content/b_pin -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > $OUT/cmake_pin.log 2>&1 || { tail -20 $OUT/cmake_pin.log; fail "cmake pin"; }
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_pin.log && echo "pin: CUDA toolkit and cuBLAS found" || fail "pin configured without CUDA"
make -C /content/b_pin -j$J tinyllama_server > $OUT/make_pin.log 2>&1; rc=$?
echo "pin build (sm_89, tinyllama_server) rc=$rc in $(( $(date +%s) - t0 )) s"
grep -E "CMAKE_CUDA_ARCHITECTURES:|CMAKE_BUILD_TYPE:" /content/b_pin/CMakeCache.txt
[ $rc -eq 0 ] && [ -x /content/b_pin/tinyllama_server ] || { tail -30 $OUT/make_pin.log; fail "pin build"; }
sha256sum /content/b_pin/tinyllama_server > $OUT/pin_server.sha256
cuobjdump --list-elf /content/b_pin/tinyllama_server 2>/dev/null | head -5 > $OUT/pin_server_elf.txt
echo "{\"commit\": \"$NEWC\", \"rc\": $rc, \"cmake\": \"cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89\", \"make\": \"make -C build -j$J tinyllama_server\", \"seconds\": $(( $(date +%s) - t0 ))}" > $OUT/pin_build.json
cp /content/b_pin/tinyllama_server /content/bin/pin_server
add_tools() {  # src dir, then tool names
  local s=$1; shift
  for t in "$@"; do
    [ -f $s/tools/$t.cpp ] || cp $SM/$t.cpp $s/tools/
    printf '\nadd_executable(%s tools/%s.cpp)\ntarget_link_libraries(%s PRIVATE tinyllama_core)\n' $t $t $t >> $s/CMakeLists.txt
  done
}
# (2) the tools (and, on a device other than sm_89, the new server at its arch)
add_tools /content/src_new x5_device_logits
if [ "$CC" = 89 ]; then BT=/content/b_pin; else BT=/content/b_tools; fi
t0=$(date +%s)
cmake -S /content/src_new -B $BT -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_tools.log 2>&1 || fail "cmake tools"
make -C $BT -j$J tinyllama_server x5_device_logits state_device_gate ir_device_gate > $OUT/make_tools.log 2>&1 || { tail -30 $OUT/make_tools.log; fail "build tools"; }
echo "tools built (sm_$CC) in $(( $(date +%s) - t0 )) s"
cp $BT/tinyllama_server /content/bin/new_server
[ "$CC" = 89 ] && { cmp /content/bin/new_server /content/bin/pin_server && echo "new_server = pin_server (same tree, sm_89)"; }
# (3) old, then oldce in the same tree
python3 $SM/x5_dump9.py /content/src_old || fail "dump9"
add_tools /content/src_old x5_device_logits
t0=$(date +%s)
cmake -S /content/src_old -B /content/b_old -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake_old.log 2>&1 || fail "cmake old"
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake_old.log || fail "old configured without CUDA"
make -C /content/b_old -j$J tinyllama_server > $OUT/make_old.log 2>&1 || { tail -30 $OUT/make_old.log; fail "build old"; }
cp /content/b_old/tinyllama_server /content/bin/old_server
echo "old built in $(( $(date +%s) - t0 )) s"
(cd /content/src_old && patch -p1 < /content/ce41ffb.patch) > $OUT/patch_ce41.log 2>&1 || fail "ce41ffb patch"
make -C /content/b_old -j$J tinyllama_server x5_device_logits ir_device_gate > $OUT/make_oldce.log 2>&1 || { tail -30 $OUT/make_oldce.log; fail "build oldce"; }
cp /content/b_old/tinyllama_server /content/bin/oldce_server
echo "oldce built"
sha256sum /content/bin/* > $OUT/bins.sha256

echo "--- (a) ar_x5 on the device $(date -u +%T)"
IN=$SM/ar_x5_in.json
for g in ar_x5 ar_x5_q8_0; do
  $BT/x5_device_logits $M/$g.gguf < $IN > $OUT/dev_new_$g.json 2> $OUT/dev_new_$g.log; echo "x5_device_logits new $g rc=$?"
  /content/b_old/x5_device_logits $M/$g.gguf < $IN > $OUT/dev_oldce_$g.json 2> $OUT/dev_oldce_$g.log; echo "x5_device_logits oldce $g rc=$?"
  $BT/state_device_gate $M/$g.gguf < $IN > $OUT/sdg_$g.json 2> $OUT/sdg_$g.log
  echo "state_device_gate $g rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['summary']), 'pass', d['pass'])" $OUT/sdg_$g.json 2>/dev/null || tail -2 $OUT/sdg_$g.log)"
done
for t in new oldce; do
  if [ $t = new ]; then B=$BT; else B=/content/b_old; fi
  $B/ir_device_gate $M/ar_x5.gguf --n 40 --len 8 --gen 8 > $OUT/dg_${t}_ar_x5.json 2> $OUT/dg_${t}_ar_x5.log
  echo "ir_device_gate $t rc=$?: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('rows',None); print(d['greedy_identical'], d['top1_agree'], d['worst_prefill_rel'], d['worst_decode_rel'], 'pass', d['pass'])" $OUT/dg_${t}_ar_x5.json 2>/dev/null || tail -2 $OUT/dg_${t}_ar_x5.log)"
done

echo "--- (b) plain llama through tinyllama_server, gpu_layers -1, EMBER_GEMM_TF32=0 $(date -u +%T)"
P=$SM/prompts.jsonl; C5=$SM/carry5_v1.jsonl
python3 $SM/x5_plain_collect.py /content/bin/new_server $M/pick_L16.gguf -1 8411 $P $C5 /content/plain/new_cuda > $OUT/plain_new_cuda.txt 2>&1; echo "collect new rc=$? $(tail -1 $OUT/plain_new_cuda.txt)"
python3 $SM/x5_plain_collect.py /content/bin/old_server $M/pick_L16.gguf -1 8412 $P $C5 /content/plain/old_cuda > $OUT/plain_old_cuda.txt 2>&1; echo "collect old rc=$? $(tail -1 $OUT/plain_old_cuda.txt)"
python3 $SM/x5_plain_compare.py /content/plain/new_cuda /content/plain/old_cuda > $OUT/plain_cuda_new_vs_old.json; echo "compare rc=$? $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['bit_identical'], json.dumps({k: {x: v[x] for x in ('n','replies_identical','all_positions_logits_bit_equal','first_token_logits_bit_equal')} for k, v in d['sets'].items()}))" $OUT/plain_cuda_new_vs_old.json)"
for t in new_cuda old_cuda; do
  mkdir -p $OUT/plain/$t
  cp /content/plain/$t/records.jsonl /content/plain/$t/meta.json /content/plain/$t/first_logits.f32 $OUT/plain/$t/
  tail -c 20000 /content/plain/$t/server.log > $OUT/plain/$t/server_tail.log
done

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
