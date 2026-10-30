#!/usr/bin/env bash
# VM side of the R2 KV-cache quantisation session (experiments/kv_quant_r2_2026-10-28), on a Colab L4.
# Launched detached by tools/colab_kvq_drive.sh; never run locally (GPU work is remote only).
#
#   bash colab_kvq_vm.sh <ember_head> [phases]
#
# Inputs in /content: ember.tgz (ember HEAD + the R2 files), ember_base.tgz (ember HEAD alone), inputs.tgz
# (inputs/{inputs,long}_{qwen3,qwen25}.json, made at home by tools/kvq_inputs.py: tokenising is light), this
# script. The checkpoints and tokenizers are fetched here: Qwen3-0.6B
# Q8_0 and Qwen2.5-0.5B-Instruct f16 GGUFs from their Qwen repositories, as the earlier device gates did.
# Phases (default all, in this order; each one's receipts are zipped to /content/gate_out.zip as it ends
# and its name appended to /content/PHASES):
#   eval_qwen3, eval_qwen25  tools/kv_quant_eval: f32 against f16, q8, q4: bench, greedy and tf_prefill
#        (speed, memory, greedy agreement, prefill-path errors);
#   g1   the default mode bit for bit: ir_device_dump from ember HEAD and from HEAD + R2, default env and
#        EMBER_DETERMINISTIC=1, 40 seeded short prompts and two 3,072-id prompts, both models;
#   g2   each new format run to run under EMBER_DETERMINISTIC=1 (ir_device_dump twice), both models;
#   g3   each format on the device against the CPU interpreter's form of the same format (ir_device_gate,
#        12 seeded prompts), and the default mode the same way, both models;
#   tfd_qwen3, tfd_qwen25  kv_quant_eval's tf_decode (decode-path errors), last: the longest runs.
# Done: /content/GATE_DONE (or /content/GATE_FAILED with the reason). Log: /content/gate_vm.log.
set -uo pipefail
cd /content
exec > /content/gate_vm.log 2>&1
HEAD_C=${1:?ember head}; PHASES=${2:-eval_qwen3,eval_qwen25,tfd_qwen3,tfd_qwen25,g1,g2,g3}
FORMATS=${KVQ_FORMATS:-f16,q8,q4}
unset EMBER_KV EMBER_DETERMINISTIC
CHECK=/content/src/new/ember/tools/kvq_gate_check.py
OUT=/content/out; M=/content/models
mkdir -p "$OUT" "$M" /content/src/new /content/src/base
fail() { echo "FAILED: $*"; echo "$*" > /content/GATE_FAILED; pack; exit 1; }
pack() {  # atomic: the driver may download the zip at any moment
  python3 -c "import os, shutil; t = '/content/gate_out_%d' % os.getpid(); shutil.make_archive(t, 'zip', '/content/out'); os.replace(t + '.zip', '/content/gate_out.zip')"
}
phase_done() { pack; echo "$1" >> /content/PHASES; echo "=== phase $1 done $(date -u +%T)"; }
want() { case ",$PHASES," in *",$1,"*) return 0;; esac; return 1; }
echo "=== kvq vm $(date -u) ember HEAD $HEAD_C, phases $PHASES ==="
{ nvidia-smi; nvcc --version; nproc; free -g; python3 -c "import transformers; print('transformers', transformers.__version__)"; } > "$OUT/hardware.txt" 2>&1

echo "--- deps, builds and downloads $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
tar xzf ember.tgz -C /content/src/new && tar xzf ember_base.tgz -C /content/src/base || fail "untar"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d . | tr -d " ")
echo "GPU compute capability $ARCH"
build() {  # tree targets...: a Release CUDA build for this GPU (sm_$ARCH) in /content/build_<tree>
  local t=$1; shift
  mkdir -p /content/build_$t && cd /content/build_$t
  CMAKE_CUDA_ARCHITECTURES=$ARCH cmake /content/src/$t/ember -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=ON > cmake.log 2>&1 || return 1
  grep -q "CUDA toolkit and cuBLAS found" cmake.log || return 1
  make "$@" -j"$JOBS" > make.log 2>&1
}
# The new tree first on every core; the base tree (g1 only) in the background once the eval is running.
JOBS=${KVQ_BUILD_JOBS:-4} build new kv_quant_eval ir_device_dump ir_device_gate || { tail -30 /content/build_new/make.log; fail "build new"; }
cd /content
echo "--- build new done $(date -u +%T)"
( JOBS=${KVQ_BUILD_JOBS:-4} build base ir_device_dump; echo $? > /content/build_base.rc ) &
BASE_PID=$!
tar xzf /content/inputs.tgz -C /content && cp /content/inputs/*.json $OUT/ || fail "inputs"
python3 - <<'PY' || fail "downloads"
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", local_dir="/content/models/qwen3")
hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf", local_dir="/content/models/qwen25")
import hashlib
from pathlib import Path
expected = {
    '/content/models/qwen3/Qwen3-0.6B-Q8_0.gguf': '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031',
    '/content/models/qwen25/qwen2.5-0.5b-instruct-fp16.gguf': '8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc',
}
for path, digest in expected.items():
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    if h.hexdigest() != digest: raise SystemExit('model hash mismatch: ' + path)
print("downloads verified")
PY
QWEN3=$M/qwen3/Qwen3-0.6B-Q8_0.gguf; QWEN25=$M/qwen25/qwen2.5-0.5b-instruct-fp16.gguf
sha256sum $QWEN3 $QWEN25 /content/inputs/*.json > $OUT/models.sha256; cat $OUT/models.sha256
NEW=/content/build_new; BASE=/content/build_base
# One kv capacity for every run: 8,192 positions plus the bench's and greedy's continuations.
export EMBER_IR_CTX=8448

declare -A GG=([qwen3]=$QWEN3 [qwen25]=$QWEN25)
# The receipts zipped every 2 minutes, so a reclaimed VM still leaves what it measured (the driver fetches
# the zip at every poll; two VMs were reclaimed 23 minutes in on 28 Oct 2026).
( while [ ! -e /content/GATE_DONE ] && [ ! -e /content/GATE_FAILED ]; do sleep 120; pack; done ) &
evals() {  # spec...: "phase model list" runs of tools/kv_quant_eval
 for spec in "$@"; do
  set -- $spec; ph=$1; m=$2; list=$3
  want ${ph}_$m || continue
  echo "--- ${ph} $m ($list) $(date -u +%T)"
  mkdir -p $OUT/eval_$m
  $NEW/kv_quant_eval ${GG[$m]} $OUT/inputs_$m.json $OUT/eval_$m --formats "$FORMATS" --phases $list > $OUT/eval_$m/stdout_$ph.log \
    2> $OUT/eval_$m/stderr_$ph.log || fail "${ph} $m: $(tail -3 $OUT/eval_$m/stderr_$ph.log)"
  grep -v "^\[" $OUT/eval_$m/stderr_$ph.log | tail -12 | cut -c1-600
  phase_done ${ph}_$m
 done
}
evals "eval qwen3 bench,greedy,tf_prefill" "eval qwen25 bench,greedy,tf_prefill"

dump() {  # bin out gguf args...: one ir_device_dump receipt
  local bin=$1 out=$2 g=$3; shift 3
  $bin/ir_device_dump "$g" "$out" "$@" > "$out.json" 2> "$out.log" || fail "dump $out: $(tail -2 $out.log)"
  python3 "$CHECK" dump "$out" "$@" > "$out.check.json" || fail "invalid dump $out"
}
same() {  # a b label: byte equality of two dumps, one line per comparison in $OUT/gates.txt
  if [ -s "$1" ] && cmp -s "$1" "$2"; then r=IDENTICAL; else r=DIFFER; fi
  echo "$3 $r $(sha256sum < "$1" | cut -c1-16) $(sha256sum < "$2" | cut -c1-16)" | tee -a $OUT/gates.txt
  [ "$r" = IDENTICAL ] || fail "byte comparison failed: $3"
}
if want g1; then
  wait $BASE_PID  # the base build only (a bare wait would also wait for the receipt packer below)
  [ "$(cat /content/build_base.rc)" = 0 ] || { tail -30 /content/build_base/make.log; fail "build base"; }
  echo "--- g1: default mode, HEAD vs HEAD + R2 $(date -u +%T)"
  mkdir -p $OUT/g1
  for m in qwen3 qwen25; do
    for det in 0 1; do
      for kind in short long; do
        args="--n 40 --len 24 --gen 32"; [ $kind = long ] && args="--ids $OUT/long_$m.json --gen 32"
        for t in base new; do
          b=$BASE; [ $t = new ] && b=$NEW
          if [ $det = 1 ]; then EMBER_DETERMINISTIC=1 dump $b $OUT/g1/${m}_${kind}_det${det}_$t.bin ${GG[$m]} $args
          else dump $b $OUT/g1/${m}_${kind}_det${det}_$t.bin ${GG[$m]} $args; fi
        done
        # A changed row is not evidence against the feature when the old build itself varies.
        if ! cmp -s $OUT/g1/${m}_${kind}_det${det}_base.bin $OUT/g1/${m}_${kind}_det${det}_new.bin; then
          repeat=$OUT/g1/${m}_${kind}_det${det}_base_run2.bin
          if [ $det = 1 ]; then EMBER_DETERMINISTIC=1 dump $BASE "$repeat" ${GG[$m]} $args
          else dump $BASE "$repeat" ${GG[$m]} $args; fi
          if ! cmp -s $OUT/g1/${m}_${kind}_det${det}_base.bin "$repeat"; then
            fail "BASELINE_UNSTABLE: g1 $m $kind det=$det; old build differs from itself"
          fi
        fi
        same $OUT/g1/${m}_${kind}_det${det}_base.bin $OUT/g1/${m}_${kind}_det${det}_new.bin "g1 $m $kind det=$det base-vs-new"
      done
    done
    # the default mode run to run under EMBER_DETERMINISTIC=1 (the base run above is the first)
    EMBER_DETERMINISTIC=1 dump $NEW $OUT/g1/${m}_long_det1_new_run2.bin ${GG[$m]} --ids $OUT/long_$m.json --gen 32
    same $OUT/g1/${m}_long_det1_new.bin $OUT/g1/${m}_long_det1_new_run2.bin "g1 $m long det=1 new run-to-run"
    rm -f "$OUT/g1/${m}_"*.bin  # hashes and validated byte counts remain in the receipts
  done
  phase_done g1
fi
if want g2; then
  echo "--- g2: each format run to run, EMBER_DETERMINISTIC=1 $(date -u +%T)"
  mkdir -p $OUT/g2
  for m in qwen3 qwen25; do
    for f in ${FORMATS//,/ }; do
      for r in 1 2; do
        EMBER_KV=$f EMBER_DETERMINISTIC=1 dump $NEW $OUT/g2/${m}_${f}_long_run$r.bin ${GG[$m]} --ids $OUT/long_$m.json --gen 32
        EMBER_KV=$f EMBER_DETERMINISTIC=1 dump $NEW $OUT/g2/${m}_${f}_short_run$r.bin ${GG[$m]} --n 40 --len 24 --gen 32
      done
      same $OUT/g2/${m}_${f}_long_run1.bin $OUT/g2/${m}_${f}_long_run2.bin "g2 $m $f long det=1 run-to-run"
      same $OUT/g2/${m}_${f}_short_run1.bin $OUT/g2/${m}_${f}_short_run2.bin "g2 $m $f short det=1 run-to-run"
    done
    # without EMBER_DETERMINISTIC, only when q8 is part of this requested campaign
    if [[ ",$FORMATS," == *,q8,* ]]; then
    for r in 1 2; do EMBER_KV=q8 dump $NEW $OUT/g2/${m}_q8_long_nodet_run$r.bin ${GG[$m]} --ids $OUT/long_$m.json --gen 32; done
    same $OUT/g2/${m}_q8_long_nodet_run1.bin $OUT/g2/${m}_q8_long_nodet_run2.bin "g2 $m q8 long det=0 run-to-run"
    fi
    rm -f "$OUT/g2/${m}_"*.bin
  done
  rm -f $OUT/g2/*.bin $OUT/g1/*.bin  # equality is recorded in gates.txt with each file's hash
  phase_done g2
fi
if want g3; then
  echo "--- g3: device against the CPU's form of each format $(date -u +%T)"
  mkdir -p $OUT/g3
  for m in qwen3 qwen25; do
    for f in f32 ${FORMATS//,/ }; do
      EMBER_KV=$f $NEW/ir_device_gate ${GG[$m]} --n 12 --len 24 --gen 32 > $OUT/g3/${m}_$f.json 2> $OUT/g3/${m}_$f.log || fail "device/CPU gate: $m $f"
      echo "g3 $m $f rc=0 $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('rows',None); print(json.dumps(d))" $OUT/g3/${m}_$f.json 2>&1 | cut -c1-400)" | tee -a $OUT/gates.txt
    done
  done
  phase_done g3
fi
evals "tfd qwen3 tf_decode" "tfd qwen25 tf_decode"
# Qwen2.5-0.5B-Instruct's GGUF declares 8,192 positions, so its 8K bench runs at 8,064 (+ 128 steps).
if want tfd_qwen25; then
  mkdir -p $OUT/eval_qwen25_8064
  $NEW/kv_quant_eval $QWEN25 $OUT/inputs_qwen25.json $OUT/eval_qwen25_8064 --formats "$FORMATS" --phases bench --contexts 8064 \
    > $OUT/eval_qwen25_8064/stdout.log 2> $OUT/eval_qwen25_8064/stderr.log || fail "bench 8064"
  phase_done bench_qwen25_8064
fi
rm -f $OUT/g1/*.bin $OUT/g2/*.bin
python3 "$CHECK" summary "$OUT" --phases "$PHASES" --formats "$FORMATS" > "$OUT/gate_summary.json" || fail "receipt validation"
pack
echo "=== done $(date -u) ==="
touch /content/GATE_DONE
