#!/usr/bin/env bash
# VM side of the d=192 CUDA rmsnorm check (ember.cpp, 24 Oct 2026). Launched detached by drive.sh.
#   bash vm.sh <ember_fix_commit> <paperkiln_commit>
# Inputs in /content: fix.patch (cuda_kernels.cu only), rmsnorm_batch_test.cu, dumpprec.py, sp_compare.py,
# cmp_spot.py, prompts.jsonl, sft.part*, l16.part*, SHA256SUMS. Outputs: /content/out.zip, /content/VM_DONE.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
FIX=${1:?fix commit}; PK=${2:?paperkiln commit}
OUT=/content/out; M=/content/models
mkdir -p "$OUT" "$M"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
echo "=== vm $(date -u) fix=$FIX paperkiln=$PK ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
python3 -c "import tokenizers" || pip install -q tokenizers || fail "pip tokenizers"

echo "--- models $(date -u +%T)"
cat sft.part* > $M/tc_sft_L32_r1.gguf && cat l16.part* > $M/sp_L16.gguf || fail "reassemble"
(cd $M && sha256sum tc_sft_L32_r1.gguf sp_L16.gguf) > $OUT/models.sha256
diff <(sort $OUT/models.sha256) <(sort SHA256SUMS) || fail "model sha256 mismatch"
rm -f sft.part* l16.part*

echo "--- paperkiln tools $(date -u +%T)"
git clone -q --filter=blob:none https://github.com/JohnnyTeutonic/paperkiln.git /content/pk && git -C /content/pk checkout -q "$PK" || fail "paperkiln clone"

echo "--- ember builds $(date -u +%T)"
git clone -q --filter=blob:none https://github.com/JohnnyTeutonic/ember.cpp.git /content/ember || fail "ember clone"
E=/content/ember; B=/content/build
build() {  # tag commit fix(0|1)
  local tag=$1 c=$2 f=$3
  git -C $E checkout -q -f --detach "$c" || fail "checkout $c"
  [ "$f" = 1 ] && { git -C $E apply /content/fix.patch || fail "apply fix on $c"; }
  python3 /content/dumpprec.py $E/api.cpp || fail "dumpprec on $c"
  if [ ! -f $B/Makefile ]; then
    cmake -S $E -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || fail "cmake"
    grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log || fail "ember configured without CUDA"
  fi
  local t0=$(date +%s)
  make -C $B -j"$(nproc)" tinyllama_server > $OUT/make_$tag.log 2>&1 || { tail -30 $OUT/make_$tag.log; fail "build $tag"; }
  cp $B/tinyllama_server /content/srv_$tag
  echo "built $tag ($c fix=$f) in $(( $(date +%s) - t0 )) s"
  git -C $E checkout -q -f .
  git -C $E clean -fdq
}
build pin cbdcb915d26a66a276919328b8d808a450d51307 0
build pinfix cbdcb915d26a66a276919328b8d808a450d51307 1
build head 2886a97 0
build fix 2886a97 1

echo "--- kernel test across commits $(date -u +%T)"
for c in b72ade8 ef4b30f cbdcb91 2886a97 FIX; do
  d=/content/k_$c; rm -rf $d; mkdir -p $d
  if [ $c = FIX ]; then git -C $E archive 2886a97 | tar x -C $d && (cd $d && git apply /content/fix.patch) || fail "fix tree"
  else git -C $E archive $c | tar x -C $d; fi
  cp /content/rmsnorm_batch_test.cu $d/tools/rmsnorm_batch_test.cu
  (cd $d && nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_$CC -I. tools/rmsnorm_batch_test.cu cuda_kernels.cu logger.cpp -lcublas -o rt > $OUT/nvcc_$c.log 2>&1) \
    && $d/rt $OUT/rmsnorm_out_$c.f32 > $OUT/rmsnorm_test_$c.txt 2>&1
  echo "$c: $(tail -1 $OUT/rmsnorm_test_$c.txt 2>/dev/null || tail -2 $OUT/nvcc_$c.log)"
done

echo "--- spot-check leg (carry5_eval, gpu_layers -1, EMBER_GEMM_TF32=0) $(date -u +%T)"
mkdir -p /content/sft
cp $M/tc_sft_L32_r1.gguf /content/sft/
python3 - <<'PY'
import json
s = {"name": "tc_sft_L32_r1", "arch": {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 192, "layers": 32, "heads": 3, "d_ff": 832}},
     "data": {"vocab": "/content/sft/tc_sft_L32_r1.gguf", "vocab_cap": 16384, "T": 512}, "out_dir": "/content/sft"}
json.dump(s, open("/content/sft/sft_spec.json", "w"), indent=1)
PY
port=8400
for tag in pin pinfix head fix; do
  port=$((port + 1))
  (cd /content/pk && EMBER_GEMM_TF32=0 python3 tools/carry5_eval.py run --model m=/content/sft/sft_spec.json --out $OUT/spot_$tag \
     --scorer none --gen ember --ember-bin /content/srv_$tag --gpu-layers -1 --set corpus/eval/carry5_v1.jsonl \
     --every 26 --limit 40 --port $port > $OUT/spot_$tag.log 2>&1)
  echo "spot $tag rc=$?"
  rm -rf $OUT/spot_$tag/www $OUT/spot_$tag/ember_cwd/debugging.log
done

echo "--- tinychat prompts, both request variants $(date -u +%T)"
serve() {  # tag gguf gpu_layers port [dump]
  mkdir -p /content/run/$1_$3/www; cd /content/run/$1_$3; ln -sf /dev/null debugging.log
  if [ -n "${5:-}" ]; then TINYLLAMA_LOGITS_DUMP=$5 EMBER_GEMM_TF32=0 /content/srv_$1 $2 $4 127.0.0.1 /content/run/$1_$3/www $3 > /dev/null 2> err.log &
  else EMBER_GEMM_TF32=0 /content/srv_$1 $2 $4 127.0.0.1 /content/run/$1_$3/www $3 > /dev/null 2> err.log & fi
  SP=$!
  for i in $(seq 1 120); do curl -s -m 2 127.0.0.1:$4/health > /dev/null && break; sleep 1; done
  cd /content
}
for model in sft L16; do
  g=$M/tc_sft_L32_r1.gguf; [ $model = L16 ] && g=$M/sp_L16.gguf
  for tag in pin fix; do
    for gl in -1 0; do
      [ $gl = 0 ] && [ $tag = pin ] && continue
      port=$((port + 1)); serve $tag $g $gl $port
      dev=cuda; [ $gl = 0 ] && dev=cpu
      for V in spot p1; do
        python3 /content/sp_compare.py collect http://127.0.0.1:$port /content/prompts.jsonl 40 48 $V $OUT/tc_${model}_${tag}_${dev}_$V.jsonl || echo "collect $model $tag $dev $V failed"
      done
      kill $SP; wait $SP 2>/dev/null
    done
  done
done

echo "--- logits at the first divergent steps (TINYLLAMA_LOGITS_DUMP, precision 9) $(date -u +%T)"
python3 - <<'PY'
import json
items = [json.loads(x) for x in open("/content/pk/corpus/eval/carry5_v1.jsonl") if x.strip()][::26][:40]
rows = [{"prompt": it["turns"][0]["user"], "id": it["id"]} for it in items[:4]]
with open("/content/spot_first4.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
PY
for k in 0 1 2 3; do
  sed -n "$((k + 1))p" /content/spot_first4.jsonl > /content/one.jsonl
  for tag in pin fix; do
    for gl in -1 0; do
      [ $gl = 0 ] && [ $tag = pin ] && continue
      dev=cuda; [ $gl = 0 ] && dev=cpu
      port=$((port + 1)); rm -f $OUT/dump_${k}_${tag}_$dev.txt
      serve $tag $M/tc_sft_L32_r1.gguf $gl $port $OUT/dump_${k}_${tag}_$dev.txt
      python3 /content/sp_compare.py collect http://127.0.0.1:$port /content/one.jsonl 1 24 spot $OUT/dumpreq_${k}_${tag}_$dev.jsonl
      kill $SP; wait $SP 2>/dev/null
    done
  done
done
cp /content/spot_first4.jsonl $OUT/

echo "--- pack $(date -u +%T)"
cd /content && rm -f out.zip && zip -qr out.zip out && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
