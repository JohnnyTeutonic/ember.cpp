#!/usr/bin/env bash
# VM side of a TTFT measurement session (docs/TTFT.md). Launched detached by
# tools/colab_ttft_drive.sh; never run locally.
#
#   bash colab_ttft_vm.sh <mode> <ember_commit> <paperkiln_commit>
#
# mode probe: one short run per model and length, to size the full sweep and
#             to find out whether nsys exists on this image.
# mode full:  the protocol: RUNS fresh-server runs x (WARMUP + REPS) requests
#             per length (env overrides below), then the nsys kernel pass.
#
# Inputs in /content: ember.tgz, paperkiln.tgz, story3k.gguf, this script.
# Outputs: /content/ttft_out.zip, then a marker /content/TTFT_DONE (or
# /content/TTFT_FAILED holding the reason). Log: /content/ttft_vm.log.
set -uo pipefail
cd /content
exec > /content/ttft_vm.log 2>&1

MODE=${1:?mode}; EMBER_COMMIT=${2:?ember commit}; PK_COMMIT=${3:?paperkiln commit}
RUNS=${RUNS:-5}; WARMUP=${WARMUP:-10}; REPS=${REPS:-50}; FIRST_RUN=${FIRST_RUN:-1}
TIMING_LENGTHS=${TIMING_LENGTHS:-"128 512 1024 2048 4096"}
# Seconds-scale lengths get fewer requests (protocol deviation, docs/TTFT.md).
LONG_LENGTHS=${LONG_LENGTHS:-""}; LONG_WARMUP=${LONG_WARMUP:-2}; LONG_REPS=${LONG_REPS:-5}
# GPU legs of the timing-model parity check; NSYS=0 skips the kernel pass.
PARITY_LENGTHS=${PARITY_LENGTHS:-"128 512 1024 2048 4096"}
NSYS=${NSYS:-1}; NSYS_LENGTHS=${NSYS_LENGTHS:-"128 1024"}
STAMP=$(date -u +%Y-%m-%d)
OUT=/content/out/ttft_${STAMP}_${MODE}
fail() { echo "FAILED: $*"; echo "$*" > /content/TTFT_FAILED; exit 1; }

echo "=== ttft vm $(date -u) mode=$MODE ember=$EMBER_COMMIT paperkiln=$PK_COMMIT ==="
mkdir -p "$OUT" /content/src
{ nvidia-smi; nvidia-smi -q -d CLOCK,PERFORMANCE,POWER; nvcc --version; uname -a;
  nproc; free -g; } > "$OUT/hardware.txt" 2>&1

echo "--- deps + build $(date -u +%T)"
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
tar xzf ember.tgz -C /content/src && tar xzf paperkiln.tgz -C /content/src || fail "untar"
( mkdir -p /content/build_mt && cd /content/build_mt &&
  cmake /content/src/microtorch -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 &&
  make make_timing_gguf -j"$(nproc)" > make.log 2>&1 ) &
mt_pid=$!
mkdir -p /content/build_ember && cd /content/build_ember
CMAKE_CUDA_ARCHITECTURES=89 cmake /content/src/ember -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=ON \
    > cmake.log 2>&1 || fail "ember cmake"
grep -q "CUDA toolkit and cuBLAS found" cmake.log || fail "ember configured without CUDA"
make tinyllama_server -j"$(nproc)" > make.log 2>&1 || { tail -30 make.log; fail "ember build"; }
cd /content
wait $mt_pid || fail "make_timing_gguf build"
SERVER=/content/build_ember/tinyllama_server

echo "--- timing model $(date -u +%T)"
/content/build_mt/make_timing_gguf story3k.gguf timing.gguf | tail -1 || fail "timing gguf"
sha256sum story3k.gguf timing.gguf > "$OUT/models.sha256"

BENCH="python3 /content/src/ember/tools/ttft_bench.py --server $SERVER --commit $EMBER_COMMIT"
echo "paperkiln_commit=$PK_COMMIT ember_commit=$EMBER_COMMIT" > "$OUT/commits.txt"

# Numerical gate before any timing: the GPU path must compute what the CPU
# path computes (FP32, tight), and the shipped BF16 path must stay close to
# FP32 (loose). Gates are fixed here, before the numbers are seen. The CPU
# leg of the timing model is slow (~70 s per 128-token request), so it runs
# at 128 only; GPU legs cover every length. A failure ends a full run (no
# baseline of a wrong path); a probe records it and carries on.
echo "--- parity $(date -u +%T)"
PARITY="python3 /content/src/ember/tools/ttft_parity.py --server $SERVER --prompts 2"
mkdir -p "$OUT/parity"
parity_ok=1
$PARITY --model story3k.gguf --ref cpu --test gpu_fp32 --lengths 128 \
    --max-rel-diff 1e-3 --min-cosine 0.99999 --out "$OUT/parity/story3k_cpu_vs_gpu_fp32.jsonl" || parity_ok=0
$PARITY --model story3k.gguf --ref gpu_fp32 --test gpu_bf16 --lengths 128 \
    --max-rel-diff 5e-2 --min-cosine 0.999 --out "$OUT/parity/story3k_gpu_fp32_vs_bf16.jsonl" || parity_ok=0
$PARITY --model timing.gguf --ref cpu --test gpu_fp32 --lengths 128 \
    --max-rel-diff 1e-3 --min-cosine 0.99999 --out "$OUT/parity/timing_cpu_vs_gpu_fp32.jsonl" || parity_ok=0
$PARITY --model timing.gguf --ref gpu_fp32 --test gpu_bf16 --lengths $PARITY_LENGTHS \
    --max-rel-diff 5e-2 --min-cosine 0.999 --out "$OUT/parity/timing_gpu_fp32_vs_bf16.jsonl" || parity_ok=0
echo "parity_ok=$parity_ok" | tee "$OUT/parity/RESULT"
[ "$MODE" = full ] && [ $parity_ok = 0 ] && fail "parity gate failed: no baseline of a wrong path"

if [ "$MODE" = probe ]; then
    $BENCH --model story3k.gguf --model-label "story3k (paperkiln, d128 L2 ctx256)" \
        --lengths 128 --runs 1 --warmup 1 --reps 3 --out-dir "$OUT/story3k" || fail "bench story3k"
    $BENCH --model timing.gguf --model-label "timing (paperkiln random, TinyLlama-1.1B shape, GQA 32/4)" \
        --lengths $TIMING_LENGTHS --runs 1 --warmup 1 --reps 2 --out-dir "$OUT/timing" || fail "bench timing"
elif [ "$MODE" = full ]; then
    $BENCH --model story3k.gguf --model-label "story3k (paperkiln, d128 L2 ctx256)" \
        --lengths 128 --runs "$RUNS" --first-run "$FIRST_RUN" --warmup "$WARMUP" --reps "$REPS" \
        --out-dir "$OUT/story3k" || fail "bench story3k"
    $BENCH --model timing.gguf --model-label "timing (paperkiln random, TinyLlama-1.1B shape, GQA 32/4)" \
        --lengths $TIMING_LENGTHS --runs "$RUNS" --first-run "$FIRST_RUN" --warmup "$WARMUP" \
        --reps "$REPS" --long-lengths $LONG_LENGTHS --long-warmup "$LONG_WARMUP" \
        --long-reps "$LONG_REPS" --out-dir "$OUT/timing" || fail "bench timing"
else
    fail "unknown mode $MODE"
fi

# Kernel-time pass: nsys under the same harness, separate from the timed runs
# (profiling perturbs timing, so these numbers never enter the TTFT records).
echo "--- nsys $(date -u +%T)"
WANT_NSYS=$NSYS
NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ "$WANT_NSYS" = 0 ]; then
    echo "nsys pass skipped in this session (NSYS=0)"; NSYS=SKIP
elif [ -z "$NSYS" ]; then
    apt-get install -y -q nsight-systems-cli > /dev/null 2>&1 ||
        apt-get install -y -q "cuda-nsight-systems-$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')" > /dev/null 2>&1
    NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
fi
if [ "$NSYS" = SKIP ]; then
    :
elif [ -z "$NSYS" ]; then
    echo "nsys unavailable on this image" | tee "$OUT/NSYS_UNAVAILABLE"
else
    "$NSYS" --version > "$OUT/nsys_version.txt" 2>&1
    for L in $NSYS_LENGTHS; do
        D="$OUT/nsys_timing_$L"
        $BENCH --model timing.gguf --lengths "$L" --runs 1 --warmup 1 --reps 3 --out-dir "$D" \
            --wrapper "$NSYS profile -t cuda,nvtx -s none --cpuctxsw=none --force-overwrite=true -o {run_dir}/prof" \
            || { echo "nsys bench $L failed"; continue; }
        for R in nvtx_gpu_proj_sum nvtx_kern_sum cuda_gpu_kern_sum nvtx_sum; do
            "$NSYS" stats -r "$R" -f csv -o "$D/stats" "$D/run1/prof.nsys-rep" > /dev/null 2>&1 || true
        done
        rm -f "$D/run1/prof.sqlite"
        ls "$D"
    done
fi

echo "--- package $(date -u +%T)"
find "$OUT" -name '*.nsys-rep' -size +50M -delete
find "$OUT" -name debugging.log -size +20M -exec sh -c 'tail -c 2000000 "$1" > "$1.tail" && rm "$1"' _ {} \;
cp /content/build_ember/make.log "$OUT/ember_make.log" 2>/dev/null
# Python's zipfile: the Colab image has no zip binary.
python3 -c "import shutil; shutil.make_archive('/content/ttft_out_new', 'zip', '/content/out')" \
    && mv -f /content/ttft_out_new.zip /content/ttft_out.zip || fail "packaging"
touch /content/TTFT_DONE
echo "=== done $(date -u) ==="
