#!/usr/bin/env bash
# Runs ON the Colab VM (AMENDMENT.md). Usage: bash vm_gemma.sh FULL_BRANCH_COMMIT RUN_ID
# The driver has uploaded the six Google tokenizer/config files and converter.tgz to /content/upload_RUN_ID/.
# Everything goes under a fresh /content/run_RUN_ID. status.json (each gate's own exit status, the incoming shell
# status, the final status, run id, commits) is written atomically, then DONE only if that succeeded. A failed
# prerequisite stops further paid work; no forward runs unless the regenerated GGUF is the canonical file.
set -u
COMMIT=${1:-}
RUN=${2:-}
SOURCE=5bfda89345b3bcf1f0352eee555f0fd4eb73ac7d          # ember compiled source of the frozen comparison
CANONICAL=0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa
CONTROLLER=ddd3104489dd2935187271d48dfabe77501a2b3c5fa02350aad1ca5e54a2f04c
MIRROR=unsloth/gemma-3-270m-it
MIRROR_REV=23cf460f6bb16954176b3ddcc8d4f250501458a9
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]{1,32}$ ]] || { echo "RUN_ID: 1-32 letters, digits, - or _" >&2; exit 2; }
ROOT=/content
[ "${VM_TEST_HARNESS:-}" = 1 ] && ROOT=${VM_ROOT:-/content}   # test harness only
W=$ROOT/run_$RUN
O=$W/out
U=$ROOT/upload_$RUN
[ -e "$W" ] && { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p "$O" || exit 2
declare -A G=()
REQUIRED="analyze benchmark build checkout contract inputs packages provenance regenerate weights"
finish() {
    local rc=$? pass=true keys final
    keys=$(printf '%s\n' "${!G[@]}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
    [ "$keys" = "$REQUIRED" ] || pass=false
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ "$rc" = 0 ] || pass=false
    [ "$pass" = true ] && final=0 || final=1
    {
        printf '{"run_id": "%s", "commit": "%s", "head": "%s", "source": "%s", "shell_status": %d, "exit_status": %d, "pass": %s, "gates": {' \
            "$RUN" "$COMMIT" "$(git -C "$W/branch" rev-parse HEAD 2>/dev/null)" "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" "$rc" "$final" "$pass"
        local first=1
        for k in $(printf '%s\n' "${!G[@]}" | LC_ALL=C sort); do
            [ $first = 1 ] || printf ', '
            first=0
            printf '"%s": %d' "$k" "${G[$k]}"
        done
        printf '}}\n'
    } > "$O/status.json.tmp" && mv "$O/status.json.tmp" "$O/status.json" || exit 1
    echo "$RUN" > "$O/DONE.tmp" && mv "$O/DONE.tmp" "$O/DONE" || exit 1
    exit $final
}
trap finish EXIT
fail() { G[$1]=$2; exit 1; }
cd "$W"
E=experiments/hf_cuda_2026-10-30
A=$E/gemma_colab_20261103

# sources: the amendment branch (contract, manifests) and the frozen engine source; never modified
git clone -q https://github.com/JohnnyTeutonic/ember.cpp.git branch > "$O/clone.log" 2>&1 || fail checkout $?
git -C branch -c advice.detachedHead=false checkout -q --detach "$COMMIT" >> "$O/clone.log" 2>&1 || fail checkout $?
git -C branch worktree add -q --detach "$W/src" "$SOURCE" >> "$O/clone.log" 2>&1 || fail checkout $?
{ [ "$(git -C branch rev-parse HEAD)" = "$COMMIT" ] && [ "$(git -C src rev-parse HEAD)" = "$SOURCE" ] \
  && [ -z "$(git -C branch status --porcelain)" ] && [ -z "$(git -C src status --porcelain)" ] \
  && [ "$(sha256sum src/$E/benchmark.py | cut -c1-64)" = "$CONTROLLER" ]; } || fail checkout 99
G[checkout]=0

# provenance of this Colab image (recorded, not digest-checked: AMENDMENT.md)
python3 - "$O/provenance.json" <<'PY' || fail provenance $?
import json, platform, subprocess, sys
def out(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return "error: " + repr(e)
p = {"provider": "colab",
     "gpu": out("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "driver": out("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "cuda_toolkit": out("ls -d /usr/local/cuda-* 2>/dev/null | tr '\\n' ' '"),
     "python": platform.python_version(), "os": out("cat /etc/os-release | grep PRETTY_NAME"),
     "cpu": out("grep -m1 'model name' /proc/cpuinfo"), "nproc": out("nproc")}
json.dump(p, open(sys.argv[1], "w"), indent=2)
print(json.dumps(p))
PY
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1 ] || fail provenance 98
G[provenance]=0

# pinned packages, exactly the frozen contract's
apt-get install -y -q libboost-regex-dev > "$O/apt.log" 2>&1 || fail packages $?
python3 -m pip install -q 'torch==2.4.1' --index-url https://download.pytorch.org/whl/cu124 > "$O/pip_torch.log" 2>&1 || fail packages $?
python3 -m pip install -q 'numpy==1.26.4' 'transformers==4.57.0' 'tokenizers==0.22.1' 'gguf==0.19.0' \
    'safetensors==0.6.2' 'huggingface-hub==0.35.3' 'sentencepiece==0.2.2' 'accelerate==1.13.0' > "$O/pip.log" 2>&1 || fail packages $?
python3 -m pip list --format=json > "$O/packages.json"
G[packages]=0

# engine build at the frozen source; a CUDA 12 toolkit if the image has one, else its default
CUDA_HOME=$(ls -d /usr/local/cuda-12* 2>/dev/null | sort -V | tail -1); CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "$CUDA_HOME" > "$O/cuda_home.txt"; "$CUDA_HOME/bin/nvcc" --version > "$O/nvcc.txt" 2>&1
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    > "$O/cmake.log" 2>&1 || fail build $?
grep -q 'CUDA toolkit and cuBLAS found' "$O/cmake.log" || fail build 97
make -C build -j"$(nproc)" ir_hf_cuda_bench > "$O/make.log" 2>&1 || fail build $?
sha256sum build/ir_hf_cuda_bench > "$O/binary.sha256"
G[build]=0

# inputs: the uploaded Google files and converter, against the committed manifest
(cd "$U" && sha256sum -c "$W/branch/$A/inputs.sha256") > "$O/inputs_check.txt" 2>&1 || fail inputs $?
G[inputs]=0

# weights from the ungated mirror, pinned revision, byte-identical to Google's (manifest)
HF=$W/hf/gemma-3-270m-it
mkdir -p "$HF"
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python3 - "$HF" "$MIRROR" "$MIRROR_REV" <<'PY' > "$O/weights.log" 2>&1 || fail weights $?
import shutil, sys
from huggingface_hub import hf_hub_download
dest, repo, rev = sys.argv[1:4]
for name in ("model.safetensors", "tokenizer.model"):
    shutil.copy(hf_hub_download(repo, name, revision=rev), f"{dest}/{name}")
print("downloaded")
PY
(cd "$HF" && sha256sum -c "$W/branch/$A/weights.sha256") >> "$O/weights.log" 2>&1 || fail weights $?
for f in config.json tokenizer_config.json tokenizer.json special_tokens_map.json added_tokens.json generation_config.json; do
    cp "$U/$f" "$HF/$f" || fail weights $?
done
G[weights]=0

# regenerate the canonical GGUF; no forward unless it is byte-identical
mkdir -p conv && tar xzf "$U/converter.tgz" -C conv || fail regenerate $?
CV=$(find conv -name convert_hf_to_gguf.py | head -1)
PYTHONPATH=$(dirname "$CV")/gguf-py python3 "$CV" "$HF" --outtype f32 --outfile "$W/gemma-3-270m-it-f32.gguf" \
    > "$O/conversion.log" 2>&1 || fail regenerate $?
actual=$(sha256sum "$W/gemma-3-270m-it-f32.gguf" | cut -c1-64)
echo "{\"sha256\": \"$actual\", \"expected\": \"$CANONICAL\"}" > "$O/canonical_gguf.json"
[ "$actual" = "$CANONICAL" ] || fail regenerate 96
G[regenerate]=0

# the frozen launch environment, then the contract
unset EMBER_IR EMBER_KV EMBER_DETERMINISTIC EMBER_DECODE_LINEAR EMBER_DECODE_OPT EMBER_DECODE_PICK
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 EMBER_IR_BUILD=1 EMBER_IR_CTX=1024 \
    NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
python3 "branch/$A/launch_contract_gemma.py" --out "$O/launch_contract.json" --provenance "$O/provenance.json" \
    --binary build/ir_hf_cuda_bench --gguf "$W/gemma-3-270m-it-f32.gguf" > "$O/contract.log" 2>&1 || fail contract $?
G[contract]=0

# the frozen controller, unchanged arguments; then its own offline analysis
python3 "src/$E/benchmark.py" --gguf "$W/gemma-3-270m-it-f32.gguf" --binary build/ir_hf_cuda_bench \
    --out "$O/gemma3-270m" --lengths 16 128 512 --steps 32 --cycles 2 > "$O/benchmark.log" 2>&1
G[benchmark]=$?
python3 "src/$E/benchmark.py" --out "$O/gemma3-270m" --analyze > "$O/gate_summary.json" 2> "$O/analyze.log"
G[analyze]=$?
[ "${G[benchmark]}" = 0 ] && [ "${G[analyze]}" = 0 ]
