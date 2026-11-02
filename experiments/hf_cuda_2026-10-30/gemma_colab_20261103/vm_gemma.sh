#!/usr/bin/env bash
# Runs ON the Colab VM (AMENDMENT.md). Usage: bash vm_gemma.sh FULL_BRANCH_COMMIT RUN_ID
# The driver has uploaded the six Google tokenizer/config files and converter.tgz to /content/upload_RUN_ID/.
# Everything goes under a fresh /content/run_RUN_ID. Once that directory exists, status.json (each gate's own exit
# status, the incoming shell status, the final status, run id, commits, whether scientific work started) is written
# atomically on every exit path, then DONE only if that succeeded. A failed prerequisite stops further paid work; no
# forward runs unless the regenerated GGUF is the canonical file. SCIENCE_STARTED is written immediately before the
# frozen controller starts: a run that reached it may not be repeated (AMENDMENT.md, outcomes).
set -u
COMMIT=${1:-}
RUN=${2:-}
# ember compiled source of the frozen comparison: 5bfda893 before the 3 Nov history purge renamed its commit; the
# tree, which is the source content, is unchanged and is checked as well
SOURCE=4d276ff91ab9b191be8556010b05d74cd4cb2520
SOURCE_TREE=ee183b61772737b9b6f659ddc25f34d1a75ba332
CANONICAL=0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa
CONTROLLER=ddd3104489dd2935187271d48dfabe77501a2b3c5fa02350aad1ca5e54a2f04c
MIRROR=unsloth/gemma-3-270m-it
MIRROR_REV=23cf460f6bb16954176b3ddcc8d4f250501458a9
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]{1,32}$ ]] || { echo "RUN_ID: 1-32 letters, digits, - or _" >&2; exit 2; }
ROOT=/content
[ "${VM_TEST_HARNESS:-}" = 1 ] && ROOT=${VM_ROOT:-/content}   # test harness only
signal_point() { [ "${VM_TEST_HARNESS:-}" = 1 ] && [ "${VM_TEST_SIGNAL_AT:-}" = "$1" ] && kill -TERM $$; }  # tests
W=$ROOT/run_$RUN
O=$W/out
U=$ROOT/upload_$RUN
[ -e "$W" ] && { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p "$O/binaries" "$O/inputs" || exit 2
declare -A G=()
REQUIRED="analyze benchmark build checkout contract environment inputs packages provenance regenerate toolkit weights"
finish() {
    local rc=$? pass=true keys final science=false
    trap '' TERM INT   # publication is never interrupted; a stop arriving now finds DONE shortly
    keys=$(printf '%s\n' "${!G[@]}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
    [ "$keys" = "$REQUIRED" ] || pass=false
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ "$rc" = 0 ] || pass=false
    [ "$pass" = true ] && final=0 || final=1
    [ -e "$O/SCIENCE_STARTED" ] && science=true
    {
        printf '{"run_id": "%s", "commit": "%s", "head": "%s", "source": "%s", "science_started": %s, "shell_status": %d, "exit_status": %d, "pass": %s, "gates": {' \
            "$RUN" "$COMMIT" "$(git -C "$W/branch" rev-parse HEAD 2>/dev/null)" "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" "$science" "$rc" "$final" "$pass"
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
# Stop on SIGTERM/SIGINT. The controller runs in its own process group (setsid), which its native child inherits
# (the frozen controller starts it with a plain Popen). Containment is by that group: a process that left it on its
# own would not be seen. Every wait is bounded and nothing waits on a PID:
# - a stop that arrives while the controller is being forked and registered is held in STOP_PENDING and delivered
#   right after registration, so CPID is always known once a controller exists;
# - live() sees any non-zombie process whose PID or process group is CPID, so it covers the child before its
#   setsid() (when the group does not exist yet) as well as the group and any member the controller left behind;
# - TERM goes to the group and to CPID itself, repeated every second for up to 60 s, then KILL; stop.json records
#   whether anything is left. Only then does finish publish status.json and DONE.
# Other children of this shell (pip, make) are sent TERM too.
CPID=""
STOP_PENDING=""
live() {   # a failed or empty process listing counts as live: absence is never assumed
    [ -n "$CPID" ] || return 1
    local t
    t=$(ps -eo pid=,pgid=,stat=) && [ -n "$t" ] || return 0
    # exit 1 only for a well-formed listing with no live match; a malformed row or an awk failure counts as live
    awk -v c="$CPID" 'NF != 3 || $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/ {bad = 1}
        ($1 == c || $2 == c) && $3 !~ /^Z/ {f = 1} END {exit (f || bad) ? 0 : 1}' <<< "$t"
    [ $? -ne 1 ]
}
on_stop() {
    trap '' TERM INT
    local gone=true started=false
    if [ -n "$CPID" ]; then
        started=true
        for _ in $(seq 60); do
            live || break
            kill -TERM -- -"$CPID" "$CPID" 2>/dev/null
            sleep 1
        done
        if live; then kill -KILL -- -"$CPID" "$CPID" 2>/dev/null; sleep 2; fi
        live && { gone=false; : > "$O/QUIESCENCE_UNCONFIRMED"; }
        G[benchmark]=143
    fi
    pkill -TERM -P $$ 2>/dev/null
    printf '{"stopped_at": "%s", "controller_started": %s, "controller_group_gone": %s}\n' \
        "$(date -u +%FT%TZ)" "$started" "$gone" > "$O/stop.json"
    exit 1
}
trap on_stop TERM INT
# the driver stops this run through this PID (SIGTERM; see on_stop); published only once the traps are in place
{ echo $$ > "$ROOT/vm_$RUN.pid.tmp" && mv "$ROOT/vm_$RUN.pid.tmp" "$ROOT/vm_$RUN.pid"; } || exit 2
cd "$W"
E=experiments/hf_cuda_2026-10-30
A=$E/gemma_colab_20261103

# sources: the amendment branch (contract, manifests) and the frozen engine source; never modified
git clone -q https://github.com/JohnnyTeutonic/ember.cpp.git branch > "$O/clone.log" 2>&1 || fail checkout $?
git -C branch -c advice.detachedHead=false checkout -q --detach "$COMMIT" >> "$O/clone.log" 2>&1 || fail checkout $?
git -C branch worktree add -q --detach "$W/src" "$SOURCE" >> "$O/clone.log" 2>&1 || fail checkout $?
{ [ "$(git -C branch rev-parse HEAD)" = "$COMMIT" ] && [ "$(git -C src rev-parse HEAD)" = "$SOURCE" ] \
  && [ "$(git -C src rev-parse 'HEAD^{tree}')" = "$SOURCE_TREE" ] \
  && [ -z "$(git -C branch status --porcelain)" ] && [ -z "$(git -C src status --porcelain)" ] \
  && [ "$(sha256sum src/$E/benchmark.py | cut -c1-64)" = "$CONTROLLER" ]; } || fail checkout 99
# the evidence bundle keeps the exact controller, amendment, contract and manifests (the frozen run kept its inputs)
cp src/$E/benchmark.py branch/$A/AMENDMENT.md branch/$A/launch_contract_gemma.py branch/$A/inputs.sha256 \
   branch/$A/weights.sha256 branch/$A/vm_gemma.sh "$O/inputs/" || fail checkout $?
G[checkout]=0

# provenance of this Colab image: recorded, and checked for what this run needs (AMENDMENT.md)
python3 - "$O/provenance.json" > "$O/provenance.log" 2>&1 <<'PY' || fail provenance $?
import json, platform, subprocess, sys
def out(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    if r.returncode or not r.stdout.strip():
        raise SystemExit(f"provenance command failed: {cmd}: {r.stderr.strip()}")
    return r.stdout.strip()
p = {"provider": "colab",
     "gpu": out("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "driver": out("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "cuda_toolkit": out("ls -d /usr/local/cuda-* | tr '\\n' ' '"),
     "python": platform.python_version(), "os": out("grep PRETTY_NAME /etc/os-release"),
     "cpu": out("grep -m1 'model name' /proc/cpuinfo"), "nproc": out("nproc"),
     "nvidia_smi": out("nvidia-smi")}
if p["gpu"].count("\n") or "L4" not in p["gpu"]:
    raise SystemExit("exactly one NVIDIA L4 required, found: " + p["gpu"])
json.dump(p, open(sys.argv[1], "w"), indent=2)
print(json.dumps({k: v for k, v in p.items() if k != "nvidia_smi"}))
PY
G[provenance]=0

# CUDA toolkit 12.6 from NVIDIA's ubuntu2404 repository, exact versions (the image had only CUDA 13.0 on 3 Nov; ember
# 5bfda893 was verified to compile under 12.6). It runs before the interpreter and packages, so an APT problem costs
# minutes, not the torch download. The image may already configure NVIDIA's repository, possibly with another
# Signed-By, which APT refuses to combine with a second entry, and dpkg would not restore a package-owned list file
# the image has edited. So no package manages the entry here:
# - every NVIDIA source entry is recorded first; if the image's own configuration offers both pinned versions
#   exactly and a simulated install of them succeeds, it is used unchanged;
# - otherwise the keyring is taken from NVIDIA's keyring package (package and keyring both checked by hash) and
#   installed under a name of our own, and apt_reconcile.py sets aside only the entries for this exact repository
#   (copies kept in out/apt_disabled), writes one entry signed by that keyring, and fails unless that entry is the
#   only active one for the repository; apt-get update and the same exact checks must then pass.
CUDA_KEYRING_DEB_SHA=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba
CUDA_KEYRING_SHA=25100d6f2eccaee7d6719a70e3c0c6145064aa9edc2c32148f693ac4c524a376
CUDA_REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
APT_ETC=/etc/apt KEYRINGS=/usr/share/keyrings SYS_PY=python3
if [ "${VM_TEST_HARNESS:-}" = 1 ]; then   # test harness only
    APT_ETC=${VM_APT_ETC:-$APT_ETC} KEYRINGS=${VM_KEYRINGS:-$KEYRINGS} SYS_PY=${VM_SYS_PYTHON:-$SYS_PY}
fi
# cuda-nvtx-12-6: ember includes nvtx3/nvToolsExt.h (header-only), which the minimal build lacks (g4-20261103)
CUDA_PINS="cuda-minimal-build-12-6=12.6.3-1 libcublas-dev-12-6=12.6.4.1-1 cuda-nvtx-12-6=12.6.77-1"
offered() {   # package, exact version: listed by apt-cache madison
    apt-cache madison "$1" | awk -F'|' -v v="$2" '{gsub(/ /, "", $2)} $2 == v {f = 1} END {exit !f}'
}
pinned_available() {
    offered cuda-minimal-build-12-6 12.6.3-1 && offered libcublas-dev-12-6 12.6.4.1-1 \
        && offered cuda-nvtx-12-6 12.6.77-1 \
        && apt-get install --simulate -q --no-install-recommends $CUDA_PINS > "$O/apt_simulate.txt"
}
toolkit_gate() {
    grep -rn -F developer.download.nvidia.com "$APT_ETC/sources.list" "$APT_ETC/sources.list.d" \
        > "$O/apt_nvidia_sources_before.txt" 2>&1
    if apt-get update -q && pinned_available; then
        echo "toolkit source: the image's own APT configuration, unchanged"
    else
        echo "toolkit source: our entry for $CUDA_REPO; the image's entries for it set aside"
        curl -sSfL -o "$W/cuda-keyring.deb" "$CUDA_REPO/cuda-keyring_1.1-1_all.deb" || return $?
        echo "$CUDA_KEYRING_DEB_SHA  $W/cuda-keyring.deb" | sha256sum -c - || return $?
        dpkg-deb -x "$W/cuda-keyring.deb" "$W/cuda-keyring" || return $?
        echo "$CUDA_KEYRING_SHA  $W/cuda-keyring/usr/share/keyrings/cuda-archive-keyring.gpg" | sha256sum -c - \
            || return $?
        cat "$W/cuda-keyring/usr/share/keyrings/cuda-archive-keyring.gpg" \
            > "$KEYRINGS/vm-gemma-cuda-archive-keyring.gpg" || return $?
        "$SYS_PY" "branch/$A/apt_reconcile.py" --apt-etc "$APT_ETC" --uri "$CUDA_REPO" \
            --keyring "$KEYRINGS/vm-gemma-cuda-archive-keyring.gpg" \
            --entry "$APT_ETC/sources.list.d/vm-gemma-cuda-12-6.list" --backup "$O/apt_disabled" \
            > "$O/apt_reconcile.json" || return $?
        apt-get update -q || return $?
        pinned_available || { echo "the pinned CUDA 12.6 versions are not offered after reconciliation"; return 98; }
    fi
    grep -rn -F developer.download.nvidia.com "$APT_ETC/sources.list" "$APT_ETC/sources.list.d" \
        > "$O/apt_nvidia_sources_after.txt" 2>&1
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends $CUDA_PINS || return $?
    dpkg -l | grep -E 'cuda|cublas' > "$O/cuda_packages.txt"
}
toolkit_gate > "$O/toolkit.log" 2>&1 || fail toolkit $?
G[toolkit]=0

# Pinned interpreter and packages (AMENDMENT.md, v2). The frozen pod ran CPython 3.11; the Colab image's own Python
# (3.13 on 3 Nov, for which torch 2.4.1 has no wheel) is used only for provenance and to install uv. uv 0.11.29
# (wheel pinned by hash) installs its CPython 3.11.15 build into a seeded venv, first on PATH from here on, so every
# later python3 (packages, contract, conversion, controller, analysis) is that interpreter. The nine pins are
# installed into it by pip, exactly as before; then the real import path is checked.
UV_WHEEL_SHA=eec03a8b63d55915694db3af4e91324b39ced49e2aeac7af37851c7eb3f470ea
{
    apt-get install -y -q libboost-regex-dev \
    && echo "uv==0.11.29 --hash=sha256:$UV_WHEEL_SHA" > "$W/uv.requirements" \
    && python3 -m pip install -q --require-hashes -r "$W/uv.requirements" \
    && python3 -m uv python install 3.11.15 \
    && python3 -m uv venv --seed --python 3.11.15 "$W/venv" \
    && export PATH="$W/venv/bin:$PATH" VIRTUAL_ENV="$W/venv" \
    && python3 -c "import sys; assert sys.version_info[:3] == (3, 11, 15) and sys.prefix == '$W/venv', sys.version" \
    && python3 -m pip install -q 'torch==2.4.1' --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install -q 'numpy==1.26.4' 'transformers==4.57.0' 'tokenizers==0.22.1' 'gguf==0.19.0' \
        'safetensors==0.6.2' 'huggingface-hub==0.35.3' 'sentencepiece==0.2.2' 'accelerate==1.13.0' \
    && python3 -m pip list --format=json > "$O/packages.json" \
    && python3 -c "import sys; print(sys.version)" > "$O/interpreter.txt" \
    && python3 -c "import torch, transformers; from transformers import AutoModelForCausalLM, Gemma3ForCausalLM; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, transformers.__version__)"
} > "$O/packages.log" 2>&1 || fail packages $?
G[packages]=0

# the frozen launch environment (set where the frozen wrapper set it: after packages, before download/conversion)
unset EMBER_IR EMBER_KV EMBER_DETERMINISTIC EMBER_DECODE_LINEAR EMBER_DECODE_OPT EMBER_DECODE_PICK
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 EMBER_IR_BUILD=1 EMBER_IR_CTX=1024 \
    NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
python3 "branch/$A/launch_contract_gemma.py" --out "$O/environment_contract.json" --provenance "$O/provenance.json" \
    > "$O/environment_contract.log" 2>&1 || fail environment $?
G[environment]=0

# engine build at the frozen source with the toolkit installed above (/usr/local/cuda-12.6, nvcc release 12.6);
# sm_89; Release. A new build: the executable is archived and its hash recorded.
CUDA_HOME=/usr/local/cuda-12.6
[ "${VM_TEST_HARNESS:-}" = 1 ] && CUDA_HOME=${VM_CUDA_HOME:-$CUDA_HOME}   # test harness only
# evidence: compiler and loader search variables inherited from the image
env | grep -E '^(CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|LIBRARY_PATH|LD_LIBRARY_PATH|CUDA_PATH|CUDA_HOME|CUDACXX|NVCC_[A-Z_]*|CFLAGS|CXXFLAGS|LDFLAGS)=' \
    > "$O/build_env.txt" 2>&1
{
    echo "$CUDA_HOME" > "$O/cuda_home.txt" \
    && "$CUDA_HOME/bin/nvcc" --version > "$O/nvcc.txt" \
    && grep -q 'release 12\.6' "$O/nvcc.txt" \
    && cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
        -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" > "$O/cmake.log" \
    && { cp build/CMakeCache.txt "$O/CMakeCache.txt"; true; } \
    && grep -q 'CUDA toolkit and cuBLAS found' "$O/cmake.log" \
    && grep -qF "Found CUDA: $CUDA_HOME (found version \"12.6" "$O/cmake.log" \
    && make -C build -j"$(nproc)" ir_hf_cuda_bench > "$O/make.log" \
    && { echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"; ldd build/ir_hf_cuda_bench || echo "ldd exit $?"; true; } \
        > "$O/ldd.txt" 2>&1 \
    && cp build/ir_hf_cuda_bench "$O/binaries/ir_hf_cuda_bench" \
    && [ "$(sha256sum build/ir_hf_cuda_bench | cut -c1-64)" = "$(sha256sum "$O/binaries/ir_hf_cuda_bench" | cut -c1-64)" ] \
    && sha256sum "$O/binaries/ir_hf_cuda_bench" > "$O/binary.sha256"
} 2> "$O/build_errors.log" || fail build $?
G[build]=0

# inputs: the uploaded Google files and converter, against the committed manifest
(cd "$U" && sha256sum -c "$W/branch/$A/inputs.sha256") > "$O/inputs_check.txt" 2>&1 || fail inputs $?
G[inputs]=0

# weights from the ungated mirror, pinned revision, byte-identical to Google's (manifest)
HF=$W/hf/gemma-3-270m-it
mkdir -p "$HF"
python3 - "$HF" "$MIRROR" "$MIRROR_REV" <<'PY' > "$O/weights.log" 2>&1 || fail weights $?
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
[ -n "$CV" ] || fail regenerate 95
PYTHONPATH=$(dirname "$CV")/gguf-py python3 "$CV" "$HF" --outtype f32 --outfile "$W/gemma-3-270m-it-f32.gguf" \
    > "$O/conversion.log" 2>&1 || fail regenerate $?
actual=$(sha256sum "$W/gemma-3-270m-it-f32.gguf" | cut -c1-64)
echo "{\"sha256\": \"$actual\", \"expected\": \"$CANONICAL\"}" > "$O/canonical_gguf.json"
[ "$actual" = "$CANONICAL" ] || fail regenerate 96
G[regenerate]=0

# the full contract, with the binary's readiness inventory
python3 "branch/$A/launch_contract_gemma.py" --out "$O/launch_contract.json" --provenance "$O/provenance.json" \
    --binary build/ir_hf_cuda_bench --gguf "$W/gemma-3-270m-it-f32.gguf" > "$O/contract.log" 2>&1 || fail contract $?
G[contract]=0

# the frozen controller, unchanged arguments; scientific work starts here. It runs in its own process group and this
# shell waits on it, so a stop signal is handled at once (bash defers traps while a foreground command runs). A stop
# that arrives during the fork and registration is held, then delivered (see on_stop).
# With job control off (this non-interactive shell), a background child is not a group leader, so setsid(1) calls
# setsid() and execs in place: $! is the controller and its group id. With job control on it would fork instead.
case $- in *m*) fail benchmark 95;; esac
trap 'STOP_PENDING=1' TERM INT
date -u +%FT%TZ > "$O/SCIENCE_STARTED" || fail benchmark 94
setsid python3 "src/$E/benchmark.py" --gguf "$W/gemma-3-270m-it-f32.gguf" --binary build/ir_hf_cuda_bench \
    --out "$O/gemma3-270m" --lengths 16 128 512 --steps 32 --cycles 2 > "$O/benchmark.log" 2>&1 &
signal_point after_fork
CPID=$!
echo "$CPID" > "$O/controller.pid"
trap on_stop TERM INT
[ -n "$STOP_PENDING" ] && on_stop
signal_point after_register
wait "$CPID"
G[benchmark]=$?
# Nothing the controller started may outlive it: a member of its group still running now could still be writing.
if live; then
    ps -eo pid=,pgid=,stat=,args= | awk -v c="$CPID" '$2 == c' > "$O/leftover_processes.txt"
    kill -KILL -- -"$CPID" 2>/dev/null; sleep 2
    # still there: the outputs may still change, so the archive cannot be certified final (drive.py reads this)
    live && { echo "still present after KILL" >> "$O/leftover_processes.txt"; : > "$O/QUIESCENCE_UNCONFIRMED"; }
    G[benchmark]=97
fi
CPID=""
python3 "src/$E/benchmark.py" --out "$O/gemma3-270m" --analyze > "$O/gate_summary.json" 2> "$O/analyze.log"
G[analyze]=$?
# what stopped the controller, if anything: HALTED's own content and the logs' tails (AMENDMENT.md, outcomes)
python3 - "$O" <<'PY' > "$O/failure_summary.json" 2>&1
import json, pathlib, sys
o = pathlib.Path(sys.argv[1])
h = o / "gemma3-270m" / "HALTED"
tail = lambda p: p.read_text(errors="replace").splitlines()[-30:] if p.exists() else []
print(json.dumps({"halted": h.exists(), "halted_text": h.read_text(errors="replace") if h.exists() else None,
                  "benchmark_log_tail": tail(o / "benchmark.log"), "analyze_log_tail": tail(o / "analyze.log")},
                 indent=2))
PY
[ "${G[benchmark]}" = 0 ] && [ "${G[analyze]}" = 0 ]
