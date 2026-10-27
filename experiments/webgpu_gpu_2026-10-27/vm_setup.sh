#!/usr/bin/env bash
# E6 on the GPU VM: the builds and the official checkpoints (VM side, after vm_probe.sh).
#   E6_W=work dir (the ember tree at $E6_W/ember) E6_OUT=out root bash vm_setup.sh
# - ember CPU (the EXACT reference): cmake -DHAS_CUDA=OFF Release of the unpacked tree (/root/e6/ember):
#   tinyllama_server, ir_logits, libtinyllama_core.a; tools/ir_wasm_gate linked with it, as
#   ../wasm_families_2026-10-25/build_gate.sh does.
# - The page's WebAssembly, deploy/wasm/build.sh with the laptop's emsdk version (6.0.6): ember.js (the
#   Space's build: single-threaded, SIMD, 1gb heap), ember_4gb.js (EMBER_WASM_MAX_MEMORY=4gb, for the
#   official checkpoints) and ember-mt.js (EMBER_WASM_THREADS=8, 4gb; cross-origin-isolated pages only).
# - In the background: Qwen3-0.6B Q8_0 and Qwen2.5-0.5B-Instruct fp16 (the official GGUFs), and
#   unsloth/gemma-3-270m-it, EleutherAI/pythia-160m (pinned revision) and state-spaces/mamba-130m-hf
#   through llama.cpp b6000's convert_hf_to_gguf.py --outtype f32, as the 22-25 Oct VM legs made them;
#   sha256 against the laptop's files. $E6_W/TP_DONE when done.
# Outputs: $E6_OUT/setup/ (logs, hashes); builds and checkpoints in $E6_W.
set -uo pipefail
W=${E6_W:-/root/e6}; SRC=$W/ember; O=${E6_OUT:-/workspace/e6/out}/setup
mkdir -p "$O" "$W/tp"
exec > >(tee -a "$O/setup.log") 2>&1
echo "=== setup $(date -u) ==="
export DEBIAN_FRONTEND=noninteractive
# The container's CPU quota (nproc shows the host's).
CPUS=$(python3 -c "
q = open('/sys/fs/cgroup/cpu.max').read().split()
print(max(1, int(int(q[0]) / int(q[1]))) if q[0] != 'max' else __import__('os').cpu_count())" 2>/dev/null || nproc)
echo "cpu quota: $CPUS"; echo "$CPUS" > "$O/cpus.txt"
apt-get install -y -q libboost-regex-dev git build-essential > /dev/null 2>&1 || { apt-get update -qq; apt-get install -y -q libboost-regex-dev git build-essential > /dev/null; }
python3 -m pip -q install --root-user-action=ignore "cmake==3.31.6" huggingface_hub gguf sentencepiece "transformers==4.57.0" 2>&1 | tail -2
hash -r

( set +e
  cd "$W/tp"
  [ -d llamacpp_b6000 ] || git clone -q --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp llamacpp_b6000
  python3 - << 'PY'
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
import json
api, rev = HfApi(), {}
for repo, f, d in (("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", "qwen3-0.6b"),
                   ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf", "qwen2.5-0.5b-instruct")):
    hf_hub_download(repo, f, local_dir=d)
    rev[repo] = api.model_info(repo).sha
snapshot_download("unsloth/gemma-3-270m-it", local_dir="gemma-3-270m-it", allow_patterns=["*.json", "*.safetensors", "*.model"])
snapshot_download("EleutherAI/pythia-160m", revision="50f5173d932e8e61f858120bcb800b97af589f46", local_dir="pythia-160m",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
snapshot_download("state-spaces/mamba-130m-hf", local_dir="mamba-130m", allow_patterns=["*.json", "*.safetensors", "*.txt"])
for repo in ("unsloth/gemma-3-270m-it", "state-spaces/mamba-130m-hf"):
    rev[repo] = api.model_info(repo).sha
rev["EleutherAI/pythia-160m"] = "50f5173d932e8e61f858120bcb800b97af589f46"
json.dump(rev, open("revisions.json", "w"), indent=1)
print("downloads done")
PY
  for m in gemma-3-270m-it pythia-160m mamba-130m; do
    [ -f $m/$m-f32.gguf ] || python3 llamacpp_b6000/convert_hf_to_gguf.py $m --outtype f32 --outfile $m/$m-f32.gguf > "$O/convert_$m.log" 2>&1 || rm -f $m/$m-f32.gguf
  done
  sha256sum qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf gemma-3-270m-it/gemma-3-270m-it-f32.gguf \
    pythia-160m/pythia-160m-f32.gguf mamba-130m/mamba-130m-f32.gguf > "$O/thirdparty.sha256"
  cp revisions.json "$O/thirdparty_revisions.json"
  touch "$W/TP_DONE" ) > "$O/thirdparty_fetch.log" 2>&1 &

echo "--- ember CPU build $(date -u +%T)"
cmake -S "$SRC" -B "$W/build_cpu" -DHAS_CUDA=OFF -DCMAKE_BUILD_TYPE=Release > "$O/cmake_cpu.log" 2>&1 || { tail -30 "$O/cmake_cpu.log"; exit 1; }
cmake --build "$W/build_cpu" -j "$CPUS" --target tinyllama_server ir_logits tinyllama_core > "$O/build_cpu.log" 2>&1 || { tail -30 "$O/build_cpu.log"; exit 1; }
JI=$W/build_cpu/_deps/nlohmann_json-src
(cd "$SRC" && g++ -O3 -DNDEBUG -std=c++17 -I. -I"$JI/include" tools/ir_wasm_gate.cpp "$W/build_cpu/libtinyllama_core.a" -fopenmp -lpthread \
  -o "$W/build_cpu/ir_wasm_gate") || exit 1
echo "--- emsdk $(date -u +%T)"
if [ ! -f "$W/emsdk/upstream/emscripten/emcc" ]; then
  git clone -q https://github.com/emscripten-core/emsdk "$W/emsdk"
  (cd "$W/emsdk" && ./emsdk install 6.0.6 > "$O/emsdk.log" 2>&1 && ./emsdk activate 6.0.6 >> "$O/emsdk.log" 2>&1) || { tail -20 "$O/emsdk.log"; exit 1; }
fi
source "$W/emsdk/emsdk_env.sh" > /dev/null 2>&1
em++ --version | head -1 | tee "$O/emcc_version.txt"
echo "--- WebAssembly builds $(date -u +%T)"
export JSON_INC=$JI/single_include
mkdir -p "$W/wasm"
bash "$SRC/deploy/wasm/build.sh" "$W/wasm" > "$O/wasm_st.log" 2>&1 & P1=$!
EMBER_WASM_MAX_MEMORY=4gb EMBER_WASM_NAME=ember_4gb bash "$SRC/deploy/wasm/build.sh" "$W/wasm" > "$O/wasm_4gb.log" 2>&1 & P2=$!
EMBER_WASM_THREADS=8 EMBER_WASM_MAX_MEMORY=4gb EMBER_WASM_NAME=ember-mt bash "$SRC/deploy/wasm/build.sh" "$W/wasm" > "$O/wasm_mt.log" 2>&1 & P3=$!
wait $P1 $P2 $P3
ls -la "$W/wasm"
(cd "$W" && sha256sum build_cpu/tinyllama_server build_cpu/ir_logits build_cpu/ir_wasm_gate wasm/*.wasm wasm/*.js) > "$O/builds.sha256"
cat "$O/builds.sha256"
echo "setup builds done $(date -u)"
