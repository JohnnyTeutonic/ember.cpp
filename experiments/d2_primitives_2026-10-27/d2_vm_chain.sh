#!/bin/bash
# VM side of the chain model's CUDA leg (ember.cpp, 27 Oct 2026). Launched detached by d2_drive.sh (VM=d2_vm_chain.sh).
# Inputs in /content: COMMITS, ember.tgz (git archive), small.tgz (d2_stage_chain.sh: the chain's Shaw GGUF, the
# 42 prompts of chain_parity.sh, SHA256SUMS).
# Build (HAS_CUDA on, Release, the device's architecture): tinyllama_server, ir_device_gate.
# Legs: ir_device_gate on the chain model (device_supported false: relative_kv has no CUDA kernel); the CUDA build's
# tinyllama_server with 99 GPU layers on the chain model, the 42 prompts through /chat (max_new_tokens 24, top_k 1,
# chat_parity.py's body), replies to out/cuda_chat.jsonl, and the log line naming the path the interpreter took.
set -uo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; SM=/content/small
mkdir -p "$OUT"
fail() { echo "FAILED: $*"; echo "$*" > /content/VM_FAILED; exit 1; }
read -r C < /content/COMMITS
echo "=== vm $(date -u) commit=$C ==="
{ nvidia-smi; nvcc --version; nproc; free -g; } > "$OUT/hardware.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "compute capability $CC, nproc $(nproc)"
tar xzf small.tgz || fail "untar small"
(cd small && sha256sum -c --quiet SHA256SUMS) || fail "small sha256"
cp $SM/SHA256SUMS $OUT/small.sha256
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1 || fail "apt libboost-regex-dev"
echo "--- build $(date -u +%T)"
mkdir -p /content/src
tar xzf ember.tgz -C /content/src --strip-components=1 || fail "untar ember"
find /content/src -name "*.sh" -exec sed -i 's/\r$//' {} +
B=/content/b
t0=$(date +%s)
cmake -S /content/src -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC > $OUT/cmake.log 2>&1 || { tail -20 $OUT/cmake.log; fail "cmake"; }
grep -q "CUDA toolkit and cuBLAS found" $OUT/cmake.log && echo "CUDA toolkit and cuBLAS found" || fail "configured without CUDA"
grep -E "HAS_CUDA" $B/CMakeCache.txt | head -3
make -C $B -j$(nproc) tinyllama_server ir_device_gate > $OUT/make.log 2>&1 || { tail -30 $OUT/make.log; fail "build"; }
echo "built (sm_$CC) in $(( $(date +%s) - t0 )) s"
sha256sum $B/tinyllama_server $B/ir_device_gate > $OUT/bins.sha256
G=$SM/arxiv_1803_02155.gguf
$B/ir_device_gate $G --n 40 --len 16 --gen 16 > $OUT/dg_chain.json 2> $OUT/dg_chain.log
echo "ir_device_gate chain rc=$?: $(head -c 300 $OUT/dg_chain.json | tr -d '\n')"
mkdir -p /content/run/www
(cd /content/run && exec $B/tinyllama_server $G 8125 127.0.0.1 /content/run/www 99) > $OUT/server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:8125/ && break; sleep 1; done
python3 - $SM/prompts.jsonl $OUT/cuda_chat.jsonl <<'EOF'
import json, sys, urllib.request
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
with open(sys.argv[2], "w") as f:
    for i, r in enumerate(rows):
        body = {"user_input": r["prompt"], "max_new_tokens": 24, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}
        req = urllib.request.Request("http://127.0.0.1:8125/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        e = json.load(urllib.request.urlopen(req, timeout=300))
        f.write(json.dumps({"i": i, "prompt": r["prompt"], "reply": e["reply"], "tokens": e["tokens"],
                            "stop_reason": e["stop_reason"], "prompt_tokens": e["prompt_tokens"]}) + "\n")
print(len(rows), "replies")
EOF
kill $SP 2> /dev/null
grep -h "interpreter" /content/run/debugging.log $OUT/server.log 2> /dev/null | head -5 | tee $OUT/server_path.txt
cp /content/run/debugging.log $OUT/server_debugging.log 2> /dev/null
cd /content && rm -f out.zip && python3 -c "import shutil; shutil.make_archive('/content/out', 'zip', '/content', 'out')" && echo "packed $(du -sh out.zip | cut -f1)"
echo done > /content/VM_DONE
echo "=== done $(date -u) ==="
