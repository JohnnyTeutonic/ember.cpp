#!/bin/bash
# CPU greedy parity on one model: mtstudio chat --kv-cache (CPU) vs ember's tinyllama_server (CPU build,
# gpu_layers 0) on the 40 frozen prompts, 48 new tokens, both request variants.
# Usage: parity_cpu.sh TAG CHAT_SPEC GGUF
set -u
TAG=$1; SPEC=$2; GGUF=$3
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/rmsnorm_2026-10-24
W=~/ember_wt/shape_parity
MT=~/pk_state_build/mtstudio
EB=~/ember_wt/build_cpu/tinyllama_server
P=/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl
mkdir -p $W/run_$TAG/www && cd $W/run_$TAG
ln -sf /dev/null debugging.log
$MT chat $SPEC --port 8391 --kv-cache > mts.log 2>&1 &
MP=$!
$EB $GGUF 8390 127.0.0.1 $W/run_$TAG/www 0 > /dev/null 2> ember.err &
EP=$!
for i in $(seq 1 180); do
  curl -s -m 2 127.0.0.1:8390/health > /dev/null && curl -s -m 2 127.0.0.1:8391/health > /dev/null && break
  sleep 1
done
for V in spot p1; do
  python3 $S/sp_compare.py collect http://127.0.0.1:8390 $P 40 48 $V ember_cpu_$V.jsonl
  python3 $S/sp_compare.py collect http://127.0.0.1:8391 $P 40 48 $V mtstudio_cpu_$V.jsonl
  python3 $S/sp_compare.py compare ember_cpu_$V.jsonl mtstudio_cpu_$V.jsonl > cmp_cpu_$V.json
  python3 -c "import json;d=json.load(open('cmp_cpu_$V.json'));print('$TAG $V', d['replies_identical'], '/', d['n'])"
done
kill $MP $EP 2>/dev/null; wait 2>/dev/null
