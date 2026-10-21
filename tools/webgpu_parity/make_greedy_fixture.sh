#!/bin/bash
# ember CPU greedy replies (exact ids) for the 40 frozen prompts, one thread (light under the shared CPU rule).
#   bash make_greedy_fixture.sh MODEL.gguf OUT.json
set -eu
M=$1; OUT=$2
B=~/ember_bin/338b24a/cpu/tinyllama_server
P=/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl
for p in $(pgrep -f '[t]inyllama_server .* 8099 '); do kill $p; done
OMP_NUM_THREADS=1 nice -n 19 $B "$M" 8099 127.0.0.1 ~/ember_bin/338b24a/www 0 > /dev/null 2>&1 &
S=$!
for i in $(seq 1 60); do curl -s 127.0.0.1:8099/health > /dev/null && break; sleep 1; done
python3 - "$P" "$OUT" << 'PY'
import json, sys, urllib.request
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")][:40]
out = []
for i, r in enumerate(rows):
    body = {"user_input": r["prompt"], "max_new_tokens": 60, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0,
            "return_ids": True}
    req = urllib.request.Request("http://127.0.0.1:8099/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    out.append({"i": i, "prompt_ids": d["prompt_ids"], "ids": d["ids"], "stop_id": d["stop_id"], "reply": d["reply"]})
json.dump({"prompts": out}, open(sys.argv[2], "w"))
print(len(out), "replies; tokens", sum(len(o["ids"]) for o in out))
PY
kill $S
