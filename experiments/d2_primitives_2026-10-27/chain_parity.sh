#!/bin/bash
# Greedy parity on the arXiv chain's Shaw model (paperkiln's chain on 1803.02155; flex d 64, 2 layers, 4 heads,
# relative_kv per head, max_distance 16): ember's CPU tinyllama_server against `mtstudio chat` (run as a black box
# from a copy of the chain's sft/ run, so the chain's own directories are not touched), through tools/chat_parity.py:
# the 40 frozen tinychat_hf_prompts_v1 prompts and the chain's 2 chat turns (each a one-turn request, as the chain
# sent them), max_new_tokens 24, top_k 1. Then mtstudio's replies here against the chain's recorded
# mtstudio_greedy.jsonl. WSL, under the CPU budget.
#   bash chain_parity.sh BUILD_DIR OUT_DIR
set -u
B=${1:?build}; O=${2:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
G=/mnt/c/ml_artifacts/arxiv_chain_2026-10-27_d2/arxiv_1803_02155.gguf
RUN=$HOME/arxiv_chain_runs/d2_shaw
MTS=$HOME/d2_build/mtstudio
W=$HOME/d2e/chain
mkdir -p "$O" "$W/www"
rm -rf "$W/sft" && cp -a "$RUN/sft" "$W/sft"
python3 - "$RUN/sft_spec.json" "$W/sft_spec.json" "$W/sft" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); s["out_dir"] = sys.argv[3]; json.dump(s, open(sys.argv[2], "w"), indent=1)
EOF
python3 - "$W/prompts.jsonl" "$RUN/chat.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open("/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl") if l.strip()][:40]
rows += [{"prompt": json.loads(l)["user"]} for l in open(sys.argv[2]) if l.strip()]
open(sys.argv[1], "w").write("".join(json.dumps({"prompt": r["prompt"]}) + "\n" for r in rows))
print(len(rows), "prompts")
EOF
sha256sum $G | tee "$O/chain_gguf.sha256"
$MTS chat "$W/sft_spec.json" --port 8126 > "$W/mtstudio.log" 2>&1 &
MP=$!
(cd "$W" && exec $B/tinyllama_server $G 8125 127.0.0.1 "$W/www" 0) > "$W/ember.log" 2>&1 &
EP=$!
trap 'kill $MP $EP 2> /dev/null' EXIT
for i in $(seq 1 120); do
  curl -s -o /dev/null http://127.0.0.1:8126/ && curl -s -o /dev/null http://127.0.0.1:8125/ && break
  sleep 1
done
head -3 "$W/mtstudio.log"
cd $R/ember.cpp
PARITY_GGUF=$G PARITY_PROMPTS=$W/prompts.jsonl PARITY_MAX_NEW=24 PARITY_EMBER_PORT=8125 PARITY_MTS_PORT=8126 \
  python3 tools/chat_parity.py 42 "$O/chain_parity.jsonl" | tee "$O/chain_parity.txt"
python3 - "$O/chain_parity.jsonl" "$RUN/mtstudio_greedy.jsonl" <<'EOF' | tee -a "$O/chain_parity.txt"
import json, sys
a = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
b = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
same = sum(x["mtstudio"] == y["mtstudio"] and x["prompt"] == y["prompt"] for x, y in zip(a, b))
print(f"mtstudio here = the chain's recorded mtstudio_greedy.jsonl: {same}/{len(b)}; ember = recorded: "
      f"{sum(x['ember'] == y['mtstudio'] for x, y in zip(a, b))}/{len(b)}")
EOF
