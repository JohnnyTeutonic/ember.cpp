#!/bin/bash
# CPU legs of the X5 gate (WSL, under the CPU budget at 2 cores), on ~/x5g's builds:
#   the nine-digit logits dump into old and oldce (x5_dump9.py; tinyllama_server rebuilt); x5_device_logits
#   compiled against each tree (compile check; it runs on the VM); x5_gate_cpu.py (new gated, oldce and old
#   reported); plain llama (the L16 control's SFT GGUF) through new's and old's tinyllama_server at gpu_layers 0.
#   bash x5_cpu_legs.sh OUT
set -u
OUT=${1:?out}
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/attnres_x5_2026-10-25
W=~/x5g
A=/mnt/c/ml_artifacts
mkdir -p $OUT
for t in old oldce; do
  python3 $S/x5_dump9.py $W/$t/src && make -C $W/$t/build_cpu -j2 tinyllama_server > $W/$t/build_cpu/build_dump9.log 2>&1
  echo "$t dump9 rebuild rc=$?"
done
JSON_INC=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
for t in new oldce; do
  g++ -O2 -std=c++17 -I$W/$t/src -I$JSON_INC $S/x5_device_logits.cpp $W/$t/build_cpu/libtinyllama_core.a -fopenmp -lpthread -o $W/$t/build_cpu/x5_device_logits_cpu
  echo "x5_device_logits compiles against $t: rc=$?"
done
python3 $S/x5_gate_cpu.py $OUT/x5_gate_cpu.json new=$W/new/build_cpu oldce=$W/oldce/build_cpu old=$W/old/build_cpu > $OUT/x5_gate_cpu.txt 2>&1
echo "x5_gate_cpu rc=$?"
P=$W/inputs_plain; mkdir -p $P
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > $P/prompts.jsonl
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/eval/carry5_v1.jsonl $P/
G=$A/tiny_chat/e1/pick_e1_L16/model.gguf
python3 $S/x5_plain_collect.py $W/new/build_cpu/tinyllama_server $G 0 8421 $P/prompts.jsonl $P/carry5_v1.jsonl $W/plain/new_cpu; echo "collect new cpu rc=$?"
python3 $S/x5_plain_collect.py $W/old/build_cpu/tinyllama_server $G 0 8422 $P/prompts.jsonl $P/carry5_v1.jsonl $W/plain/old_cpu; echo "collect old cpu rc=$?"
python3 $S/x5_plain_compare.py $W/plain/new_cpu $W/plain/old_cpu > $OUT/plain_cpu_new_vs_old.json; echo "plain cpu compare rc=$?"
head -c 1500 $OUT/plain_cpu_new_vs_old.json
