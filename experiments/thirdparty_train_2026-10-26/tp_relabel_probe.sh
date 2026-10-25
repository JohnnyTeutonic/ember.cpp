#!/bin/bash
# The relabelled-arch probe: llama-arch copies of cases 2-5 (tp_relabel.py) through ir_path_check, then the served
# path's logits (tp_session_logits, gpu_layers 0) bit-compared with the original file's (~/tptrain/work). Seconds.
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/thirdparty_train_2026-10-26
D=/mnt/c/ml_artifacts/thirdparty_train_gate
W=~/tptrain/relabel
B=~/tptrain/build_cpu
rm -rf $W; mkdir -p $W; cd $W
for n in tp_qwen tp_mqa_partial tp_neox_partial tp_mamba1; do
  python3 $S/tp_relabel.py $D/$n/$n.gguf $W/${n}_as_llama.gguf > /dev/null || { echo "relabel $n failed"; continue; }
  $B/ir_path_check $W/${n}_as_llama.gguf
  python3 $S/tp_inputs.py $D/$n/score_in.jsonl $W/in.json > /dev/null
  $B/tp_session_logits $W/${n}_as_llama.gguf 0 < $W/in.json > $W/${n}_as_llama.json 2> $W/err.txt || { echo "session $n: $(tail -2 $W/err.txt)"; continue; }
  python3 $S/tp_relabel_cmp.py $W/${n}_as_llama.json ~/tptrain/work/cpu_session_$n.json $n
done
python3 $S/tp_inspect_relabel.py $W
