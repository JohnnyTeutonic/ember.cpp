#!/bin/bash
# The spot-check on ember's CPU path: carry5_eval's ember leg on the exact SFT GGUF (gpu_layers 0), 40 items,
# then compare_gens against the VM's mtstudio and ember (CUDA) records.
set -u
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/rmsnorm_2026-10-24
W=~/ember_wt/shape_parity
H=/mnt/c/ml_artifacts/tiny_chat/e1/post_e1_L32_r1
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
mkdir -p $W/sft
[ -f $W/sft/tc_sft_L32_r1.gguf ] || cp $H/sft/tc_sft_L32_r1.gguf $W/sft/
sha256sum $W/sft/tc_sft_L32_r1.gguf
python3 - <<PY
import json
s = json.load(open("$H/sft_spec.json"))
s["out_dir"] = "$W/sft"
s["data"]["vocab"] = "/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
json.dump(s, open("$W/sft/sft_spec.json", "w"), indent=1)
PY
cd $PK
python3 tools/carry5_eval.py run --model m=$W/sft/sft_spec.json --out $W/spot_cpu --scorer none --gen ember \
  --ember-bin ~/ember_wt/build_cpu/tinyllama_server --gpu-layers 0 --set corpus/eval/carry5_v1.jsonl \
  --every 26 --limit 40 --port 8392 > $W/spot_cpu.log 2>&1
echo "carry5 rc=$?"
echo "ember CPU vs mtstudio (VM):"; python3 $S/cmp_spot.py $W/spot_cpu/m/gen.jsonl $H/spot/mtstudio/m/gen.jsonl corpus/eval/carry5_v1.jsonl $W/spot_cpu_vs_mtstudio.json
echo "ember CPU vs ember CUDA (VM):"; python3 $S/cmp_spot.py $W/spot_cpu/m/gen.jsonl $H/spot/ember_default/m/gen.jsonl corpus/eval/carry5_v1.jsonl
echo "VM ember CUDA vs mtstudio:"; python3 $S/cmp_spot.py $H/spot/ember_default/m/gen.jsonl $H/spot/mtstudio/m/gen.jsonl corpus/eval/carry5_v1.jsonl
