#!/bin/bash
# The GPT-2 builder gate on ext4 (~/gpt2_gate, set up by prep.sh and build.sh), every heavy step through
# the CPU budget (2 cores). Steps: check (tools/thirdparty_ember_check.py), f64 (its sequences against
# float64), greedy (40 raw prompts), f64g (the greedy sequences whose decode steps left HF by > 1e-5,
# against float64).
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file> [check|f64|greedy|f64g|all]"
set -u
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$ROOT/ember.cpp
X=$E/experiments/gpt2_2026-10-25
W=~/gpt2_gate
M=$W/model/gpt2-124m-f32.gguf
STEP=${1:-all}
B="python3 $ROOT/scripts/cpu_budget.py run --session ember --cores 2"
cd $W
cp $X/greedy.py $X/f64.py $W/
if [ ! -x $W/bin/ir_logits ]; then
  mkdir -p $W/bin
  cp $W/build/ir_logits $W/build/ir_adapter $W/bin/
  sha256sum $W/bin/* > $W/bin/SHA256
fi
cat $W/bin/SHA256
export PYTHONDONTWRITEBYTECODE=1
on() { [ $STEP = $1 ] || [ $STEP = all ]; }
if on check; then
  $B --job "ember GPT-2 builder gate: thirdparty_ember_check" --resume "rerun $X/run.sh check" -- \
    bash -c "python3 $E/tools/thirdparty_ember_check.py --gguf $M --paperkiln $W/pk --bin $W/bin/ir_logits --work $W/check --n 4 --len 32 > $W/check.log 2>&1; echo check rc=\$?"
  grep -vE '^\{"i"' $W/check.log | tail -30
fi
if on f64; then
  $B --job "ember GPT-2 builder gate: float64 reference" --resume "rerun $X/run.sh f64" -- \
    bash -c "python3 $W/f64.py --gguf $M --bin $W/bin/ir_logits --paperkiln $W/pk --out $W/f64.json --n 4 --len 32 > $W/f64.log 2>&1; echo f64 rc=\$?"
  tail -30 $W/f64.log
fi
if on greedy; then
  $B --job "ember GPT-2 builder gate: greedy 40 raw prompts" --resume "rerun $X/run.sh greedy (resumes after the last row)" -- \
    bash -c "python3 $W/greedy.py --model $W/model --adapter $W/bin/ir_adapter --receipt $W/gate_n40.jsonl --prompts $W/pk/corpus/eval/chat_prompts_v1.jsonl --out $W/greedy.jsonl > $W/greedy.log 2>&1; echo greedy rc=\$?"
  tail -25 $W/greedy.log
fi
if on f64g; then
  $B --job "ember GPT-2 builder gate: float64 on the greedy sequences" --resume "rerun $X/run.sh f64g" -- \
    bash -c "python3 $W/f64.py --gguf $M --bin $W/bin/ir_logits --paperkiln $W/pk --out $W/f64_greedy.json --greedy $W/greedy.jsonl > $W/f64_greedy.log 2>&1; echo f64g rc=\$?"
  tail -30 $W/f64_greedy.log
fi
