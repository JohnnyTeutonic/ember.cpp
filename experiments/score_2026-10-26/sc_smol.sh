#!/bin/bash
# SmolLM2-135M (F32 GGUF, llama.cpp b6000's converter; the coordinator's addition, 26 Oct 2026) in the /score CPU gate
# (WSL; budget, 2 cores, session ember): ember default on the gate set and the 256-request batch, EMBER_SCORE_FULL=1 on
# the 60-request subset, HF transformers float32 from the same GGUF, and bench_eval.py on a slice. Resumes per leg.
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
B=~/ember_score/build_cpu; W=~/ember_score/work
G=/mnt/c/ml_artifacts/thirdparty/smollm2-135m/SmolLM2-135M-f32.gguf
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: $1" --cores 2 \
  --resume "bash $S/sc_smol.sh (legs resume)" -- "${@:2}"; }
sha256sum $G > $W/out/smol.sha256; cat $W/out/smol.sha256
[ -s $S/req/smol_sub60.jsonl ] || { head -48 $S/req/smol_requests.jsonl; tail -12 $S/req/smol_requests.jsonl; } > $S/req/smol_sub60.jsonl
[ -s $W/out/smol_def.stats.json ] || budget "ember /score smol_def" python3 $S/sc_client.py serve $B/tinyllama_server $G 8491 \
  $W/out smol_def $S/req/smol_requests.jsonl $S/req/smol_bench256.jsonl --errors --chat 2>&1 | grep -v "^\[" | tail -20
[ -s $W/out/smol_full.stats.json ] || budget "ember /score smol_full" env EMBER_SCORE_FULL=1 python3 $S/sc_client.py serve \
  $B/tinyllama_server $G 8492 $W/out smol_full $S/req/smol_sub60.jsonl 2>&1 | grep -v "^\[" | tail -5
grep -h "interpreter\|Interpreter\|tuned\|Chat template\|No chat template\|tokenizer" $W/out/smol_def.cwd/debugging.log 2>/dev/null \
  | grep -v "^\s*$" | head -12 | cut -c1-220 > $W/out/smol_path.txt; cat $W/out/smol_path.txt
if [ ! -s $W/out/hf_smol.done ]; then
  budget "HF reference smol" python3 $S/sc_hf_gguf.py $(dirname $G) $(basename $G) $S/req/smol_requests.jsonl $W/out/hf_smol.jsonl \
    > $W/out/hf_smol.log 2>&1 && touch $W/out/hf_smol.done
  tail -1 $W/out/hf_smol.log
fi
budget "bench_eval slice smol" bash $S/sc_smol_bench.sh $B/tinyllama_server $G 8493 $W/smolbench 2>&1 | tail -30
echo "smol done"
