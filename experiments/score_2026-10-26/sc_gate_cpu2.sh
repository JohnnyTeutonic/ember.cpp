#!/bin/bash
# The /score CPU gate, second part (run in WSL; every job under the CPU budget at 2 cores, session ember): the
# legs after pick_e1_L16's ember runs (sc_gate_cpu.sh did those and pk_base_v1's mtstudio run). Trimmed for time
# (WSL's 4 CPUs are shared): pk_base_v1 without the calls-of-50 rerun and with its full-reference run on the gate
# set only; Qwen without a 256-request batch. Legs are skipped when their output exists.
#   bash sc_gate_cpu2.sh
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
A=/mnt/c/ml_artifacts
B=~/ember_score/build_cpu
W=~/ember_score/work
MTS=~/tpops_build/mtstudio
mkdir -p $W/out $W/specs
python3 $S/sc_specs.py $W/specs > /dev/null
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: $1" --cores 2 \
  --resume "bash $S/sc_gate_cpu2.sh (legs resume)" -- "${@:2}"; }
emb() {  # tag gguf port [env...] -- client args
  local tag=$1 g=$2 port=$3; shift 3
  [ -s $W/out/$tag.stats.json ] && { echo "$tag: done"; return; }
  budget "ember /score $tag" env "$@" python3 $S/sc_client.py serve $B/tinyllama_server $g $port $W/out $tag \
    ${ARGS[@]} 2>&1 | grep -v "^\[" | tail -20
}
P=$A/next_base/base/pk_base_v1.gguf
Y=$A/thirdparty/pythia-160m/pythia-160m-f32.gguf
Q=$A/thirdparty/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf
ARGS=($S/req/pk_requests.jsonl $S/req/pk_bench256.jsonl --errors --chat); emb pkbase_def $P 8481 X=1
ARGS=($S/req/pk_requests.jsonl); emb pkbase_full $P 8482 EMBER_SCORE_FULL=1
ARGS=($S/req/pythia_requests.jsonl $S/req/pythia_bench256.jsonl --chunk 50 --errors --chat); emb pythia_def $Y 8483 X=1
ARGS=($S/req/pythia_requests.jsonl $S/req/pythia_bench256.jsonl); emb pythia_full $Y 8484 EMBER_SCORE_FULL=1
ARGS=($S/req/qwen_requests.jsonl --chunk 50 --errors --chat); emb qwen_def $Q 8485 X=1
ARGS=($S/req/qwen_requests.jsonl); emb qwen_full $Q 8486 EMBER_SCORE_FULL=1
for m in pythia qwen; do
  [ -s $W/out/hf_$m.done ] && { echo "hf $m: done"; continue; }
  g=$([ $m = pythia ] && echo $Y || echo $Q)
  budget "HF reference $m" python3 $S/sc_hf.py $(dirname $g) $(basename $g) $S/req/${m}_requests.jsonl $W/out/hf_$m.jsonl \
    > $W/out/hf_$m.log 2>&1 && touch $W/out/hf_$m.done
  echo "hf $m: $(tail -1 $W/out/hf_$m.log)"
done
if [ ! -s $W/out/mts_pick.jsonl ]; then
  t0=$(date +%s%N)
  budget "mtstudio score pick" $MTS score $W/specs/pick_e1_L16.json --in $S/req/pk_requests.jsonl \
    --out $W/out/mts_pick.jsonl.part > $W/out/mts_pick.log 2>&1 && mv $W/out/mts_pick.jsonl.part $W/out/mts_pick.jsonl
  echo "mts pick rc=$? $(tail -1 $W/out/mts_pick.log) $(( ($(date +%s%N) - t0) / 1000000 )) ms" | tee $W/out/mts_pick.time
fi
[ -s $W/out/mts_errors.json ] || budget "mtstudio score refusals" python3 $S/sc_mts_errors.py $MTS $W/specs/pick_e1_L16.json \
  16384 $W/mts_err $W/out/mts_errors.json > $W/out/mts_errors.log 2>&1
tail -14 $W/out/mts_errors.log
echo "cpu gate part 2 done"
