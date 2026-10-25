#!/bin/bash
# The /score CPU gate, third part (WSL; budget, 2 cores, session ember), after pk_base_v1's default run: ember's CPU
# prefill runs at ~50-130 positions/s per 2 cores here, so the EMBER_SCORE_FULL=1 reference runs on a 60-request
# subset of each gate set (its first 48 requests: whole multiple-choice groups, where prefixes are reused; and its
# last 12: the long prompts and, for paperkiln's vocabulary, the over_ctx edges), and the calls-of-50 rerun only on
# pick_e1_L16 (done). Legs are skipped when their output exists.
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
A=/mnt/c/ml_artifacts
B=~/ember_score/build_cpu
W=~/ember_score/work
MTS=~/tpops_build/mtstudio
for r in pk pythia qwen; do
  [ -s $S/req/${r}_sub60.jsonl ] || { head -48 $S/req/${r}_requests.jsonl; tail -12 $S/req/${r}_requests.jsonl; } > $S/req/${r}_sub60.jsonl
done
python3 $S/sc_specs.py $W/specs > /dev/null
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: $1" --cores 2 \
  --resume "bash $S/sc_gate_cpu3.sh (legs resume)" -- "${@:2}"; }
emb() {  # tag gguf port env -- client args in ARGS
  local tag=$1 g=$2 port=$3; shift 3
  [ -s $W/out/$tag.stats.json ] && { echo "$tag: done"; return; }
  budget "ember /score $tag" env "$@" python3 $S/sc_client.py serve $B/tinyllama_server $g $port $W/out $tag \
    ${ARGS[@]} 2>&1 | grep -v "^\[" | tail -20
}
P=$A/next_base/base/pk_base_v1.gguf
Y=$A/thirdparty/pythia-160m/pythia-160m-f32.gguf
Q=$A/thirdparty/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf
if [ ! -s $W/out/mts_pick.jsonl ]; then
  t0=$(date +%s%N)
  budget "mtstudio score pick" $MTS score $W/specs/pick_e1_L16.json --in $S/req/pk_requests.jsonl \
    --out $W/out/mts_pick.jsonl.part > $W/out/mts_pick.log 2>&1 && mv $W/out/mts_pick.jsonl.part $W/out/mts_pick.jsonl
  echo "mts pick rc=$? $(tail -1 $W/out/mts_pick.log) $(( ($(date +%s%N) - t0) / 1000000 )) ms" | tee $W/out/mts_pick.time
fi
[ -s $W/out/mts_errors.json ] || budget "mtstudio score refusals" python3 $S/sc_mts_errors.py $MTS $W/specs/pick_e1_L16.json \
  16384 $W/mts_err $W/out/mts_errors.json > $W/out/mts_errors.log 2>&1
tail -14 $W/out/mts_errors.log
ARGS=($S/req/pk_sub60.jsonl); emb pkbase_full $P 8482 EMBER_SCORE_FULL=1
ARGS=($S/req/pythia_requests.jsonl $S/req/pythia_bench256.jsonl --errors --chat); emb pythia_def $Y 8483 X=1
ARGS=($S/req/pythia_sub60.jsonl); emb pythia_full $Y 8484 EMBER_SCORE_FULL=1
for m in pythia qwen; do
  [ -s $W/out/hf_$m.done ] && { echo "hf $m: done"; continue; }
  g=$([ $m = pythia ] && echo $Y || echo $Q)
  budget "HF reference $m" python3 $S/sc_hf.py $(dirname $g) $(basename $g) $S/req/${m}_requests.jsonl $W/out/hf_$m.jsonl \
    > $W/out/hf_$m.log 2>&1 && touch $W/out/hf_$m.done
  echo "hf $m: $(tail -1 $W/out/hf_$m.log)"
done
ARGS=($S/req/qwen_requests.jsonl --errors --chat); emb qwen_def $Q 8485 X=1
ARGS=($S/req/qwen_sub60.jsonl); emb qwen_full $Q 8486 EMBER_SCORE_FULL=1
echo "cpu gate part 3 done"
