#!/bin/bash
# The /score CPU gate (run in WSL; every job under the CPU budget at 2 cores, session ember). Legs are skipped
# when their output exists, so a rerun resumes.
#   bash sc_gate_cpu.sh [leg ...]     legs: mts emb hf errs (default: all)
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
A=/mnt/c/ml_artifacts
B=~/ember_score/build_cpu
W=~/ember_score/work
MTS=~/tpops_build/mtstudio
mkdir -p $W/out $W/specs
python3 $S/sc_specs.py $W/specs > /dev/null
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: $1" --cores 2 \
  --resume "bash $S/sc_gate_cpu.sh (legs resume)" -- "${@:2}"; }
want() { [ -z "$LEGS" ] && return 0; for g in $LEGS; do [ "$g" = "$1" ] && return 0; done; return 1; }
LEGS="$*"
declare -A GG=([pick]=$A/tiny_chat/e1/pick_e1_L16/model.gguf [pkbase]=$A/next_base/base/pk_base_v1.gguf
               [pythia]=$A/thirdparty/pythia-160m/pythia-160m-f32.gguf
               [qwen]=$A/thirdparty/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf)
declare -A SPEC=([pick]=$W/specs/pick_e1_L16.json [pkbase]=$W/specs/pk_base_v1.json)
declare -A REQ=([pick]=pk [pkbase]=pk [pythia]=pythia [qwen]=qwen)
if want mts; then
  if [ ! -s $W/pick_w/pick_e1_L16.safetensors ]; then
    mkdir -p $W/pick_w && python3 $S/sc_strip.py $A/tiny_chat/e1/pick_e1_L16/model.safetensors $W/pick_w/pick_e1_L16.safetensors
  fi
  for m in pick pkbase; do
    [ -s $W/out/mts_$m.jsonl ] && { echo "mts $m: done"; continue; }
    t0=$(date +%s%N)
    budget "mtstudio score $m" $MTS score ${SPEC[$m]} --in $S/req/pk_requests.jsonl --out $W/out/mts_$m.jsonl.part \
      > $W/out/mts_$m.log 2>&1 && mv $W/out/mts_$m.jsonl.part $W/out/mts_$m.jsonl
    echo "mts $m rc=$? $(tail -1 $W/out/mts_$m.log) $(( ($(date +%s%N) - t0) / 1000000 )) ms"
    if [ ! -s $W/out/mts_${m}_bench256.jsonl ]; then
      t0=$(date +%s%N)
      budget "mtstudio score $m bench256" $MTS score ${SPEC[$m]} --in $S/req/pk_bench256.jsonl \
        --out $W/out/mts_${m}_bench256.jsonl > $W/out/mts_${m}_bench256.log 2>&1
      echo "mts $m bench256 $(( ($(date +%s%N) - t0) / 1000000 )) ms" | tee $W/out/mts_${m}_bench256.time
    fi
  done
fi
if want emb; then
  port=8471
  for m in pick pkbase pythia qwen; do
    r=${REQ[$m]}
    extra=""; [ -f $S/req/${r}_bench256.jsonl ] && extra=$S/req/${r}_bench256.jsonl
    for mode in def full; do
      port=$((port + 1))
      [ -s $W/out/${m}_$mode.stats.json ] && { echo "emb $m $mode: done"; continue; }
      if [ $mode = def ]; then
        budget "ember /score $m default" python3 $S/sc_client.py serve $B/tinyllama_server ${GG[$m]} $port $W/out \
          ${m}_$mode $S/req/${r}_requests.jsonl $extra --chunk 50 --errors --chat 2>&1 | grep -v "^\[" | tail -20
      else
        budget "ember /score $m full" env EMBER_SCORE_FULL=1 python3 $S/sc_client.py serve $B/tinyllama_server ${GG[$m]} \
          $port $W/out ${m}_$mode $S/req/${r}_requests.jsonl $extra 2>&1 | grep -v "^\[" | tail -5
      fi
    done
  done
fi
if want hf; then
  for m in pythia qwen; do
    [ -s $W/out/hf_$m.done ] && { echo "hf $m: done"; continue; }
    d=$(dirname ${GG[$m]}); g=$(basename ${GG[$m]})
    budget "HF reference $m" python3 $S/sc_hf.py $d $g $S/req/${m}_requests.jsonl $W/out/hf_$m.jsonl \
      > $W/out/hf_$m.log 2>&1 && touch $W/out/hf_$m.done
    echo "hf $m: $(tail -1 $W/out/hf_$m.log)"
  done
fi
if want errs; then
  [ -s $W/out/mts_errors.json ] || budget "mtstudio score refusals" python3 $S/sc_mts_errors.py $MTS ${SPEC[pick]} 16384 \
    $W/mts_err $W/out/mts_errors.json > $W/out/mts_errors.log 2>&1
  tail -14 $W/out/mts_errors.log
fi
echo "cpu gate legs done: $LEGS"
