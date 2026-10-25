#!/bin/bash
# The commit's tree (ember 17b3999, the WebGPU work, plus this work): pick_e1_L16's gate set byte for byte against the
# gate's run, and the 60-request subsets of Pythia (interpreter) and SmolLM2 (tied head) value for value.
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
W=~/ember_score/work; O=$W/out; A=/mnt/c/ml_artifacts
run() { python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: commit tree $1" --cores 2 \
  --resume "bash $S/sc_final2.sh" -- python3 $S/sc_client.py serve ~/ember_score/build_cpu/tinyllama_server $2 $3 $O $1 $4 2>&1 | grep -v "^\[" | tail -1; }
run pick_commit $A/tiny_chat/e1/pick_e1_L16/model.gguf 8496 $S/req/pk_requests.jsonl
cmp $O/pick_commit.pk_requests.jsonl $O/pick_def.pk_requests.jsonl && echo "pick: commit tree byte-identical to the gate run"
run pythia_commit $A/thirdparty/pythia-160m/pythia-160m-f32.gguf 8497 $S/req/pythia_sub60.jsonl
bash $S/sc_peek.sh $O/pythia_commit.pythia_sub60.jsonl $O/pythia_def.pythia_requests.jsonl
run smol_commit $A/thirdparty/smollm2-135m/SmolLM2-135M-f32.gguf 8498 $S/req/smol_sub60.jsonl
bash $S/sc_peek.sh $O/smol_commit.smol_sub60.jsonl $O/smol_def.smol_requests.jsonl
sha256sum ~/ember_score/build_cpu/tinyllama_server
