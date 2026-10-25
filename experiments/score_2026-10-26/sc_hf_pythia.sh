#!/bin/bash
# HF reference for Pythia-160m from the hub checkpoint at the pinned revision (transformers reads no gptneox GGUF;
# the F32 GGUF ember runs was converted from this checkpoint, sha256 97522db7..., as ../determinism_2026-10-26/vm.sh
# reproduces). Budget, 2 cores.
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
O=~/ember_score/work/out
[ -s $O/hf_pythia.done ] && exit 0
rm -f $O/hf_pythia.jsonl
python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score gate: HF reference pythia (hub)" --cores 2 \
  --resume "bash $S/sc_hf_pythia.sh" -- python3 $S/sc_hf_gguf.py /mnt/c/ml_artifacts/thirdparty/pythia-160m \
  hub:EleutherAI/pythia-160m@50f5173d932e8e61f858120bcb800b97af589f46 $S/req/pythia_requests.jsonl $O/hf_pythia.jsonl \
  > $O/hf_pythia.log 2>&1 && touch $O/hf_pythia.done
tail -2 $O/hf_pythia.log
