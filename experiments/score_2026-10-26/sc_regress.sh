#!/bin/bash
# CPU regression of this work (WSL): ../cuda_state_2026-10-25/cpu_regress.sh with this work's build against the
# committed receipts (REF: git show HEAD copies), then the two gates whose committed receipts are not reproducible
# from the committed builds (as in ../determinism_2026-10-26): state_tokens_gate rerun with ember HEAD's own build
# (this work's receipt must equal HEAD's byte for byte), and regress_bits with HEAD's ir_logits as the old side.
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/score
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp/experiments
H=$E/cuda_state_2026-10-25
B=~/ember_score/build_cpu; BH=~/ember_score/build_head; O=~/ember_score/regress
mkdir -p $O
bash $H/cpu_regress.sh $B $O/new $S/ref 2>&1 | grep -v "^\[" | tee $O/new.txt
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "ember /score regression: $1" --cores 2 \
  --resume "bash $S/sc_regress.sh" -- "${@:2}"; }
budget "state_tokens_gate with HEAD" python3 $H/rungate.py $BH $O/head $E/state_tokens_2026-10-24/state_tokens_gate.py \
  > $O/head_state_tokens.out 2>&1
python3 -c "import sys; s = open(sys.argv[1]).read().replace(sys.argv[2], '/home/jonat/ember_wt/build_cpu'); open(sys.argv[1] + '.norm', 'w').write(s)" \
  $O/head/state_tokens_gate_cpu.json $BH
cmp -s $O/new/state_tokens_gate_cpu.json.norm $O/head/state_tokens_gate_cpu.json.norm && echo "state_tokens_gate: this work = HEAD byte for byte" \
  || echo "state_tokens_gate: this work DIFFERS from HEAD"
# regress_bits: its old build (~/ember_69d7ca2/build) is gone; HEAD's build linked there for the run.
if [ ! -e ~/ember_69d7ca2 ]; then
  mkdir -p ~/ember_69d7ca2 && ln -s $BH ~/ember_69d7ca2/build && LINKED=1
  budget "regress_bits vs HEAD" python3 $H/rungate.py $B $O/bits $E/chat_template_2026-10-24/regress.py bits > $O/bits.out 2>&1
  echo "regress_bits vs HEAD rc=$?: $(tail -2 $O/bits.out | tr '\n' ' ')"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: v for k, v in d.items() if isinstance(v, (bool, dict))})" $O/bits/regress_bits.json 2>&1 | cut -c1-900
  rm ~/ember_69d7ca2/build && rmdir ~/ember_69d7ca2
else
  echo "~/ember_69d7ca2 exists: regress_bits not rerun against HEAD"
fi
echo "regression done"
