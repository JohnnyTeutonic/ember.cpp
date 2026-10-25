#!/bin/bash
# CPU regression of this work (run in WSL): the existing CPU gate scripts rerun unchanged against this work's
# CPU build through rungate.py (receipts written to OUT, not over the committed ones), each receipt then
# compared with the committed file byte for byte; then the IR conformance adapter's self-test. Every job
# under the CPU budget at 2 cores (session ember).
#   bash cpu_regress.sh <build dir> <out dir> <reference dir> [gate ...]   (default: all)
# <reference dir> holds the committed receipts as `git show HEAD:<path>` gives them (the working tree's copies
# may be rewritten by another session's in-place reruns meanwhile).
B=${1:?build}; O=${2:?out}; REF=${3:?reference receipts}; shift 3
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp/experiments
H=$E/cuda_state_2026-10-25
mkdir -p "$O" && cd "$O"
budget() { python3 $R/scripts/cpu_budget.py run --session ember --job "ember CUDA items CPU regression: $1" --cores 2 \
  --resume "rerun ember.cpp/experiments/cuda_state_2026-10-25/cpu_regress.sh $B $O $REF $1" -- "${@:2}"; }
gate() {  # name script receipt [args]
  local name=$1 script=$2 receipt=$3; shift 3
  budget "$name" python3 $H/rungate.py "$B" "$O" "$script" "$@" > "$O/$name.out" 2>&1
  local rc=$?
  # Receipts that name the binary ("bin", "new") name this build: that path is put back before comparing.
  python3 -c "import sys; s = open(sys.argv[1]).read().replace(sys.argv[2], '/home/jonat/ember_wt/build_cpu'); open(sys.argv[1] + '.norm', 'w').write(s)" \
    "$O/$(basename $receipt)" "$B" 2> /dev/null
  if cmp -s "$O/$(basename $receipt).norm" "$REF/$(basename $receipt)"; then echo "$name rc=$rc: $(basename $receipt) byte-identical"
  else echo "$name rc=$rc: $(basename $receipt) DIFFERS or missing"; fi
}
want() { [ $# -eq 0 ] && return 0; for g in $WANT; do [ "$g" = "$1" ] && return 0; done; return 1; }
WANT="$*"
run() { if [ -z "$WANT" ] || want "$1"; then gate "$@"; fi; }
run recursion_gate $E/recursion_2026-10-23/recursion_gate.py $E/recursion_2026-10-23/recursion_gate_cpu.json
run mor_gate $E/recursion_2026-10-23/mor_gate.py $E/recursion_2026-10-23/mor_gate_cpu.json
run mor_refusals $E/recursion_2026-10-23/mor_refusals.py $E/recursion_2026-10-23/mor_refusals.json
run attnres_gate $E/attnres_2026-10-24/attnres_gate.py $E/attnres_2026-10-24/attnres_gate_cpu.json
run state_tokens_gate $E/state_tokens_2026-10-24/state_tokens_gate.py $E/state_tokens_2026-10-24/state_tokens_gate_cpu.json
run state4_gate $E/state_tokens_2026-10-24/state4_gate.py $E/state_tokens_2026-10-24/state4_gate_cpu.json
run regress_bits $E/chat_template_2026-10-24/regress.py $E/chat_template_2026-10-24/regress_bits.json bits
RUNGATE_OUTPUTS=regress_parity.json,regress_parity.json.tmp run regress_parity $E/chat_template_2026-10-24/regress.py $E/chat_template_2026-10-24/regress_parity.json parity
run g5_thirdparty $E/chat_template_2026-10-24/g5_thirdparty.py $E/chat_template_2026-10-24/g5_thirdparty.json
if [ -z "$WANT" ] || want adapter; then
  IR_ADAPTER_BIN=$B/ir_adapter budget "adapter self-test" python3 $E/ir_adapter_2026-10-25/adapter_selftest.py > "$O/adapter.out" 2>&1
  echo "adapter self-test rc=$?: $(tail -2 $O/adapter.out | tr '\n' ' ')"
fi
echo "cpu regression done"
