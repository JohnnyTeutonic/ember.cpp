#!/bin/bash
# CPU regression of the pinned build (WSL): ../cuda_state_2026-10-25/cpu_regress.sh (the CPU gate scripts rerun
# unchanged through rungate.py, each receipt compared byte for byte with the committed one as `git show HEAD`
# gives it, each gate under the CPU budget at 2 cores), then the adapter self-test compared check by check with its
# committed receipt (../ir_adapter_2026-10-25/adapter_selftest_cpu.json).
#   bash regress.sh BUILD_DIR OUT_DIR
set -u
B=${1:?build}; O=${2:?out}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp/experiments
mkdir -p "$O/ref"
for p in recursion_2026-10-23/recursion_gate_cpu.json recursion_2026-10-23/mor_gate_cpu.json \
         recursion_2026-10-23/mor_refusals.json attnres_2026-10-24/attnres_gate_cpu.json \
         state_tokens_2026-10-24/state_tokens_gate_cpu.json state_tokens_2026-10-24/state4_gate_cpu.json \
         chat_template_2026-10-24/regress_bits.json chat_template_2026-10-24/regress_parity.json \
         chat_template_2026-10-24/g5_thirdparty.json ir_adapter_2026-10-25/adapter_selftest_cpu.json; do
  git -C $R/ember.cpp show HEAD:experiments/$p > "$O/ref/$(basename $p)"
done
bash $E/cuda_state_2026-10-25/cpu_regress.sh "$B" "$O" "$O/ref" recursion_gate mor_gate mor_refusals attnres_gate \
  state_tokens_gate state4_gate regress_bits regress_parity g5_thirdparty
cd ~
python3 $R/scripts/cpu_budget.py run --session ember --job "d2: adapter self-test" --cores 2 --resume "rerun regress.sh" -- \
  env IR_ADAPTER_BIN=$B/ir_adapter ADAPTER_SELFTEST_OUT=$O/adapter_selftest_cpu.json ADAPTER_SCRATCH=$O/adapter_scratch \
  python3 $E/ir_adapter_2026-10-25/adapter_selftest.py > $O/adapter_selftest.txt 2>&1; echo "adapter_selftest rc=$?"
python3 - "$O" <<'EOF'
import json, sys
O = sys.argv[1]
a = json.load(open(O + "/ref/adapter_selftest_cpu.json")); b = json.load(open(O + "/adapter_selftest_cpu.json"))
ca, cb = a["checks"], b["checks"]
print("adapter self-test committed:", sum(1 for v in ca.values() if v.get("pass")), "/", len(ca), "requests", a["requests"])
print("adapter self-test this build:", sum(1 for v in cb.values() if v.get("pass")), "/", len(cb), "requests", b["requests"],
      "failures", len(b["failures"]), "pass", b["pass"])
print("same check names:", sorted(ca) == sorted(cb))
same = [k for k in ca if k in cb and json.dumps(ca[k], sort_keys=True) == json.dumps(cb[k], sort_keys=True)]
print("checks with identical records:", len(same), "of", len(ca))
for k in ca:
    if k not in same:
        print("  differs:", k, json.dumps(ca[k])[:200], "->", json.dumps(cb.get(k))[:200])
EOF
