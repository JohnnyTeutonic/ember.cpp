#!/bin/bash
# CPU regression on the gated build (~/tig/ember/build_cpu, WSL; under the CPU budget, tig_job.sh):
#   (1) ../attnres_2026-10-24/attnres_gate.py (ar3) with its BIN pointed at this build and its receipt written to
#       OUT/attnres_gate_cpu.json, against the committed receipt (every field but "bin" equal);
#   (2) ../ir_adapter_2026-10-25/adapter_selftest.py with IR_ADAPTER_BIN = this build's ir_adapter (report to
#       OUT/adapter_selftest_cpu.json, scratch under ~/tig): check count and passes, against the committed receipt.
#   bash tig_regress.sh OUT
set -u
OUT=${1:?out}
X=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments
B=$HOME/tig/ember/build_cpu
W=$HOME/tig/regress
mkdir -p $OUT $W
sed -e "s|^BIN = .*|BIN = \"$B/ir_logits\"|" -e "s|^HERE = .*|HERE = \"$OUT\"|" \
    -e "s|~/ember_wt/attnres_gate|~/tig/regress/attnres_gate|" $X/attnres_2026-10-24/attnres_gate.py > $W/attnres_gate.py
python3 $W/attnres_gate.py > $W/attnres_gate.txt 2>&1; echo "ar3 attnres_gate rc=$?"
IR_ADAPTER_BIN=$B/ir_adapter ADAPTER_SELFTEST_OUT=$OUT/adapter_selftest_cpu.json ADAPTER_SCRATCH=$W/adapter_scratch \
  python3 $X/ir_adapter_2026-10-25/adapter_selftest.py > $W/adapter_selftest.txt 2>&1; echo "adapter_selftest rc=$?"
python3 - "$X" "$OUT" <<'EOF'
import json, sys
X, OUT = sys.argv[1], sys.argv[2]
a = json.load(open(X + "/attnres_2026-10-24/attnres_gate_cpu.json")); b = json.load(open(OUT + "/attnres_gate_cpu.json"))
a.pop("bin"); b.pop("bin")
print("ar3 receipt equal to the committed one (all fields but bin):", a == b)
a = json.load(open(X + "/ir_adapter_2026-10-25/adapter_selftest_cpu.json")); b = json.load(open(OUT + "/adapter_selftest_cpu.json"))
ca, cb = a["checks"], b["checks"]
print("adapter self-test committed:", sum(1 for v in ca.values() if v.get("pass")), "/", len(ca), "requests", a["requests"])
print("adapter self-test this build:", sum(1 for v in cb.values() if v.get("pass")), "/", len(cb), "requests", b["requests"],
      "failures", len(b["failures"]), "pass", b["pass"])
print("same check names:", sorted(ca) == sorted(cb))
same = [k for k in ca if k in cb and json.dumps(ca[k], sort_keys=True) == json.dumps(cb[k], sort_keys=True)]
print("checks with identical records:", len(same), "of", len(ca))
EOF
