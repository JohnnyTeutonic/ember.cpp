#!/bin/bash
# The revision 1.3 gates of a pinned build (build_pinned.sh of ../ir_rev12_2026-10-26, with the gates' targets):
# keramos's suite at v1.3.0 (git archive), the rule citations, the adapter self-test on the relabelled gate models
# against the committed revision 1.1 receipt, and agreement.py at v1.3.0's tools. The CPU regression gates run
# through ../cuda_state_2026-10-25/cpu_regress.sh, the CUDA compile check through ../ir_rev11_2026-10-25/.
# Usage (WSL, under scripts/cpu_budget.py with 2 cores): bash gates.sh BUILD_DIR OUT_DIR REF_SELFTEST.json
set -e
B=${1:?build dir}; O=${2:?out dir}; REF=${3:?the committed adapter_selftest_cpu.json, as git show HEAD gives it}
E=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$O"
bash $E/ir_rev12_2026-10-26/run_suite.sh v1.3.0 $B/ir_adapter $O/suite_v1.3.0_cpu.json
KERAMOS=~/keramos_v1.3.0 python3 $E/ir_rev12_2026-10-26/rule_citations.py $B $O/rule_citations_cpu.json | tail -1
ls ~/ember_wt/adapter_models > /dev/null || bash $E/ir_adapter_2026-10-25/copy_models.sh
bash $E/ir_rev13_2026-10-27/relabel_models.sh > /dev/null
K=~/keramos_v1.3.0/conformance
IR_ADAPTER_BIN=$B/ir_adapter IR_SUITE=$K IR12_SUITE=$K IR13_SUITE=$K IR_ADAPTER_MODELS=~/ember_wt/adapter_models_r13 \
  ADAPTER_SCRATCH=$O/scratch ADAPTER_SELFTEST_OUT=$O/adapter_selftest_cpu.json \
  python3 $E/ir_adapter_2026-10-25/adapter_selftest.py | tail -1
python3 $E/ir_rev13_2026-10-27/compare_selftest.py "$REF" $O/adapter_selftest_cpu.json > $O/selftest_vs_rev11_receipt.txt
head -3 $O/selftest_vs_rev11_receipt.txt
(cd $E/ir_spec_2026-10-25 && KERAMOS_TOOLS=~/keramos_v1.3.0/tools python3 agreement.py --build $B > $O/agreement.log 2>&1; \
  echo "agreement rc=$?"; tail -1 $O/agreement.log)
