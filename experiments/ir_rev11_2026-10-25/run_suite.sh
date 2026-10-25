#!/bin/bash
# keramos's conformance suite (tools/ir_conformance.py, the checkout at its tag) against an ember adapter binary,
# with no EMBER_* variable passed through (section 11.3). Writes OUT (the runner's --json) and OUT.log.
# Usage (WSL, under scripts/cpu_budget.py, 1 core): bash run_suite.sh ADAPTER OUT [--only SUBSTRING]
set -e
BIN=${1:?adapter}
OUT=${2:?out json}
shift 2
K=${KERAMOS:-/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos}
cd "$K"
echo "keramos $(git describe --tags --exact-match 2>/dev/null || echo untagged) $(git rev-parse HEAD)"
python3 tools/ir_conformance.py --adapter "cmd:$BIN --conformance" --json "$OUT" "$@" > "$OUT.log" 2>&1 && rc=0 || rc=$?
grep -E "^(FAIL|XPASS)" "$OUT.log" | head -40 || true
tail -1 "$OUT.log"
echo "suite rc=$rc"
