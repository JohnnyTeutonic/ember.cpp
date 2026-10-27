#!/bin/bash
# keramos's conformance suite at v1.4.0 (tools/ir_conformance.py from a git archive of the tag, c1a53197, on ext4 at
# ~/d2e/keramos_v1.4.0) against an ember adapter binary, with no EMBER_* variable passed through (section 11.3).
# Writes OUT (the runner's --json) and OUT.log.
# Usage (WSL, under scripts/cpu_budget.py, 2 cores): bash run_suite.sh ADAPTER OUT [--only SUBSTRING]
set -e
BIN=${1:?adapter}
OUT=${2:?out json}
shift 2
K=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos
D=${KERAMOS_TREE:-$HOME/d2e/keramos_v1.4.0}
C=$(git -C "$K" rev-parse "v1.4.0^{commit}")
if [ ! -f "$D/COMMIT" ] || [ "$(cat "$D/COMMIT")" != "$C" ]; then
  rm -rf "$D" && mkdir -p "$D" && git -C "$K" archive "$C" | tar -x -C "$D" && echo "$C" > "$D/COMMIT"
fi
cd "$D"
for v in $(env | grep -o '^EMBER_[A-Z0-9_]*'); do unset "$v"; done
echo "keramos v1.4.0 $C"
python3 tools/ir_conformance.py --adapter "cmd:$BIN --conformance" --json "$OUT" "$@" > "$OUT.log" 2>&1 && rc=0 || rc=$?
grep -E "^(FAIL|XPASS)" "$OUT.log" | head -40 || true
tail -1 "$OUT.log"
echo "suite rc=$rc"
