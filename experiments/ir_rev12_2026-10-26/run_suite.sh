#!/bin/bash
# keramos's conformance suite at a tag (tools/ir_conformance.py from a git archive of the tag in ~/keramos_<tag>/,
# so keramos's checkout and branch are not touched) against an ember adapter binary, with no EMBER_* variable
# passed through (section 11.3). Writes OUT (the runner's --json) and OUT.log.
# Usage (WSL, under scripts/cpu_budget.py, 1 core): bash run_suite.sh TAG ADAPTER OUT [--only SUBSTRING]
set -e
TAG=${1:?tag}
BIN=${2:?adapter}
OUT=${3:?out json}
shift 3
K=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos
D=~/keramos_$TAG
C=$(git -C "$K" rev-parse "$TAG^{commit}")
if [ ! -f "$D/COMMIT" ] || [ "$(cat "$D/COMMIT")" != "$C" ]; then
  rm -rf "$D" && mkdir -p "$D" && git -C "$K" archive "$C" | tar -x -C "$D" && echo "$C" > "$D/COMMIT"
fi
cd "$D"
echo "keramos $TAG $C"
python3 tools/ir_conformance.py --adapter "cmd:$BIN --conformance" --json "$OUT" "$@" > "$OUT.log" 2>&1 && rc=0 || rc=$?
grep -E "^(FAIL|XPASS)" "$OUT.log" | head -40 || true
tail -1 "$OUT.log"
echo "suite rc=$rc"
