#!/usr/bin/env bash
# Local driver for one Colab session running vm.sh (path C of microtorch/AGENTS.md: new, upload, launch,
# poll, download, stop; the session is stopped on every exit path). Run from WSL as one foreground process.
#   bash drive.sh <session> <gpu> <max-minutes> <fix-commit> <paperkiln-commit>
set -uo pipefail
S=${1:?session}; GPU=${2:?gpu}; MAXMIN=${3:?max minutes}; FIX=${4:?fix}; PKC=${5:?paperkiln}
COL=~/.local/bin/colab
COLAB_PY=~/.local/share/uv/tools/google-colab-cli/bin/python
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
STAGE=~/sp_stage; OUT=/mnt/c/ml_artifacts/ember_rmsnorm/$S
log() { echo "$(date +%H:%M:%S) $*"; }
c() { local t=$1; shift; timeout "$t" $COL "$@"; }
[ -e "$OUT" ] && { log "$OUT exists: pick a new session name"; exit 2; }
for f in vm.sh fix.patch rmsnorm_batch_test.cu dumpprec.py sp_compare.py prompts.jsonl SHA256SUMS; do
  [ -s "$STAGE/$f" ] || { log "stage lacks $f"; exit 2; }
done
mkdir -p "$OUT"
sessions=$(c 180 sessions 2>&1)
echo "$sessions" | grep -q "\[$S\]" && { log "session $S already exists"; exit 2; }
echo "$sessions" | grep -q "\[?\]" && { log "orphan [?] sessions present: resolve them first"; exit 2; }
stop_session() { log "stopping $S"; c 120 stop -s "$S" > /dev/null 2>&1; c 180 sessions 2>&1 | grep -v "^\[colab\]" | grep -v "^$"; }
trap stop_session EXIT
start=$(date +%s); last_refresh=$start
past_deadline() { [ $(( $(date +%s) - start )) -ge $((MAXMIN * 60)) ]; }
maybe_refresh() {
  if [ $(( $(date +%s) - last_refresh )) -ge 900 ]; then
    timeout 180 "$COLAB_PY" "$PK/tools/colab_adopt.py" --refresh > /dev/null 2>&1 && last_refresh=$(date +%s)
  fi
}
log "new $GPU session $S"
out=$(c 600 new -s "$S" --gpu "$GPU" 2>&1)
echo "$out" | grep -q READY || { log "session not ready: ${out: -300}"; exit 1; }
for f in vm.sh fix.patch rmsnorm_batch_test.cu dumpprec.py sp_compare.py prompts.jsonl SHA256SUMS $(cd $STAGE && ls sft.part* l16.part*); do
  maybe_refresh
  ok=0
  for attempt in 1 2 3; do
    c 900 upload -s "$S" "$STAGE/$f" "/content/$f" > /dev/null 2>&1 && { ok=1; break; }
    log "upload $f attempt $attempt failed"
  done
  [ $ok = 1 ] || { log "upload $f failed"; exit 1; }
  log "uploaded $f"
done
cat > "$STAGE/launch.py" <<EOF
import os, subprocess
if not os.path.exists("/content/vm.log"):
    subprocess.Popen("cd /content && bash vm.sh $FIX $PKC", shell=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
print("LAUNCH_SENT")
EOF
launched=0
for attempt in 1 2 3; do
  maybe_refresh
  c 180 exec -s "$S" -f "$STAGE/launch.py" --timeout 120 > /dev/null 2>&1
  sleep 20
  if c 300 download -s "$S" /content/vm.log "$STAGE/vm.log" > /dev/null 2>&1; then launched=1; break; fi
  log "launch attempt $attempt not confirmed"
done
[ $launched = 1 ] || { log "launch failed"; exit 1; }
log "launched"
state=RUNNING
while :; do
  sleep 60
  maybe_refresh
  if c 300 download -s "$S" /content/VM_DONE "$STAGE/VM_DONE" > /dev/null 2>&1; then state=DONE
  elif c 300 download -s "$S" /content/VM_FAILED "$STAGE/VM_FAILED" > /dev/null 2>&1; then state=FAILED; fi
  if c 300 download -s "$S" /content/vm.log "$STAGE/vm.log" > /dev/null 2>&1; then
    log "state=$state $(grep -v '^[[:space:]]*$' "$STAGE/vm.log" | tail -1 | cut -c1-200)"
  fi
  [ "$state" = DONE ] || [ "$state" = FAILED ] && break
  past_deadline && { log "deadline"; state=DEADLINE; break; }
done
c 600 download -s "$S" /content/vm.log "$OUT/vm.log" > /dev/null 2>&1
if [ "$state" != DONE ]; then
  cat > "$STAGE/partial.py" <<'EOF'
import shutil
shutil.make_archive("/content/out", "zip", "/content", "out")
print("PACKED")
EOF
  c 400 exec -s "$S" -f "$STAGE/partial.py" --timeout 300 > /dev/null 2>&1
fi
maybe_refresh
if c 1800 download -s "$S" /content/out.zip "$OUT/out.zip" > /dev/null 2>&1 && (cd "$OUT" && unzip -oq out.zip); then
  log "results ($state) in $OUT"
else
  log "no results downloaded ($state)"
fi
echo "$state" > "$OUT/DRIVER_STATE"
