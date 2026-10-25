#!/usr/bin/env bash
# Local driver for one Colab session running vm.sh (the /score CUDA gate; adapted from ember.cpp/experiments/determinism_2026-10-26/drive.sh) (microtorch/AGENTS.md path C: new, upload, launch, poll,
# download, stop; the session is stopped on every exit path). Run from WSL as one foreground process.
#   bash drive.sh <session> <max-minutes> <stage dir>
# An L4 is asked for first (the deployment GPU, sm_89); when none is granted, a T4 under <session>-t4.
# Results land in /mnt/c/ml_artifacts/ember_score/<session actually used>/ (outside OneDrive).
set -uo pipefail
S0=${1:?session}; MAXMIN=${2:?max minutes}; STAGE=${3:?stage}
COL=~/.local/bin/colab
COLAB_PY=~/.local/share/uv/tools/google-colab-cli/bin/python
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
log() { echo "$(date +%H:%M:%S) $*"; }
c() { local t=$1; shift; timeout "$t" $COL "$@"; }
for f in vm.sh ember_new.tgz small.tgz BIG_SHA256SUMS; do
  [ -s "$STAGE/$f" ] || { log "stage lacks $f"; exit 2; }
done
sessions=$(c 180 sessions 2>&1)
echo "$sessions" | grep -q "\[$S0\]\|\[$S0-t4\]" && { log "session $S0 already exists"; exit 2; }
echo "$sessions" | grep -q "\[?\]" && { log "orphan [?] sessions present: resolve them first"; exit 2; }
S=""
stop_session() {
  for s in "$S0" "$S0-t4"; do c 180 sessions 2>&1 | grep -q "\[$s\]" && { log "stopping $s"; c 120 stop -s "$s" > /dev/null 2>&1; }; done
  c 180 sessions 2>&1 | grep -v "^\[colab\]" | grep -v "^$"
}
trap stop_session EXIT
start=$(date +%s); last_refresh=$start
past_deadline() { [ $(( $(date +%s) - start )) -ge $((MAXMIN * 60)) ]; }
maybe_refresh() {
  if [ $(( $(date +%s) - last_refresh )) -ge 900 ]; then
    timeout 180 "$COLAB_PY" "$PK/tools/colab_adopt.py" --refresh > /dev/null 2>&1 && last_refresh=$(date +%s)
  fi
}
log "new L4 session $S0"
out=$(c 600 new -s "$S0" --gpu L4 2>&1)
if echo "$out" | grep -q READY; then S=$S0; GPU=L4
else
  log "no L4: ${out: -300}"
  c 120 stop -s "$S0" > /dev/null 2>&1
  log "new T4 session $S0-t4"
  out=$(c 600 new -s "$S0-t4" --gpu T4 2>&1)
  echo "$out" | grep -q READY || { log "no T4 either: ${out: -300}"; exit 1; }
  S=$S0-t4; GPU=T4
fi
OUT=/mnt/c/ml_artifacts/ember_score/$S
mkdir -p "$OUT"
echo "$GPU" > "$OUT/GPU"
for f in vm.sh BIG_SHA256SUMS ember_new.tgz small.tgz $(cd $STAGE && ls pick.part* pkbase.part*); do
  maybe_refresh
  ok=0
  for attempt in 1 2 3; do
    c 900 upload -s "$S" "$STAGE/$f" "/content/$f" > /dev/null 2>&1 && { ok=1; break; }
    log "upload $f attempt $attempt failed"
  done
  [ $ok = 1 ] || { log "upload $f failed"; exit 1; }
  log "uploaded $f"
done
cat > "$STAGE/launch.py" <<'EOF'
import os, subprocess
if not os.path.exists("/content/vm.log"):
    subprocess.Popen("cd /content && bash vm.sh", shell=True, stdin=subprocess.DEVNULL,
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
got1=0
while :; do
  sleep 60
  maybe_refresh
  if c 300 download -s "$S" /content/VM_DONE "$STAGE/VM_DONE" > /dev/null 2>&1; then state=DONE
  elif c 300 download -s "$S" /content/VM_FAILED "$STAGE/VM_FAILED" > /dev/null 2>&1; then state=FAILED; fi
  # The partial pack (all but the third-party dumps), fetched once as soon as the VM has it.
  if [ "$got1" = 0 ] && c 300 download -s "$S" /content/OUT1_READY "$STAGE/OUT1_READY" > /dev/null 2>&1; then
    if c 1200 download -s "$S" /content/out1.zip "$OUT/out1.zip" > /dev/null 2>&1 && (cd "$OUT" && unzip -oq out1.zip); then
      got1=1; log "partial results (out1) in $OUT"
    fi
  fi
  if c 300 download -s "$S" /content/vm.log "$STAGE/vm.log" > /dev/null 2>&1; then
    log "state=$state $(grep -v '^[[:space:]]*$' "$STAGE/vm.log" | tail -1 | cut -c1-220)"
  fi
  [ "$state" = DONE ] || [ "$state" = FAILED ] && break
  # The session gone from the control-plane listing twice running: the VM is lost (the first session's was).
  if c 180 sessions 2>&1 | grep -q "\[$S\]"; then gone=0; else gone=$((${gone:-0} + 1)); fi
  [ "${gone:-0}" -ge 2 ] && { log "session $S left the listing"; state=LOST; break; }
  past_deadline && { log "deadline"; state=DEADLINE; break; }
done
c 600 download -s "$S" /content/vm.log "$OUT/vm.log" > /dev/null 2>&1
if [ "$state" != DONE ]; then
  cat > "$STAGE/partial.py" <<'EOF'
import shutil, subprocess
subprocess.run("pkill -f '[v]m.sh'; pkill -f '[t]inyllama_server'; pkill -f '[s]c_client'", shell=True)
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
