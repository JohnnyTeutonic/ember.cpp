#!/usr/bin/env bash
# Local driver for the R2 KV-cache quantisation session on Colab (tools/colab_kvq_vm.sh). Run from WSL as
# ONE foreground process that lives for the whole job: path C of microtorch/AGENTS.md (new, upload,
# launch, poll, download, stop), the session stopped on every exit path. Modelled on
# tools/colab_device_gate_drive.sh; the receipts are downloaded at every poll (the VM zips them every 2 minutes).
#
#   bash tools/colab_kvq_drive.sh <session-name> <max-minutes> <stage-dir> [phases] [gpu]
#
# <stage-dir> (made on Windows, outside OneDrive): ember.tgz, ember_base.tgz, inputs.tgz,
# colab_kvq_vm.sh, HEAD. Every poll also checks the control plane for the session and restarts its
# keep-alive when it has died (ember-r2's first VM was reclaimed 23 min in after its keep-alive died). Results land in /mnt/c/ml_artifacts/kvq/<session>/ (outside OneDrive).
set -uo pipefail
S=${1:?session}; MAXMIN=${2:?max minutes}; STAGE=${3:?stage dir}; PHASES=${4:-}; GPU=${5:-L4}
COL=~/.local/bin/colab
COLAB_PY=~/.local/share/uv/tools/google-colab-cli/bin/python
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
OUT=/mnt/c/ml_artifacts/kvq/$S; WORK=~/kvq_drive/$S
log() { echo "$(date +%H:%M:%S) $*"; }
c() { local t=$1; shift; timeout "$t" $COL "$@"; }
[ -e "$OUT" ] && { log "$OUT exists: results are immutable, pick a new session name"; exit 2; }
for f in ember.tgz ember_base.tgz inputs.tgz colab_kvq_vm.sh HEAD; do
  [ -s "$STAGE/$f" ] || { log "stage lacks $f"; exit 2; }
done
read -r HEAD_C < "$STAGE/HEAD"
mkdir -p "$OUT" "$WORK"
sessions=$(c 180 sessions 2>&1)
echo "$sessions"
echo "$sessions" | grep -q "\[$S\]" && { log "session $S already exists: someone owns it"; exit 2; }

stop_session() { log "stopping $S"; c 120 stop -s "$S" > /dev/null 2>&1; c 180 sessions 2>&1 | tail -5; }
trap stop_session EXIT
start=$(date +%s); last_refresh=$start
past_deadline() { [ $(( $(date +%s) - start )) -ge $((MAXMIN * 60)) ]; }
maybe_refresh() {
  if [ $(( $(date +%s) - last_refresh )) -ge 900 ] && [ -x "$COLAB_PY" ]; then
    timeout 180 "$COLAB_PY" "$PK/tools/colab_adopt.py" --refresh > /dev/null 2>&1 && last_refresh=$(date +%s)
  fi
}
log "new $GPU session $S (ember HEAD $HEAD_C)"
out=$(c 600 new -s "$S" --gpu "$GPU" 2>&1)
echo "$out" | grep -q READY || { log "session not ready: ${out: -300}"; exit 1; }
for f in ember.tgz ember_base.tgz inputs.tgz colab_kvq_vm.sh; do
  c 900 upload -s "$S" "$STAGE/$f" "/content/$f" > /dev/null 2>&1 || { log "upload $f failed"; exit 1; }
done
log "uploaded"
cat > "$WORK/launch.py" <<EOF
import os, subprocess
if not os.path.exists("/content/gate_vm.log"):
    subprocess.Popen("cd /content && bash colab_kvq_vm.sh $HEAD_C $PHASES", shell=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
print("KVQ_LAUNCH_SENT")
EOF
launched=0
for attempt in 1 2 3; do
  maybe_refresh
  c 180 exec -s "$S" -f "$WORK/launch.py" --timeout 120 > /dev/null 2>&1
  sleep 20
  if c 300 download -s "$S" /content/gate_vm.log "$WORK/vm.log" > /dev/null 2>&1; then launched=1; break; fi
  log "launch attempt $attempt not confirmed"
done
[ $launched = 1 ] || { log "launch failed after 3 attempts"; exit 1; }
log "launched; deadline $MAXMIN min from session start"

fetch() {  # the VM's current zip of /content/out, unpacked into OUT
  maybe_refresh
  if c 1200 download -s "$S" /content/gate_out.zip "$WORK/gate_out.zip" > /dev/null 2>&1; then
    (cd "$OUT" && unzip -oq "$WORK/gate_out.zip") && log "receipts fetched ($(ls "$OUT" | tr '\n' ' '))"
  else
    log "receipt download failed"
  fi
}
keepalive() {  # the session's endpoint in the listing; its keep-alive restarted when it is not running
  local ep
  ep=$(c 180 sessions 2>/dev/null | sed -n "s/^\[$S\] \([^ ]*\) .*/\1/p" | head -1)
  if [ -z "$ep" ]; then missing=$((missing + 1)); log "session $S not in the listing ($missing)"; return; fi
  missing=0
  if ! pgrep -f "[k]eep-alive $ep" > /dev/null; then
    setsid nohup "$COLAB_PY" -m colab_cli.cli --auth=oauth2 keep-alive "$ep" "$S" > /dev/null 2>&1 < /dev/null &
    log "keep-alive restarted for $ep"
  fi
}
echo 'print("KVQ_PING")' > "$WORK/ping.py"
state=RUNNING; seen=""; missing=0
while :; do
  sleep 90
  maybe_refresh
  keepalive
  if [ $missing -ge 3 ]; then log "session gone"; state=GONE; break; fi
  if c 300 download -s "$S" /content/GATE_DONE "$WORK/GATE_DONE" > /dev/null 2>&1; then state=DONE
  elif c 300 download -s "$S" /content/GATE_FAILED "$WORK/GATE_FAILED" > /dev/null 2>&1; then state=FAILED; fi
  if c 300 download -s "$S" /content/PHASES "$WORK/PHASES" > /dev/null 2>&1; then
    now=$(tr '\n' ' ' < "$WORK/PHASES")
    if [ "$now" != "$seen" ]; then log "phases done: $now"; seen=$now; fi
  fi
  fetch   # every poll: the VM zips its receipts every 2 minutes
  # A trivial exec every poll: three VMs that only served downloads were reclaimed 23 minutes in, each as
  # another session was created at the account's limit (28 Oct 2026).
  c 150 exec -s "$S" -f "$WORK/ping.py" --timeout 60 > /dev/null 2>&1 || log "ping exec failed"
  if c 300 download -s "$S" /content/gate_vm.log "$WORK/vm.log" > /dev/null 2>&1; then
    log "state=$state $(grep -v '^[[:space:]]*$' "$WORK/vm.log" | tail -1 | cut -c1-240)"
  else
    log "state=$state (log download failed)"
  fi
  if [ "$state" = DONE ] || [ "$state" = FAILED ]; then break; fi
  if past_deadline; then log "deadline reached"; state=DEADLINE; break; fi
done
if [ "$state" = DEADLINE ]; then
  cat > "$WORK/partial.py" <<'EOF'
import shutil, subprocess
subprocess.run("pkill -f colab_kvq_vm.sh; pkill -f kv_quant_eval; pkill -f ir_device_dump; pkill -f ir_device_gate", shell=True)
shutil.make_archive("/content/gate_out", "zip", "/content/out")
print("PARTIAL_PACKED")
EOF
  maybe_refresh
  c 400 exec -s "$S" -f "$WORK/partial.py" --timeout 300 > /dev/null 2>&1
  sleep 10
fi
fetch
c 900 download -s "$S" /content/gate_vm.log "$OUT/gate_vm.log" > /dev/null 2>&1
echo "$state" > "$OUT/DRIVER_STATE"
log "$state: results in $OUT"
