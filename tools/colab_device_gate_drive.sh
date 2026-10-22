#!/usr/bin/env bash
# Local driver for one device-gate session on Colab (tools/colab_device_gate_vm.sh): ember's IR
# interpreter on CUDA against its CPU interpreter for the third-party families, since GPU work runs
# on Colab only (22 Oct 2026). Run from WSL as ONE foreground process that lives for the whole job;
# path C of microtorch/AGENTS.md (one-off: new, upload, launch, poll, download, stop), the session
# stopped on every exit path. Modelled on tools/colab_ttft_drive.sh.
#
#   bash tools/colab_device_gate_drive.sh <session-name> <max-minutes> <ids-dir>
#
# <ids-dir> holds gemma3_270m_pastwin.ids.json, qwen25_long.ids.json and qwen3_long.ids.json
# (tools/chat_prompt_ids.py --min-tokens 513). Only committed code ships: ember HEAD and paperkiln
# HEAD's tools/ (git archive). The checkpoints are fetched on the VM; nothing large is uploaded.
# PRESTAGED=dir uses a stage made on Windows instead (ember.tgz, paperkiln.tgz, ids.tgz, the VM
# script and COMMITS = "<ember> <paperkiln>"), so this script touches no OneDrive tree.
# Results land in /mnt/c/ml_artifacts/devgate/<session>/ (outside OneDrive).
set -uo pipefail
S=${1:?session}; MAXMIN=${2:?max minutes}; IDS=${3:?ids dir}
COL=~/.local/bin/colab
COLAB_PY=~/.local/share/uv/tools/google-colab-cli/bin/python
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
EMBER=$ROOT/ember.cpp; PK=$ROOT/microtorch
STAGE=~/devgate_stage/$S; OUT=/mnt/c/ml_artifacts/devgate/$S
log() { echo "$(date +%H:%M:%S) $*"; }
c() { local t=$1; shift; timeout "$t" $COL "$@"; }   # every CLI call carries a shell timeout

[ -e "$OUT" ] && { log "$OUT exists: results are immutable, pick a new session name"; exit 2; }
if [ -n "${PRESTAGED:-}" ]; then
    # Staged on Windows (git archive with Windows git, outside OneDrive): WSL then reads nothing from
    # the OneDrive trees. A WSL git diff across them preceded the WSL service fault of 22 Oct 18:13.
    STAGE=$PRESTAGED
    for f in ember.tgz paperkiln.tgz ids.tgz colab_device_gate_vm.sh COMMITS; do
        [ -s "$STAGE/$f" ] || { log "PRESTAGED lacks $f"; exit 2; }
    done
    read -r EC PC < "$STAGE/COMMITS"
else
    for f in gemma3_270m_pastwin qwen25_long qwen3_long; do
        [ -s "$IDS/$f.ids.json" ] || { log "missing $IDS/$f.ids.json"; exit 2; }
    done
    EC=$(git -C "$EMBER" rev-parse --short=12 HEAD); PC=$(git -C "$PK" rev-parse --short=12 HEAD)
    rm -rf "$STAGE"; mkdir -p "$STAGE/ids"
    git -C "$EMBER" archive --format=tar.gz --prefix=ember/ -o "$STAGE/ember.tgz" HEAD
    git -C "$PK" archive --format=tar.gz --prefix=microtorch/ -o "$STAGE/paperkiln.tgz" HEAD tools
    cp "$IDS"/gemma3_270m_pastwin.ids.json "$IDS"/qwen25_long.ids.json "$IDS"/qwen3_long.ids.json "$STAGE/ids/"
    tar czf "$STAGE/ids.tgz" -C "$STAGE" ids
    git -C "$EMBER" show HEAD:tools/colab_device_gate_vm.sh > "$STAGE/colab_device_gate_vm.sh"
fi
mkdir -p "$OUT"
log "ember $EC, paperkiln $PC (HEADs; uncommitted edits are not shipped); staged $(du -sh "$STAGE" | cut -f1)"

sessions=$(c 180 sessions 2>&1)
echo "$sessions" | grep -q "\[$S\]" && { log "session $S already exists: someone owns it"; exit 2; }
echo "$sessions" | grep -q "\[?\]" && { log "orphan [?] sessions present: resolve them first"; echo "$sessions"; exit 2; }

stop_session() { log "stopping $S"; c 120 stop -s "$S" > /dev/null 2>&1; }
trap stop_session EXIT

start=$(date +%s); last_refresh=$start
past_deadline() { [ $(( $(date +%s) - start )) -ge $((MAXMIN * 60)) ]; }
maybe_refresh() {   # the session key expires after ~1 h
    if [ $(( $(date +%s) - last_refresh )) -ge 900 ] && [ -x "$COLAB_PY" ]; then
        timeout 180 "$COLAB_PY" "$PK/tools/colab_adopt.py" --refresh > /dev/null 2>&1 && last_refresh=$(date +%s)
    fi
}
log "new L4 session $S"
out=$(c 600 new -s "$S" --gpu L4 2>&1)
echo "$out" | grep -q READY || { log "session not ready: ${out: -300}"; exit 1; }
for f in ember.tgz paperkiln.tgz ids.tgz colab_device_gate_vm.sh; do
    c 900 upload -s "$S" "$STAGE/$f" "/content/$f" > /dev/null 2>&1 || { log "upload $f failed"; exit 1; }
done
log "uploaded"

cat > "$STAGE/launch.py" <<EOF
import os, subprocess
if not os.path.exists("/content/gate_vm.log"):
    subprocess.run("cd /content && tar xzf ids.tgz", shell=True)
    subprocess.Popen("cd /content && bash colab_device_gate_vm.sh $EC $PC", shell=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
print("GATE_LAUNCH_SENT")
EOF
launched=0
for attempt in 1 2 3; do
    maybe_refresh
    c 180 exec -s "$S" -f "$STAGE/launch.py" --timeout 120 > /dev/null 2>&1
    sleep 20
    if c 300 download -s "$S" /content/gate_vm.log "$STAGE/vm.log" > /dev/null 2>&1; then launched=1; break; fi
    log "launch attempt $attempt not confirmed"
done
[ $launched = 1 ] || { log "launch failed after 3 attempts"; exit 1; }
log "launched; deadline ${MAXMIN} min from session start"

state=RUNNING
while :; do
    sleep 90
    maybe_refresh
    if c 300 download -s "$S" /content/GATE_DONE "$STAGE/GATE_DONE" > /dev/null 2>&1; then state=DONE
    elif c 300 download -s "$S" /content/GATE_FAILED "$STAGE/GATE_FAILED" > /dev/null 2>&1; then state=FAILED; fi
    if c 300 download -s "$S" /content/gate_vm.log "$STAGE/vm.log" > /dev/null 2>&1; then
        log "state=$state $(grep -v '^[[:space:]]*$' "$STAGE/vm.log" | tail -1 | cut -c1-240)"
    else
        log "state=$state (log download failed)"
    fi
    if [ "$state" = DONE ] || [ "$state" = FAILED ]; then break; fi
    if past_deadline; then log "deadline reached"; state=DEADLINE; break; fi
done

c 900 download -s "$S" /content/gate_vm.log "$OUT/gate_vm.log" > /dev/null 2>&1
if [ "$state" != DONE ]; then   # partial results are still results
    cat > "$STAGE/partial.py" <<'EOF'
import shutil, subprocess
subprocess.run("pkill -f colab_device_gate_vm.sh; pkill -f ir_device_gate; pkill -f ir_logits", shell=True)
shutil.make_archive("/content/gate_out", "zip", "/content/out")
print("PARTIAL_PACKED")
EOF
    maybe_refresh
    c 400 exec -s "$S" -f "$STAGE/partial.py" --timeout 300 > /dev/null 2>&1
    sleep 10
fi
maybe_refresh
if c 1800 download -s "$S" /content/gate_out.zip "$OUT/gate_out.zip" > /dev/null 2>&1 && (cd "$OUT" && unzip -oq gate_out.zip); then
    log "results ($state) in $OUT"
else
    log "no results downloaded ($state); log at $OUT/gate_vm.log"
fi
echo "$state" > "$OUT/DRIVER_STATE"
