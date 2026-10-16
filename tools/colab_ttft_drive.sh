#!/usr/bin/env bash
# Local driver for one TTFT session on Colab (docs/TTFT.md). Run from WSL as
# ONE foreground process that lives for the whole job; it follows path C of
# microtorch/AGENTS.md (one-off: new, upload, launch, poll, download, stop)
# and stops the session on every exit path.
#
#   bash tools/colab_ttft_drive.sh <probe|full> <session-name> <max-minutes> [VAR=value ...]
#
# VAR=value pairs (RUNS, WARMUP, REPS, TIMING_LENGTHS, NSYS_LENGTHS) are
# passed to tools/colab_ttft_vm.sh. The code measured is the COMMITTED HEAD of
# ember.cpp and paperkiln (git archive), so a receipt's commit hash is exactly
# the code that ran; uncommitted edits are refused.
#
# Results land in /mnt/c/ml_artifacts/ttft/<session>/ (outside OneDrive).
set -uo pipefail
MODE=${1:?mode}; S=${2:?session}; MAXMIN=${3:?max minutes}; shift 3
VM_ENV=$(printf '%q ' "$@")   # shell-quoted: values may hold spaces (LONG_LENGTHS="2048 4096")
COL=~/.local/bin/colab
COLAB_PY=~/.local/share/uv/tools/google-colab-cli/bin/python
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
EMBER=$ROOT/tinyllama.cpp; PK=$ROOT/microtorch
STAGE=~/ttft_stage/$S; OUT=/mnt/c/ml_artifacts/ttft/$S
log() { echo "$(date +%H:%M:%S) $*"; }

# Every CLI call carries a SHELL timeout: `colab exec --timeout` does not
# bound a hung exec (probe 2, 16 Oct 2026: the launch exec hung for 60 min,
# the session key expired, the CLI pruned the session and orphaned the VM).
# Exec is used only to launch and to package; progress is read by
# DOWNLOADING the VM log and marker files, a channel that kept working
# while exec was dead.
c() { local t=$1; shift; timeout "$t" $COL "$@"; }

# ---- only committed code reaches the VM
# Everything shipped comes from HEAD (git archive, git show), so uncommitted
# edits in either working tree cannot leak into a measurement and are only
# noted. (Refusing on them blocked runs on another session's work in
# progress in paperkiln, 17 Oct 2026.) filemode/autocrlf: the OneDrive
# checkout is a Windows (CRLF) working tree; without these, WSL git calls
# every file modified.
for repo in "$EMBER" "$PK"; do
    if ! git -c core.filemode=false -c core.autocrlf=true -C "$repo" diff --quiet HEAD; then
        log "note: $repo has uncommitted changes; they are NOT shipped (HEAD is)"
    fi
done
EC=$(git -C "$EMBER" rev-parse --short=12 HEAD); PC=$(git -C "$PK" rev-parse --short=12 HEAD)
[ -e "$OUT" ] && { log "$OUT exists: results are immutable, pick a new session name"; exit 2; }

# ---- preflight (AGENTS.md section 2)
sessions=$(c 180 sessions 2>&1)
echo "$sessions" | grep -q "\[$S\]" && { log "session $S already exists: someone owns it"; exit 2; }
echo "$sessions" | grep -q "\[?\]" && { log "orphan [?] sessions present: resolve them first"; echo "$sessions"; exit 2; }

# ---- package the committed trees
rm -rf "$STAGE"; mkdir -p "$STAGE" "$OUT"
git -C "$EMBER" archive --format=tar.gz --prefix=ember/ -o "$STAGE/ember.tgz" HEAD
git -C "$PK" archive --format=tar.gz --prefix=microtorch/ -o "$STAGE/paperkiln.tgz" HEAD \
    CMakeLists.txt include src third_party tools studio web python tests
unzip -jo /mnt/c/ml_artifacts/paperkiln_story3k.zip paperkiln_story3k/story3k.gguf -d "$STAGE" > /dev/null
git -C "$EMBER" show HEAD:tools/colab_ttft_vm.sh > "$STAGE/colab_ttft_vm.sh"
log "ember $EC, paperkiln $PC; staged $(du -sh "$STAGE" | cut -f1)"

stop_session() { log "stopping $S"; c 120 stop -s "$S" > /dev/null 2>&1; }
trap stop_session EXIT

# ---- session (the deadline counts from here, not from launch)
start=$(date +%s); last_refresh=$start
past_deadline() { [ $(( $(date +%s) - start )) -ge $((MAXMIN * 60)) ]; }
# The key `colab new` stores expires after ~1 h; refresh well inside it, at
# every step, not only while polling (colab_transfer_runner.py refresh_key).
maybe_refresh() {
    if [ $(( $(date +%s) - last_refresh )) -ge 900 ] && [ -x "$COLAB_PY" ]; then
        timeout 180 "$COLAB_PY" "$PK/tools/colab_adopt.py" --refresh > /dev/null 2>&1 \
            && last_refresh=$(date +%s)
    fi
}
log "new L4 session $S"
out=$(c 600 new -s "$S" --gpu L4 2>&1)
echo "$out" | grep -q READY || { log "session not ready: ${out: -300}"; exit 1; }
for f in ember.tgz paperkiln.tgz story3k.gguf colab_ttft_vm.sh; do
    c 900 upload -s "$S" "$STAGE/$f" "/content/$f" > /dev/null 2>&1 || { log "upload $f failed"; exit 1; }
done
log "uploaded"

# ---- launch detached (Popen, every fd on /dev/null, own session: see
#      colab_transfer_runner.py on why exec must not hold the job's pipes).
#      Confirmed by the VM log appearing, and retried: the exec reply itself
#      is not trustworthy. The launcher is idempotent (no second job once
#      the log exists).
cat > "$STAGE/launch.py" <<EOF
import os, subprocess
if not os.path.exists("/content/ttft_vm.log"):
    subprocess.Popen(r"""cd /content && $VM_ENV bash colab_ttft_vm.sh $MODE $EC $PC""", shell=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
print("TTFT_LAUNCH_SENT")
EOF
launched=0
for attempt in 1 2 3; do
    maybe_refresh
    c 180 exec -s "$S" -f "$STAGE/launch.py" --timeout 120 > /dev/null 2>&1
    sleep 20
    if c 300 download -s "$S" /content/ttft_vm.log "$STAGE/vm.log" > /dev/null 2>&1; then
        launched=1; break
    fi
    log "launch attempt $attempt not confirmed"
done
[ $launched = 1 ] || { log "launch failed after 3 attempts"; exit 1; }
log "launched mode=$MODE $VM_ENV; deadline ${MAXMIN} min from session start"

# ---- poll by download
state=RUNNING
while :; do
    sleep 90
    maybe_refresh
    if c 300 download -s "$S" /content/TTFT_DONE "$STAGE/TTFT_DONE" > /dev/null 2>&1; then
        state=DONE
    elif c 300 download -s "$S" /content/TTFT_FAILED "$STAGE/TTFT_FAILED" > /dev/null 2>&1; then
        state=FAILED
    fi
    if c 300 download -s "$S" /content/ttft_vm.log "$STAGE/vm.log" > /dev/null 2>&1; then
        log "state=$state $(grep -v '^[[:space:]]*$' "$STAGE/vm.log" | tail -2 | tr '\n' ' ')"
    else
        log "state=$state (log download failed)"
    fi
    # Relay: the bench flushes raw/summary after every completed cell, so
    # copying them home each poll keeps finished cells if the VM is
    # reclaimed (16-17 Oct 2026: two of three L4s vanished mid-baseline).
    # Files are replaced only by a successful, non-shrinking download.
    if [ -z "${vm_out:-}" ] && c 120 download -s "$S" /content/TTFT_OUT_DIR "$STAGE/OUT_DIR" > /dev/null 2>&1; then
        vm_out=$(tr -d '\n' < "$STAGE/OUT_DIR")
    fi
    if [ -n "${vm_out:-}" ]; then
        for rel in story3k/raw.jsonl story3k/summary.jsonl timing/raw.jsonl timing/summary.jsonl; do
            mkdir -p "$OUT/relay/$(dirname "$rel")"
            if c 300 download -s "$S" "$vm_out/$rel" "$STAGE/relay.tmp" > /dev/null 2>&1 &&
               [ "$(stat -c %s "$STAGE/relay.tmp")" -ge "$(stat -c %s "$OUT/relay/$rel" 2>/dev/null || echo 0)" ]; then
                mv -f "$STAGE/relay.tmp" "$OUT/relay/$rel"
            fi
        done
    fi
    if [ "$state" = DONE ] || [ "$state" = FAILED ]; then break; fi
    if past_deadline; then log "deadline reached"; state=DEADLINE; break; fi
done

# ---- bring everything home, then the trap stops the session
c 900 download -s "$S" /content/ttft_vm.log "$OUT/ttft_vm.log" > /dev/null 2>&1
if [ "$state" != DONE ]; then
    # Partial results are still results: package whatever exists.
    cat > "$STAGE/partial.py" <<'EOF'
import shutil, subprocess
subprocess.run("pkill -f colab_ttft_vm.sh; pkill -f ttft_bench.py; pkill -f ttft_parity.py; "
               "pkill -f tinyllama_server", shell=True)
shutil.make_archive("/content/ttft_out", "zip", "/content/out")
print("PARTIAL_PACKED")
EOF
    maybe_refresh
    c 400 exec -s "$S" -f "$STAGE/partial.py" --timeout 300 > /dev/null 2>&1
    sleep 10
fi
maybe_refresh
if c 1800 download -s "$S" /content/ttft_out.zip "$OUT/ttft_out.zip" > /dev/null 2>&1 \
        && (cd "$OUT" && unzip -oq ttft_out.zip); then
    log "results ($state) in $OUT"
else
    log "no results downloaded ($state); log at $OUT/ttft_vm.log"
fi
echo "$state" > "$OUT/DRIVER_STATE"
