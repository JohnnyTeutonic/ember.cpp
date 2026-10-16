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
VM_ENV="$*"
COL=~/.local/bin/colab
COLAB_PY=~/.local/share/uv/tools/google-colab-cli/bin/python
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
EMBER=$ROOT/tinyllama.cpp; PK=$ROOT/microtorch
STAGE=~/ttft_stage/$S; OUT=/mnt/c/ml_artifacts/ttft/$S
log() { echo "$(date +%H:%M:%S) $*"; }

# ---- refuse to measure code that is not a commit
for repo in "$EMBER" "$PK"; do
    if [ "$repo" = "$EMBER" ]; then paths=(.); else paths=(CMakeLists.txt include src third_party tools); fi
    # filemode/autocrlf: the OneDrive checkout is a Windows (CRLF) working
    # tree; without these WSL git calls every file modified. The archive
    # below reads committed blobs, so it ships LF regardless.
    if ! git -c core.filemode=false -c core.autocrlf=true -C "$repo" diff --quiet HEAD -- "${paths[@]}"; then
        log "uncommitted changes in $repo (${paths[*]}): commit first"; exit 2
    fi
done
EC=$(git -C "$EMBER" rev-parse --short=12 HEAD); PC=$(git -C "$PK" rev-parse --short=12 HEAD)
[ -e "$OUT" ] && { log "$OUT exists: results are immutable, pick a new session name"; exit 2; }

# ---- preflight (AGENTS.md section 2)
sessions=$($COL sessions 2>&1)
echo "$sessions" | grep -q "\[$S\]" && { log "session $S already exists: someone owns it"; exit 2; }
echo "$sessions" | grep -q "\[?\]" && { log "orphan [?] sessions present: resolve them first"; echo "$sessions"; exit 2; }

# ---- package the committed trees
rm -rf "$STAGE"; mkdir -p "$STAGE" "$OUT"
git -C "$EMBER" archive --format=tar.gz --prefix=ember/ -o "$STAGE/ember.tgz" HEAD
git -C "$PK" archive --format=tar.gz --prefix=microtorch/ -o "$STAGE/paperkiln.tgz" HEAD \
    CMakeLists.txt include src third_party tools studio web python tests
unzip -jo /mnt/c/ml_artifacts/paperkiln_story3k.zip paperkiln_story3k/story3k.gguf -d "$STAGE" > /dev/null
cp "$EMBER/tools/colab_ttft_vm.sh" "$STAGE/"
log "ember $EC, paperkiln $PC; staged $(du -sh "$STAGE" | cut -f1)"

stop_session() { log "stopping $S"; $COL stop -s "$S" > /dev/null 2>&1; }
trap stop_session EXIT

# ---- session
log "new L4 session $S"
out=$(timeout 600 $COL new -s "$S" --gpu L4 2>&1)
echo "$out" | grep -q READY || { log "session not ready: ${out: -300}"; exit 1; }
for f in ember.tgz paperkiln.tgz story3k.gguf colab_ttft_vm.sh; do
    timeout 900 $COL upload -s "$S" "$STAGE/$f" "/content/$f" > /dev/null 2>&1 || { log "upload $f failed"; exit 1; }
done
log "uploaded"

# ---- launch detached (Popen, every fd on /dev/null, own session: see
#      colab_transfer_runner.py on why exec must not hold the job's pipes)
cat > "$STAGE/launch.py" <<EOF
import subprocess, time
cmd = "cd /content && $VM_ENV bash colab_ttft_vm.sh $MODE $EC $PC"
subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
time.sleep(5)
r = subprocess.run("pgrep -f colab_ttft_vm.sh", shell=True, capture_output=True, text=True)
print("TTFT_LAUNCHED" if r.stdout.strip() else "TTFT_NOT_RUNNING")
EOF
cat > "$STAGE/status.py" <<'EOF'
import os, subprocess
state = ("DONE" if os.path.exists("/content/TTFT_DONE") else
         "FAILED" if os.path.exists("/content/TTFT_FAILED") else "RUNNING")
tail = subprocess.run("tail -n 4 /content/ttft_vm.log", shell=True, capture_output=True, text=True).stdout
print("TTFT_STATE=" + state); print(tail)
EOF
$COL exec -s "$S" -f "$STAGE/launch.py" --timeout 120 2>&1 | grep -q TTFT_LAUNCHED || { log "launch failed"; exit 1; }
log "launched mode=$MODE $VM_ENV; deadline ${MAXMIN} min"

# ---- poll
start=$(date +%s); last_refresh=$start; state=RUNNING
while :; do
    sleep 90
    now=$(date +%s)
    if [ $((now - last_refresh)) -ge 900 ] && [ -x "$COLAB_PY" ]; then
        "$COLAB_PY" "$PK/tools/colab_adopt.py" --refresh > /dev/null 2>&1 && last_refresh=$now
    fi
    st=$(timeout 600 $COL exec -s "$S" -f "$STAGE/status.py" --timeout 300 2>&1)
    state=$(echo "$st" | sed -n 's/.*TTFT_STATE=\([A-Z]*\).*/\1/p' | tail -1)
    log "state=${state:-unknown} $(echo "$st" | grep -v '^\[colab\]\|TTFT_STATE\|^\s*$' | tail -2 | tr '\n' ' ')"
    [ "$state" = DONE ] || [ "$state" = FAILED ] && break
    [ $((now - start)) -ge $((MAXMIN * 60)) ] && { log "deadline reached"; state=DEADLINE; break; }
done

# ---- bring everything home, then the trap stops the session
timeout 900 $COL download -s "$S" /content/ttft_vm.log "$OUT/ttft_vm.log" > /dev/null 2>&1
if [ "$state" != DONE ]; then
    # Partial results are still results: package whatever exists.
    cat > "$STAGE/partial.py" <<'EOF'
import shutil, subprocess
subprocess.run("pkill -f colab_ttft_vm.sh; pkill -f ttft_bench.py; pkill -f ttft_parity.py; "
               "pkill -f tinyllama_server", shell=True)
shutil.make_archive("/content/ttft_out", "zip", "/content/out")
print("PARTIAL_PACKED")
EOF
    $COL exec -s "$S" -f "$STAGE/partial.py" --timeout 300 > /dev/null 2>&1
fi
timeout 1800 $COL download -s "$S" /content/ttft_out.zip "$OUT/ttft_out.zip" > /dev/null 2>&1 \
    && (cd "$OUT" && unzip -oq ttft_out.zip) && log "results ($state) in $OUT" \
    || log "no results downloaded ($state); log at $OUT/ttft_vm.log"
echo "$state" > "$OUT/DRIVER_STATE"
