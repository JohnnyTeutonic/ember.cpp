#!/usr/bin/env bash
# Local driver for E6's Colab session (path C of microtorch/AGENTS.md: one foreground process for the whole
# job; new, upload, launch, poll, download, stop; the session stopped on every exit path), modelled on
# tools/colab_device_gate_drive.sh. Run from WSL:
#   bash drive.sh SESSION MAX_MINUTES STAGE_DIR [PART ...]
# STAGE_DIR (made on Windows, outside OneDrive): ember.tgz (git archive of the ember commit), COMMIT,
# vm_colab.sh, and stage.part.NN (a tar of the small models and inputs with SHA256SUMS, split at 40 MB).
# The official checkpoints are fetched on the VM. Every poll packs /content/e6out (Chrome logs cut to 2 MB)
# and brings it home to /mnt/c/ml_artifacts/e6gpu/SESSION/ (outside OneDrive). GPU: L4, else T4.
set -uo pipefail
S=${1:?session}; MAXMIN=${2:?max minutes}; STAGE=${3:?stage dir}; shift 3
PARTS="$*"
COL=~/.local/bin/colab
COLAB_PY=~/.local/share/uv/tools/google-colab-cli/bin/python
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
OUT=/mnt/c/ml_artifacts/e6gpu/$S
log() { echo "$(date +%H:%M:%S) $*"; }
c() { local t=$1; shift; timeout "$t" $COL "$@"; }
for f in ember.tgz COMMIT vm_colab.sh stage.part.00; do [ -s "$STAGE/$f" ] || { log "stage lacks $f"; exit 2; }; done
[ -e "$OUT" ] && { log "$OUT exists: pick a new session name"; exit 2; }
mkdir -p "$OUT"
sessions=$(c 180 sessions 2>&1)
echo "$sessions" | grep -q "\[$S\]" && { log "session $S already exists"; exit 2; }
echo "$sessions" | grep -q "\[?\]" && { log "orphan [?] sessions present: resolve them first"; echo "$sessions"; exit 2; }

stop_session() { log "stopping $S"; c 120 stop -s "$S" > /dev/null 2>&1; log "session time $(( ($(date +%s) - start) / 60 )) min on $GPU"; }
start=$(date +%s); last_refresh=$start; GPU=none
trap stop_session EXIT
maybe_refresh() {
  if [ $(( $(date +%s) - last_refresh )) -ge 900 ] && [ -x "$COLAB_PY" ]; then
    timeout 180 "$COLAB_PY" "$PK/tools/colab_adopt.py" --refresh > /dev/null 2>&1 && last_refresh=$(date +%s)
  fi
}
for g in L4 T4; do
  log "new $g session $S"
  out=$(c 600 new -s "$S" --gpu $g 2>&1)
  if echo "$out" | grep -q READY; then GPU=$g; break; fi
  log "no $g: ${out: -300}"
done
[ $GPU = none ] && exit 1
start=$(date +%s); echo "$GPU $(date -u +%FT%TZ)" > "$OUT/SESSION"
for f in ember.tgz COMMIT vm_colab.sh; do
  c 900 upload -s "$S" "$STAGE/$f" "/content/$f" > /dev/null 2>&1 || { log "upload $f failed"; exit 1; }
done
cat > "$STAGE/launch.py" << EOF
import os, subprocess
if not os.path.exists("/content/e6_vm.log"):
    subprocess.Popen("cd /content && bash vm_colab.sh $PARTS", shell=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
print("E6_LAUNCH_SENT")
EOF
c 180 exec -s "$S" -f "$STAGE/launch.py" --timeout 120 > /dev/null 2>&1
log "launched on $GPU; uploading the stage"
for p in "$STAGE"/stage.part.*; do
  ok=0
  for t in 1 2 3; do
    maybe_refresh
    c 900 upload -s "$S" "$p" "/content/stage/$(basename "$p")" > /dev/null 2>&1 && { ok=1; break; }
    log "upload $(basename "$p") attempt $t failed"
  done
  [ $ok = 1 ] || exit 1
done
cat > "$STAGE/unstage.py" << 'EOF'
import subprocess
r = subprocess.run("cd /content/stage && cat stage.part.* | tar --no-same-owner -xf - && rm -f stage.part.* && touch UPLOAD_DONE && ls",
                   shell=True, capture_output=True, text=True)
print("UNSTAGED", r.returncode, r.stdout[-300:], r.stderr[-300:])
EOF
c 400 exec -s "$S" -f "$STAGE/unstage.py" --timeout 300 2>&1 | grep UNSTAGED
log "stage uploaded"

cat > "$STAGE/pack.py" << 'EOF'
import os, shutil, zipfile
src, dst = "/content/e6out", "/content/e6out.zip.tmp"
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        for f in files:
            p = os.path.join(root, f)
            a = os.path.relpath(p, src)
            if f.endswith(".chrome.log") and os.path.getsize(p) > 2 << 20:
                z.writestr(a, open(p, "rb").read(2 << 20))
            else:
                z.write(p, a)
    if os.path.exists("/content/e6_vm.log"):
        z.write("/content/e6_vm.log", "e6_vm.log")
os.replace(dst, "/content/e6out.zip")
print("PACKED", os.path.getsize("/content/e6out.zip"))
EOF
fetch() {
  maybe_refresh
  c 400 exec -s "$S" -f "$STAGE/pack.py" --timeout 300 > /dev/null 2>&1
  if c 900 download -s "$S" /content/e6out.zip "$OUT/e6out.zip.part" > /dev/null 2>&1; then
    mv "$OUT/e6out.zip.part" "$OUT/e6out.zip" && (cd "$OUT" && rm -rf e6out && mkdir e6out && cd e6out && unzip -oq ../e6out.zip)
    log "home: $(grep -v '^[[:space:]]*$' "$OUT/e6out/e6_vm.log" 2>/dev/null | tail -1 | cut -c1-260)"
  else
    log "fetch failed"
  fi
}
state=RUNNING
while :; do
  sleep 150
  maybe_refresh
  if c 300 download -s "$S" /content/E6_DONE "$OUT/E6_DONE" > /dev/null 2>&1; then state=DONE
  elif c 300 download -s "$S" /content/E6_FAILED "$OUT/E6_FAILED" > /dev/null 2>&1; then state=FAILED; fi
  fetch
  [ "$state" != RUNNING ] && break
  if [ $(( $(date +%s) - start )) -ge $((MAXMIN * 60)) ]; then log "deadline"; state=DEADLINE; break; fi
done
fetch
log "state $state; results in $OUT"
echo "$state" > "$OUT/DRIVER_STATE"
