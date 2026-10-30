#!/bin/bash
# E10.2: complete self-output pools, frozen candidates and paired held-out evaluation.
set -euo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; B=/content; SRC=/content/src
mkdir -p "$OUT" "$SRC"
pack() {
 flock /content/self_pack.lock python3 - <<'PACK'
import os, pathlib, zipfile
root = pathlib.Path('/content')
temporary = root / ('self_tmp_' + str(os.getpid()) + '.zip')
try:
 with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
  for path in sorted((root / 'out').rglob('*')):
   if path.is_file():
    archive.write(path, path.relative_to(root))
 os.replace(temporary, root / 'out.zip')
finally:
 temporary.unlink(missing_ok=True)
PACK
}
packer=""
finish() { rc=$?; trap - EXIT; if [ -n "$packer" ]; then kill "$packer" 2>/dev/null || true; wait "$packer" 2>/dev/null || true; fi; cp /content/vm.log "$OUT/vm.log"; pack || true; exit "$rc"; }
trap finish EXIT
trap 'exit 130' INT TERM
sha256sum -c INPUT_SHA256SUMS > "$OUT/input_verification.txt"
cp INPUT_SHA256SUMS COMMITS "$OUT/"
{ date -u; nvidia-smi; nvcc --version; python3 --version; } >> "$OUT/hardware.txt" 2>&1
tar --no-same-owner --no-same-permissions -xzf ember.tgz -C "$SRC" --strip-components=1
# Restore the verified first100 journals at identical paths without rewriting metadata.
tar --no-same-owner --no-same-permissions -xzf pilot_prefix.tgz -C /content
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1
(while sleep 120; do pack || true; done) &
packer=$!
chmod +x /content/tinyllama_server
mkdir -p "$OUT/binaries"
cp "$B/tinyllama_server" "$OUT/binaries/tinyllama_server"
sha256sum "$B/tinyllama_server" > "$OUT/binaries.sha256"
# Bound remote CPU reference threads, independently of the laptop's CPU budget.
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
P="$SRC/experiments/spec_r1_2026-10-28"
# Only the audited health-schema repair differs from the original frozen controllers.
cp /content/self_output_pool.py "$P/self_output_pool.py"
cp /content/self_output_gate.py "$P/self_output_gate.py"
# Preserve the exact audited Python sources beside the raw journal and binary.
mkdir -p "$OUT/source"
cp -a "$SRC/experiments/spec_r1_2026-10-28" "$SRC/experiments/draft_2026-10-27" "$OUT/source/"
for model in pick pk; do
 table=/content/${model}_table.bin
 pool=/content/${model}_pool.jsonl
 # The immutable pilot prefix already contains100 verified completed requests.
 python3 "$P/self_output_pool.py" run "$B/tinyllama_server" /content/$model.gguf "$pool" "$table" "$OUT/${model}_pool"
 python3 "$P/self_output_pool.py" analyze "$OUT/${model}_pool" > "$OUT/${model}_pool_audit.json"
 python3 "$P/self_output_candidate.py" "$OUT/${model}_pool" "$table" "$OUT/${model}_candidate"
 rows=(); [ "$model" != pick ] || rows=(--rows)
 if [ ! -e "$OUT/${model}_evaluation/plan.json" ]; then
  python3 "$P/self_output_gate.py" plan "$B/tinyllama_server" /content/$model.gguf /content/eval100.jsonl "$table" "$OUT/${model}_candidate/merged.bin" "$OUT/${model}_pool/outputs.jsonl" "$OUT/${model}_candidate/recipe.json" "$OUT/${model}_evaluation" "${rows[@]}"
 fi
 python3 "$P/self_output_gate.py" run "$OUT/${model}_evaluation"
 rc=0
 python3 "$P/self_output_gate.py" analyze "$OUT/${model}_evaluation" > "$OUT/${model}_gate_summary.json" || rc=$?
 [ "$rc" -le 1 ]
 python3 - "$OUT/${model}_gate_summary.json" <<'CHECK'
import json, sys
assert json.load(open(sys.argv[1]))['status'] in ('PASS', 'NO_GAIN', 'FAIL', 'BASELINE_UNSTABLE', 'INCONCLUSIVE_TIMING')
CHECK
 pack
 done
python3 - "$OUT" <<'CHECK'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
r={m:json.loads((p/(m+'_gate_summary.json')).read_text()) for m in ('pick','pk')}
(p/'gate_summary.json').write_text(json.dumps({'complete':True,'models':r,'status':'PASS' if all(v['status']=='PASS' for v in r.values()) else 'NOT_ADOPTED'},indent=1)+'\n')
CHECK
echo done > /content/VM_DONE
