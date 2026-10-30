#!/bin/bash
# E10.3 priority pick only. Negative scientific outcomes are preserved, never promoted.
set -euo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out; B=/content/b; SRC=/content/src
mkdir -p "$OUT" "$SRC"
pack() { flock /content/e10_pack.lock python3 -c "import os,shutil; p=shutil.make_archive('/content/e10_tmp_'+str(os.getpid()),'zip','/content','out'); os.replace(p,'/content/out.zip')"; }
packer=""
finish() { rc=$?; trap - EXIT; if [ -n "$packer" ]; then kill "$packer" 2>/dev/null || true; wait "$packer" 2>/dev/null || true; fi; cp /content/vm.log "$OUT/vm.log"; pack || true; exit "$rc"; }
trap finish EXIT
trap 'exit 130' INT TERM
cp COMMITS "$OUT/COMMIT"
{ nvidia-smi; nvcc --version; python3 --version; } > "$OUT/hardware.txt" 2>&1
sha256sum ember.tgz model.gguf eval100.jsonl decode_pick_vm.sh > "$OUT/inputs.sha256"
tar --no-same-owner --no-same-permissions -xzf ember.tgz -C "$SRC" --strip-components=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
(while sleep 120; do pack || true; done) &
packer=$!
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CC" > "$OUT/cmake.log" 2>&1
grep -q 'CUDA toolkit and cuBLAS found' "$OUT/cmake.log"
J=$(nproc); [ "$J" -le 8 ] || J=8
make -C "$B" -j"$J" tinyllama_server decode_pick_rows > "$OUT/make.log" 2>&1
sha256sum "$B/tinyllama_server" "$B/decode_pick_rows" > "$OUT/binaries.sha256"
"$B/decode_pick_rows" > "$OUT/handcrafted.json"
P="$SRC/experiments/spec_r1_2026-10-28"
python3 - "$P" "$B/tinyllama_server" "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import decode_pick_fallback as fallback
import decode_pick_gate as gate
binary, out = sys.argv[2], Path(sys.argv[3])
f = out / 'fallback'
if (f / 'summary.json').exists():
    summary = json.loads((f / 'summary.json').read_text())
    records = {(graph, pick): json.loads((f / f'graph{graph}_pick{pick}.json').read_text())['rows']
               for graph in (0, 1) for pick in (0, 1)}
    assert fallback.compare(records)['pass']
    assert summary['binary_sha256'] == gate.sha(binary)
    assert summary['model_sha256'] == gate.sha('/content/model.gguf')
else:
    # An interrupted fallback has no final summary: retain diagnostics, rerun it.
    if f.exists():
        import time
        f.rename(out / ('fallback_partial_' + str(time.time_ns())))
    fallback.run(binary, '/content/model.gguf', f)
p = out / 'pick' / 'plan.json'
if not p.exists():
    gate.plan(binary, '/content/model.gguf', '/content/eval100.jsonl', p.parent, 8080)
manifest = json.loads(p.read_text())
assert manifest['inputs'] == {'binary': binary, 'model': '/content/model.gguf', 'prompts': '/content/eval100.jsonl'}
assert all(gate.sha(path) == manifest['sha256'][key] for key, path in manifest['inputs'].items())
import subprocess
for run in manifest['runs']:
    try:
        gate.load_run(p.parent, run, manifest)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    else:
        print('verified completed leg', run['name'], flush=True)
        continue
    subprocess.run(run["command"], check=True)
    (p.parent / "progress.json").write_text(json.dumps({"completed": run["name"]}) + "\n")
PY
rc=0
python3 "$P/decode_pick_gate.py" analyze "$OUT/pick" > "$OUT/gate_summary.json" || rc=$?
[ "$rc" -le 1 ]
python3 - "$OUT/gate_summary.json" <<'PY'
import json,sys
result=json.load(open(sys.argv[1]))
assert result['status'] in ('PASS','NO_GAIN','BASELINE_UNSTABLE'), result
PY
echo done > /content/VM_DONE
