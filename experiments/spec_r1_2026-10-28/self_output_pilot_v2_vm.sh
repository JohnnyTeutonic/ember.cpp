#!/bin/bash
# Versioned timing-only pilot: preserve v1 failure; no candidate or evaluation.
set -euo pipefail
cd /content
exec > /content/vm.log 2>&1
OUT=/content/out
mkdir -p "$OUT"
pack() { flock /content/self_pilot_pack.lock python3 -c "import os,shutil; p=shutil.make_archive('/content/self_pilot_tmp_'+str(os.getpid()),'zip','/content','out'); os.replace(p,'/content/out.zip')"; }
finish() { rc=$?; trap - EXIT; cp /content/vm.log "$OUT/vm.log"; pack || true; exit "$rc"; }
trap finish EXIT
trap 'exit 130' INT TERM
sha256sum -c INPUT_SHA256SUMS > "$OUT/input_verification.txt"
cp INPUT_SHA256SUMS COMMITS "$OUT/"
{ date -u; nvidia-smi; python3 --version; } > "$OUT/hardware.txt" 2>&1
apt-get update -qq > /dev/null 2>&1
apt-get install -y -q libboost-regex-dev > /dev/null 2>&1
chmod +x /content/tinyllama_server
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$OUT/source" "$OUT/binaries"
cp /content/self_output_pool.py /content/self_output_pilot_v2.py "$OUT/source/"
cp /content/tinyllama_server "$OUT/binaries/"
for model in pick pk; do
 # Idempotent resume never generates another100 after a completed pilot.
 python3 /content/self_output_pilot_v2.py /content/tinyllama_server /content/$model.gguf /content/${model}_pool.jsonl /content/${model}_table.bin "$OUT/${model}_pool"
 python3 /content/self_output_pool.py analyze "$OUT/${model}_pool" > "$OUT/${model}_audit.json"
 pack
done
python3 - <<'PY'
import json,pathlib
p=pathlib.Path('/content/out'); report={'status':'PILOT_ONLY','scientific_complete':False,'full_generation_started':False,'evaluation_forecast_available':False,'forecast_limit':'Pilot extrapolates remaining generation only; evaluation and transfer costs excluded. Interrupted downtime is included in elapsed time.','models':{}}
for m in ('pick','pk'):
 r=json.loads((p/(m+'_pilot.json')).read_text());a=json.loads((p/(m+'_audit.json')).read_text())
 assert r['completed']==100 and a['completed']==100 and a['status']=='VERIFIED_PARTIAL'
 elapsed=r['elapsed_seconds'];count=r['new_requests']
 assert count>0
 report['models'][m]={'pilot':r,'audit':a,'seconds_per_request_including_startup':elapsed/count,'remaining_pool_seconds_at_pilot_rate':(r['requests']-r['completed'])*elapsed/count,'conservative_remaining_pool_seconds_2x':2*(r['requests']-r['completed'])*elapsed/count}
(p/'pilot_summary.json').write_text(json.dumps(report,indent=2)+'\n')
PY
echo done > /content/VM_DONE
