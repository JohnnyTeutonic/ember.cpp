"""Bounded HTTP/control profile; separate from the frozen full adoption gate."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
HERE=Path(__file__).resolve().parent


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def schedule():
 rows=[]
 def append(family,mode,pick,repeat,position,source,graph,timer):
  rows.append(dict(family=family,mode=mode,pick=pick,repeat=repeat,position=position,source=source,graph=graph,timer=timer,name=f'{mode}_{family}_pick{pick}_r{repeat}_p{position}_{source}'))
 for mode in ['default','det']:
  for pick in [0,1]:
   for repeat in range(3):
    for pos,source in enumerate(['baseline','candidate','candidate','baseline']):append('primary',mode,pick,repeat,pos,source,1,0)
 for mode in ['default','det']:
  for pick in [0,1]:
   for pos,source in enumerate(['baseline','candidate','candidate','baseline']):append('graph1_timer',mode,pick,0,pos,source,1,1)
 for source in ['baseline','candidate']:
  for mode in ['default','det']:
   for pos,timer in enumerate([0,1,1,0]):append('graph0_timer',mode,0,0,pos,source,0,timer)
 return rows


def sanitize():
 e={k:v for k,v in os.environ.items() if not k.startswith(('EMBER_','TINYLLAMA_')) and k!='CUBLAS_WORKSPACE_CONFIG'}
 e.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TZ='UTC')
 return e


def phase_rows(text):
 # Original API emits per-request wall/synchronized phases, including warm passes.
 return [{'phase':m.group(1),'ms_per_step':float(m.group(2))} for m in re.finditer(r'^\s*((?:gpu\.|host\.)[^\s]+)\s+([0-9.]+) ms/step$',text,re.M)]


def run(binary_old,binary_new,model,eval_file,roots,out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);model=Path(model).resolve();eval_file=Path(eval_file).resolve();proto=json.loads((HERE/'protocol.json').read_text())
 report={'status':'ERROR_OR_INCOMPLETE','complete':False,'scope':proto['scope'],'legs':[]};telemetry=None;telemetry_file=None
 try:
  if sha(model)!=proto['model_sha256'] or sha(eval_file)!=proto['eval_sha256']:raise ValueError('canonical inputs changed')
  lines=[x for x in eval_file.read_bytes().splitlines(keepends=True) if x.strip()];chosen=[lines[i] for i in proto['subset_original_indices']];subset=out/'subset.jsonl';subset.write_bytes(b''.join(chosen))
  if [json.loads(x)['set'] for x in chosen]!=['chat40']*4+['copy20']*2+['carry40']*2:raise ValueError('registered subset families')
  binaries={'baseline':str(Path(binary_old).resolve()),'candidate':str(Path(binary_new).resolve())};gates={};manifests={}
  for src,root in zip(['baseline','candidate'],map(Path,roots)):
   checker=root/'experiments/spec_r1_2026-10-28/decode_pick_gate.py';client=root/'experiments/draft_2026-10-27/dr_e2e.py'
   if sha(checker)!=proto['checker_sha256'] or sha(client)!=proto['client_sha256']:raise ValueError('frozen client/checker drift')
   spec=importlib.util.spec_from_file_location('checker_'+src,checker);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate);gates[src]=gate
   manifests[src]={'schema':'profile-only-not-adoption','inputs':{'binary':binaries[src],'model':str(model),'prompts':str(subset)},'sha256':{'binary':sha(binaries[src]),'model':sha(model),'prompts':sha(subset)},'request_keys':gate.request_keys(subset),'runs':[]}
   if len(manifests[src]['request_keys'])!=16:raise ValueError('subset request inventory')
  hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True);(out/'hardware.txt').write_text(hardware)
  if len(hardware.strip().splitlines())!=1 or not hardware.startswith('NVIDIA L4,'):raise ValueError('single actual L4 required')
  telemetry_file=(out/'gpu_telemetry.csv').open('w');telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu','--format=csv','-l','1'],stdout=telemetry_file,stderr=subprocess.STDOUT,env=sanitize())
  atomic_plan={'protocol':proto,'subset_sha256':sha(subset),'schedule':schedule(),'manifests':manifests};(out/'profile_plan.json').write_text(json.dumps(atomic_plan,indent=2)+'\n')
  for descriptor in schedule():
   if telemetry.poll() is not None:raise ValueError('common telemetry process failed')
   src=descriptor['source'];mode=descriptor['mode'];name=descriptor['name'];directory=out/src;directory.mkdir(exist_ok=True);path=directory/(name+'.jsonl')
   envargs=[f"EMBER_DECODE_PICK={descriptor['pick']}",'EMBER_DECODE_PICK_PARTS=0','EMBER_SPEC=0',f"EMBER_DECODE_GRAPH={descriptor['graph']}",'EMBER_DECODE_OPT=1','EMBER_DECODE_LINEAR=cublas','EMBER_DETERMINISTIC='+('1' if mode=='det' else '0')]
   if descriptor['timer']:envargs.append('EMBER_DECODE_TIMING=1')
   client=Path(roots[0 if src=='baseline' else 1])/'experiments/draft_2026-10-27/dr_e2e.py';cmd=['python3',str(client),'run',binaries[src],str(model),'999','8080',str(subset),str(path),*envargs]
   start=time.time_ns()
   with (directory/(name+'.log')).open('w') as log:
    p=subprocess.Popen(cmd,env=sanitize(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:p.wait(timeout=180)
    except BaseException:
     if p.poll() is None:
      os.killpg(p.pid,signal.SIGTERM)
      try:p.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     raise
   if p.returncode:raise ValueError('profile leg command failed '+name)
   runrow={'name':name,'env':envargs,'dump':False,'command':cmd};records,meta=gates[src].load_run(directory,runrow,manifests[src]);manifests[src]['runs'].append(runrow)
   serverlog=Path(str(path)+'.server.log');phases=phase_rows(serverlog.read_text()) if serverlog.exists() else []
   if descriptor['timer'] and not phases:raise ValueError('missing requested timing phases '+name)
   report['legs'].append({**descriptor,'start_utc_ns':start,'end_utc_ns':time.time_ns(),'throughput':gates[src].speed(records),'ids':gates[src].ids(records),'phase_rows':phases,'metadata_sha256':sha(Path(str(path)+'.meta.json'))})
   (out/'checkpoint.json.tmp').write_text(json.dumps(report,indent=2)+'\n');(out/'checkpoint.json.tmp').replace(out/'checkpoint.json')
  primary=[]
  for offset in range(0,48,4):
   group=report['legs'][offset:offset+4]
   for old,new in [(group[0],group[1]),(group[3],group[2])]:primary.append({'mode':old['mode'],'pick':old['pick'],'repeat':old['repeat'],'new_over_old_throughput':new['throughput']/old['throughput'],'identical_generated_token_paths':old['ids']==new['ids']})
  for source,binary in binaries.items():
   if sha(binary)!=manifests[source]['sha256']['binary']:raise ValueError('binary changed during profiling')
  report.update(status='PROFILE_COMPLETE',complete=True,primary_pairs=primary,interpretation='Descriptive diagnostic only. Instrumented/graph0 legs excluded from primary estimates. Nonmatching IDs/token paths make old/new work incomparable; never resampled. No unique causal/adoption/speed claim.')
 except BaseException as exc:report.update(status='ERROR_OR_INCOMPLETE',complete=False,error=repr(exc))
 finally:
  if telemetry is not None and telemetry.poll() is None:telemetry.terminate();telemetry.wait(timeout=10)
  if telemetry_file is not None:telemetry_file.close()
  (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--model',required=True);p.add_argument('--eval',required=True);p.add_argument('--source-old',required=True);p.add_argument('--source-new',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=run(a.old,a.new,a.model,a.eval,[a.source_old,a.source_new],a.out);raise SystemExit(0 if r['complete'] else 1)
