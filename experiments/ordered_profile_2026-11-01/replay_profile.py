"""Recompute diagnostic timing/comparability, never turn it into an adoption gate."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('profile_contract',HERE/'profile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def replay(out):
 out=Path(out);summary=json.loads((out/'summary.json').read_text());plan=json.loads((out/'profile_plan.json').read_text());expected=m.schedule()
 if plan['protocol']!=json.loads((HERE/'protocol.json').read_text()) or m.sha(out/'subset.jsonl')!=plan['subset_sha256']:raise ValueError('protocol/subset drift')
 if summary.get('status')!='PROFILE_COMPLETE' or summary.get('complete') is not True or plan['schedule']!=expected or len(summary['legs'])!=80:raise ValueError('incomplete/altered profile')
 legs=[]
 for descriptor,recorded in zip(expected,summary['legs']):
  if any(recorded.get(k)!=v for k,v in descriptor.items()):raise ValueError('descriptor mismatch')
  src=descriptor['source'];name=descriptor['name'];path=out/src/(name+'.jsonl');records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()];meta=json.loads(Path(str(path)+'.meta.json').read_text());manifest=plan['manifests'][src]
  if [[x['i'],x['turn']] for x in records]!=manifest['request_keys'] or len(records)!=16:raise ValueError('request inventory')
  if m.sha(Path(str(path)+'.meta.json'))!=recorded['metadata_sha256']:raise ValueError('metadata hash')
  if any(x.get('warm_pass_same') is not True or any(x.get(k,0)!=0 for k in ['passes','accepted','drafted']) for x in records):raise ValueError('invalid warm/spec control')
  env=meta['env']
  required=[f"EMBER_DECODE_PICK={descriptor['pick']}",'EMBER_DECODE_PICK_PARTS=0','EMBER_SPEC=0',f"EMBER_DECODE_GRAPH={descriptor['graph']}",'EMBER_DECODE_OPT=1','EMBER_DECODE_LINEAR=cublas','EMBER_DETERMINISTIC='+('1' if descriptor['mode']=='det' else '0')]
  if descriptor['timer']:required.append('EMBER_DECODE_TIMING=1')
  if env!=required or bool(meta.get('health',{}).get('deterministic'))!=(descriptor['mode']=='det') or meta.get('dump_rows') is not None or meta.get('once') is not False:raise ValueError('mode/timer/workload metadata')
  if meta.get('gpu_layers') not in (999,'999') or meta.get('health',{}).get('model')!=manifest['inputs']['model']:raise ValueError('device/model contract')
  if any(meta.get(field)!=manifest['inputs'][key] for key,field in [('binary','bin'),('model','gguf'),('prompts','prompts')]):raise ValueError('input identity')
  tokens=0;elapsed=0.
  for row in records:
   if type(row['tokens']) is not int or row['tokens']<0 or not isinstance(row['ids'],list) or not isinstance(row['prompt_ids'],list) or not row['prompt_ids'] or any(type(v) is not int or v<0 for v in row['ids']+row['prompt_ids']):raise ValueError('invalid token identity')
   wall,first=row['wall_ms'],row['first_token_ms']
   if not all(type(v) in (int,float) and math.isfinite(v) for v in [wall,first]) or not 0<=first<wall:raise ValueError('invalid timing')
   tokens+=max(0,row['tokens']-1);elapsed+=wall-first
  if tokens<=0 or elapsed<=0:raise ValueError('no decode work')
  throughput=tokens*1000/elapsed
  identities=[(x['reply'],x['ids'],x['stop_id'],x['prompt_ids'],x['tokens']) for x in records]
  phases=m.phase_rows(Path(str(path)+'.server.log').read_text())
  if throughput!=recorded['throughput'] or json.loads(json.dumps(identities))!=recorded['ids'] or phases!=recorded['phase_rows']:raise ValueError('summary/raw mismatch')
  legs.append({**descriptor,'throughput':throughput,'ids':identities})
 pairs=[]
 for offset in range(0,48,4):
  group=legs[offset:offset+4]
  for left,right in [(0,1),(3,2)]:
   old,new=group[left],group[right];pairs.append({'mode':old['mode'],'pick':old['pick'],'repeat':old['repeat'],'new_over_old_throughput':new['throughput']/old['throughput'],'identical_generated_token_paths':old['ids']==new['ids']})
 if pairs!=summary['primary_pairs']:raise ValueError('paired result mismatch')
 return {'status':'PROFILE_REPLAY_VERIFIED','legs':80,'primary_legs':48,'instrumented_legs':32,'primary_pairs':pairs,'scope':'Diagnostic estimates only; retain all pairs, unmatched-work flags and telemetry. No adoption, overall gain, isolated causal mechanism or statistical significance claimed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();print(json.dumps(replay(a.out),indent=2))
