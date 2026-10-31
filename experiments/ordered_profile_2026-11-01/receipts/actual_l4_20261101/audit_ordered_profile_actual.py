from pathlib import Path
import argparse,csv,datetime,hashlib,io,json,math,os,re,stat,statistics,subprocess,tarfile,zipfile
parser=argparse.ArgumentParser(description='Independent full raw ZIP/source/telemetry audit; compact public receipt alone is insufficient.')
parser.add_argument('--canonical-job',type=Path,default=Path('/mnt/c/ml_artifacts/ember_ordered_profile_1101/ordered_profile'))
parser.add_argument('--stage',type=Path,default=Path('/home/jonat/ordered_profile_stage_v1_20261101'))
parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[4])
parser.add_argument('--python',default='/home/jonat/pk_bf16_replay_20261031/bin/python')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();home=args.canonical_job;out=args.out;out.mkdir(exist_ok=False);stage=args.stage;source=args.source;packet=source/'experiments/ordered_profile_2026-11-01'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def stats(values):
 values=list(values)
 return {'n':len(values),'median':statistics.median(values),'min':min(values),'max':max(values)} if values else {'n':0}
before={n:((home/n).stat().st_size,(home/n).stat().st_mtime_ns) for n in ['summary.json','out.zip','transport_exit.json']};manifest=json.loads((stage/'inputs.json').read_text());assert sha(stage/'inputs.json')=='368d19cb0778b6acf97fdeec69f7b0271d913ee89d655282d8cef6866e3a8d20'
for n,h in manifest['files'].items():assert sha(stage/n)==h,n
source_counts={}
for arm in ['baseline','candidate']:
 actual={}
 with tarfile.open(stage/(arm+'.tgz')) as t:
  for member in t:
   if member.isfile():actual[member.name]=hashlib.sha256(t.extractfile(member).read()).hexdigest()
 assert actual==manifest['source_files'][arm];source_counts[arm]=len(actual)
for name in ['profile.py','replay_profile.py','protocol.json']:
 assert sha(packet/name)==manifest['source_files']['candidate']['experiments/ordered_profile_2026-11-01/'+name]
outer=json.loads((home/'summary.json').read_text());assert outer['complete'] is True and outer['status']=='PROFILE_COMPLETE' and len(outer['commands'])==11 and all(r['returncode']==0 for r in outer['commands']);assert outer['source_commits']==manifest['source_commits']
assert json.loads((home/'transport_exit.json').read_text())=={'science_returncode':0,'transport_only':True}
science=out/'science';science.mkdir();inventory=[]
with zipfile.ZipFile(home/'out.zip') as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist())) and sum(i.file_size for i in z.infolist())<64*1024*1024
 for i in z.infolist():
  p=Path(i.filename);assert not p.is_absolute() and '..' not in p.parts and not stat.S_ISLNK(i.external_attr>>16)
  if i.is_dir():continue
  raw=z.read(i);dest=science/p;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(raw);inventory.append({'path':i.filename,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
assert json.loads((science/'summary.json').read_text())==outer and sha(science/'inputs.json')==sha(stage/'inputs.json')
profile=science/'profile';cmd=[args.python,str(packet/'replay_profile.py'),str(profile)]
p=subprocess.run(cmd,capture_output=True,text=True,env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});(out/'frozen_replay.stdout').write_text(p.stdout);(out/'frozen_replay.stderr').write_text(p.stderr);assert p.returncode==0,p.stderr
replay=json.loads(p.stdout);assert replay['status']=='PROFILE_REPLAY_VERIFIED' and replay['legs']==80
summary=json.loads((profile/'summary.json').read_text());plan=json.loads((profile/'profile_plan.json').read_text());binaries=json.loads((science/'binaries.json').read_text());proto=plan['protocol']
assert proto==json.loads((stage/'protocol.json').read_text())
original=[x for x in (stage/'eval100.jsonl').read_bytes().splitlines(keepends=True) if x.strip()];assert (profile/'subset.jsonl').read_bytes()==b''.join(original[i] for i in proto['subset_original_indices'])
for arm in ['baseline','candidate']:
 m=plan['manifests'][arm];assert m['sha256']['binary']==binaries[arm] and m['sha256']['model']==proto['model_sha256'] and m['sha256']['prompts']==sha(profile/'subset.jsonl')
 assert 'CUDA toolkit and cuBLAS found' in (science/(arm+'_configure.log')).read_text()
hardware=(science/'hardware.log').read_text().strip();assert hardware.startswith('NVIDIA L4,') and hardware==(profile/'hardware.txt').read_text().strip();uuid=hardware.split(',')[1].strip()
telemetry=[]
for row in csv.reader((profile/'gpu_telemetry.csv').read_text().splitlines()[1:]):
 assert len(row)==7 and row[1].strip()==uuid
 stamp=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc);values=[]
 for cell in row[2:]:
  match=re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)\s*(?:MHz|W|%)?\s*',cell);assert match,cell;values.append(float(match.group(1)))
 telemetry.append({'utc_ns':int(stamp.timestamp()*1e9),'sm_MHz':values[0],'memory_MHz':values[1],'power_W':values[2],'temperature_C':values[3],'gpu_util_percent':values[4]})
assert telemetry and all(a['utc_ns']<b['utc_ns'] for a,b in zip(telemetry,telemetry[1:]))
perleg=[];phasegroups={};native_routes=[]
for leg in summary['legs']:
 assert type(leg['start_utc_ns']) is int and leg['start_utc_ns']<leg['end_utc_ns']
 rows=[r for r in telemetry if leg['start_utc_ns']<=r['utc_ns']<=leg['end_utc_ns']];perleg.append({'name':leg['name'],'samples':len(rows),'sm_MHz':stats(r['sm_MHz'] for r in rows),'power_W':stats(r['power_W'] for r in rows),'temperature_C':stats(r['temperature_C'] for r in rows),'gpu_util_percent':stats(r['gpu_util_percent'] for r in rows)})
 log=(profile/leg['source']/(leg['name']+'.jsonl.server.log')).read_text();effective=re.findall(r'Effective CPU layers for this init: ([0-9]+), Effective GPU layers for this init: ([0-9]+)',log);native_routes.append({'name':leg['name'],'effective_layers':effective,'cublas_success':'cuBLAS handle created successfully.' in log,'fallback':'Model will run entirely on CPU' in log})
 assert bool(leg['phase_rows'])==bool(leg['timer'])
 for phase in leg['phase_rows']:
  assert math.isfinite(phase['ms_per_step']) and phase['ms_per_step']>=0
  key='|'.join(map(str,[leg['family'],leg['mode'],leg['pick'],leg['source'],leg['timer'],phase['phase']]))
  phasegroups.setdefault(key,[]).append(phase['ms_per_step'])
primary=[]
for mode in ['default','det']:
 for pick in [0,1]:
  pairs=[r for r in replay['primary_pairs'] if r['mode']==mode and r['pick']==pick];assert len(pairs)==6
  primary.append({'mode':mode,'pick':pick,'new_over_old':stats(r['new_over_old_throughput'] for r in pairs),'matched_token_paths':sum(r['identical_generated_token_paths'] for r in pairs),'total_pairs':6,'old_throughput':stats(r['throughput'] for r in summary['legs'][:48] if r['mode']==mode and r['pick']==pick and r['source']=='baseline'),'new_throughput':stats(r['throughput'] for r in summary['legs'][:48] if r['mode']==mode and r['pick']==pick and r['source']=='candidate')})
instrumented=[]
for offset in range(48,64,4):
 group=summary['legs'][offset:offset+4]
 for left,right in [(0,1),(3,2)]:
  old,new=group[left],group[right];instrumented.append({'kind':'graph1_timer_new_over_old','mode':old['mode'],'pick':old['pick'],'ratio':new['throughput']/old['throughput'],'same_token_paths':old['ids']==new['ids']})
for offset in range(64,80,4):
 group=summary['legs'][offset:offset+4]
 for off,on in [(0,1),(3,2)]:
  a,b=group[off],group[on];instrumented.append({'kind':'graph0_timer_on_over_off','source':a['source'],'mode':a['mode'],'pick':0,'ratio':b['throughput']/a['throughput'],'same_token_paths':a['ids']==b['ids']})
assert before=={n:((home/n).stat().st_size,(home/n).stat().st_mtime_ns) for n in before}
report={'status':'INDEPENDENT_FROZEN_PROFILE_REPLAY_VERIFIED','canonical_zip':str(home/'out.zip'),'canonical_zip_sha256':sha(home/'out.zip'),'canonical_zip_bytes':before['out.zip'][0],'all_CRC_verified':True,'manifest_sha256':sha(stage/'inputs.json'),'source_commits':manifest['source_commits'],'source_archive_inventories':source_counts,'binaries':binaries,'hardware':hardware,'raw_replay':replay,'primary_summary':primary,'separate_instrumentation_pairs':instrumented,'phase_summary':{k:stats(v) for k,v in phasegroups.items()},'phase_semantics':'Synchronized walltime perstep, includes warm+measured requests; not pure GPUeventtime. No pooling with primary timing.','telemetry':{'samples':len(telemetry),'uuid':uuid,'per_leg':perleg,'all_legs_have_samples':all(r['samples'] for r in perleg),'ranges':{k:stats(r[k] for r in telemetry) for k in ['sm_MHz','memory_MHz','power_W','temperature_C','gpu_util_percent']}},'source_bound_native_route_logs':native_routes,'limitations':['Eight frozenitems/sixteenrequests, not full260request adoption benchmark or allmodelclaim.','All24primarypairs retained; unmatched tokenpaths are not comparable work and not silently filtered.','32separate instrumentation/graph diagnostic legs never pooled into primary; phase instrumentation synchronizes and can distort timings.','Observed timing/clocks/control differences do not isolate unique causal mechanism, prove absolute gain, statistical significance or adoption.','Binary bytes are not duplicated in evidence; recorded hashes bind actualbuilt binaries and source inventories.'],'provider_or_GPU_actions':0,'inventory':inventory}
(out/'independent_actual_audit.json').write_text(json.dumps(report,indent=2)+'\n');(out/'descriptive_summary.json').write_text(json.dumps({'primary_summary':primary,'instrumented_pairs':instrumented,'telemetry_samples':len(telemetry),'all80_legs_have_telemetry':report['telemetry']['all_legs_have_samples'],'limitations':report['limitations']},indent=2)+'\n');print(json.dumps({'status':report['status'],'ZIPsha':report['canonical_zip_sha256'],'receiptsha':sha(out/'independent_actual_audit.json'),'primary_summary':primary,'telemetry_samples':len(telemetry),'all80_legs_have_telemetry':report['telemetry']['all_legs_have_samples']},indent=2))
