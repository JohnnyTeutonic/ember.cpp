import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).parent;spec=importlib.util.spec_from_file_location('profile_gate',HERE/'profile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_exact_registered_matrix():
 s=m.schedule();assert len(s)==80 and len({x['name'] for x in s})==80
 assert sum(x['family']=='primary' for x in s)==48
 for offset in range(0,64,4):assert [x['source'] for x in s[offset:offset+4]]==['baseline','candidate','candidate','baseline']
 for offset in range(64,80,4):assert [x['timer'] for x in s[offset:offset+4]]==[0,1,1,0]
 assert all(x['graph']==1 and x['timer']==0 for x in s[:48])
def test_phase_parse_preserves_host_and_gpu_separation():
 rows=m.phase_rows('  host.forward_device_graph  1.200 ms/step\n    gpu.layer.norm1 .040 ms/step\nnot a phase\n')
 assert rows==[{'phase':'host.forward_device_graph','ms_per_step':1.2},{'phase':'gpu.layer.norm1','ms_per_step':.04}]
def test_default_sanitizer_clears_presence_switches(monkeypatch):
 for key in ['EMBER_DECODE_TIMING','EMBER_IR','TINYLLAMA_LOGITS_DUMP','CUBLAS_WORKSPACE_CONFIG']:monkeypatch.setenv(key,'0')
 assert not any(key in m.sanitize() for key in ['EMBER_DECODE_TIMING','EMBER_IR','TINYLLAMA_LOGITS_DUMP','CUBLAS_WORKSPACE_CONFIG'])


@pytest.fixture
def raw_profile(tmp_path):
 import json
 plan={'protocol':__import__('json').loads((HERE/'protocol.json').read_text()),'schedule':m.schedule(),'manifests':{}}
 subset=tmp_path/'subset.jsonl';subset.write_text('fixed immutable subset fixture\n');plan['subset_sha256']=m.sha(subset)
 keys=[[i,0] for i in range(6)]+[[i,turn] for i in [6,7] for turn in range(1,6)]
 for src in ['baseline','candidate']:plan['manifests'][src]={'inputs':{'binary':'/'+src+'/bin','model':'/model.gguf','prompts':'/subset.jsonl'},'request_keys':keys}
 summary={'status':'PROFILE_COMPLETE','complete':True,'legs':[]}
 for descriptor in m.schedule():
  src=descriptor['source'];name=descriptor['name'];d=tmp_path/src;d.mkdir(exist_ok=True);path=d/(name+'.jsonl')
  records=[{'i':i,'turn':turn,'wall_ms':4.,'first_token_ms':1.,'tokens':2,'reply':'x','ids':[2],'prompt_ids':[1],'stop_id':3,'warm_pass_same':True} for i,turn in keys]
  path.write_text(''.join(json.dumps(x)+'\n' for x in records))
  env=[f"EMBER_DECODE_PICK={descriptor['pick']}",'EMBER_DECODE_PICK_PARTS=0','EMBER_SPEC=0',f"EMBER_DECODE_GRAPH={descriptor['graph']}",'EMBER_DECODE_OPT=1','EMBER_DECODE_LINEAR=cublas','EMBER_DETERMINISTIC='+('1' if descriptor['mode']=='det' else '0')]
  if descriptor['timer']:env.append('EMBER_DECODE_TIMING=1')
  inputs=plan['manifests'][src]['inputs'];meta={'env':env,'gpu_layers':999,'bin':inputs['binary'],'gguf':inputs['model'],'prompts':inputs['prompts'],'health':{'model':inputs['model'],'deterministic':descriptor['mode']=='det'},'once':False,'dump_rows':None}
  metadata=Path(str(path)+'.meta.json');metadata.write_text(json.dumps(meta));text=' host.forward_device_graph 1.000 ms/step\n' if descriptor['timer'] else '';Path(str(path)+'.server.log').write_text(text)
  ids=[[x['reply'],x['ids'],x['stop_id'],x['prompt_ids'],x['tokens']] for x in records]
  summary['legs'].append({**descriptor,'throughput':1000/3,'ids':ids,'phase_rows':m.phase_rows(text),'metadata_sha256':m.sha(metadata)})
 summary['primary_pairs']=[{'mode':row['mode'],'pick':row['pick'],'repeat':row['repeat'],'new_over_old_throughput':1.,'identical_generated_token_paths':True} for row in summary['legs'][:48] if row['position'] in [0,3]]
 (tmp_path/'profile_plan.json').write_text(json.dumps(plan));(tmp_path/'summary.json').write_text(json.dumps(summary));return tmp_path

def replay_module():
 spec=importlib.util.spec_from_file_location('raw_replay',HERE/'replay_profile.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_full_diagnostic_replay_without_model_execution(raw_profile):
 result=replay_module().replay(raw_profile)
 assert result['legs']==80 and len(result['primary_pairs'])==24

@pytest.mark.parametrize('mutation',['timing_nan','missing_request','ids','mode','fake_summary','fake_pair','schedule','subset'])
def test_corrupted_raw_diagnostic_rejected(raw_profile,mutation):
 import json
 summary_path=raw_profile/'summary.json';summary=json.loads(summary_path.read_text());first=summary['legs'][0];path=raw_profile/first['source']/(first['name']+'.jsonl');rows=[json.loads(x) for x in path.read_text().splitlines()]
 if mutation=='timing_nan':rows[0]['wall_ms']=float('nan');path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 if mutation=='missing_request':path.write_text(''.join(json.dumps(x)+'\n' for x in rows[:-1]))
 if mutation=='ids':rows[0]['ids']=[7];path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 if mutation=='mode':
  meta_path=Path(str(path)+'.meta.json');meta=json.loads(meta_path.read_text());meta['health']['deterministic']=True;meta_path.write_text(json.dumps(meta));summary['legs'][0]['metadata_sha256']=m.sha(meta_path)
 if mutation=='fake_summary':summary['legs'][0]['throughput']+=1
 if mutation=='fake_pair':summary['primary_pairs'][0]['new_over_old_throughput']=10
 if mutation=='schedule':
  plan_path=raw_profile/'profile_plan.json';plan=json.loads(plan_path.read_text());plan['schedule'].reverse();plan_path.write_text(json.dumps(plan))
 if mutation=='subset':(raw_profile/'subset.jsonl').write_text('changed')
 summary_path.write_text(json.dumps(summary))
 with pytest.raises(ValueError):replay_module().replay(raw_profile)

def test_vm_bad_manifest_preserves_error_without_device_call(tmp_path):
 import json,subprocess,sys
 upload=tmp_path/'upload';upload.mkdir();(upload/'inputs.json').write_text('{}');out=tmp_path/'out'
 p=subprocess.run([sys.executable,str(HERE/'profile_vm.py'),'--upload',str(upload),'--manifest-sha256','bad','--out',str(out)],capture_output=True)
 report=json.loads((out/'summary.json').read_text());assert p.returncode==1 and report['commands']==[] and report['complete'] is False and (out/'out.zip').is_file()
