import copy
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('storage',HERE/'replay_raw_v3_storage.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def members():
    protocol=json.loads(Path('/home/jonat/ordered_default_stage_20261101_v2/protocol.json').read_text())
    plans={s:{'runs':[{'name':name,'dump':'rows_' in name} for src,name in protocol['schedule'] if src==s]} for s in ('baseline','candidate')}
    cases=[];ops=[]
    for src,name in protocol['schedule'][1:]:
        dump='rows_' in name;out='/content/chat/ordered_default_gate/science/'+src+'/'+name+'.jsonl'
        cases.append({'source':src,'name':name,'dump':dump,'out':out})
    for c in sorted(cases,key=lambda c:c['dump']):
        target='/tmp/ember_ordered_default_v3_fastio/'+c['source']+'/'+c['name']
        ops.append({'source':c['source'],'name':c['name'],'dump':c['dump'],'run_link':c['out']+'.run','run_target':target,'rows_target':target+'/rows.f32' if c['dump'] else None})
    start={'status':'ROOT_APPROVED_STORAGE_ONLY_START','time_unix':1,'scratch_free_bytes':5*1024**3,'current_leg_untouched':['baseline','default_rows_off'],'first_dump_bytes':4,'cases':cases,'review_sha256':a.REVIEW_SHA,'review':json.loads((HERE/'storage_amendment_review.json').read_text())}
    complete={'status':'APPLIED_STORAGE_ONLY','time_unix':2,'operations':ops,**{k:True for k in ('first_leg_untouched','timing_placement_uniform','source_and_input_bytes_unchanged','retain_scratch_until_verified_final_archive')}}
    d={'science/storage_amendment_start.json':start,'science/storage_amendment_complete.json':complete,'science/protocol.json':protocol,'summary.json':{'complete':True},'science/baseline/plan.json':plans['baseline'],'science/candidate/plan.json':plans['candidate'],'science/storage_amendment_runtime.py.txt':Path('/home/jonat/ordered_default_storage_amendment_20261101/runtime_code.py.txt').read_bytes(),'science/baseline/default_rows_off.jsonl.rows.f32':b'1234'}
    for c in cases:
        root='science/'+c['source']+'/'+c['name']+'.jsonl'
        for suffix in ('','.meta.json','.server.log'):d[root+suffix]=b'fixture'
        if c['dump']:d[root+'.rows.f32']=b'1234'
    return d

def check(tmp_path,d):
    p=tmp_path/'f.zip'
    with zipfile.ZipFile(p,'w') as z:
        for n,v in d.items():z.writestr(n,v if isinstance(v,bytes) else json.dumps(v))
    with zipfile.ZipFile(p) as z:return a.audit_storage(z)

def test_complete_inventory_and_disclosures(tmp_path):
    r=check(tmp_path,members());assert r['future_run_links']==23 and r['uniform_local_timing_legs']==16 and r['future_dump_files']==7 and not r['missing_uncompleted_outputs']
    assert 'mixed-storage' in r['disclosures'][0] and 'excluded' in r['disclosures'][2]

@pytest.mark.parametrize('mutation',['missing_case','timing_target','row_missing','server_missing','run_subtree','runtime','scratch','first_size','flag'])
def test_negative_mutations(tmp_path,mutation):
    d=members();start=d['science/storage_amendment_start.json'];complete=d['science/storage_amendment_complete.json']
    if mutation=='missing_case':start['cases'].pop()
    if mutation=='timing_target':complete['operations'][0]['run_target']='/content/slow'
    if mutation=='row_missing':del d['science/candidate/default_rows_off.jsonl.rows.f32']
    if mutation=='server_missing':del d['science/candidate/default_rows_off.jsonl.server.log']
    if mutation=='run_subtree':d['science/candidate/default_rows_off.jsonl.run/dump.txt']=b'leaked ephemeral'
    if mutation=='runtime':d['science/storage_amendment_runtime.py.txt']=b'changed'
    if mutation=='scratch':start['scratch_free_bytes']=1
    if mutation=='first_size':start['first_dump_bytes']=5
    if mutation=='flag':complete['first_leg_untouched']=False
    with pytest.raises(ValueError):check(tmp_path,d)


def timing_members():
    protocol=json.loads(Path('/home/jonat/ordered_default_stage_20261101_v2/protocol.json').read_text())
    d={'science/protocol.json':protocol};timings={}
    for mode in ('default','det'):
        for source in ('baseline','candidate'):
            throughput={}
            for index,leg in enumerate(('off_a','on_a','on_b','off_b')):
                elapsed=(100.+index*10.)/(2. if source=='candidate' else 1.)
                row={'tokens':11,'wall_ms':elapsed+5.,'first_token_ms':5.}
                d['science/'+source+'/'+mode+'_'+leg+'.jsonl']=(json.dumps(row)+'\n').encode()
                throughput[leg]=10000./elapsed
            timings[source+'_'+mode]={'throughput':throughput,'paired_speedups':[throughput['on_'+pair]/throughput['off_'+pair] for pair in ('a','b')]}
    d['science/baseline_comparison.json']={'scope':'baseline10 controls; no full baseline adoption','timings':timings}
    return d

def timing_check(tmp_path,d):
    p=tmp_path/'timing.zip'
    with zipfile.ZipFile(p,'w') as z:
        for n,v in d.items():z.writestr(n,v if isinstance(v,bytes) else json.dumps(v))
    with zipfile.ZipFile(p) as z:return a.descriptive_timing(z)

def test_descriptive_timing_replays_all_sixteen_and_eight_pairs(tmp_path):
    r=timing_check(tmp_path,timing_members())
    assert len(r['recomputed'])==4 and all(len(x['throughput'])==4 and len(x['paired_speedups'])==2 for x in r['recomputed'].values())
    assert all(ratio==2. for mode in r['matched_candidate_over_baseline_throughput'].values() for ratio in mode.values())
    assert 'No new acceptance gate' in r['scope']

@pytest.mark.parametrize('field',['throughput','paired_speedups'])
def test_timing_summary_mutation_refused(tmp_path,field):
    d=timing_members();r=d['science/baseline_comparison.json']['timings']['baseline_default']
    if field=='throughput':r[field]['off_a']+=1
    else:r[field][0]+=1
    with pytest.raises(ValueError):timing_check(tmp_path,d)

def test_invalid_timing_raw_refused(tmp_path):
    d=timing_members();d['science/baseline/default_off_a.jsonl']=json.dumps({'tokens':11,'wall_ms':5.,'first_token_ms':5.}).encode()
    with pytest.raises(ValueError):timing_check(tmp_path,d)
