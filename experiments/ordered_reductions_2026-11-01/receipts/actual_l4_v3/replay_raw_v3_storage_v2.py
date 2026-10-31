"""Separate offline storage amendment audit wrapping immutable V3 numerical replay."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
HERE=Path(__file__).resolve().parent
ORIGINAL_SHA='0e95d323e711930ad4fefceb85102aeb2677894263fabf2e2ed020f61e3cfca5'
FULL_RUNTIME_SHA='dea45ef19129b4528f55d492f0dcd533d57d792b4a12df8cd0866c34ae5326af'
RUNTIME_SHA='ac9037a2890fa8538e819bef170b365381bf39b4a5499f66d395d5805a8328bd'
REVIEW_SHA='3d5cff5486f370d190556704a8797e13172ebd6b93f51e4123f7c5de1eb2745b'
EXECUTION=Path('/home/jonat/ordered_default_storage_amendment_20261101/execution.json')

def sha(b):return hashlib.sha256(b).hexdigest()

def reconstructed_runtime():
    transport=Path('/home/jonat/pk_publication_20261030/tools/colab_chat/runpod_transport.py')
    raw=transport.read_bytes()
    if sha(raw)!='4e9e99b1642f1e0b88edc6a8180fadef91cf0bd73749898e541429cb1ae70690':raise ValueError('pinned transport preamble source drift')
    node=next(n.value for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENV_PREAMBLE' for t in n.targets))
    preamble=ast.literal_eval(node).encode()
    frozen=json.loads(Path('/home/jonat/ordered_default_launch_20261101_v3/launch_review.json').read_text())
    if frozen['pins'][str(transport)]!=sha(raw):raise ValueError('execution transport differs from frozen launch')
    body=Path('/home/jonat/ordered_default_storage_amendment_20261101/runtime_code.py.txt').read_bytes()
    if sha(body)!=RUNTIME_SHA:raise ValueError('amendment body source drift')
    return preamble+body


def audit_storage(z):
    read=lambda n:json.loads(z.read(n))
    start=read('science/storage_amendment_start.json');complete=read('science/storage_amendment_complete.json')
    runtime=z.read('science/storage_amendment_runtime.py.txt')
    if sha(runtime)!=FULL_RUNTIME_SHA or runtime!=reconstructed_runtime():raise ValueError('runtime amendment source drift')
    if start['status']!='ROOT_APPROVED_STORAGE_ONLY_START' or start['scratch_free_bytes']<4*1024**3:raise ValueError('start/free scratch evidence')
    if start['current_leg_untouched']!=['baseline','default_rows_off'] or start['first_dump_bytes']<=0:raise ValueError('active first leg proof')
    if start['review_sha256']!=REVIEW_SHA or start['review']!=json.loads((HERE/'storage_amendment_review.json').read_text()):raise ValueError('review binding')
    if complete['status']!='APPLIED_STORAGE_ONLY' or complete['time_unix']<start['time_unix']:raise ValueError('completion ordering')
    for key in ['first_leg_untouched','timing_placement_uniform','source_and_input_bytes_unchanged','retain_scratch_until_verified_final_archive']:
        if complete[key] is not True:raise ValueError('completion scope flag '+key)
    protocol=read('science/protocol.json');plans={s:read('science/'+s+'/plan.json') for s in ('baseline','candidate')}
    if protocol['schedule'][0]!=['baseline','default_rows_off'] or len(protocol['schedule'])!=24:raise ValueError('registered schedule')
    expected=[]
    for source,name in protocol['schedule'][1:]:
        leg=next(r for r in plans[source]['runs'] if r['name']==name)
        expected.append({'source':source,'name':name,'dump':leg['dump'],'out':'/content/chat/ordered_default_gate/science/'+source+'/'+name+'.jsonl'})
    if start['cases']!=expected or len(expected)!=23 or sum(not c['dump'] for c in expected)!=16 or sum(c['dump'] for c in expected)!=7:raise ValueError('23-case snapshot inventory')
    expected_ops=[]
    for c in sorted(expected,key=lambda c:c['dump']):
        target='/tmp/ember_ordered_default_v3_fastio/'+c['source']+'/'+c['name']
        expected_ops.append({'source':c['source'],'name':c['name'],'dump':c['dump'],'run_link':c['out']+'.run','run_target':target,'rows_target':target+'/rows.f32' if c['dump'] else None})
    if complete['operations']!=expected_ops:raise ValueError('exact local link/timing symmetry inventory')
    names=set(z.namelist());missing=[]
    for c in expected:
        root='science/'+c['source']+'/'+c['name']+'.jsonl'
        if any(n.startswith(root+'.run/') for n in names):raise ValueError('linked ephemeral run subtree unexpectedly archived')
        if c['dump'] and root+'.rows.f32' not in names:raise ValueError('canonical linked row member missing')
        for suffix in ('','.meta.json','.server.log'):
            if root+suffix not in names:missing.append(root+suffix)
    if z.getinfo('science/baseline/default_rows_off.jsonl.rows.f32').file_size<start['first_dump_bytes']:raise ValueError('first active dump regressed')
    summary=read('summary.json')
    if summary.get('complete') is True and missing:raise ValueError('complete receipt missing canonical outputs/serverlogs')
    return {'status':'STORAGE_AMENDMENT_VERIFIED','future_run_links':23,'future_dump_files':7,'uniform_local_timing_legs':16,'scratch_free_bytes_at_start':start['scratch_free_bytes'],'runtime_source_sha256':RUNTIME_SHA,'archived_runtime_with_pinned_transport_preamble_sha256':sha(runtime),'runtime_decomposition':'Exact AST-literal ENV_PREAMBLE from frozen transport plus exact reviewed amendment body; no stripping or ignored bytes','missing_uncompleted_outputs':missing,'disclosures':['First baseline default_rows_off stayed on NFS; its repeat used local scratch. Baseline off/off is mixed-storage and cannot establish a unique causal storage or kernel effect.','Candidate14 and all16 timing legs used uniform local storage; source, model, numerical gates and dump flags unchanged.','The23 linked .run directory subtrees were not traversed by rglob; ephemeral last dump.txt/debugging/www contents are excluded. Canonical linked f32 files and outside serverlogs are included by the packer.','Component20-repeat check remains a recorded executable assertion plus independent cross-process first-row comparisons; all20 outputs were not retained.']}

def descriptive_timing(z):
    gatefile=Path('/home/jonat/ember_ordered_default_20261101/experiments/spec_r1_2026-10-28/decode_pick_gate.py')
    protocol=json.loads(z.read('science/protocol.json'))
    if sha(gatefile.read_bytes())!=protocol['frozen_checker_sha256']:raise ValueError('timing checker source drift')
    spec=importlib.util.spec_from_file_location('frozen_descriptive_timing',gatefile);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    captured=json.loads(z.read('science/baseline_comparison.json'))
    recomputed={}
    for mode in ('default','det'):
        for source in ('baseline','candidate'):
            throughput={}
            for leg in ('off_a','on_a','on_b','off_b'):
                name='science/'+source+'/'+mode+'_'+leg+'.jsonl'
                rows=[json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]
                throughput[leg]=gate.speed(rows)
            recomputed[source+'_'+mode]={'throughput':throughput,'paired_speedups':[throughput['on_'+pair]/throughput['off_'+pair] for pair in ('a','b')]}
    if captured['timings']!=recomputed:raise ValueError('captured descriptive throughput/paired speedups differ from frozen speed replay')
    matched={mode:{leg:recomputed['candidate_'+mode]['throughput'][leg]/recomputed['baseline_'+mode]['throughput'][leg] for leg in ('off_a','on_a','on_b','off_b')} for mode in ('default','det')}
    return {'status':'DESCRIPTIVE_TIMING_REPLAY_VERIFIED','recomputed':recomputed,'matched_candidate_over_baseline_throughput':matched,'scope':'Matched mode/leg new-over-old throughput ratios only. Original protocol interleaving retained. No new acceptance gate, pooling rescue, or isolated causal claim; candidate14 original selector gate remains unchanged.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing replay output')
    original=HERE/'replay_raw_v3.py'
    if sha(original.read_bytes())!=ORIGINAL_SHA or sha((HERE/'storage_amendment_review.json').read_bytes())!=REVIEW_SHA:raise ValueError('independent audit source drift')
    execution=json.loads(EXECUTION.read_text())
    if execution['runtime_code_sha256']!=RUNTIME_SHA or execution['ssh_rc']!=0 or json.loads(execution['output'])!={'status':'APPLIED_STORAGE_ONLY','future_legs':23,'timing_legs_uniform':16,'future_dump_files':7,'current_leg_untouched':True}:raise ValueError('root execution receipt binding')
    spec=importlib.util.spec_from_file_location('immutable_v3_replay',original);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    if a.archive.resolve()!=base.CANONICAL.resolve():raise ValueError('canonical V3 archive required')
    report={'status':'STORAGE_AUDIT_ERROR_OR_INCOMPLETE','provider_calls':0,'model_calls':0,'config_writes':0}
    try:
        with zipfile.ZipFile(a.archive) as z:
            report['storage']=audit_storage(z)
            report['descriptive_timing']=descriptive_timing(z)
        raw=a.output.with_name(a.output.name+'.numerical_base.json')
        rc=base.replay(a.archive,raw);report['numerical_base_path']=str(raw);report['numerical_base_sha256']=sha(raw.read_bytes())
        report['numerical_base_status']=json.loads(raw.read_text())['status']
        report['status']='STORAGE_AND_NUMERICAL_REPLAY_VERIFIED' if rc==0 else 'STORAGE_VERIFIED_NUMERICAL_ERROR_OR_INCOMPLETE'
    except Exception as exc:report['error']=repr(exc)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report));return 0 if report['status']=='STORAGE_AND_NUMERICAL_REPLAY_VERIFIED' else 2
if __name__=='__main__':raise SystemExit(main())
