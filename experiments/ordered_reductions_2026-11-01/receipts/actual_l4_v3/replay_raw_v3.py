"""Offline independent ZIP replay. Run through the shared CPU budget for large receipts.
Never launches model code, contacts a provider, extracts model artifacts, or changes config.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile

HOME = Path(__file__).resolve().parent
STAGE = Path('/home/jonat/ordered_default_stage_20261101_v2')
SOURCE = Path('/home/jonat/ember_ordered_default_20261101')
MANIFEST = 'b7f50738edc08228a9be1155964d48afbbfeb482a64e1a63156b3cf56661ee88'
CANONICAL = Path('/mnt/c/ml_artifacts/ember_ordered_default_20261101_v3/ordered_default_gate/out.zip')
REGISTRATION = 'c3576f977'
LAUNCH = '4985e90ba89e8b34318584911e6fd99b6749fd1e00d5e13e6594e72cfe5b2ccd'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def finite(values):
    if not all(type(v) in (float, int) and math.isfinite(v) for v in values):
        raise ValueError('nonfinite numerical row')


def reference(r):
    kind = r['kernel']
    if kind in ('rms', 'softmax'):
        a = r['input']; finite(a)
        if len(a) != r['n']:
            raise ValueError('component input shape')
        if kind == 'rms':
            w = r['weights']; finite(w)
            if len(w) != len(a): raise ValueError('weights shape')
            eps = struct.unpack('f', struct.pack('f', 1e-5))[0]
            denominator = math.sqrt(math.fsum(x*x for x in a)/len(a)+eps)
            return [x*y/denominator for x,y in zip(a,w)]
        maximum = max(a); exps = [math.exp(x-maximum) for x in a]
        total = math.fsum(exps)
        return [x/total for x in exps]
    if kind not in ('attention_legacy','attention_devpos_legacy'):
        raise ValueError('unknown component')
    q,k,v,scale = (r[x] for x in ('input_q','input_k','input_v','scale'))
    for row in (q,k,v,[scale]): finite(row)
    n = r['n']
    if len(q)!=128 or len(k)!=n*64 or len(v)!=n*64: raise ValueError('attention shape')
    output = []
    for head in range(4):
        scores = [math.fsum(q[head*32+d]*k[t*64+(head//2)*32+d] for d in range(32))*scale for t in range(n)]
        maximum = max(scores); exp = [math.exp(s-maximum) for s in scores]; total = math.fsum(exp)
        output.extend(math.fsum(exp[t]/total*v[t*64+(head//2)*32+d] for t in range(n)) for d in range(32))
    return output


def check_component(r, candidate):
    expected = reference(r); observed = r['first_output']; recorded = r['reference']
    finite(observed); finite(recorded)
    if len(observed)!=len(expected) or len(recorded)!=len(expected): raise ValueError('output shape')
    error = max(abs(a-b) for a,b in zip(observed,expected))
    ref_error = max(abs(a-b) for a,b in zip(recorded,expected))
    finite([r['error']])
    if error>1e-5 or r['error']>1e-5 or r['error']+1e-12<error or ref_error>1e-12 or not r['pass']: raise ValueError('component semantic failure')
    if candidate and r['repeat_equal'] is not True: raise ValueError('recorded candidate repetition failure')
    timings = r['milliseconds']; finite(timings)
    if len(timings)!=30 or any(t<=0 for t in timings): raise ValueError('component timing shape')
    return error


class ZipPath:
    archive = None
    def __init__(self, path): self.name = str(path)
    def __str__(self): return self.name
    def __truediv__(self, child): return ZipPath(self.name.rstrip('/')+'/'+str(child))
    def read_text(self, encoding='utf-8'): return self.archive.read(self.name).decode(encoding)
    def stat(self): return SimpleNamespace(st_size=self.archive.getinfo(self.name).file_size)


def replay(archive, output):
    output = Path(output)
    if Path(archive).resolve() != CANONICAL.resolve(): raise ValueError('V3 replay requires canonical archive path')
    if output.exists(): raise ValueError('preserve existing replay evidence')
    if digest(STAGE/'inputs.json')!=MANIFEST: raise ValueError('stage manifest drift')
    launch = Path('/home/jonat/ordered_default_launch_20261101_v3/launch_review.json')
    if digest(launch)!=LAUNCH: raise ValueError('launch review drift')
    frozen = json.loads(launch.read_text())
    if not frozen['registration_commit'].startswith(REGISTRATION): raise ValueError('registration identity differs')
    inputs = json.loads((STAGE/'inputs.json').read_text())
    protocol = json.loads((STAGE/'protocol.json').read_text())
    gatefile = SOURCE/'experiments/spec_r1_2026-10-28/decode_pick_gate.py'
    if digest(gatefile)!=protocol['frozen_checker_sha256']: raise ValueError('checker drift')
    spec=importlib.util.spec_from_file_location('frozen_gate',gatefile); gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    report={'status':'REPLAY_ERROR_OR_INCOMPLETE','scope':'offline raw evidence only','component_repeat_limit':'Only first rows and recorded20-repeat assertions retained; cross-process first rows independently compared.','binary_limit':'VM server/probe identity is captured SHA metadata; executable bytes are not in raw ZIP.','model_calls':0,'provider_calls':0,'config_writes':0}
    try:
      with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(n.startswith('/') or '..' in Path(n).parts for n in names): raise ValueError('unsafe/duplicate ZIP member')
        hashes={}
        for info in z.infolist():
            h=hashlib.sha256()
            with z.open(info) as f:
                for b in iter(lambda:f.read(1048576),b''): h.update(b)
            hashes[info.filename]=h.hexdigest()
        report['crc_members_verified']=len(hashes);report['member_sha256']=hashes
        read=lambda name:json.loads(z.read(name))
        if hashes['science/inputs.json']!=MANIFEST or read('science/inputs.json')!=inputs: raise ValueError('archived inputs drift')
        if read('science/protocol.json')!=protocol: raise ValueError('archived protocol drift')
        summary=read('summary.json');report['captured_summary']=summary
        kernel=read('science/kernel_results.json')
        expected=[(mode,quartet,position,source) for mode in ('unset','0','1') for quartet in range(3) for position,source in enumerate(('baseline','candidate','candidate','baseline'))]
        if [(x['mode'],x['quartet'],x['position'],x['source']) for x in kernel]!=expected: raise ValueError('probe inventory')
        worst=0.;firsts={};component_count=0;cross_process=0
        expected_cases=[(kind,n,pattern) for n in (1,31,257,320,1024,16384) for pattern in range(3) for kind in ('rms','softmax')]+[(kind,n,None) for n in (1,31,257,512) for kind in ('attention_legacy','attention_devpos_legacy')]
        for record in kernel:
            name='science/kernel_%s_%s_%s_%s.log'%(record['mode'],record['quartet'],record['position'],record['source'])
            raw=json.loads(z.read(name).splitlines()[-1])
            if {**raw,**{k:record[k] for k in ('source','mode','quartet','position','returncode')}}!=record: raise ValueError('raw probe aggregate mismatch')
            if record['returncode']!=0 or record['pass'] is not True or 'L4' not in record['device']: raise ValueError('probe status/device')
            cases=[(r['kernel'],r['n'],r.get('pattern')) for r in record['records']]
            if cases!=expected_cases: raise ValueError('component cases')
            for r in record['records']:
                worst=max(worst,check_component(r,record['source']=='candidate')); component_count+=1
                if record['source']=='candidate':
                    key=(record['mode'],r['kernel'],r['n'],r.get('pattern'))
                    bits=struct.pack('<%sf'%len(r['first_output']),*r['first_output'])
                    if key in firsts:
                        if bits!=firsts[key]: raise ValueError('candidate cross-process component rows differ')
                        cross_process+=1
                    firsts[key]=bits
        report['components']={'cases_replayed':component_count,'maximum_absolute_error':worst,'candidate_cross_process_row_comparisons':cross_process}
        traps=[]
        for kind in ('rms','softmax'):
          for axis in range(3):
            r=json.loads(z.read('science/trap_%s_%s.log'%(kind,axis)).splitlines()[-1])
            if r['stage']!='actual_guarded_kernel' or r['kernel']!=kind or r['axis']!=axis or r['allocation_valid'] is not True or r['launch']!=0 or r['sync'] not in (715,719) or r['pass'] is not True: raise ValueError('trap evidence')
            traps.append(r)
        report['traps']=traps
        ZipPath.archive=z;gate.Path=ZipPath
        bound={}
        for source in ('baseline','candidate'):
            plan=read('science/'+source+'/plan.json'); binary=read('science/'+source+'_binary.json')
            if binary['source_archive_sha256']!=inputs['files'][source+'.tgz']['sha256']: raise ValueError('binary source metadata')
            if binary['tinyllama_sha256']!=plan['sha256']['binary']: raise ValueError('binary plan metadata')
            bound[plan['inputs']['binary']]=binary['tinyllama_sha256']
            for key,filename in (('model','model.gguf'),('prompts','eval100.jsonl')):
                actual=digest(STAGE/filename)
                if actual!=inputs['files'][filename]['sha256'] or actual!=plan['sha256'][key]: raise ValueError('model/prompts binding')
                bound[plan['inputs'][key]]=actual
        def zipsha(path):
            key=str(path)
            if key in hashes:return hashes[key]
            if key in bound:return bound[key]
            raise ValueError('unmapped frozen hash input '+key)
        gate.sha=zipsha
        result=gate.analyze('science/candidate');report['candidate_original_analyzer']=result
        if result!=read('science/candidate_analysis.json'): raise ValueError('captured analyzer differs')
        baseline=read('science/baseline/plan.json');loaded={}
        selected=[n for s,n in protocol['schedule'] if s=='baseline']
        for run in baseline['runs']:
            if run['name'] in selected:loaded[run['name']]=gate.load_run(ZipPath('science/baseline'),run,baseline)
        comparison=read('science/baseline_comparison.json')
        equal=hashes['science/baseline/default_rows_off.jsonl.rows.f32']==hashes['science/baseline/default_rows_off_repeat.jsonl.rows.f32']
        if equal!=comparison['baseline_default_repeat_bits_equal']:raise ValueError('baseline repeat summary')
        report['baseline_partial']={'legs':len(loaded),'default_repeat_bits_equal':equal,'scope':'ten controls; not full baseline adoption'}
        report.update(status='REPLAY_VERIFIED',scientific_status=result['status'])
    except Exception as exc:
      report['error']=repr(exc)
    report['zip_sha256']=digest(archive)
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('member_sha256','captured_summary','traps')}))
    return 0 if report['status']=='REPLAY_VERIFIED' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();return replay(a.archive,a.output)
if __name__=='__main__':raise SystemExit(main())
