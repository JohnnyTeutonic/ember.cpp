"""Source-bound supervised VM payload. Never allocates resources or changes defaults."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import threading
import time
import zipfile


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def clean_env(base=None):
    env=dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith(('EMBER_','TINYLLAMA_')) or key=='CUBLAS_WORKSPACE_CONFIG':env.pop(key)
    env.update(EMBER_DECODE_PICK_PARTS='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env


def extract(archive,destination):
    destination.mkdir()
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if not m.isfile() and not m.isdir():raise ValueError('links/special source archive member')
            if not (destination/m.name).resolve().is_relative_to(destination.resolve()):raise ValueError('source traversal')
        t.extractall(destination)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--upload',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    upload=a.upload.resolve();out=a.out.resolve();out.mkdir(exist_ok=True)
    # No erased negative evidence or automatic scientific retry.
    if any(out.iterdir()):raise ValueError('output already contains evidence')
    science=out/'science';science.mkdir();work=out.parent/(out.name+'_work')
    if work.exists():raise ValueError('work directory already exists')
    work.mkdir();summary={'status':'INCOMPLETE','complete':False,'legs':[]};lock=threading.Lock();stop=threading.Event();deadline=time.monotonic()+9600
    def pack():
        with lock:
            atomic_json(out/'summary.json',summary)
            tmp=out/'out.zip.tmp'
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,strict_timestamps=False) as z:
                for p in sorted(science.rglob('*')):
                    if p.is_file():z.write(p,p.relative_to(out))
                z.write(out/'summary.json','summary.json')
            os.replace(tmp,out/'out.zip')
    def periodic():
        while not stop.wait(120):
            try:pack()
            except Exception as e:(out/'vm.log').open('a').write('CHECKPOINT_ERROR '+str(e)+'\n')
    def run(cmd,log,timeout=1800,env=None):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('scientific deadline')
        with log.open('wb') as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=clean_env() if env is None else env,timeout=min(timeout,remaining))
        return result.returncode
    def terminate(sig,frame):raise TimeoutError('signal '+str(sig))
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    (out/'vm.log').write_text('ordered-default supervised payload starting\n')
    thread=threading.Thread(target=periodic,daemon=True);thread.start()
    try:
        manifest=upload/'inputs.json'
        if sha(manifest)!=a.manifest_sha256:raise ValueError('manifest SHA mismatch')
        inputs=json.loads(manifest.read_text())
        for name,record in inputs['files'].items():
            p=upload/name
            if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('input mismatch '+name)
        atomic_json(science/'inputs.json',inputs)
        protocol=json.loads((upload/'protocol.json').read_text());atomic_json(science/'protocol.json',protocol)
        if len(protocol['schedule'])!=24 or protocol['parts']!=0:raise ValueError('registered matrix drift')
        if run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],science/'device.log',30):raise RuntimeError('device metadata')
        if 'L4' not in (science/'device.log').read_text():raise ValueError('L4 required')
        if run(['apt-get','update','-qq'],science/'apt_update.log',300):raise RuntimeError('build dependency index')
        if run(['apt-get','install','-y','-q','libboost-regex-dev','python3-dev'],science/'apt_install.log',300):raise RuntimeError('build dependencies')
        if run(['bash','-lc','nvcc --version; cmake --version; g++ --version; python3 --version; dpkg-query -W libboost-regex-dev python3-dev'],science/'toolchain.log',30):raise RuntimeError('toolchain inventory')
        plans={};gates={};binaries={};probes={}
        for source in ['baseline','candidate']:
            root=work/source;extract(upload/(source+'.tgz'),root)
            build=work/(source+'_build')
            # Baseline immutable archive lacks packet: build the same uploaded diagnostic source against it.
            configure=['cmake','-S',str(upload/'probe'),'-B',str(build),'-DEMBER_SOURCE='+str(root),'-DORDERED_CANDIDATE='+('ON' if source=='candidate' else 'OFF'),'-DHAS_CUDA=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=89','-DBUILD_TESTING=OFF']
            if run(configure,science/(source+'_configure.log'),600):raise RuntimeError(source+' configure')
            if run(['cmake','--build',str(build),'--target','tinyllama_server','ordered_probe','-j2'],science/(source+'_build.log'),1800):raise RuntimeError(source+' build')
            binaries[source]=build/'ember/tinyllama_server';probes[source]=build/'ordered_probe'
            atomic_json(science/(source+'_binary.json'),{'tinyllama_sha256':sha(binaries[source]),'probe_sha256':sha(probes[source]),'source_archive_sha256':sha(upload/(source+'.tgz'))})
            spec=importlib.util.spec_from_file_location('gate_'+source,root/'experiments/spec_r1_2026-10-28/decode_pick_gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate);gates[source]=gate
            plans[source]=gate.plan(binaries[source],upload/'model.gguf',upload/'eval100.jsonl',science/source,8080)
            if len(plans[source]['request_keys'])!=260:raise ValueError('request inventory')
        # Resident wrapper timings use paired source ABBA, all raw batches retained.
        kernel_results=[]
        for mode in ['unset','0','1']:
            for quartet in range(3):
                for position,source in enumerate(['baseline','candidate','candidate','baseline']):
                    name=f'kernel_{mode}_{quartet}_{position}_{source}';env=clean_env()
                    if mode!='unset':env['EMBER_DETERMINISTIC']=mode
                    rc=run([str(probes[source])],science/(name+'.log'),120,env)
                    record=json.loads((science/(name+'.log')).read_text().splitlines()[-1]);record.update(source=source,mode=mode,quartet=quartet,position=position,returncode=rc);kernel_results.append(record)
                    if rc or not record['pass']:raise RuntimeError('kernel semantics '+name)
        atomic_json(science/'kernel_results.json',kernel_results)
        kernel_pairs=[]
        for offset in range(0,len(kernel_results),4):
            group=kernel_results[offset:offset+4]
            for left,right in [(0,1),(3,2)]:
                for old,new in zip(group[left]['records'],group[right]['records']):
                    if (old['kernel'],old['n'],old.get('pattern'))!=(new['kernel'],new['n'],new.get('pattern')):raise ValueError('kernel timing inventory')
                    old_mean=sum(old['milliseconds'])/len(old['milliseconds']);new_mean=sum(new['milliseconds'])/len(new['milliseconds'])
                    kernel_pairs.append({'mode':group[left]['mode'],'quartet':group[left]['quartet'],'pair':[left,right],'kernel':old['kernel'],'n':old['n'],'pattern':old.get('pattern'),'old_ms':old_mean,'new_ms':new_mean,'old_over_new_speedup':old_mean/new_mean})
        atomic_json(science/'kernel_timing_comparison.json',{'scope':'resident component timing only, no automatic production opt-in','pairs':kernel_pairs})

        for kernel in ['rms','softmax']:
            for axis in range(3):
                name=f'trap_{kernel}_{axis}';rc=run([str(probes['candidate']),'trap',kernel,str(axis)],science/(name+'.log'),30)
                record=json.loads((science/(name+'.log')).read_text().splitlines()[-1])
                if rc or record!={**record,'stage':'actual_guarded_kernel','kernel':kernel,'axis':axis,'allocation_valid':True,'pass':True}:raise RuntimeError('guard trap missing '+name)
        for source,name in protocol['schedule']:
            leg=next(r for r in plans[source]['runs'] if r['name']==name)
            env=clean_env();log=science/(source+'_'+name+'.log')
            rc=run(leg['command'],log,1800,env)
            if rc:raise RuntimeError('registered leg execution '+source+'/'+name)
            # Verify actual fixed inventory immediately, with frozen source checker.
            gates[source].load_run(science/source,leg,plans[source])
            summary['legs'].append({'source':source,'name':name,'returncode':rc,'log_sha256':sha(log)})
            atomic_json(science/'completed_legs.json',summary['legs'])
        candidate=gates['candidate'].analyze(science/'candidate');atomic_json(science/'candidate_analysis.json',candidate)
        # Baseline partial controls explicitly never labelled full adoption.
        old=gates['baseline'];old_off=next(r for r in plans['baseline']['runs'] if r['name']=='default_rows_off');old_repeat=next(r for r in plans['baseline']['runs'] if r['name']=='default_rows_off_repeat')
        baseline={'scope':'10 controls, not full baseline adoption','rows_off':old.load_run(science/'baseline',old_off,plans['baseline']),'rows_off_repeat':old.load_run(science/'baseline',old_repeat,plans['baseline'])}
        comparison={'scope':'baseline10 controls; no full baseline adoption','baseline_default_repeat_bits_equal':sha(science/'baseline/default_rows_off.jsonl.rows.f32')==sha(science/'baseline/default_rows_off_repeat.jsonl.rows.f32'),'timings':{}}
        for mode in ['default','det']:
            for source in ['baseline','candidate']:
                gate=gates[source];plan=plans[source];loaded={}
                for leg in ['off_a','on_a','on_b','off_b']:
                    run_record=next(r for r in plan['runs'] if r['name']==mode+'_'+leg)
                    loaded[leg]=gate.load_run(science/source,run_record,plan)[0]
                comparison['timings'][source+'_'+mode]={'throughput':{leg:gate.speed(records) for leg,records in loaded.items()},'paired_speedups':[gate.speed(loaded['on_'+pair])/gate.speed(loaded['off_'+pair]) for pair in ['a','b']]}
        atomic_json(science/'baseline_comparison.json',comparison)
        for name,record in inputs['files'].items():
            if sha(upload/name)!=record['sha256']:raise ValueError('final input drift '+name)
        summary.update(status=candidate['status'],complete=True,candidate=candidate,baseline_scope='10 controls only; no full baseline adoption',kernel_semantics=True,release_guard=True)
    except BaseException as exc:
        summary.update(status='ERROR_OR_INCOMPLETE',error=repr(exc),complete=False)
        with (out/'vm.log').open('a') as f:f.write(repr(exc)+'\n')
    finally:
        stop.set();thread.join(timeout=5);pack()
    return 0 if summary['complete'] else 2

if __name__=='__main__':raise SystemExit(main())
