"""Source-bound single-L4 diagnostic payload, no launch/provider/config access."""
import argparse,hashlib,json,os,signal,subprocess,tarfile,threading,time,traceback,zipfile
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--upload',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(exist_ok=True,parents=True)
 if any(out.iterdir()):raise ValueError('preserve prior evidence')
 work=out.parent/(out.name+'_work');work.mkdir(exist_ok=False);summary={'status':'ERROR_OR_INCOMPLETE','complete':False,'commands':[]};deadline=time.monotonic()+3000;stop=threading.Event();lock=threading.Lock()
 def pack():
  with lock:
   tmp=out/'summary.json.tmp';tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(out/'summary.json');tmp=out/'out.zip.tmp'
   with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,strict_timestamps=False) as z:
    for p in sorted(out.rglob('*')):
     if p.is_file() and p.name not in ['out.zip','out.zip.tmp','summary.json.tmp']:z.write(p,p.relative_to(out))
   tmp.replace(out/'out.zip')
 def checkpoint():
  while not stop.wait(120):pack()
 def command(name,cmd,timeout):
  left=deadline-time.monotonic()
  if left<=0:raise TimeoutError('science deadline')
  with (out/(name+'.log')).open('w') as f:
   p=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
   try:p.wait(timeout=min(timeout,left))
   except BaseException:
    if p.poll() is None:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
  summary['commands'].append({'name':name,'returncode':p.returncode})
  if p.returncode:raise ValueError(name+' failed')
 def terminated(sig,frame):raise TimeoutError('signal '+str(sig))
 signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated);thread=threading.Thread(target=checkpoint,daemon=True);thread.start()
 try:
  if sha(a.upload/'inputs.json')!=a.manifest_sha256:raise ValueError('manifest pin')
  m=json.loads((a.upload/'inputs.json').read_text())
  if m['limits']!={'max_usd':1,'max_hours':1.25,'science_seconds':3000,'gpu':'NVIDIA L4','gpus':1}:raise ValueError('limits')
  for name,h in m['files'].items():
   if Path(name).name!=name or sha(a.upload/name)!=h:raise ValueError('flat input mismatch '+name)
  (out/'inputs.json').write_bytes((a.upload/'inputs.json').read_bytes());roots={};binaries={}
  command('hardware',['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],30)
  lines=(out/'hardware.log').read_text().strip().splitlines()
  if len(lines)!=1 or not lines[0].startswith('NVIDIA L4,'):raise ValueError('one actual L4')
  command('apt_update',['apt-get','update','-qq'],300);command('apt_dependencies',['apt-get','install','-y','-q','libboost-regex-dev','python3-dev'],300)
  command('python_dependencies',['python3','-m','pip','install','requests==2.32.3'],120)
  command('toolchain',['bash','-lc','set -e; nvcc --version; cmake --version; g++ --version; python3 --version; lscpu --json'],30)
  for source in ['baseline','candidate']:
   root=work/source;root.mkdir();roots[source]=root
   with tarfile.open(a.upload/(source+'.tgz')) as t:
    for info in t.getmembers():
     if Path(info.name).is_absolute() or '..' in Path(info.name).parts or not (info.isfile() or info.isdir()):raise ValueError('unsafe source')
    t.extractall(root)
   actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
   if actual!=set(m['source_files'][source]):raise ValueError('source inventory')
   for name,h in m['source_files'][source].items():
    if sha(root/name)!=h:raise ValueError('source drift '+name)
   build=work/(source+'_build')
   command(source+'_configure',['cmake','-S',root,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DHAS_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=89'],300)
   if 'CUDA toolkit and cuBLAS found' not in (out/(source+'_configure.log')).read_text():raise ValueError('actual CUDA configure required')
   command(source+'_build',['cmake','--build',build,'--target','tinyllama_server','--parallel','2'],900);binaries[source]=build/'tinyllama_server'
  packet=roots['candidate']/'experiments/ordered_profile_2026-11-01'
  if sha(packet/'profile_vm.py')!=sha(a.upload/'profile_vm.py') or sha(__file__)!=m['files']['profile_vm.py'] or sha(packet/'protocol.json')!=sha(a.upload/'protocol.json'):raise ValueError('VM/protocol source bindings')
  (out/'binaries.json').write_text(json.dumps({source:sha(binary) for source,binary in binaries.items()},indent=2)+'\n')
  command('profile',['python3',packet/'profile.py','--old',binaries['baseline'],'--new',binaries['candidate'],'--model',a.upload/'model.gguf','--eval',a.upload/'eval100.jsonl','--source-old',roots['baseline'],'--source-new',roots['candidate'],'--out',out/'profile'],1800)
  command('independent_raw_replay',['python3',packet/'replay_profile.py',out/'profile'],120)
  for source,root in roots.items():
   for name,h in m['source_files'][source].items():
    if sha(root/name)!=h:raise ValueError('final source drift')
  summary.update(status='PROFILE_COMPLETE',complete=True,scope='Diagnostic only; not a speed/adoption or isolated causal claim',source_commits=m['source_commits'])
 except BaseException as e:summary.update(status='ERROR_OR_INCOMPLETE',complete=False,error=repr(e));(out/'exception.txt').write_text(traceback.format_exc())
 finally:stop.set();thread.join(timeout=5);pack()
 return 0 if summary['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
