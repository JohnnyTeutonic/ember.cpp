from pathlib import Path
import subprocess,hashlib,json,datetime
root=Path('/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete');out=Path('/mnt/c/ml_artifacts/benchmarks/hf_cpu_20261030');binary=Path('/home/jonat/ember_ir_build/ir_adapter');report=out/'avx2_ir_conformance_bound.json';log=out/'avx2_ir_conformance_bound.log';receipt=out/'avx2_ir_conformance_launch.json'
assert not any(p.exists() for p in (report,log,receipt))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[binary,*[root/'ember.cpp'/x for x in ['cpu_dot.h','utils.cpp','CMakeLists.txt','graph_ir.cpp','graph_ir_ops.cpp','graph_ir_ops.h','tools/ir_adapter.cpp']]]
before={str(p):sha(p) for p in files}
cmd=['python3',str(root/'scripts/cpu_budget.py'),'run','--session','ember','--job','bound AVX2 IR conformance','--cores','1','--resume','preserve prior receipt; a repeat uses fresh filename','--','python3',str(root/'keramos/tools/ir_conformance.py'),'--adapter','cmd:'+str(binary),'--json',str(report)]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with log.open('w') as f:r=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
after={str(p):sha(p) for p in files};assert before==after
counts=json.loads(report.read_text())['counts']
record={'command':cmd,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'binary_and_sources_before_sha256':before,'binary_and_sources_after_sha256':after,'log_sha256':sha(log),'report_sha256':sha(report),'counts':counts,'launcher_sha256':sha(__file__)}
tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(receipt)
assert r.returncode==0 and counts['FAIL']==0
print('bound conformance PASS',counts)