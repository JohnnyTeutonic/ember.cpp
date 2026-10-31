import importlib.util
import json
import subprocess
from pathlib import Path
import pytest
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('vm',HERE/'ordered_vm.py');vm=importlib.util.module_from_spec(spec);spec.loader.exec_module(vm)

def test_environment_clears_override_and_default_workspace():
 env=vm.clean_env({'PATH':'x','EMBER_IR':'0','EMBER_DETERMINISTIC':'1','EMBER_DECODE_PICK_PARTS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','TINYLLAMA_LOGITS_DUMP':'x'})
 assert env['PATH']=='x' and env['EMBER_DECODE_PICK_PARTS']=='0'
 assert 'EMBER_IR' not in env and 'EMBER_DETERMINISTIC' not in env and 'CUBLAS_WORKSPACE_CONFIG' not in env and 'TINYLLAMA_LOGITS_DUMP' not in env

def test_matrix_exact_and_abba():
 p=json.loads((HERE/'protocol.json').read_text());s=p['schedule']
 assert len(s)==24 and len(set(map(tuple,s)))==24
 assert sum(x[0]=='candidate' for x in s)==14 and sum(x[0]=='baseline' for x in s)==10
 for source in ['candidate','baseline']:
  for mode in ['default','det']:
   assert [name.removeprefix(mode+'_') for src,name in s if src==source and name.startswith(mode+'_') and not name.startswith(mode+'_rows')]==['off_a','on_a','on_b','off_b']

def test_nan_semantic_negative(tmp_path):
 src=tmp_path/'check.cpp';src.write_text('#include "semantic_check.h"\n#include <limits>\nint main(){try{checked_error(std::numeric_limits<double>::quiet_NaN(),0);}catch(...){return checked_error(1,1)==0?0:2;}return 1;}')
 subprocess.run(['g++','-std=c++17','-DNDEBUG','-I'+str(HERE),str(src),'-o',str(tmp_path/'check')],check=True)
 subprocess.run([str(tmp_path/'check')],check=True)

def test_refuses_existing_evidence(tmp_path):
 out=tmp_path/'out';out.mkdir();old=out/'summary.json';old.write_text('negative evidence')
 p=subprocess.run(['python3',str(HERE/'ordered_vm.py'),'--upload',str(tmp_path),'--manifest-sha256','wrong','--out',str(out)],capture_output=True)
 assert p.returncode!=0 and old.read_text()=='negative evidence' and len(list(out.iterdir()))==1

def test_server_and_capture_contract():
 s=(HERE/'ordered_vm.py').read_text();probe=(HERE/'kernel_probe.cu').read_text()
 assert "'tinyllama_server','ordered_probe'" in s and "build/'ember/tinyllama_server'" in s
 assert 'cudaStreamNonBlocking' in probe and 'cudaStreamBeginCapture(stream,' in probe
 assert 'checked_error(out[i],reference[i])' in probe and '"first_output",first' in probe
