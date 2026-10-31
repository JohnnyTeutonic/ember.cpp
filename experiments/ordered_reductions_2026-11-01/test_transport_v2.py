"""Execute the existing runner's actual generated reassembly code, offline."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).parent
RUNNER=Path('/home/jonat/pk_publication_20261030/tools/colab_chat/runner_multi.py')
spec=importlib.util.spec_from_file_location('vm_v2',HERE/'ordered_vm.py');vm=importlib.util.module_from_spec(spec);spec.loader.exec_module(vm)

def actual_reassembly(names,directory):
    tree=ast.parse(RUNNER.read_text())
    expressions=[node.args[0] for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='exec_py' and node.args and 'parts = sorted(p for p in os.listdir(d)' in ast.unparse(node.args[0])]
    assert len(expressions)==1
    script=eval(compile(ast.Expression(expressions[0]),str(RUNNER),'eval'),{'names':names,'moves':[]})
    script=script.replace("'/content/chat/upload'",repr(str(directory)))
    subprocess.run([sys.executable,'-c',script],check=True,capture_output=True)

def test_actual_runner_reproduces_nested_loss_and_flat_reassembles(tmp_path):
    nested=tmp_path/'nested';nested.mkdir();(nested/'probe').mkdir();(nested/'probe/kernel_probe.cu.part00').write_bytes(b'original diagnostic')
    actual_reassembly(['probe/kernel_probe.cu'],nested)
    assert not (nested/'probe/kernel_probe.cu').exists()
    assert (nested/'probe/kernel_probe.cu.part00').read_bytes()==b'original diagnostic'
    flat=tmp_path/'flat';flat.mkdir()
    for name in ['probe_CMakeLists.txt','probe_kernel_probe.cu','probe_semantic_check.h']:
        (flat/(name+'.part00')).write_bytes(name.encode()+b' first')
        (flat/(name+'.part01')).write_bytes(b' second')
    names=['probe_CMakeLists.txt','probe_kernel_probe.cu','probe_semantic_check.h']
    actual_reassembly(names,flat)
    for name in names:assert (flat/name).read_bytes()==name.encode()+b' first second'
    assert sorted(p.name for p in flat.iterdir())==sorted(names)
    work=tmp_path/'work';work.mkdir();probe=vm.prepare_probe(flat,work)
    for name in ['CMakeLists.txt','kernel_probe.cu','semantic_check.h']:assert (probe/name).read_bytes()==(flat/('probe_'+name)).read_bytes()

def test_missing_input_records_actual_filename_and_traceback(tmp_path):
    upload=tmp_path/'upload';upload.mkdir();manifest=upload/'inputs.json';manifest.write_text(json.dumps({'files':{'missing_probe.txt':{'bytes':2,'sha256':'bad'}}}))
    sha=hashlib.sha256(manifest.read_bytes()).hexdigest();out=tmp_path/'out'
    result=subprocess.run([sys.executable,str(HERE/'ordered_vm.py'),'--upload',str(upload),'--manifest-sha256',sha,'--out',str(out)],capture_output=True)
    summary=json.loads((out/'summary.json').read_text())
    assert result.returncode==2 and summary['complete'] is False
    assert summary['filename']==str(upload/'missing_probe.txt')
    assert 'FileNotFoundError' in (out/'science/exception.txt').read_text()
    assert (out/'out.zip').is_file()
