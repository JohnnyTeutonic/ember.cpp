"""Real offline rsync transfer with the existing controller's actual flags."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
TRANSPORT=Path('/home/jonat/pk_publication_20261030/tools/colab_chat/runpod_transport.py')

def actual_rsync_argv():
    tree=ast.parse(TRANSPORT.read_text())
    method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='_rsync')
    assignment=next(node for node in method.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='argv' for t in node.targets))
    # Local endpoints deliberately exercise file semantics without SSH/network.
    # The actual -e option is retained; rsync ignores it for two local paths.
    return eval(compile(ast.Expression(assignment.value),str(TRANSPORT),'eval'),{'e':'ssh'})

def test_actual_rsync_skips_symlink_but_resolved_regular_source_copies(tmp_path):
    raw=bytes(range(256))*17+b'actual byte copy\x00\xff'
    original=tmp_path/'canonical.bin';original.write_bytes(raw)
    link=tmp_path/'staged.bin';link.symlink_to(original)
    destination=tmp_path/'received.part00'
    argv=actual_rsync_argv()
    skipped=subprocess.run([*argv,str(link),str(destination)],capture_output=True,text=True)
    assert skipped.returncode==0
    assert not destination.exists()
    assert 'skipping non-regular file' in skipped.stdout+skipped.stderr
    resolved=link.resolve(strict=True)
    assert resolved.is_file() and not resolved.is_symlink()
    copied=subprocess.run([*argv,str(resolved),str(destination)],capture_output=True,text=True)
    assert copied.returncode==0 and destination.is_file() and not destination.is_symlink()
    assert destination.read_bytes()==raw
    receipt=os.environ.get('ORDERED_RSYNC_RECEIPT')
    if receipt:
        path=Path(receipt)
        with path.open('x') as f:
            json.dump({'schema':1,'transport_source_sha256':hashlib.sha256(TRANSPORT.read_bytes()).hexdigest(),'argv_prefix':argv,'network_used':False,'rsync_version':subprocess.check_output(['rsync','--version'],text=True).splitlines()[0],'symlink_attempt':{'returncode':skipped.returncode,'destination_exists':False,'stdout':skipped.stdout,'stderr':skipped.stderr},'resolved_regular_attempt':{'returncode':copied.returncode,'bytes':len(raw),'source_sha256':hashlib.sha256(raw).hexdigest(),'destination_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()},'result':'PASS'},f,indent=2);f.write('\n')
