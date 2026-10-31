"""Actual CPU constructor negative/positive check; never a GPU proof."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import vm


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(loader, model, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    cases = []
    for name, env in [('v1_path_zero', dict(vm.runtime_environment({}), EMBER_IR='0')),
                      ('v2_cleared_inherited_zero', vm.runtime_environment({'EMBER_IR': '0'}))]:
        result = subprocess.run([str(loader), str(model)], env=env, capture_output=True, timeout=30)
        (output / (name + '.stdout')).write_bytes(result.stdout)
        (output / (name + '.stderr')).write_bytes(result.stderr)
        cases.append({'name': name, 'returncode': result.returncode,
                      'stdout': result.stdout.decode(), 'stderr': result.stderr.decode(),
                      'EMBER_IR_present': 'EMBER_IR' in env})
    if cases[0]['returncode'] != 1 or 'EMBER_IR: cannot read 0' not in cases[0]['stderr']:
        raise ValueError('actual production negative did not reproduce pathname failure')
    if cases[1]['returncode'] != 0 or 'LOADED bpe=1' not in cases[1]['stdout']:
        raise ValueError('corrected environment did not load actual model')
    receipt = {'scope': 'Actual existing CPU production API constructor only; no CUDA proof',
               'source_commit': vm.check.SOURCE, 'loader_sha256': sha(loader), 'model_sha256': sha(model),
               'cases': cases, 'status': 'PASS_PATHNAME_ENVIRONMENT_REGRESSION'}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('loader',type=Path);p.add_argument('model',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(run(a.loader.resolve(),a.model.resolve(),a.output.resolve()),indent=2))
