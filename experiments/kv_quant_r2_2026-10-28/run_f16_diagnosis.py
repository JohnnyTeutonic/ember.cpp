"""Fixed E9 diagnostic inventory; preserves valid negative controls and raw traces."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from f16_trace_analysis import analyze, inventory, sha

MODEL_SHA = '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031'


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    temporary.replace(path)


def completed_leg(folder, command, environment, accepted):
    identity = {'command': command, 'environment': environment}
    receipt = folder / 'leg.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] != identity or saved['files'] != inventory_without_receipt(folder):
            raise ValueError('completed diagnostic leg drift')
        return saved
    if folder.exists():
        raise ValueError('incomplete leg retained; operational review required')
    folder.mkdir()
    start = time.time()
    with (folder/'stdout.json').open('wb') as stdout, (folder/'stderr.log').open('wb') as stderr:
        result = subprocess.run(command, env=dict(os.environ, **environment), stdout=stdout, stderr=stderr)
    if result.returncode not in accepted:
        raise ValueError(f'operational failure {folder.name}: exit {result.returncode}; retained')
    value = json.loads((folder/'stdout.json').read_text())
    saved = {'identity': identity, 'returncode': result.returncode, 'result': value,
             'elapsed_seconds': time.time()-start, 'files': inventory_without_receipt(folder)}
    atomic(receipt, saved)
    return saved


def inventory_without_receipt(folder):
    return {name: value for name, value in inventory(folder).items() if name != 'leg.json'}


def execute(model, normal, trace, component, seeded_ids, ids, out):
    if (out/'diagnosis_report.json').exists():
        raise FileExistsError('completed diagnostic report is immutable')
    paths = [model, normal, trace, component, seeded_ids, ids, Path(__file__),
             Path(__file__).with_name('f16_trace_analysis.py')]
    before = {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}
    if before[str(model)]['sha256'] != MODEL_SHA:
        raise ValueError('canonical Qwen3 model mismatch before forward')
    sequences = json.loads(ids.read_text())
    if len(sequences) != 12 or any(len(s) != 24 for s in sequences):
        raise ValueError('frozen prompt inventory must be all 12x24')
    environment = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'EMBER_IR_BUILD': '1', 'EMBER_IR_CTX': '8448', 'EMBER_IR_GRAPH': '0',
                   'EMBER_KV_TRACE_MODEL_SHA256': MODEL_SHA}
    # Reject ambient overrides: controls must use the committed automatic IR/numerical recipe.
    overrides = [key for key in os.environ if key.startswith('EMBER_')
                 and key not in ('EMBER_IR_BUILD', 'EMBER_IR_CTX')]
    if any(os.environ.get(key) for key in overrides):
        raise ValueError('ambient IR/trace override is forbidden')
    regenerated = subprocess.run([str(seeded_ids), str(model)], env=dict(os.environ, **environment),
                                 check=True, capture_output=True)
    if regenerated.stdout != ids.read_bytes():
        raise ValueError('libstdc++ seeded recipe differs from frozen IDs')
    out.mkdir(exist_ok=True)
    manifest = out/'diagnosis_inputs.json'
    fixed = {'schema': 1, 'inputs': before, 'ids': sequences, 'registered_tolerance': 1e-4,
             'scope': 'Qwen3 f16 numerical diagnosis; not E9 adoption gate completion'}
    if manifest.exists() and json.loads(manifest.read_text()) != fixed:
        raise ValueError('diagnostic manifest drift')
    if not manifest.exists():
        atomic(manifest, fixed)
    probe = completed_leg(out/'component', [str(component)], environment, (0, 1))
    if probe['result'].get('complete') is not True or probe['result'].get('diagnostic_only') is not True:
        raise ValueError('component result incomplete')
    controls = {}
    for fmt in ('f32', 'f16'):
        for graph in ('0', '1'):
            env = dict(environment, EMBER_KV=fmt, EMBER_IR_GRAPH=graph)
            command = [str(normal), str(model), '--n', '12', '--len', '24', '--gen', '32', '--seed', '0', '--tol', '0.0001']
            leg = completed_leg(out/f'control_{fmt}_graph{graph}', command, env, (0, 1))
            result = leg['result']
            if (result.get('on_gpu') is not True or result.get('prompts') != 12
                    or result.get('tol') != 1e-4 or result.get('len') != 24 or result.get('gen') != 32
                    or not str(result.get('top1_agree', '')).endswith('/288')
                    or not str(result.get('greedy_identical', '')).endswith('/12')
                    or result.get('positions') not in (None, 288)
                    or result.get('pass') is not (leg['returncode'] == 0)):
                raise ValueError('invalid control contract')
            controls[f'{fmt}_graph{graph}'] = result
    # Trace tool resumes only independently hashed complete cases, refusing incomplete case folders.
    for fmt in ('f32', 'f16'):
        for backend in ('cpu', 'cuda'):
            folder = out/f'{fmt}_{backend}'
            command = [str(trace), str(model), str(ids), str(folder), backend]
            result = subprocess.run(command, env=dict(os.environ, **environment, EMBER_KV=fmt),
                                    capture_output=True)
            (out/f'{fmt}_{backend}.stdout').write_bytes(result.stdout)
            (out/f'{fmt}_{backend}.stderr').write_bytes(result.stderr)
            if result.returncode:
                raise ValueError(f'trace {fmt}_{backend} failed; original case data retained')
    report = analyze(out)
    report['controls'] = controls
    report['component'] = probe['result']
    after = {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}
    if before != after:
        raise ValueError('final model/binary/controller/input integrity differs')
    report['final_input_integrity'] = after
    atomic(out/'diagnosis_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('model', 'normal', 'trace', 'component', 'seeded-ids', 'ids', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = execute(args.model, args.normal, args.trace, args.component, args.seeded_ids, args.ids, args.out)
        print(json.dumps({'status': report['status'], 'e9_closed': False}))
    except Exception as error:
        args.out.mkdir(exist_ok=True)
        atomic(args.out/'HALTED', {'reason': str(error), 'complete': False, 'e9_closed': False})
        raise