"""E10.1 complete CPU/cublas/fixed row gate; never silently drops requests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent / 'draft_2026-10-27/dr_e2e.py'
MODELS = ('pick', 'pk')
LEGS = ('cpu', 'cublas', 'fixed')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def keys(prompts):
    rows = [json.loads(line) for line in Path(prompts).read_text().splitlines() if line.strip()]
    return [[i, turn] for i, row in enumerate(rows) for turn in
            (range(1, len(row['turns']) + 1) if 'turns' in row else [0])]


def plan(binary, pick, pk, prompts, output):
    out = Path(output).resolve()
    out.mkdir(exist_ok=False)
    inputs = {k: str(Path(v).resolve()) for k, v in
              (('binary', binary), ('pick', pick), ('pk', pk), ('prompts', prompts))}
    manifest = {'schema': 1, 'inputs': inputs, 'sha256': {k: sha(v) for k, v in inputs.items()},
                'request_keys': keys(prompts), 'rounding': 1e-4, 'runs': []}
    if not manifest['request_keys']:
        raise ValueError('empty requests')
    for model in MODELS:
        for leg in LEGS:
            env = ['EMBER_DECODE_PICK=0', 'EMBER_SPEC=0', 'EMBER_DECODE_GRAPH=1',
                   'EMBER_DECODE_OPT=1', 'EMBER_DETERMINISTIC=1', 'EMBER_GEMM_TF32=0',
                   'EMBER_DECODE_LINEAR=' + ('fixed' if leg == 'fixed' else 'cublas')]
            name = model + '_' + leg
            gpu = 0 if leg == 'cpu' else 999
            command = ['env', '-u', 'TINYLLAMA_LOGITS_DUMP', '-u', 'TINYLLAMA_TOPK_DEBUG',
                       '-u', 'EMBER_DECODE_TIMING', 'python3', str(CLIENT), 'run', inputs['binary'],
                       inputs[model], str(gpu), '8091', inputs['prompts'], str(out / (name + '.jsonl')),
                       '--dump', *env]
            manifest['runs'].append({'name': name, 'model': model, 'leg': leg,
                                     'gpu_layers': gpu, 'env': env, 'command': command})
    (out / 'plan.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def load(out, run, manifest):
    path = Path(out) / (run['name'] + '.jsonl')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    if [[r['i'], r['turn']] for r in records] != manifest['request_keys']:
        raise ValueError(run['name'] + ': incomplete/reordered requests')
    if meta.get('env') != run['env'] or meta.get('once') is not False:
        raise ValueError(run['name'] + ': wrong environment/warm protocol')
    if int(meta['gpu_layers']) != run['gpu_layers']:
        raise ValueError(run['name'] + ': wrong backend')
    for field, key in (('bin', 'binary'), ('gguf', run['model']), ('prompts', 'prompts')):
        if meta.get(field) != manifest['inputs'][key]:
            raise ValueError(run['name'] + ': input path mismatch')
    health = meta.get('health') or {}
    if health.get('model') != manifest['inputs'][run['model']] or health.get('deterministic') is not True:
        raise ValueError(run['name'] + ': wrong model/deterministic mode')
    offset = 0
    for r in records:
        if (r.get('warm_pass_same') is not True or not r.get('prompt_ids')
                or not isinstance(r.get('ids'), list) or r.get('rows', 0) <= 0
                or r.get('row0') != offset
                or any(r.get(k, 0) != 0 for k in ('passes', 'accepted', 'drafted'))):
            raise ValueError(run['name'] + ': invalid warm/IDs/dump/speculation')
        offset += r['rows']
    vocab = meta.get('vocab')
    raw = Path(str(path) + '.rows.f32')
    if (not isinstance(vocab, int) or vocab <= 0 or meta.get('dump_rows') != offset
            or raw.stat().st_size != offset * vocab * 4):
        raise ValueError(run['name'] + ': truncated/extra rows')
    return records, np.memmap(raw, dtype=np.float32, mode='r', shape=(offset, vocab))


def identities(records):
    return [(r['reply'], r['ids'], r['prompt_ids'], r['stop_id'], r['tokens'], r['rows']) for r in records]


def compare(reference, cublas, fixed, limit):
    rr, xr = reference
    cr, xc = cublas
    fr, xf = fixed
    if identities(rr) != identities(cr) or identities(rr) != identities(fr):
        return {'pass': False, 'reason': 'CPU/cublas/fixed request identities differ; no rows omitted'}
    if xr.shape != xc.shape or xr.shape != xf.shape:
        raise ValueError('row shape mismatch')
    rows = []
    for r in rr:
        for j in range(r['rows']):
            pos = r['row0'] + j
            a, b, c = (np.asarray(x[pos], dtype=np.float64) for x in (xr, xc, xf))
            if not all(np.isfinite(x).all() for x in (a, b, c)):
                raise ValueError('nonfinite row')
            scale = max(1.0, float(np.abs(a).max()))
            blas_error = float(np.abs(a - b).max())
            fixed_error = float(np.abs(a - c).max())
            rows.append({'request': r['i'], 'turn': r['turn'], 'row': j,
                         'scale': scale, 'cublas_abs': blas_error, 'fixed_abs': fixed_error,
                         'fixed_within_rounding': fixed_error <= limit * scale,
                         'fixed_no_farther': fixed_error <= blas_error})
    bad = [r for r in rows if not r['fixed_within_rounding'] or not r['fixed_no_farther']]
    return {'pass': not bad, 'requests': len(rr), 'rows': len(rows), 'failed_rows': len(bad),
            'worst_fixed_relative': max(r['fixed_abs'] / r['scale'] for r in rows),
            'worst_fixed_minus_cublas': max(r['fixed_abs'] - r['cublas_abs'] for r in rows),
            'row_errors': rows}


def validate_plan(m, input_map=None):
    if (m.get('schema') != 1 or m.get('rounding') != 1e-4 or not m.get('request_keys')
            or [r['name'] for r in m['runs']] != [model + '_' + leg for model in MODELS for leg in LEGS]):
        raise ValueError('altered/incomplete plan')
    mapping = input_map or {}
    if set(mapping) - set(m['inputs']):
        raise ValueError('unknown input mapping')
    paths = {key: mapping.get(key, path) for key, path in m['inputs'].items()}
    if any(sha(path) != m['sha256'][key] for key, path in paths.items()):
        raise ValueError('frozen input changed')
    if m['request_keys'] != keys(paths['prompts']):
        raise ValueError('request inventory differs from frozen prompts')
    for run, (model, leg) in zip(m['runs'], ((model, leg) for model in MODELS for leg in LEGS)):
        if run['model'] != model or run['leg'] != leg or run['gpu_layers'] != (0 if leg == 'cpu' else 999):
            raise ValueError('altered model/leg/backend assignment')
        expected = ['EMBER_DECODE_PICK=0', 'EMBER_SPEC=0', 'EMBER_DECODE_GRAPH=1',
                    'EMBER_DECODE_OPT=1', 'EMBER_DETERMINISTIC=1', 'EMBER_GEMM_TF32=0',
                    'EMBER_DECODE_LINEAR=' + ('fixed' if leg == 'fixed' else 'cublas')]
        if run['env'] != expected:
            raise ValueError('altered gate environment')


def analyze(out, input_map=None):
    out = Path(out)
    m = json.loads((out / 'plan.json').read_text())
    validate_plan(m, input_map)
    results = {}
    for model in MODELS:
        loaded = [load(out, r, m) for r in m['runs'] if r['model'] == model]
        results[model] = compare(*loaded, m['rounding'])
    return {'status': 'PASS' if all(r['pass'] for r in results.values()) else 'FAIL',
            'scope': 'E10.1 complete CPU-reference row gate, both models; no default change', 'models': results}


def run_plan(out):
    out = Path(out)
    m = json.loads((out / 'plan.json').read_text())
    validate_plan(m)
    for run in m['runs']:
        if (out / (run['name'] + '.jsonl.meta.json')).exists():
            load(out, run, m)  # complete invalid legs stop; never retry for a pass
        else:
            subprocess.run(run['command'], check=True)
            load(out, run, m)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('plan')
    for name in ('binary', 'pick', 'pk', 'prompts', 'output'):
        q.add_argument(name)
    sub.add_parser('run').add_argument('output')
    q = sub.add_parser('analyze')
    q.add_argument('output')
    q.add_argument('--input-map')
    args = p.parse_args()
    try:
        if args.action == 'plan':
            plan(args.binary, args.pick, args.pk, args.prompts, args.output)
        elif args.action == 'run':
            run_plan(args.output)
        else:
            result = analyze(args.output, json.loads(Path(args.input_map).read_text()) if args.input_map else None)
            print(json.dumps(result, indent=1))
            sys.exit(0 if result['status'] == 'PASS' else 1)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'INCOMPLETE_OR_INVALID', 'error': str(exc)}))
        sys.exit(2)
