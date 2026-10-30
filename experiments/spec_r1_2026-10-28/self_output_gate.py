"""E10.2 frozen candidate evaluation: exact IDs, paired per-set speed, optional deterministic rows."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import decode_pick_gate as common
import self_output_pool as pool

LEGS = ('plain', 'corpus_a', 'merged_a', 'merged_b', 'corpus_b')
MODES = ('default', 'det')
MODEL_REQUESTS = {'96f6afdcbbcf1a4cf53ad6c0e451e75301dc4187f726f3c2218600fbe72a8cc9': 19994,
                  '510d288cf10ead4d3902247e3d898b6fa39e1d31d7465d3a8b6820761d6c9178': 8000}


def finite_row_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            if len(chunk) % 4 or not np.isfinite(np.frombuffer(chunk, dtype='<f4')).all():
                raise ValueError('nonfinite/truncated candidate row bytes')
            h.update(chunk)
    return h.hexdigest()


def plan(binary, model, prompts, corpus, merged, journal, recipe, out, rows=False):
    out = Path(out).resolve()
    out.mkdir(exist_ok=False)
    inputs = {k: str(Path(v).resolve()) for k, v in
              (('binary', binary), ('model', model), ('prompts', prompts), ('corpus', corpus),
               ('merged', merged), ('journal', journal), ('recipe', recipe))}
    inputs['client'] = str(common.CLIENT)
    m = {'schema': 1, 'inputs': inputs, 'sha256': {k: common.sha(v) for k, v in inputs.items()},
         'request_keys': common.request_keys(prompts), 'minimum_speedup': 1.03,
         'deterministic_row_gate': rows, 'output': str(out), 'runs': []}
    for mode in MODES:
        for leg in LEGS:
            m['runs'].append(make_run(m, out, mode, leg, False))
    if rows:
        for leg in ('rows_plain', 'rows_plain_repeat', 'rows_merged'):
            m['runs'].append(make_run(m, out, 'det', leg, True))
    validate_recipe(m, inputs)
    pool_dir = Path(inputs['journal']).parent
    pool_manifest = json.loads((pool_dir / 'manifest.json').read_text())
    for key, path in (('pool_manifest', pool_dir / 'manifest.json'),
                      ('pool_summary', pool_dir / 'summary.json'),
                      ('request_pool', Path(pool_manifest['inputs']['pool']))):
        inputs[key] = str(path)
        m['sha256'][key] = common.sha(path)
    validate_pool(m, inputs)
    (out / 'plan.json').write_text(json.dumps(m, indent=1) + '\n')
    return m


def make_run(m, out, mode, leg, dump):
    plain = leg in ('plain', 'rows_plain', 'rows_plain_repeat')
    table = 'merged' if 'merged' in leg else 'corpus'
    env = ['EMBER_DECODE_PICK=0', 'EMBER_SPEC=' + ('0' if plain else '8'),
           'EMBER_DRAFT=table', 'EMBER_DRAFT_TABLE=' + m['inputs'][table], 'EMBER_DRAFT_THETA=0.02',
           'EMBER_DECODE_GRAPH=1', 'EMBER_DECODE_OPT=1', 'EMBER_GEMM_TF32=0',
           'EMBER_DETERMINISTIC=' + ('1' if mode == 'det' else '0'),
           'EMBER_DECODE_LINEAR=' + ('fixed' if mode == 'det' else 'cublas')]
    name = mode + '_' + leg
    command = ['env', '-u', 'TINYLLAMA_LOGITS_DUMP', '-u', 'TINYLLAMA_TOPK_DEBUG',
               '-u', 'EMBER_DECODE_TIMING', 'python3', m['inputs']['client'], 'run',
               m['inputs']['binary'], m['inputs']['model'], '999', '8095', m['inputs']['prompts'],
               str(out / (name + '.jsonl')), *(['--dump'] if dump else []), *env]
    return {'name': name, 'mode': mode, 'leg': leg, 'plain': plain, 'dump': dump,
            'env': env, 'command': command}


def validate_recipe(m, paths):
    r = json.loads(Path(paths['recipe']).read_text())
    if (r.get('orders') != 6 or r.get('min_count') != 2 or r.get('min_conf') != 0.10
            or r.get('corpus_first') is not True or r.get('pool_status') != 'VERIFIED_COMPLETE'):
        raise ValueError('wrong/incomplete self-output recipe')
    for key in ('model', 'corpus', 'merged', 'journal'):
        if r['sha256'].get(key) != m['sha256'][key]:
            raise ValueError('recipe input/output hash differs: ' + key)


def validate_pool(m, paths):
    if m['sha256']['model'] not in MODEL_REQUESTS:
        raise ValueError('not one of the two frozen model checkpoints')
    audit = pool.analyze(Path(paths['journal']).parent,
                         {'binary': paths['binary'], 'model': paths['model'],
                          'pool': paths['request_pool'], 'table': paths['corpus']})
    required = MODEL_REQUESTS[m['sha256']['model']]
    if (audit['status'] != 'VERIFIED_COMPLETE' or audit['requests'] != required
            or audit['completed'] != required or audit['journal_sha256'] != m['sha256']['journal']):
        raise ValueError('complete original self-output pool required; no pilot adoption')
    manifest = json.loads(Path(paths['pool_manifest']).read_text())
    if any(manifest['sha256'][a] != m['sha256'][b] for a, b in
           (('binary', 'binary'), ('model', 'model'), ('table', 'corpus'), ('pool', 'request_pool'))):
        raise ValueError('pool generation inputs differ from evaluation')


def validate_plan(m, out, mapping=None):
    mapping = mapping or {}
    if set(mapping) - set(m['inputs']):
        raise ValueError('unknown input mapping')
    paths = {k: mapping.get(k, v) for k, v in m['inputs'].items()}
    slots = [(mode, leg, False) for mode in MODES for leg in LEGS]
    if m.get('deterministic_row_gate'):
        slots += [('det', leg, True) for leg in ('rows_plain', 'rows_plain_repeat', 'rows_merged')]
    if (m.get('schema') != 1 or m.get('minimum_speedup') != 1.03
            or type(m.get('deterministic_row_gate')) is not bool
            or [r['name'] for r in m['runs']] != [mode + '_' + leg for mode, leg, dump in slots]):
        raise ValueError('altered/incomplete candidate plan')
    if any(common.sha(path) != m['sha256'][key] for key, path in paths.items()):
        raise ValueError('frozen input bytes changed')
    if common.request_keys(paths['prompts']) != m['request_keys']:
        raise ValueError('altered request inventory')
    validate_recipe(m, paths)
    validate_pool(m, paths)
    for run, (mode, leg, dump) in zip(m['runs'], slots):
        expected_run = make_run(m, Path(m['output']), mode, leg, dump)
        if run != expected_run:
            raise ValueError('altered run environment')
    return paths


def load(out, run, m):
    path = Path(out) / (run['name'] + '.jsonl')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    if [[r['i'], r['turn']] for r in records] != m['request_keys']:
        raise ValueError('missing/reordered candidate requests')
    if (meta.get('env') != run['env'] or meta.get('once') is not False
            or int(meta['gpu_layers']) != 999):
        raise ValueError('wrong candidate environment/backend/warm protocol')
    for field, key in (('bin', 'binary'), ('gguf', 'model'), ('prompts', 'prompts')):
        if meta.get(field) != m['inputs'][key]:
            raise ValueError('wrong candidate input path')
    health = meta.get('health') or {}
    if health.get('model') != m['inputs']['model'] or health.get('deterministic') != (run['mode'] == 'det'):
        raise ValueError('wrong candidate model/mode')
    offset = 0
    for r in records:
        if not r.get('prompt_ids') or not isinstance(r.get('ids'), list) or r.get('warm_pass_same') is not True:
            raise ValueError('invalid candidate IDs/warm pass')
        if any(type(r.get(k)) is not int or r[k] < 0 for k in ('passes', 'drafted', 'accepted')):
            raise ValueError('invalid speculation counts')
        if r['accepted'] > r['drafted'] or (run['plain'] and any(r[k] for k in ('passes', 'drafted', 'accepted'))):
            raise ValueError('wrong plain/speculative path')
        if run['dump']:
            if r.get('row0') != offset or type(r.get('rows')) is not int or r['rows'] <= 0:
                raise ValueError('invalid candidate dump offsets')
            offset += r['rows']
    if not run['plain'] and sum(r['passes'] for r in records) == 0:
        raise ValueError('speculation never used')
    if run['dump']:
        vocab = meta.get('vocab')
        if (type(vocab) is not int or vocab <= 0 or meta.get('dump_rows') != offset
                or Path(str(path) + '.rows.f32').stat().st_size != offset * vocab * 4):
            raise ValueError('truncated/extra candidate rows')
    elif meta.get('dump_rows') is not None:
        raise ValueError('candidate timing contaminated by dumping')
    hardware = json.loads(Path(str(path) + '.hardware.json').read_text())
    if hardware.get('name') != 'NVIDIA L4' or not hardware.get('uuid'):
        raise ValueError('missing/wrong timing device')
    return records, meta, hardware


def analyze(out, mapping=None):
    out = Path(out)
    m = json.loads((out / 'plan.json').read_text())
    paths = validate_plan(m, out, mapping)
    loaded = {run['name']: load(out, run, m) for run in m['runs']}
    prompt_rows = [json.loads(line) for line in Path(paths['prompts']).read_text().splitlines() if line.strip()]
    expected_sets = [prompt_rows[i]['set'] for i, turn in m['request_keys']]
    if any([r.get('set') for r in rows] != expected_sets for rows, meta, hardware in loaded.values()):
        raise ValueError('wrong dataset assignment')
    result = {'status': 'PASS', 'modes': {}, 'scope': 'self-output candidate; deterministic mode explicitly fixed; no default change'}
    for mode in MODES:
        runs = [run for run in m['runs'] if run['mode'] == mode]
        base = loaded[mode + '_plain'][0]
        if any(common.ids(loaded[run['name']][0]) != common.ids(base) for run in runs):
            result['status'] = 'FAIL'
            result['modes'][mode] = {'reason': 'plain/corpus/merged IDs or replies differ'}
            continue
        ratios, devices = {}, []
        for pair in ('a', 'b'):
            corpus = loaded[mode + '_corpus_' + pair]
            merged = loaded[mode + '_merged_' + pair]
            devices.extend((corpus[2]['uuid'], merged[2]['uuid']))
            for dataset in ('chat40', 'carry40'):
                a = [r for r in corpus[0] if r['set'] == dataset]
                b = [r for r in merged[0] if r['set'] == dataset]
                ratios.setdefault(dataset, []).append(common.speed(b) / common.speed(a))
        device_ok = len(set(devices)) == 1
        gain = all(value >= m['minimum_speedup'] for values in ratios.values() for value in values)
        result['modes'][mode] = {'requests': len(base), 'ids_equal': True,
                                 'paired_speedups': ratios, 'same_device_all_timing_legs': device_ok,
                                 'throughput_pass': gain and device_ok}
        if not device_ok and result['status'] != 'FAIL':
            result['status'] = 'INCONCLUSIVE_TIMING'
        elif not gain and result['status'] == 'PASS':
            result['status'] = 'NO_GAIN'
    if m['deterministic_row_gate']:
        files = [out / ('det_rows_' + leg + '.jsonl.rows.f32') for leg in ('plain', 'plain_repeat', 'merged')]
        hashes = [finite_row_sha(path) for path in files]
        same_shapes = len({(loaded['det_rows_' + leg][1]['vocab'],
                           tuple(r['rows'] for r in loaded['det_rows_' + leg][0]))
                          for leg in ('plain', 'plain_repeat', 'merged')}) == 1
        result['deterministic_rows'] = {'baseline_repeat_bits_equal': hashes[0] == hashes[1],
                                         'merged_bits_equal': same_shapes and hashes[0] == hashes[2], 'sha256': hashes}
        if hashes[0] != hashes[1] and result['status'] != 'FAIL':
            result['status'] = 'BASELINE_UNSTABLE'
        elif not same_shapes or hashes[0] != hashes[2]:
            result['status'] = 'FAIL'
    return result


def run(out):
    out = Path(out)
    m = json.loads((out / 'plan.json').read_text())
    validate_plan(m, out)
    for leg in m['runs']:
        output = out / (leg['name'] + '.jsonl')
        if Path(str(output) + '.meta.json').exists():
            load(out, leg, m)  # Complete invalid legs stop, never retry for a pass.
            continue
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
        if len(gpu) != 1:
            raise ValueError('expected one GPU')
        name, uuid = (s.strip() for s in gpu[0].split(','))
        Path(str(output) + '.hardware.json').write_text(json.dumps({'name': name, 'uuid': uuid}) + '\n')
        subprocess.run(leg['command'], check=True)
        load(out, leg, m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('plan')
    for name in ('binary', 'model', 'prompts', 'corpus', 'merged', 'journal', 'recipe', 'out'):
        q.add_argument(name)
    q.add_argument('--rows', action='store_true')
    sub.add_parser('run').add_argument('out')
    q = sub.add_parser('analyze')
    q.add_argument('out')
    q.add_argument('--input-map')
    a = p.parse_args()
    if a.action == 'plan':
        plan(a.binary, a.model, a.prompts, a.corpus, a.merged, a.journal, a.recipe, a.out, a.rows)
    elif a.action == 'run':
        run(a.out)
    else:
        result = analyze(a.out, json.loads(Path(a.input_map).read_text()) if a.input_map else None)
        print(json.dumps(result, indent=1))
        return 0 if result['status'] == 'PASS' else 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'INCOMPLETE_OR_INVALID', 'error': str(exc)}))
        sys.exit(2)
