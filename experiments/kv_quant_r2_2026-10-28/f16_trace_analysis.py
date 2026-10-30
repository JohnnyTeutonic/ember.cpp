"""Read-only E9 stage diagnosis. No adoption verdict or tolerance revision is made here."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(folder):
    return {str(p.relative_to(folder)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'case.json'}


def load_forward(folder, layers, backend, fmt, tokens, start):
    if type(layers) is not int or layers <= 0:
        raise ValueError('layer count must be positive')
    rows = [json.loads(line) for line in (folder / 'trace.jsonl').read_text().splitlines()]
    if [r.get('layer') for r in rows] != list(range(layers)):
        raise ValueError('missing/reordered/duplicate layer trace')
    payload = np.memmap(folder / 'payload.bin', dtype=np.uint8, mode='r')
    offset = 0
    result = []
    for row in rows:
        if (row.get('schema') != 1 or row.get('complete') is not True or row.get('backend') != backend
                or row.get('format') != fmt or row.get('T') != tokens or row.get('start_pos') != start):
            raise ValueError('trace identity differs')
        hq, hkv, hd = (row[k] for k in ('hq', 'hkv', 'hd'))
        if any(type(v) is not int or v <= 0 for v in (hq, hkv, hd)) or hq % hkv:
            raise ValueError('invalid trace geometry')
        expected = {k: (tokens * (hq if k in ('q', 'attention') else hkv) * hd, 'f32')
                    for k in ('q', 'k', 'v', 'k_cache', 'v_cache', 'attention')}
        if fmt == 'f16':
            expected.update({k: (tokens * hkv * hd, 'u16') for k in ('k_codes', 'v_codes')})
        if set(row['arrays']) != set(expected):
            raise ValueError('missing/extra trace stage')
        arrays = {}
        ordered = sorted(row['arrays'].items(), key=lambda item: item[1]['offset'])
        for name, spec in ordered:
            count, dtype = expected[name]
            width = 4 if dtype == 'f32' else 2
            if (spec != {'offset': offset, 'bytes': count * width, 'count': count, 'dtype': dtype}
                    or offset + count * width > payload.size):
                raise ValueError('truncated/overlapping/altered trace payload')
            values = np.frombuffer(payload, dtype='<f4' if dtype == 'f32' else '<u2',
                                   count=count, offset=offset).reshape(tokens, -1)
            if dtype == 'f32' and not np.isfinite(values).all():
                raise ValueError('nonfinite trace value')
            arrays[name] = values
            offset += count * width
        result.append({'meta': row, 'arrays': arrays})
    if offset != payload.size:
        raise ValueError('extra trailing payload bytes')
    return result


def difference(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('invalid comparison geometry/values')
    delta = np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)), axis=1)
    span = np.ptp(reference.astype(np.float32), axis=1).astype(np.float64)
    relative = np.divide(delta, span, out=delta.copy(), where=span > 0)
    return {'rows': actual.shape[0], 'worst_absolute': float(delta.max()),
            'worst_relative': float(relative.max()),
            'top1_equal': int(np.count_nonzero(actual.argmax(1) == reference.argmax(1)))}


def merge_metric(target, value):
    for key, item in value.items():
        target[key] = target.get(key, 0) + item if key in ('rows', 'top1_equal') else max(target.get(key, 0), item)


def check_storage(record):
    a, fmt = record['arrays'], record['meta']['format']
    result = {}
    for name in ('k', 'v'):
        raw, cache = a[name], a[name + '_cache']
        if fmt == 'f16':
            codes = raw.astype(np.float16).view(np.uint16)
            saved = a[name + '_codes']
            result[name + '_conversion_mismatches'] = int(np.count_nonzero(codes != saved))
            expanded = saved.view(np.float16).astype(np.float32)
            result[name + '_readback_mismatches'] = int(np.count_nonzero(expanded.view(np.uint32) != cache.view(np.uint32)))
        else:
            result[name + '_store_mismatches'] = int(np.count_nonzero(raw.view(np.uint32) != cache.view(np.uint32)))
    return result


def compare_stages(target, left, right, example_identity):
    if not left or len(left) != len(right):
        raise ValueError('empty/unequal stage comparison')
    for layer, (a, b) in enumerate(zip(left, right)):
        if any(a['meta'][k] != b['meta'][k] for k in ('hq', 'hkv', 'hd', 'scale')):
            raise ValueError('stage comparison geometry drift')
        output = target.setdefault(str(layer), {'stages': {}, 'half_code_differences': 0, 'examples': []})
        for name in ('q', 'k', 'v', 'k_cache', 'v_cache', 'attention'):
            merge_metric(output['stages'].setdefault(name, {}), difference(a['arrays'][name], b['arrays'][name]))
        if a['meta']['format'] == 'f16':
            for name in ('k', 'v'):
                aa, bb = a['arrays'], b['arrays']
                changed = np.flatnonzero((aa[name + '_codes'] != bb[name + '_codes']).reshape(-1))
                output['half_code_differences'] += int(changed.size)
                if changed.size and len(output['examples']) < 2:
                    i = int(changed[0])
                    x, y = float(aa[name].flat[i]), float(bb[name].flat[i])
                    hx, hy = float(aa[name + '_cache'].flat[i]), float(bb[name + '_cache'].flat[i])
                    middle = (hx + hy) / 2
                    def half_order(code):
                        code = int(code)
                        return 0x8000 - (code & 0x7fff) if code & 0x8000 else 0x8000 + code
                    adjacent = abs(half_order(aa[name + '_codes'].flat[i])
                                   - half_order(bb[name + '_codes'].flat[i])) == 1
                    output['examples'].append(dict(example_identity, kind=name, index=i, left_raw=x,
                        right_raw=y, left_rounded=hx, right_rounded=hy, raw_gap=abs(x-y), rounded_gap=abs(hx-hy),
                        amplification=abs(hx-hy)/abs(x-y) if x != y else None,
                        adjacent_half_values=adjacent,
                        midpoint_is_single_rounding_boundary=adjacent,
                        opposite_sides_of_rounded_midpoint=min(x,y) <= middle <= max(x,y)))


def one_row(records, index):
    return [{'meta': row['meta'], 'arrays': {k: v[index:index+1] for k, v in row['arrays'].items()}}
            for row in records]


def analyze(root):
    root = Path(root)
    runs = {(fmt, backend): root / (fmt + '_' + backend) for fmt in ('f32', 'f16') for backend in ('cpu', 'cuda')}
    manifests = {key: json.loads((path / 'manifest.json').read_text()) for key, path in runs.items()}
    first = manifests['f16', 'cpu']
    ids, vocab = first['ids'], first['vocab']
    layers = json.loads((runs['f16', 'cpu'] / 'ir.json').read_text())['block']['repeat']
    if type(layers) is not int or layers <= 0:
        raise ValueError('layer count must be positive')
    for (fmt, backend), m in manifests.items():
        ir_text = (runs[fmt, backend] / 'ir.json').read_text()
        if hashlib.sha256(ir_text.removesuffix('\n').encode()).hexdigest() != m['ir_sha256']:
            raise ValueError('IR snapshot hash differs')
        progress = json.loads((runs[fmt, backend] / 'progress.json').read_text())
        if (m['ids'] != ids or m['vocab'] != vocab or m['format'] != fmt or m['backend'] != backend
                or m['actual_gpu'] is not (backend == 'cuda') or m['graph'] is not False
                or m['model_sha256'] != first['model_sha256'] or m['ir_sha256'] != first['ir_sha256']
                or progress != {'complete': True, 'completed': len(ids), 'total': len(ids)}):
            raise ValueError('incomplete/incompatible diagnostic matrix')
    report = {'status': 'DIAGNOSTIC_COMPLETE', 'e9_closed': False, 'registered_tolerance': 1e-4,
              'prompts': len(ids), 'ids': ids, 'model_sha256': first['model_sha256'],
              'numpy_version': np.__version__, 'storage': {}, 'logits': {}, 'stages': {}}
    for case, sequence in enumerate(ids):
        values = {}
        for (fmt, backend), folder in runs.items():
            part = folder / ('case_' + str(case).zfill(4))
            saved = json.loads((part / 'case.json').read_text())
            if saved['ids'] != sequence or saved['files'] != inventory(part):
                raise ValueError('completed case payload inventory changed')
            for method in ('prefill', 'decode'):
                path = part / (method + '.f32')
                if path.stat().st_size != len(sequence) * vocab * 4:
                    raise ValueError('wrong full logit byte count')
                values[fmt, backend, method] = np.memmap(path, mode='r', dtype='<f4').reshape(len(sequence), vocab)
        for fmt in ('f32', 'f16'):
            for left, right, label in [(('cuda','prefill'),('cpu','prefill'),'cuda_prefill_vs_cpu_prefill'),
                (('cuda','decode'),('cpu','prefill'),'cuda_decode_vs_cpu_prefill'),
                (('cpu','decode'),('cpu','prefill'),'cpu_decode_vs_cpu_prefill'),
                (('cuda','decode'),('cuda','prefill'),'cuda_decode_vs_cuda_prefill'),
                (('cuda','decode'),('cpu','decode'),'cuda_decode_vs_cpu_decode')]:
                merge_metric(report['logits'].setdefault(fmt + ':' + label, {}),
                             difference(values[(fmt,) + left], values[(fmt,) + right]))
            traces = {}
            for backend in ('cpu', 'cuda'):
                base = runs[fmt, backend] / ('case_' + str(case).zfill(4))
                traces[backend] = load_forward(base/'prefill',layers,'gpu' if backend=='cuda' else 'cpu',fmt,len(sequence),0)
                for record in traces[backend]:
                    totals = report['storage'].setdefault(fmt + ':' + backend, {})
                    for key, value in check_storage(record).items():
                        totals[key] = totals.get(key, 0) + value
            compare_stages(report['stages'].setdefault(fmt + ':cuda_vs_cpu_prefill', {}),traces['cuda'],traces['cpu'],{'case':case,'method':'prefill'})
            for position in range(len(sequence)):
                decoded = {}
                for backend in ('cpu','cuda'):
                    path = runs[fmt,backend]/('case_'+str(case).zfill(4))/('decode_'+str(position).zfill(4))
                    decoded[backend] = load_forward(path,layers,'gpu' if backend=='cuda' else 'cpu',fmt,1,position)
                    for record in decoded[backend]:
                        totals = report['storage'].setdefault(fmt+':'+backend,{})
                        for key, value in check_storage(record).items():
                            totals[key] = totals.get(key, 0) + value
                    compare_stages(report['stages'].setdefault(fmt+':'+backend+'_decode_vs_prefill',{}),decoded[backend],one_row(traces[backend],position),{'case':case,'position':position})
                compare_stages(report['stages'].setdefault(fmt+':cuda_vs_cpu_decode',{}),decoded['cuda'],decoded['cpu'],{'case':case,'position':position})
    for result in report['logits'].values():
        result['within_registered_bound'] = result['worst_relative'] <= 1e-4
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('diagnostic report is immutable')
    result = analyze(args.root)
    temporary = args.output.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'status':result['status'],'e9_closed':False,'logits':result['logits'],'storage':result['storage']}))


if __name__ == '__main__':
    main()