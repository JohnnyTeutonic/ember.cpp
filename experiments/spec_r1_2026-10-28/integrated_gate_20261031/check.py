"""Independent, read-only replay of the frozen synthetic selector packet.

This checker never changes the historical speed/adoption decision. Missing CUDA,
missing cases, shortened row files, or internally consistent substitutions fail.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import struct

SOURCE = '4836b7de0ce4a96c030d491d942bb265fe393302'
MODEL = 'e270235b70fa47369ad7bab19e231dedefd1d5b9d2966f6e2d84562e93250f87'
PREFIXES = (1, 7, 16, 31, 64)
MODES = ('rows', 'original', 'partition4')
CASES = ('finite', 'ties', 'near_tie', 'all_nan', 'all_negative_inf',
         'positive_inf', 'mixed_nan', 'signed_zero', 'subnormal', 'threshold',
         'boundary_tie', 'last_max')
VOCABS = (1, 31, 257, 16384)
ROWS = (1, 4)
V = 16384

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def require(value, message):
    if not value:
        raise ValueError(message)

def safe_file(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    require(p.is_relative_to(root) and p.is_file(), 'missing/escaped artifact: ' + name)
    return p

def kernel_row(case, vocab, row):
    x = [((i * 17 + row * 13) % 101 - 50) / 16 for i in range(vocab)]
    if case == 'ties': x = [1.0] * vocab
    elif case == 'near_tie':
        x = [-2.0] * vocab; x[0] = 1.0; x[-1] = 1.0 - 2 ** -23 if vocab > 1 else 1.0
    elif case == 'all_nan': x = [float('nan')] * vocab
    elif case == 'all_negative_inf': x = [-float('inf')] * vocab
    elif case == 'positive_inf': x[0] = float('inf'); x[-1] = float('inf')
    elif case == 'mixed_nan': x[vocab // 2] = float('nan')
    elif case == 'signed_zero': x = [-0.0 if i % 2 else 0.0 for i in range(vocab)]
    elif case == 'subnormal': x = [(i % 4) * 2 ** -149 for i in range(vocab)]
    elif case == 'threshold':
        x = [-2.0] * vocab; x[0] = 1.0; x[-1] = 1.0 - 2 ** -17 if vocab > 1 else 1.0
    elif case == 'boundary_tie':
        x = [-4.0] * vocab
        for i in (0, vocab // 4, vocab // 2, vocab - 1): x[i] = 3.0
    elif case == 'last_max': x[-1] = 9.0
    return x

def kernel_inventory():
    return [(v, n, c) for v in VOCABS for n in ROWS for c in CASES]

def kernel_input(v, n, c):
    x = [z for r in range(n) for z in kernel_row(c, v, r)]
    return struct.pack('<' + str(len(x)) + 'f', *x)

def oracle(x):
    finite_or_inf = [(i, z) for i, z in enumerate(x) if not math.isnan(z)]
    nan = int(len(finite_or_inf) != len(x))
    if not finite_or_inf:
        return (-float('inf'), -float('inf'), 2147483647, -1, nan)
    maximum = max(z for _, z in finite_or_inf)
    indices = [i for i, z in finite_or_inf if z == maximum]
    second = max((z for _, z in finite_or_inf if z < maximum), default=-float('inf'))
    return maximum, second, indices[0], indices[-1], nan

def audit_kernel(root, rec, expected):
    v, n, c = expected
    require((rec['vocab'], rec['rows'], rec['case']) == expected, 'kernel inventory changed')
    raw = safe_file(root, rec['input']).read_bytes()
    require(raw == kernel_input(v, n, c), 'registered kernel inputs changed')
    a = safe_file(root, rec['original']).read_bytes()
    b = safe_file(root, rec['partition4']).read_bytes()
    require(len(a) == n * 20 and a == b, 'kernel five-field bit equality failed')
    for r in range(n):
        actual = struct.unpack_from('<ffiii', a, r * 20)
        reference = oracle(kernel_row(c, v, r))
        # Python equality intentionally ignores signed-zero representation. Actual
        # original/partitioned field bits have already been compared above.
        require(actual == reference, 'independent semantic oracle failed')

def graph_inventory():
    return [(mode, repeat, prefix) for mode in MODES for repeat in range(2) for prefix in PREFIXES]

def graph_ids(prefix):
    # Same public deterministic teacher-forcing rule used by the C++ harness.
    return [4 + ((1103515245 * (i + 17) + 12345) & 0xffffffff) % (V - 4)
            for i in range(prefix + 8)]

def audit(root):
    root = Path(root)
    receipt = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    require(receipt['schema'] == 'ember.integrated-pick-correctness.v1', 'wrong schema')
    require(receipt['source_commit'] == SOURCE and receipt['model_sha256'] == MODEL, 'source/model substitution')
    require(receipt['complete'] is True, 'incomplete packet')
    require(receipt['protocol_sha256'] == sha(Path(__file__).with_name('protocol.json')), 'protocol substitution')
    device = receipt['device']
    require('L4' in device['name'] and device['cuda_device_count'] > 0 and device['major'] == 8 and device['minor'] == 9, 'actual L4 required')
    require(receipt['all_layers_gpu'] is True and receipt['ir'] is False, 'CPU/IR fallback refused')
    require(receipt['deterministic'] is True, 'deterministic mode required')
    require(len(receipt['kernels']) == len(kernel_inventory()), 'kernel cases missing')
    for rec, exp in zip(receipt['kernels'], kernel_inventory(), strict=True): audit_kernel(root, rec, exp)
    records = receipt['graphs']
    require(len(records) == len(graph_inventory()), 'graph legs missing')
    references = {}
    reference_samples = {}
    for rec, expected in zip(records, graph_inventory(), strict=True):
        mode, repeat, prefix = expected
        require((rec['mode'], rec['repeat'], rec['prefix']) == expected, 'graph inventory changed')
        require(rec['teacher_ids'] == graph_ids(prefix), 'teacher-forcing input changed')
        require(rec['successful_graph_calls'] == 8, 'graph fallback/incomplete execution')
        require(rec['graph_has_pick'] == (mode != 'rows'), 'capture key pick presence wrong')
        require(rec['graph_pick_parts'] == (4 if mode == 'partition4' else 0), 'capture key partitions wrong')
        if mode == 'partition4':
            require(rec['scratch_nonnull'] is True and rec['scratch_stable'] is True, 'scratch/capture contract failed')
            require(rec['transitions'] == [{'has_pick': False, 'parts': 0}, {'has_pick': True, 'parts': 4}, {'has_pick': False, 'parts': 0}], 'capture-key transition inventory changed')
            transition = safe_file(root, rec.get('transition_file', f'{mode}_{repeat}_{prefix}.transitions.bin')).read_bytes()
            width = V * 4
            require(len(transition) == 3 * width and transition[:width] == transition[width:2*width] == transition[2*width:], 'capture-key transition row bits differ')
        raw = safe_file(root, rec['row_file']).read_bytes()
        require(len(raw) == 8 * V * 4, 'full row count/width changed')
        require(sha(safe_file(root, rec['row_file'])) == rec['row_sha256'], 'row hash mismatch')
        values = struct.unpack('<' + str(8 * V) + 'f', raw)
        require(all(math.isfinite(x) for x in values), 'nonfinite synthetic model rows')
        if prefix in references: require(raw == references[prefix], 'same-kernel rows/repeats differ')
        else: references[prefix] = raw
        require(len(rec['sampler']) == 8 * 4, 'sampler inventory missing')
        full_samples = [(x['step'], x['temperature'], x['full_id'], x['full_rng']) for x in rec['sampler']]
        if prefix in reference_samples: require(full_samples == reference_samples[prefix], 'full sampler controls differ')
        else: reference_samples[prefix] = full_samples
        for j, result in enumerate(rec['sampler']):
            step, temp = j // 4, (0.01, 0.05, 0.7, 1.0)[j % 4]
            require(result['step'] == step and result['temperature'] == temp, 'sampler matrix changed')
            row = values[step * V:(step + 1) * V]
            maximum = max(row)
            maxima = [k for k, z in enumerate(row) if z == maximum]
            require(type(result['full_id']) is int and type(result['compact_id']) is int and 0 <= result['full_id'] < V and
                    result['compact_id'] == result['full_id'], 'sampler ID mismatch')
            if temp < 0.05:
                require(result['full_id'] == maxima[0], 'independent low-temperature argmax mismatch')
            require(isinstance(result['full_rng'], str) and len(result['full_rng']) > 100 and
                    result['full_rng'] == result['compact_rng'], 'sampler RNG mismatch')
    require(receipt['fallback_parts'] == 4, 'four-part API fallback environment missing')
    fallback = receipt['fallback_summary']
    require(fallback['model_sha256'] == MODEL and fallback['binary_sha256'] == receipt['server_sha256'], 'fallback binary/model binding failed')
    prompts = ['Repeat the word blue three times.', 'What is two plus two?', 'Tell me about a garden.']
    inventory = [(i, prompt) for i in range(4) for prompt in prompts]
    for graph in (0, 1):
        controls = []
        for pick in (0, 1):
            leg = json.loads(safe_file(root, f'fallback/graph{graph}_pick{pick}.json').read_text())
            require(leg['health']['deterministic'] is True, 'fallback deterministic control missing')
            require(leg['env']['EMBER_DECODE_GRAPH'] == str(graph) and leg['env']['EMBER_DECODE_PICK'] == str(pick), 'fallback environment changed')
            rows = leg['rows']
            require([(r['config'], r['prompt']) for r in rows] == inventory, 'fallback request inventory changed')
            for row in rows:
                require(isinstance(row['reply'], str) and isinstance(row['ids'], list) and bool(row['prompt_ids']), 'fallback response incomplete')
                require(all(type(x) is int and 0 <= x < V for x in row['ids'] + row['prompt_ids']), 'fallback token IDs invalid')
            controls.append(rows)
        require(controls[0] == controls[1], 'API fallback responses differ')
    controls = receipt['cpu_source_contracts']
    require(controls['passed'] is True and controls['count'] == 12, 'source-bound CPU controls missing')
    log = safe_file(root, 'logs/cpu_source_contracts.log')
    require(sha(log) == controls['log_sha256'] and re.search(r'\b12 passed\b', log.read_text()) and 'skipped' not in log.read_text(), 'CPU source control log/inventory mismatch')
    require(receipt['adoption'] == 'NOT_EVALUATED_HISTORICAL_GATE_NOT_PASSED', 'adoption claim refused')
    return {'complete': True, 'scope': 'synthetic deterministic integrated correctness only',
            'kernel_cases': len(kernel_inventory()), 'graph_legs': len(records),
            'full_rows': len(records) * 8, 'adoption': receipt['adoption']}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    args = p.parse_args(); print(json.dumps(audit(args.root), indent=2))
