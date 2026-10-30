import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import f16_trace_analysis as analysis


def write_forward(folder, fmt='f16', backend='cpu', layers=2):
    folder.mkdir()
    offset = 0
    records = []
    with (folder / 'payload.bin').open('wb') as payload:
        for layer in range(layers):
            raw = np.array([[0.125, 0.2, -0.3, 0.4], [1, 1.0006, 0.25, -0.0]], dtype='<f4')
            cache = raw.astype(np.float16).astype('<f4') if fmt == 'f16' else raw.copy()
            arrays = {'q': raw, 'k': raw, 'v': raw, 'k_cache': cache, 'v_cache': cache, 'attention': raw}
            if fmt == 'f16':
                arrays.update(k_codes=raw.astype(np.float16).view('<u2'), v_codes=raw.astype(np.float16).view('<u2'))
            specs = {}
            for name, value in arrays.items():
                specs[name] = {'offset': offset, 'bytes': value.nbytes, 'count': value.size,
                               'dtype': 'u16' if value.dtype.itemsize == 2 else 'f32'}
                payload.write(value.tobytes())
                offset += value.nbytes
            records.append({'schema': 1, 'backend': backend, 'format': fmt, 'layer': layer, 'T': 2,
                            'start_pos': 0, 'hq': 1, 'hkv': 1, 'hd': 4, 'scale': 0.5,
                            'complete': True, 'arrays': specs})
    (folder / 'trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return folder


@pytest.mark.parametrize('fmt', ['f32', 'f16'])
def test_exact_storage_and_complete_reader(tmp_path, fmt):
    folder = write_forward(tmp_path/'forward', fmt=fmt)
    rows = analysis.load_forward(folder, 2, 'cpu', fmt, 2, 0)
    assert len(rows) == 2
    assert all(not any(analysis.check_storage(r).values()) for r in rows)


@pytest.mark.parametrize('mutation', ['trailing', 'truncate', 'duplicate', 'incomplete', 'missing_stage', 'overlap', 'wrong_dtype', 'wrong_backend'])
def test_corrupt_trace_is_rejected(tmp_path, mutation):
    folder = write_forward(tmp_path/'forward')
    index = folder/'trace.jsonl'
    rows = [json.loads(x) for x in index.read_text().splitlines()]
    if mutation == 'trailing':
        with (folder/'payload.bin').open('ab') as f:
            f.write(b'bad')
    elif mutation == 'truncate':
        path = folder/'payload.bin'
        path.write_bytes(path.read_bytes()[:-1])
    elif mutation == 'duplicate':
        rows[1]['layer'] = 0
    elif mutation == 'incomplete':
        rows[0]['complete'] = False
    elif mutation == 'missing_stage':
        del rows[0]['arrays']['k_codes']
    elif mutation == 'overlap':
        rows[0]['arrays']['k']['offset'] = 0
    elif mutation == 'wrong_dtype':
        rows[0]['arrays']['k']['dtype'] = 'u16'
    else:
        rows[0]['backend'] = 'gpu'
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        analysis.load_forward(folder, 2, 'cpu', 'f16', 2, 0)


def test_original_range_normalization_and_zero_range():
    result = analysis.difference(np.array([[2, 3], [2, 3]], dtype=np.float32),
                                 np.array([[1, 3], [2, 2]], dtype=np.float32))
    assert result == {'rows': 2, 'worst_absolute': 1.0, 'worst_relative': 1.0, 'top1_equal': 1}


def test_midpoint_discontinuity_is_reported_without_changing_bound(tmp_path):
    left = analysis.load_forward(write_forward(tmp_path/'left'), 2, 'cpu', 'f16', 2, 0)
    right = analysis.load_forward(write_forward(tmp_path/'right'), 2, 'cpu', 'f16', 2, 0)
    # Build independent in-memory copies, leaving frozen payloads untouched.
    for rows, direction in ((left, np.inf), (right, -np.inf)):
        for row in rows:
            row['arrays'] = {k: v.copy() for k,v in row['arrays'].items()}
            x = np.nextafter(np.float32(1.00048828125), np.float32(direction))
            for name in ('k','v'):
                row['arrays'][name][0,0] = x
                row['arrays'][name+'_codes'][0,0] = np.float16(x).view(np.uint16)
                row['arrays'][name+'_cache'][0,0] = np.float32(np.float16(x))
    result = {}
    analysis.compare_stages(result,left,right,{'case':0})
    example = result['0']['examples'][0]
    assert example['amplification'] == 4096
    assert example['opposite_sides_of_rounded_midpoint']
    assert result['0']['half_code_differences'] == 2


def test_incomplete_matrix_has_no_diagnostic_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        analysis.analyze(tmp_path)
@pytest.mark.parametrize('left,right', [([], []), ([], [{}]), ([{}], []), ([{}], [{}, {}])])
def test_empty_or_unequal_stage_lists_rejected(left, right):
    with pytest.raises(ValueError, match='empty/unequal'):
        analysis.compare_stages({}, left, right, {})


@pytest.mark.parametrize('layers', [0, -1, True, 1.5])
def test_nonpositive_or_noninteger_layers_rejected(tmp_path, layers):
    with pytest.raises(ValueError, match='layer count'):
        analysis.load_forward(tmp_path, layers, 'cpu', 'f16', 2, 0)