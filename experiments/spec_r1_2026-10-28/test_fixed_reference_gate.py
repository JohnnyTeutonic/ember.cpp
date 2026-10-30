import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fixed_reference_gate as g


@pytest.fixture
def receipt(tmp_path):
    paths = []
    for name in ('server', 'pick.gguf', 'pk.gguf', 'prompts.jsonl'):
        p = tmp_path / name
        p.write_text('{"user_input":"test"}\n' if name.endswith('jsonl') else name)
        paths.append(str(p))
    out = tmp_path / 'out'
    m = g.plan(*paths, out)
    for run in m['runs']:
        records = [{'i': 0, 'turn': 0, 'reply': 'a', 'ids': [1], 'prompt_ids': [2],
                    'stop_id': 0, 'tokens': 1, 'rows': 2, 'row0': 0, 'warm_pass_same': True}]
        path = out / (run['name'] + '.jsonl')
        path.write_text(json.dumps(records[0]) + '\n')
        meta = {'env': run['env'], 'once': False, 'gpu_layers': run['gpu_layers'],
                'bin': paths[0], 'gguf': m['inputs'][run['model']], 'prompts': paths[3],
                'health': {'model': m['inputs'][run['model']], 'deterministic': True},
                'vocab': 3, 'dump_rows': 2}
        Path(str(path) + '.meta.json').write_text(json.dumps(meta))
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        if run['leg'] == 'cublas':
            array += np.float32(1e-5)
        array.tofile(str(path) + '.rows.f32')
    return out, m


def mutate_record(out, name, fn):
    p = out / (name + '.jsonl')
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r) + '\n')


def test_complete_pass(receipt):
    out, _ = receipt
    r = g.analyze(out)
    assert r['status'] == 'PASS'
    assert all(v['requests'] == 1 and v['rows'] == 2 for v in r['models'].values())


def test_ids_never_silently_omitted(receipt):
    out, _ = receipt
    mutate_record(out, 'pick_fixed', lambda r: r.update(ids=[99]))
    assert g.analyze(out)['status'] == 'FAIL'


@pytest.mark.parametrize('field,value', [('row0', 1), ('warm_pass_same', False), ('passes', 1)])
def test_invalid_records_fail_closed(receipt, field, value):
    out, _ = receipt
    mutate_record(out, 'pick_fixed', lambda r: r.update({field: value}))
    with pytest.raises(ValueError):
        g.analyze(out)


def test_missing_request_fails(receipt):
    out, _ = receipt
    (out / 'pk_fixed.jsonl').write_text('')
    with pytest.raises(ValueError, match='requests'):
        g.analyze(out)


def test_truncated_rows_fail(receipt):
    out, _ = receipt
    (out / 'pk_fixed.jsonl.rows.f32').write_bytes(b'1234')
    with pytest.raises(ValueError, match='rows'):
        g.analyze(out)


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_rows_fail(receipt, value):
    out, _ = receipt
    p = out / 'pk_fixed.jsonl.rows.f32'
    x = np.fromfile(p, dtype=np.float32)
    x[0] = value
    x.tofile(p)
    with pytest.raises(ValueError, match='nonfinite'):
        g.analyze(out)


def test_fixed_farther_is_failure_even_within_rounding(receipt):
    out, _ = receipt
    p = out / 'pk_fixed.jsonl.rows.f32'
    x = np.fromfile(p, dtype=np.float32)
    x += np.float32(2e-5)
    x.tofile(p)
    r = g.analyze(out)
    assert r['status'] == 'FAIL' and r['models']['pk']['failed_rows'] == 2


def test_rounding_failure(receipt):
    out, _ = receipt
    p = out / 'pick_fixed.jsonl.rows.f32'
    x = np.fromfile(p, dtype=np.float32) + np.float32(0.01)
    x.tofile(p)
    assert g.analyze(out)['status'] == 'FAIL'


def test_wrong_backend_fails(receipt):
    out, _ = receipt
    p = out / 'pick_cpu.jsonl.meta.json'
    m = json.loads(p.read_text())
    m['gpu_layers'] = 999
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='backend'):
        g.analyze(out)


def test_input_drift_fails(receipt):
    out, m = receipt
    Path(m['inputs']['pick']).write_text('changed')
    with pytest.raises(ValueError, match='frozen'):
        g.analyze(out)


def test_complete_negative_leg_is_not_rerun(receipt, monkeypatch):
    out, _ = receipt
    mutate_record(out, 'pick_cpu', lambda r: r.update(warm_pass_same=False))
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **k: pytest.fail('complete invalid leg retried'))
    with pytest.raises(ValueError):
        g.run_plan(out)


def test_identical_reference_is_no_farther():
    r = [{'reply': '', 'ids': [], 'prompt_ids': [1], 'stop_id': 0, 'tokens': 0,
          'rows': 1, 'row0': 0, 'i': 0, 'turn': 0}]
    a = np.array([[1, 2]], dtype=np.float32)
    result = g.compare((r, a), (copy.deepcopy(r), a), (copy.deepcopy(r), a), 1e-4)
    assert result['pass']


def test_relay_mapping_preserves_original_metadata_and_checks_exact_bytes(receipt, tmp_path):
    out, m = receipt
    mapping = {}
    for key, source in m['inputs'].items():
        source = Path(source)
        target = tmp_path / ('relayed_' + key)
        target.write_bytes(source.read_bytes())
        source.unlink()
        mapping[key] = str(target)
    assert g.analyze(out, mapping)['status'] == 'PASS'
    Path(mapping['binary']).write_bytes(b'altered')
    with pytest.raises(ValueError, match='frozen'):
        g.analyze(out, mapping)


def test_unknown_relay_mapping_rejected(receipt):
    out, _ = receipt
    with pytest.raises(ValueError, match='unknown input'):
        g.analyze(out, {'unexpected': 'path'})
