import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import self_output_gate as g


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    inputs = {}
    for name in ('binary', 'model', 'prompts', 'corpus', 'merged', 'journal'):
        p = tmp_path / name
        p.write_text(''.join(json.dumps({'set': s, 'user_input': s}) + '\n' for s in ('chat40', 'carry40'))
                     if name == 'prompts' else name)
        inputs[name] = str(p)
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'orders': 6, 'min_count': 2, 'min_conf': 0.10,
                                 'corpus_first': True, 'pool_status': 'VERIFIED_COMPLETE',
                                 'sha256': {k: g.common.sha(inputs[k]) for k in ('model', 'corpus', 'merged', 'journal')}}))
    request_pool = tmp_path / 'request_pool'
    request_pool.write_text('pool')
    (tmp_path / 'manifest.json').write_text(json.dumps({'inputs': {'pool': str(request_pool)},
        'sha256': {'binary': g.common.sha(inputs['binary']), 'model': g.common.sha(inputs['model']),
                   'table': g.common.sha(inputs['corpus']), 'pool': g.common.sha(request_pool)}}))
    (tmp_path / 'summary.json').write_text('{}')
    monkeypatch.setattr(g, 'MODEL_REQUESTS', {g.common.sha(inputs['model']): 2})
    monkeypatch.setattr(g.pool, 'analyze', lambda *a, **k: {'status': 'VERIFIED_COMPLETE',
                        'requests': 2, 'completed': 2, 'journal_sha256': g.common.sha(inputs['journal'])})
    out = tmp_path / 'out'
    m = g.plan(*(inputs[k] for k in ('binary', 'model', 'prompts', 'corpus', 'merged', 'journal')), str(recipe), out, True)
    for run in m['runs']:
        p = out / (run['name'] + '.jsonl')
        records = []
        for i, dataset in enumerate(('chat40', 'carry40')):
            r = {'i': i, 'turn': 0, 'set': dataset, 'reply': 'ok', 'ids': [1, 2, 3], 'prompt_ids': [4],
                 'stop_id': 0, 'tokens': 3, 'warm_pass_same': True, 'passes': 0 if run['plain'] else 1,
                 'drafted': 0 if run['plain'] else 2, 'accepted': 0 if run['plain'] else 1,
                 'first_token_ms': 10, 'wall_ms': 90 if 'merged' in run['leg'] else 100}
            if run['dump']:
                r.update(rows=1, row0=i)
            records.append(r)
        p.write_text(''.join(json.dumps(r) + '\n' for r in records))
        meta = {'env': run['env'], 'once': False, 'gpu_layers': 999, 'bin': inputs['binary'],
                'gguf': inputs['model'], 'prompts': inputs['prompts'],
                'health': {'model': inputs['model'], 'deterministic': run['mode'] == 'det'},
                'dump_rows': 2 if run['dump'] else None, 'vocab': 3 if run['dump'] else None}
        Path(str(p) + '.meta.json').write_text(json.dumps(meta))
        Path(str(p) + '.hardware.json').write_text(json.dumps({'name': 'NVIDIA L4', 'uuid': 'GPU-one'}))
        if run['dump']:
            np.ones((2, 3), dtype=np.float32).tofile(str(p) + '.rows.f32')
    return out, m


def change_record(out, name, index, **values):
    p = out / (name + '.jsonl')
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[index].update(values)
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_complete_candidate_pass(receipt):
    out, _ = receipt
    r = g.analyze(out)
    assert r['status'] == 'PASS'
    assert r['deterministic_rows']['merged_bits_equal']
    assert r['modes']['det']['paired_speedups']['carry40'] == [1.125, 1.125]


def test_single_pair_single_set_miss_is_no_gain(receipt):
    out, _ = receipt
    change_record(out, 'det_merged_b', 1, wall_ms=98)
    assert g.analyze(out)['status'] == 'NO_GAIN'


def test_changed_ids_fail_not_omitted(receipt):
    out, _ = receipt
    change_record(out, 'default_merged_a', 0, ids=[99])
    assert g.analyze(out)['status'] == 'FAIL'


def test_cross_pod_abba_is_inconclusive(receipt):
    out, _ = receipt
    p = out / 'det_corpus_b.jsonl.hardware.json'
    p.write_text(json.dumps({'name': 'NVIDIA L4', 'uuid': 'GPU-two'}))
    assert g.analyze(out)['status'] == 'INCONCLUSIVE_TIMING'


def test_baseline_row_instability_not_attributed_to_candidate(receipt):
    out, _ = receipt
    p = out / 'det_rows_plain_repeat.jsonl.rows.f32'
    x = np.ones((2, 3), dtype=np.float32)
    x[0, 0] = 2
    x.tofile(p)
    assert g.analyze(out)['status'] == 'BASELINE_UNSTABLE'


def test_merged_row_difference_fails(receipt):
    out, _ = receipt
    p = out / 'det_rows_merged.jsonl.rows.f32'
    np.zeros((2, 3), dtype=np.float32).tofile(p)
    assert g.analyze(out)['status'] == 'FAIL'


def test_all_equal_nan_rows_still_invalid(receipt):
    out, _ = receipt
    for p in out.glob('*.rows.f32'):
        np.full((2, 3), np.nan, dtype=np.float32).tofile(p)
    with pytest.raises(ValueError, match='nonfinite'):
        g.analyze(out)


@pytest.mark.parametrize('change', [{'warm_pass_same': False}, {'passes': -1}, {'accepted': 99},
                                    {'set': 'changed'}])
def test_corrupt_record_fails_closed(receipt, change):
    out, _ = receipt
    change_record(out, 'default_merged_a', 0, **change)
    with pytest.raises(ValueError):
        g.analyze(out)


def test_wrong_recipe_rejected_before_plan(tmp_path):
    paths = []
    for i in range(7):
        p = tmp_path / str(i)
        p.write_text('{"user_input":"x"}\n' if i == 2 else '{}')
        paths.append(str(p))
    with pytest.raises(ValueError, match='recipe'):
        g.plan(*paths, tmp_path / 'out')


def test_manifest_cannot_enable_dump_for_timing(receipt):
    out, _ = receipt
    p = out / 'plan.json'
    m = json.loads(p.read_text())
    m['runs'][1]['dump'] = True
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='environment'):
        g.analyze(out)


def test_complete_invalid_leg_not_retried(receipt, monkeypatch):
    out, _ = receipt
    change_record(out, 'default_plain', 0, warm_pass_same=False)
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **k: pytest.fail('complete invalid leg retried'))
    with pytest.raises(ValueError):
        g.run(out)


def test_truncated_rows_fail(receipt):
    out, _ = receipt
    (out / 'det_rows_merged.jsonl.rows.f32').write_bytes(b'1234')
    with pytest.raises(ValueError, match='rows'):
        g.analyze(out)


def test_partial_pool_cannot_be_adopted(receipt, monkeypatch):
    out, _ = receipt
    monkeypatch.setattr(g.pool, 'analyze', lambda *a, **k: {'status': 'VERIFIED_PARTIAL',
                        'requests': 2, 'completed': 1, 'journal_sha256': ''})
    with pytest.raises(ValueError, match='complete original'):
        g.analyze(out)


def test_short_pilot_cannot_claim_full_pool(receipt, monkeypatch):
    out, m = receipt
    monkeypatch.setattr(g, 'MODEL_REQUESTS', {m['sha256']['model']: 20000})
    with pytest.raises(ValueError, match='no pilot adoption'):
        g.analyze(out)


def test_default_health_absent_flag_matches_server_contract(receipt):
    out, m = receipt
    for run in m['runs']:
        if run['mode'] == 'default':
            path = out / (run['name'] + '.jsonl.meta.json')
            meta = json.loads(path.read_text())
            meta['health'].pop('deterministic')
            path.write_text(json.dumps(meta))
    assert g.analyze(out)['status'] == 'PASS'


@pytest.mark.parametrize('flag', [True, None, 0, 'false'])
def test_default_health_rejects_wrong_explicit_flag(receipt, flag):
    out, _ = receipt
    path = out / 'default_plain.jsonl.meta.json'
    meta = json.loads(path.read_text())
    meta['health']['deterministic'] = flag
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='mode'):
        g.analyze(out)


def test_deterministic_health_requires_explicit_true(receipt):
    out, _ = receipt
    path = out / 'det_plain.jsonl.meta.json'
    meta = json.loads(path.read_text())
    meta['health'].pop('deterministic')
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='mode'):
        g.analyze(out)
