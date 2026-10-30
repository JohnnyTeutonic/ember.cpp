import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import self_output_pilot_v2 as pilot


@pytest.mark.parametrize('count,remaining', [(0, 100), (37, 63), (100, 0)])
def test_resume_stays_within_first100(tmp_path, monkeypatch, count, remaining):
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text(''.join(json.dumps({'user_input': str(i)})+'\n' for i in range(150)))
    out = tmp_path / 'pick_pool'
    out.mkdir()
    (out / 'pilot_started.json').write_text('{"wall_time":1}')
    calls = []
    monkeypatch.setattr(pilot.pool, 'recover', lambda *a: (count, 'hash'))
    monkeypatch.setattr(pilot.pool, 'run', lambda *a, **kw: calls.append(kw['max_new_requests']))
    monkeypatch.setattr(pilot.pool, 'analyze', lambda *a: {'status':'VERIFIED_PARTIAL','completed':100,'journal_sha256':'hash'})
    result = pilot.run('binary','model',prompts,'table',out)
    assert calls == ([remaining] if remaining else [])
    assert result['completed'] == 100 and result['requests'] == 150
    assert not result['complete']
    pilot.run('binary','model',prompts,'table',out)
    assert calls == ([remaining] if remaining else [])


def test_out_of_scope_journal_rejected(tmp_path, monkeypatch):
    prompts = tmp_path / 'p'
    prompts.write_text('{}\n'*150)
    monkeypatch.setattr(pilot.pool,'recover',lambda *a:(101,'hash'))
    with pytest.raises(ValueError,match='scope'):
        pilot.run('b','m',prompts,'t',tmp_path/'pick_pool')
