import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import self_output_pool as p

ROWS = [{'user_input': 'one'}, {'user_input': 'two', 'history': [{'user': 'old', 'assistant': 'reply'}]}]
RESPONSE = {'reply': 'hello', 'ids': [4], 'prompt_ids': [1, 2], 'stop_id': 3, 'tokens': 1}


def test_journal_resume_and_hash_chain(tmp_path):
    journal = tmp_path / 'out.jsonl'
    previous = p.append(journal, 0, ROWS[0], RESPONSE, '0' * 64, 16384)
    assert p.recover(journal, ROWS, 16384) == (1, previous)
    last = p.append(journal, 1, ROWS[1], RESPONSE, previous, 16384)
    assert p.recover(journal, ROWS, 16384) == (2, last)


def test_only_torn_final_append_can_be_repaired(tmp_path):
    journal = tmp_path / 'out.jsonl'
    last = p.append(journal, 0, ROWS[0], RESPONSE, '0' * 64, 16384)
    original = journal.read_bytes()
    with journal.open('ab') as f:
        f.write(b'{"i":1,')
    with pytest.raises(ValueError, match='final append'):
        p.recover(journal, ROWS, 16384)
    assert p.recover(journal, ROWS, 16384, repair_torn=True) == (1, last)
    assert journal.read_bytes() == original


def test_complete_corrupt_line_never_truncated(tmp_path):
    journal = tmp_path / 'out.jsonl'
    journal.write_bytes(b'{broken}\n')
    with pytest.raises(ValueError):
        p.recover(journal, ROWS, 16384, repair_torn=True)
    assert journal.read_bytes() == b'{broken}\n'


@pytest.mark.parametrize('change', [{'reply': 'changed'}, {'i': 1}, {'previous_sha256': 'f' * 64},
                                    {'request_sha256': 'e' * 64}])
def test_modified_records_fail(tmp_path, change):
    journal = tmp_path / 'out.jsonl'
    p.append(journal, 0, ROWS[0], RESPONSE, '0' * 64, 16384)
    r = json.loads(journal.read_text())
    r.update(change)
    journal.write_bytes(p.canonical(r) + b'\n')
    with pytest.raises(ValueError):
        p.recover(journal, ROWS, 16384, repair_torn=True)


@pytest.mark.parametrize('change', [{'ids': [16384]}, {'prompt_ids': []}, {'tokens': 2},
                                    {'stop_id': 16384}, {'ids': [True]}])
def test_bad_response_contract(change):
    with pytest.raises(ValueError):
        p.validate_response({**RESPONSE, **change}, 16384)


@pytest.fixture
def live_fake(tmp_path, monkeypatch):
    files = []
    for name in ('binary', 'model', 'pool.jsonl', 'table'):
        f = tmp_path / name
        f.write_text(''.join(json.dumps(r) + '\n' for r in ROWS) if name.endswith('jsonl') else name)
        files.append(str(f))
    calls = []
    class Server:
        def __init__(self, *a, **k):
            calls.append('server')
        def terminate(self):
            pass
        def wait(self, *a):
            return 0
    def request(url, payload=None):
        if url.endswith('/health'):
            return {'model': files[1], 'deterministic': False, 'vocab_size': 16384}
        calls.append(payload['user_input'])
        return dict(RESPONSE)
    monkeypatch.setattr(p.subprocess, 'Popen', Server)
    monkeypatch.setattr(p, 'request', request)
    return files, tmp_path / 'out', calls


def test_complete_requests_never_resampled(live_fake):
    files, out, calls = live_fake
    assert not p.run(*files, out, max_new_requests=1)['complete']
    assert p.run(*files, out)['complete']
    assert p.run(*files, out)['new_requests'] == 0
    assert calls == ['server', 'one', 'server', 'two']


def test_frozen_input_drift_refused(live_fake):
    files, out, calls = live_fake
    p.run(*files, out, max_new_requests=1)
    Path(files[1]).write_text('changed model')
    with pytest.raises(ValueError, match='frozen'):
        p.run(*files, out)
    assert calls == ['server', 'one']


def test_complete_summary_corruption_refused(live_fake):
    files, out, _ = live_fake
    p.run(*files, out)
    summary = out / 'summary.json'
    r = json.loads(summary.read_text())
    r['journal_sha256'] = 'bad'
    summary.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='summary'):
        p.run(*files, out)


def test_invalid_response_halts_without_retry(live_fake, monkeypatch):
    files, out, _ = live_fake
    original = p.request
    monkeypatch.setattr(p, 'request', lambda url, payload=None:
                        original(url, payload) if url.endswith('/health') else {**RESPONSE, 'tokens': 2})
    with pytest.raises(ValueError):
        p.run(*files, out)
    assert (out / 'HALTED').exists()
    with pytest.raises(ValueError, match='HALTED'):
        p.run(*files, out)


def test_read_only_partial_and_complete_audit(live_fake):
    files, out, calls = live_fake
    p.run(*files, out, max_new_requests=1)
    assert p.analyze(out)['status'] == 'VERIFIED_PARTIAL'
    assert calls == ['server', 'one']
    p.run(*files, out)
    assert p.analyze(out)['status'] == 'VERIFIED_COMPLETE'
    assert calls == ['server', 'one', 'server', 'two']


def test_relayed_inputs_require_exact_bytes(live_fake, tmp_path):
    files, out, _ = live_fake
    p.run(*files, out)
    replacement = tmp_path / 'relayed_model'
    replacement.write_bytes(Path(files[1]).read_bytes())
    assert p.analyze(out, {'model': str(replacement)})['status'] == 'VERIFIED_COMPLETE'
    replacement.write_text('different')
    with pytest.raises(ValueError, match='bytes'):
        p.analyze(out, {'model': str(replacement)})


def test_health_corruption_refused(live_fake):
    files, out, _ = live_fake
    p.run(*files, out)
    health = out / 'health_0.json'
    r = json.loads(health.read_text())
    r['model'] = 'different'
    health.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='health'):
        p.analyze(out)


def test_environment_corruption_refused(live_fake):
    files, out, _ = live_fake
    p.run(*files, out)
    manifest = out / 'manifest.json'
    r = json.loads(manifest.read_text())
    r['environment']['EMBER_SPEC'] = '0'
    manifest.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='environment'):
        p.analyze(out)


def test_frozen_auditor_source_bundled_and_checked(live_fake):
    files, out, _ = live_fake
    p.run(*files, out)
    source = out / 'source/self_output_pool.py'
    assert source.read_bytes() == Path(p.__file__).read_bytes()
    source.write_text('corrupted')
    with pytest.raises(ValueError, match='source snapshot'):
        p.analyze(out)
