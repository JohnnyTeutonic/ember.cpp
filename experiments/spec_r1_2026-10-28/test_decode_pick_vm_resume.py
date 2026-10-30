"""The VM may resume incomplete work, but never retry a complete negative leg."""
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest


def exercise(tmp_path, monkeypatch, *, complete=False, invalid=False, fallback_invalid=False):
    source = Path(__file__).with_name('decode_pick_vm.sh').read_text()
    block = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)[0]
    out = tmp_path / 'out'
    fallback = out / 'fallback'
    fallback.mkdir(parents=True)
    for graph in (0, 1):
        for pick in (0, 1):
            (fallback / f'graph{graph}_pick{pick}.json').write_text(json.dumps({'rows': [1]}))
    pick = out / 'pick'
    pick.mkdir()
    run = {'name': 'default_rows_off', 'command': ['fake-device-command']}
    manifest = {'inputs': {'binary': '/fake/bin', 'model': '/content/model.gguf',
                           'prompts': '/content/eval100.jsonl'},
                'sha256': {k: 'hash' for k in ('binary', 'model', 'prompts')}, 'runs': [run]}
    (pick / 'plan.json').write_text(json.dumps(manifest))
    if complete:
        (pick / 'default_rows_off.jsonl.meta.json').write_text('{}')
    calls = []

    def load(*args):
        if invalid:
            raise ValueError('complete negative leg')

    def compare(records):
        if fallback_invalid:
            raise ValueError('complete fallback mismatch')
        return {'pass': True, 'requests_per_leg': 12, 'legs': 4}

    monkeypatch.setitem(sys.modules, 'decode_pick_gate', SimpleNamespace(sha=lambda path: 'hash', load_run=load))
    monkeypatch.setitem(sys.modules, 'decode_pick_fallback', SimpleNamespace(compare=compare))
    monkeypatch.setattr(sys, 'argv', ['vm', '/fake/import', '/fake/bin', str(out)])
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: calls.append(command))
    try:
        exec(compile(block, 'vm-resume', 'exec'), {})
    finally:
        assert calls == ([] if complete or fallback_invalid else [['fake-device-command']])
    return calls


def test_complete_valid_leg_is_not_repeated(tmp_path, monkeypatch):
    exercise(tmp_path, monkeypatch, complete=True)


def test_complete_negative_leg_stops_without_retry(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='complete negative'):
        exercise(tmp_path, monkeypatch, complete=True, invalid=True)


def test_incomplete_leg_runs(tmp_path, monkeypatch):
    exercise(tmp_path, monkeypatch)


def test_complete_negative_fallback_stops_without_retry(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='fallback mismatch'):
        exercise(tmp_path, monkeypatch, fallback_invalid=True)
