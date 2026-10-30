from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import run_f16_diagnosis as runner


def test_valid_negative_leg_is_preserved_and_not_rerun(tmp_path, monkeypatch):
    command = [sys.executable, '-c', 'import json; print(json.dumps({"pass":False})); raise SystemExit(1)']
    folder = tmp_path/'leg'
    first = runner.completed_leg(folder, command, {}, (0, 1))
    assert first['returncode'] == 1 and first['result'] == {'pass': False}
    before = runner.inventory_without_receipt(folder)
    def forbidden(*args, **kwargs):
        raise AssertionError('valid negative must never be rerun')
    monkeypatch.setattr(runner.subprocess, 'run', forbidden)
    assert runner.completed_leg(folder, command, {}, (0, 1)) == first
    assert runner.inventory_without_receipt(folder) == before


def test_completed_leg_corruption_rejected_without_rerun(tmp_path, monkeypatch):
    command = [sys.executable, '-c', 'print("{}")']
    folder = tmp_path/'leg'
    runner.completed_leg(folder, command, {}, (0,))
    (folder/'stdout.json').write_text('{"altered":true}')
    with pytest.raises(ValueError, match='drift'):
        runner.completed_leg(folder, command, {}, (0,))
    assert (folder/'stdout.json').read_text() == '{"altered":true}'


def test_incomplete_leg_is_retained(tmp_path):
    folder = tmp_path/'leg'
    folder.mkdir()
    (folder/'partial').write_text('raw bytes')
    with pytest.raises(ValueError, match='incomplete'):
        runner.completed_leg(folder, ['must-not-run'], {}, (0,))
    assert (folder/'partial').read_text() == 'raw bytes'


def test_canonical_model_failure_precedes_execution(tmp_path, monkeypatch):
    paths = [tmp_path/name for name in ('model','normal','trace','component','seeded','ids')]
    for path in paths:
        path.write_text('not canonical')
    def forbidden(*args, **kwargs):
        raise AssertionError('model mismatch must precede any executable')
    monkeypatch.setattr(runner.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='canonical Qwen3 model'):
        runner.execute(*paths, tmp_path/'out')
    assert not (tmp_path/'out').exists()
@pytest.mark.parametrize('failure', ['compiler', 'interrupt', 'different_object', 'success'])
def test_same_path_proof_always_restores_source(tmp_path, monkeypatch, failure):
    import f16_normal_object_proof as proof
    source = tmp_path/'source'
    build = tmp_path/'build'
    source.mkdir()
    flags = build/'CMakeFiles/tinyllama_core.dir'
    flags.mkdir(parents=True)
    original = tmp_path/'original.cpp'
    original.write_bytes(b'original')
    (source/'graph_ir.cpp').write_bytes(b'diagnostic')
    monkeypatch.setattr(proof, 'ORIGINAL_SHA', proof.sha(original))
    (flags/'flags.make').write_text('CXX_DEFINES = -DHAS_CUDA\nCXX_INCLUDES =\nCXX_FLAGS = -O3\n')
    (build/'CMakeCache.txt').write_text('CMAKE_CXX_COMPILER:FILEPATH=/frozen/c++\n')
    for path in (flags/'graph_ir.cpp.o', build/'ir_device_gate', build/'libtinyllama_core.a'):
        path.write_bytes(b'object')
    def compile_original(command, **kwargs):
        assert (source/'graph_ir.cpp').read_bytes() == b'original'
        assert command[-1] == str(source/'graph_ir.cpp')
        if failure == 'compiler':
            raise RuntimeError('compiler failure')
        if failure == 'interrupt':
            raise SystemExit(143)
        (build/'graph_ir_original_proof.o').write_bytes(b'wrong' if failure == 'different_object' else b'object')
    monkeypatch.setattr(proof.subprocess, 'run', compile_original)
    if failure == 'success':
        result = proof.execute(source, build, original, tmp_path/'receipt.json')
        assert result['object_identical'] and result['source_restored']
    else:
        with pytest.raises((RuntimeError, SystemExit, ValueError)):
            proof.execute(source, build, original, tmp_path/'receipt.json')
    assert (source/'graph_ir.cpp').read_bytes() == b'diagnostic'

def test_saved_semantic_result_tampering_is_rejected(tmp_path):
    import json
    command = [sys.executable, '-c', 'import json; print(json.dumps({"pass":False})); raise SystemExit(1)']
    folder = tmp_path/'leg'
    runner.completed_leg(folder, command, {}, (0, 1))
    receipt = folder/'leg.json'
    saved = json.loads(receipt.read_text())
    saved['result'] = {'pass': True}
    receipt.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='drift'):
        runner.completed_leg(folder, command, {}, (0, 1))


def test_historical_halted_precedes_hashing_or_execution(tmp_path):
    out = tmp_path/'out'
    out.mkdir()
    (out/'HALTED').write_text('historical failure')
    with pytest.raises(ValueError, match='existing HALTED'):
        runner.execute(*([tmp_path/'missing']*6), out)
    assert (out/'HALTED').read_text() == 'historical failure'

@pytest.mark.parametrize('mutation', ['none', 'missing', 'duplicate', 'input_drift', 'rc_drift'])
def test_component_requires_paired_real_capacity_inventory(mutation):
    cases = [dict(zip(('hq','hkv','hd'),shape), length=length, capacity=capacity, input_sha256='a'*64)
             for shape in ((16,8,128),(14,2,64))
             for length in (1,2,24,32,33,55,127,128,129,2048,2049,4096)
             for capacity in (length,8448)]
    if mutation == 'missing':
        cases.pop()
    if mutation == 'duplicate':
        cases.append(dict(cases[0]))
    if mutation == 'input_drift':
        cases[1]['input_sha256'] = 'b'*64
    result = {'cases':cases,'complete':True,'diagnostic_only':True,'exact_component_mismatches':0}
    if mutation == 'none':
        runner.validate_component(result,0)
    else:
        with pytest.raises(ValueError):
            runner.validate_component(result,1 if mutation=='rc_drift' else 0)