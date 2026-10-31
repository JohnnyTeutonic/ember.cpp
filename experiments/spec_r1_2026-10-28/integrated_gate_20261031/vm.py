"""Fail-closed remote controller for the synthetic integrated correctness packet.

No timing or adoption test is performed. This controller is prepared for parent
review, and must not be launched without an exact archive/input receipt.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import check
import fixture
import prepare


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def command(argv, log, timeout, env=None):
    # Every completed/failed command retains its exact log. No retry-for-pass.
    if log.exists(): raise FileExistsError(log)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, env=env)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {argv}; see {log.name}')


def run(source, output, manifest):
    source, output = Path(source).resolve(), Path(output).resolve()
    required = {'api.cpp', 'model.cpp', 'model.h', 'verify_kernels.cu', 'verify_kernels.h', 'row_pick_partitioned.cuh', 'decode_pick_policy.h', 'CMakeLists.txt'}
    if manifest.get('production_source_commit') != check.SOURCE or not required.issubset(manifest.get('source_files', {})):
        raise ValueError('incomplete production source manifest')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('existing evidence is immutable; no automatic scientific retry')
    output.mkdir(parents=True, exist_ok=True)
    for relative, expected in manifest['source_files'].items():
        if check.sha(source / relative) != expected:
            raise ValueError('source hash mismatch: ' + relative)
    atomic(output / 'inputs.json', manifest)
    work = output / 'work'
    work.mkdir()
    logs = output / 'logs'
    logs.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               EMBER_DETERMINISTIC='1', EMBER_DECODE_GRAPH='1', EMBER_SPEC='0',
               EMBER_GEMM_TF32='0', EMBER_IR='0', EMBER_DECODE_PICK_PARTS='4',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.pop('PYTEST_ADDOPTS', None)
    env.pop('PYTEST_PLUGINS', None)
    started = time.monotonic()
    execution = {'complete': False, 'scientific_pass': False, 'source_commit': check.SOURCE,
                 'scope': 'synthetic deterministic correctness; no timing/adoption gate'}
    try:
        fixture_path = work / 'synthetic.gguf'
        model = fixture.build(fixture_path)
        if model['sha256'] != check.MODEL: raise ValueError('fixture model differs')
        prepared = work / 'prepared'
        inputs = prepare.prepare(source, prepared)
        atomic(output / 'prepared.json', inputs)
        packet = Path(__file__).resolve().parent
        build = output.parent / (output.name + '_build')
        if build.exists(): raise FileExistsError('fresh registered build directory required')
        command(['cmake', '-S', str(packet), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 '-DEMBER_SOURCE=' + str(source), '-DPREPARED_INPUTS=' + str(prepared),
                 '-DHAS_CUDA=ON', '-DHAS_MPI=OFF', '-DCMAKE_CUDA_ARCHITECTURES=89'],
                logs / 'configure.log', 300, env)
        command(['cmake', '--build', str(build), '--target', 'integrated_pick_probe',
                 'tinyllama_server', '-j2'], logs / 'build.log', 1200, env)
        probe = build / 'integrated_pick_probe'
        server = build / 'ember' / 'tinyllama_server'
        if not probe.is_file() or not server.is_file(): raise ValueError('compiled binaries missing')
        controls = source / 'tools/row_pick_profile/test_integration.py'
        command([sys.executable, '-m', 'pytest', str(controls), '-q', '--color=no'], logs / 'cpu_source_contracts.log', 180, env)
        control_text = (logs / 'cpu_source_contracts.log').read_text()
        if not re.search(r'\b12 passed\b', control_text) or 'skipped' in control_text:
            raise ValueError('production source control inventory incomplete')
        report = {'schema': 'ember.integrated-pick-correctness.v1', 'complete': False,
                  'source_commit': check.SOURCE, 'model_sha256': model['sha256'],
                  'protocol_sha256': check.sha(packet / 'protocol.json'),
                  'binary_sha256': check.sha(probe), 'server_sha256': check.sha(server),
                  'all_layers_gpu': True, 'ir': False, 'deterministic': True,
                  'graphs': [], 'kernels': [], 'cpu_source_contracts': {'passed': True, 'count': 12, 'log_sha256': check.sha(logs / 'cpu_source_contracts.log')},
                  'adoption': 'NOT_EVALUATED_HISTORICAL_GATE_NOT_PASSED'}
        for mode in ('kernels', *check.MODES):
            destination = output / mode
            command([str(probe), str(fixture_path), str(prepared), str(destination), mode],
                    logs / (mode + '.log'), 300, env)
            device = json.loads((destination / 'device.json').read_text())
            if 'device' in report and report['device'] != device: raise ValueError('device changed')
            report['device'] = device
            if mode == 'kernels':
                for rec in inputs['kernels']:
                    rec = dict(rec)
                    # Retain each raw artifact once. The staged input is moved only
                    # after all kernel cells have completed, never rewritten.
                    original_input = prepared / rec['input']
                    new_input = destination / rec['input']
                    original_input.rename(new_input)
                    for key in ('input', 'original', 'partition4'): rec[key] = mode + '/' + rec[key]
                    report['kernels'].append(rec)
            else:
                leg = json.loads((destination / (mode + '.json')).read_text())
                if leg['all_layers_gpu'] is not True or leg['ir'] is not False: raise ValueError('fallback')
                for rec in leg['graphs']:
                    rec['row_file'] = mode + '/' + rec['row_file']
                    rec['row_sha256'] = check.sha(output / rec['row_file'])
                    if mode == 'partition4':
                        rec['transition_file'] = f'{mode}/{mode}_{rec["repeat"]}_{rec["prefix"]}.transitions.bin'
                    report['graphs'].append(rec)
            atomic(output / 'progress.json', report)
        # This existing frozen API control exercises graph-disabled host embedding
        # fallback plus repetition/top-k/low-temperature routes. PARTS=4 is added
        # by the outer environment; neither frozen script nor API is rewritten.
        fallback = source / 'experiments/spec_r1_2026-10-28/decode_pick_fallback.py'
        command([sys.executable, str(fallback), str(server), str(fixture_path), str(output / 'fallback')],
                logs / 'fallback.log', 600, env)
        report['fallback_parts'] = 4
        report['fallback_summary'] = json.loads((output / 'fallback/summary.json').read_text())
        for relative, expected in manifest['source_files'].items():
            if check.sha(source / relative) != expected: raise ValueError('final source integrity failure')
        if check.sha(probe) != report['binary_sha256'] or check.sha(fixture_path) != check.MODEL:
            raise ValueError('final binary/model integrity failure')
        report['complete'] = True
        atomic(output / 'report.json', report)
        verdict = check.audit(output)
        atomic(output / 'audit.json', verdict)
        execution.update(complete=True, scientific_pass=True)
    except BaseException as error:
        execution.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        execution['elapsed_seconds'] = time.monotonic() - started
        atomic(output / 'execution.json', execution)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source', type=Path); p.add_argument('output', type=Path)
    p.add_argument('manifest', type=Path); a = p.parse_args()
    run(a.source, a.output, json.loads(a.manifest.read_text(encoding='utf-8')))
