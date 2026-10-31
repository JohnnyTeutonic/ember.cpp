"""Transport wrapper: preserves negative/partial evidence and never retries science."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def pack(source, destination):
    temporary = destination.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1,
                         strict_timestamps=False) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file() and not path.name.endswith('.partial'):
                archive.write(path, path.relative_to(source))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None: raise ValueError('packed archive CRC failure')
    temporary.replace(destination)


def extract(archive, destination):
    if destination.exists(): raise FileExistsError('refusing to reuse a source extraction')
    destination.mkdir()
    with tarfile.open(archive, 'r:*') as source:
        for member in source.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError('unsafe source archive entry')
        source.extractall(destination)


def main(job_output, uploads=Path('/content/chat/upload')):
    job_output = Path(job_output)
    job_output.mkdir(parents=True, exist_ok=True)
    if any((job_output / name).exists() for name in ('out.zip', 'summary.json', 'RUNNING')):
        raise FileExistsError('existing campaign evidence must be retained')
    (job_output / 'RUNNING').write_text('no automatic scientific retry\n')
    output = Path('/content/e10_integrated_out')
    source = Path('/content/e10_integrated_src')
    if output.exists(): raise FileExistsError('existing raw evidence must be retained')
    output.mkdir()
    manifest = json.loads((uploads / 'e10_integrated_manifest.json').read_text())
    archive = uploads / 'e10_integrated_source.tar'
    if sha(archive) != manifest['archive_sha256']: raise ValueError('archive hash mismatch')
    if sha(Path(__file__)) != manifest['boot_sha256']: raise ValueError('boot hash mismatch')
    extract(archive, source)
    source = source / 'ember'
    packet = source / 'experiments/spec_r1_2026-10-28/integrated_gate_20261031'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    errors = []
    def checkpoint():
        while not stop.wait(120):
            try: pack(output, job_output / 'out.zip')
            except Exception as error: errors.append(str(error))
    thread = threading.Thread(target=checkpoint, daemon=True)
    thread.start()
    status = {'scope': 'synthetic integrated selector correctness; NO speed/adoption result',
              'complete': False, 'scientific_pass': False, 'returncode': None}
    try:
        for name, argv, seconds in (
            ('hardware', ['bash', '-lc', 'nvidia-smi; nvcc --version; g++ --version; cmake --version'], 30),
            ('apt_update', ['apt-get', 'update', '-qq'], 180),
            ('apt_install', ['apt-get', 'install', '-y', '-q', 'libboost-regex-dev'], 180),
            ('pytest_install', [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', 'pytest==8.3.5'], 120)):
            with (output / (name + '.log')).open('w') as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=seconds, check=True)
        vm = packet / 'vm.py'
        with (output / 'vm.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(vm), str(source), str(output / 'science'),
                                        str(uploads / 'e10_integrated_manifest.json')],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try: rc = process.wait(timeout=2100)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise TimeoutError('35-minute scientific controller ceiling; partial evidence retained')
        status['returncode'] = rc
        execution = output / 'science/execution.json'
        if execution.is_file():
            status['execution'] = json.loads(execution.read_text())
            status['complete'] = status['execution']['complete']
            status['scientific_pass'] = status['execution']['scientific_pass']
        status['status'] = 'PASS_CORRECTNESS_ONLY' if status['scientific_pass'] else 'FAILED_OR_INCOMPLETE'
    except BaseException as error:
        status.update(status='FAILED_OR_INCOMPLETE', error=str(error))
    finally:
        stop.set(); thread.join(timeout=60)
        if thread.is_alive(): raise RuntimeError('checkpoint writer did not exit')
        status['checkpoint_errors'] = errors
        (output / 'transport_summary.json').write_text(json.dumps(status, indent=2) + '\n')
        pack(output, job_output / 'out.zip')
        if not (job_output / 'out.zip').is_file(): raise ValueError('missing final archive')
        status['archive_sha256'] = sha(job_output / 'out.zip')
        status['archive_bytes'] = (job_output / 'out.zip').stat().st_size
        (job_output / 'summary.json').write_text(json.dumps(status, indent=2) + '\n')
    # Successful transport includes retained negative science, preventing a
    # capacity driver from rerunning an outcome to obtain a pass.
    return 0

if __name__ == '__main__': sys.exit(main(sys.argv[1]))
