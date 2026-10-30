"""Compile frozen original at the same pathname; restore diagnostic bytes in finally."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

from f16_trace_analysis import sha

ORIGINAL_SHA = 'c5bfa090027ae1ba448707228e5bc0e961f216c3cfb8f5c1c31c309fbdd8bfd9'


def replace(path, data):
    temporary = path.with_suffix(path.suffix + '.proof-replace')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def execute(source, build, original, output):
    if sha(original) != ORIGINAL_SHA:
        raise ValueError('original production source differs from frozen 5b96961')
    path = source/'graph_ir.cpp'
    patched = path.read_bytes()
    patched_sha = sha(path)
    flags_file = build/'CMakeFiles/tinyllama_core.dir/flags.make'
    entries = {}
    for line in flags_file.read_text().splitlines():
        if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
            key, value = line.split('=', 1)
            entries[key.strip()] = shlex.split(value)
    if set(entries) != {'CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'}:
        raise ValueError('missing actual normal compiler flags')
    flags = sum((entries[key] for key in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS')), [])
    if any('EMBER_KV_DIAGNOSTICS' in flag for flag in flags):
        raise ValueError('normal macro must be absent')
    if not any('HAS_CUDA' in flag for flag in flags):
        raise ValueError('this proof requires actual CUDA build flags')
    compiler = None
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if line.startswith('CMAKE_CXX_COMPILER:FILEPATH='):
            compiler = line.split('=', 1)[1]
    if not compiler:
        raise ValueError('missing actual compiler path')
    target = build/'graph_ir_original_proof.o'
    command = [compiler, *flags, '-o', str(target), '-c', str(path)]
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        replace(path, original.read_bytes())
        subprocess.run(command, check=True)
    finally:
        replace(path, patched)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if sha(path) != patched_sha:
            raise ValueError('diagnostic source restoration failed; no forwards permitted')
    current = build/'CMakeFiles/tinyllama_core.dir/graph_ir.cpp.o'
    result = {'scope': 'CUDA normal macro-OFF object at identical source pathname',
              'original_source_sha256': ORIGINAL_SHA, 'diagnostic_source_sha256': patched_sha,
              'original_object_sha256': sha(target), 'normal_object_sha256': sha(current),
              'normal_binary_sha256': sha(build/'ir_device_gate'),
              'normal_core_sha256': sha(build/'libtinyllama_core.a'), 'command': command,
              'source_restored': True}
    result['object_identical'] = result['original_object_sha256'] == result['normal_object_sha256']
    output.write_text(json.dumps(result, indent=1) + '\n')
    if not result['object_identical']:
        raise ValueError('ordinary CUDA object changed; no forwards permitted')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'build', 'original', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    execute(args.source, args.build, args.original, args.output)