"""Prepare source-exact sampler copies and registered kernel input files."""
import argparse
import hashlib
import json
from pathlib import Path
import check


def extract(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    level = 1
    end = brace + 1
    while level:
        if text[end] == '{': level += 1
        elif text[end] == '}': level -= 1
        end += 1
    return text[start:end]


def prepare(source, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('do not overwrite prepared inputs')
    output.mkdir(parents=True, exist_ok=True)
    text = (Path(source) / 'api.cpp').read_text(encoding='utf-8')
    signatures = ['static int sample_top_k_top_p_temperature(', 'static int greedy_from_pick(']
    bodies = [extract(text, s) for s in signatures]
    header = '// Exact function bodies copied from pinned api.cpp, diagnostic TU only.\n' + '\n\n'.join(bodies) + '\n'
    (output / 'sampler_exact.h').write_text(header, encoding='utf-8')
    records = []
    for v, n, c in check.kernel_inventory():
        stem = f'kernel_{v}_{n}_{c}'
        path = output / (stem + '.input.bin')
        path.write_bytes(check.kernel_input(v, n, c))
        records.append({'vocab': v, 'rows': n, 'case': c, 'input': path.name,
                        'original': stem + '.original.bin', 'partition4': stem + '.partition4.bin'})
    manifest = {'source_commit': check.SOURCE, 'api_sha256': check.sha(Path(source) / 'api.cpp'),
                'sampler_exact_sha256': check.sha(output / 'sampler_exact.h'),
                'function_sha256': [hashlib.sha256(b.encode()).hexdigest() for b in bodies], 'kernels': records}
    (output / 'prepared.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); print(json.dumps(prepare(a.source, a.output), indent=2))
