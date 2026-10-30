"""Resumable E10.2 self-output pool; one canonical hash-chained JSONL journal.

Pool/model/table/binary bytes and request bodies are frozen. Completed requests
are never re-sampled. An interrupted final append alone may be truncated.
The journal itself is accepted by the existing r1_sim.py --pool builder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as f:
        f.write(canonical(value) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def body(row):
    return {'user_input': row['user_input'], 'history': row.get('history', []),
            'temperature': 1.0, 'top_k': 1, 'top_p': 1.0, 'seed': 0,
            'max_new_tokens': row.get('max_new_tokens', 60), 'return_ids': True}


def validate_response(r, vocab):
    if (not isinstance(r.get('reply'), str) or not isinstance(r.get('ids'), list)
            or not isinstance(r.get('prompt_ids'), list) or not r['prompt_ids']
            or 'stop_id' not in r or type(r.get('tokens')) is not int or r['tokens'] != len(r['ids'])):
        raise ValueError('invalid response contract')
    if any(type(x) is not int or not 0 <= x < vocab for x in r['ids'] + r['prompt_ids']):
        raise ValueError('invalid token IDs')
    stop = r.get('stop_id')
    if stop is not None and (type(stop) is not int or not -1 <= stop < vocab):
        raise ValueError('invalid stop ID')


def recover(journal, rows, vocab, *, repair_torn=False):
    """Validate every byte/request/chain; never discard a complete corrupt line."""
    journal = Path(journal)
    if not journal.exists():
        return 0, '0' * 64
    count, previous, offset = 0, '0' * 64, 0
    with journal.open('rb') as f:
        while line := f.readline():
            end = f.tell()
            if not line.endswith(b'\n'):
                if not repair_torn:
                    raise ValueError('interrupted final append')
                # A final non-newline append is not an acknowledged receipt.
                with journal.open('r+b') as target:
                    target.truncate(offset)
                    target.flush()
                    os.fsync(target.fileno())
                break
            r = json.loads(line)
            if count >= len(rows) or r.get('i') != count or r.get('previous_sha256') != previous:
                raise ValueError('extra/reordered/broken-chain request')
            if r.get('request_sha256') != digest(body(rows[count])):
                raise ValueError('request changed')
            validate_response(r, vocab)
            recorded = r.pop('record_sha256', None)
            if digest(r) != recorded:
                raise ValueError('corrupt record hash')
            previous = recorded
            count += 1
            offset = end
    return count, previous


def append(journal, index, row, response, previous, vocab):
    validate_response(response, vocab)
    r = {'i': index, 'request_sha256': digest(body(row)), 'previous_sha256': previous,
         **{k: response[k] for k in ('reply', 'ids', 'prompt_ids', 'stop_id', 'tokens')},
         'ttft': response.get('ttft')}
    r['record_sha256'] = digest(r)
    with open(journal, 'ab') as f:
        f.write(canonical(r) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    return r['record_sha256']


def request(url, payload=None):
    req = urllib.request.Request(url, data=None if payload is None else canonical(payload),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def pool_environment(table):
    return {'EMBER_SPEC': '8', 'EMBER_DRAFT': 'table', 'EMBER_DRAFT_TABLE': table,
            'EMBER_DRAFT_THETA': '0.02', 'EMBER_DECODE_PICK': '0', 'EMBER_DECODE_GRAPH': '1',
            'EMBER_DECODE_OPT': '1', 'EMBER_DECODE_LINEAR': 'cublas', 'EMBER_DETERMINISTIC': '0',
            'EMBER_GEMM_TF32': '0'}


def run(binary, model, pool, table, out, max_new_requests=None, port=8093):
    if max_new_requests is not None and max_new_requests < 1:
        raise ValueError('request cap must be positive')
    out = Path(out).resolve()
    out.mkdir(exist_ok=True)
    if (out / 'HALTED').exists():
        raise ValueError('pool HALTED; preserve diagnostic and use a new versioned run after repair')
    rows = [json.loads(line) for line in Path(pool).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty pool')
    inputs = {k: str(Path(v).resolve()) for k, v in
              (('binary', binary), ('model', model), ('pool', pool), ('table', table))}
    environment = pool_environment(inputs['table'])
    manifest = {'schema': 1, 'inputs': inputs, 'sha256': {k: sha(p) for k, p in inputs.items()},
                'client_source_sha256': sha(__file__),
                'environment': environment, 'requests': len(rows), 'vocab': 16384}
    frozen = out / 'manifest.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != manifest:
            raise ValueError('frozen source/model/table/pool/environment changed')
    else:
        atomic_json(frozen, manifest)
    journal = out / 'outputs.jsonl'
    count, previous = recover(journal, rows, manifest['vocab'], repair_torn=True)
    start, started = count, time.monotonic()
    if count < len(rows):
        env = dict(os.environ, **environment)
        for k in ('TINYLLAMA_LOGITS_DUMP', 'TINYLLAMA_TOPK_DEBUG', 'EMBER_DECODE_TIMING'):
            env.pop(k, None)
        (out / 'www').mkdir(exist_ok=True)
        with (out / f'server_{count}.log').open('w') as log:
            server = subprocess.Popen([inputs['binary'], inputs['model'], str(port), '127.0.0.1',
                                       str(out / 'www'), '999'], cwd=out, env=env, stdout=subprocess.DEVNULL,
                                      stderr=log, stdin=subprocess.DEVNULL)
            try:
                for _ in range(900):
                    try:
                        health = request(f'http://127.0.0.1:{port}/health')
                        break
                    except OSError:
                        if server.poll() is not None:
                            raise RuntimeError('pool server exited')
                        time.sleep(1)
                else:
                    raise RuntimeError('pool server did not become ready')
                if (health.get('model') != inputs['model'] or health.get('deterministic') is not False
                        or health.get('vocab_size') != manifest['vocab']):
                    raise ValueError('pool server model/mode/vocab mismatch')
                atomic_json(out / f'health_{count}.json', health)
                stop = len(rows) if max_new_requests is None else min(len(rows), count + max_new_requests)
                while count < stop:
                    response = request(f'http://127.0.0.1:{port}/chat', body(rows[count]))
                    try:
                        previous = append(journal, count, rows[count], response, previous, manifest['vocab'])
                    except (ValueError, KeyError, TypeError) as exc:
                        (out / 'HALTED').write_text('Invalid completed server response; never retry for a pass.\n')
                        (out / f'invalid_response_{count}.json').write_text(json.dumps(
                            {'i': count, 'request_sha256': digest(body(rows[count])),
                             'response': response, 'error': str(exc)}, ensure_ascii=False) + '\n')
                        raise
                    count += 1
            finally:
                server.terminate()
                try:
                    server.wait(20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    result = {'complete': count == len(rows), 'requests': len(rows), 'completed': count,
              'new_requests': count - start, 'elapsed_seconds': time.monotonic() - started,
              'last_record_sha256': previous, 'journal_sha256': sha(journal) if journal.exists() else None}
    destination = out / ('summary.json' if result['complete'] else f'partial_{count}.json')
    # A complete receipt is immutable; an already complete journal is only audited.
    if not destination.exists():
        atomic_json(destination, result)
    elif result['complete']:
        saved = json.loads(destination.read_text())
        if any(saved.get(k) != result[k] for k in
               ('complete', 'requests', 'completed', 'last_record_sha256', 'journal_sha256')):
            raise ValueError('immutable complete summary differs from journal')
    return result


def analyze(out, input_map=None):
    """Read-only audit; exact-hash path mapping permits checking relayed artifacts."""
    out = Path(out)
    m = json.loads((out / 'manifest.json').read_text())
    mapping = input_map or {}
    if set(mapping) - set(m['inputs']):
        raise ValueError('unknown mapped input')
    if m.get('schema') != 1 or m.get('client_source_sha256') != sha(__file__) or m.get('vocab') != 16384:
        raise ValueError('wrong schema/source/vocabulary')
    if m.get('environment') != pool_environment(m['inputs']['table']):
        raise ValueError('altered pool environment')
    paths = {k: mapping.get(k, path) for k, path in m['inputs'].items()}
    if any(sha(path) != m['sha256'][k] for k, path in paths.items()):
        raise ValueError('frozen input bytes changed')
    rows = [json.loads(line) for line in Path(paths['pool']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != m['requests']:
        raise ValueError('wrong pool inventory')
    count, previous = recover(out / 'outputs.jsonl', rows, m['vocab'])
    for file in out.glob('health_*.json'):
        h = json.loads(file.read_text())
        if h.get('model') != m['inputs']['model'] or h.get('deterministic') is not False or h.get('vocab_size') != m['vocab']:
            raise ValueError('health model/mode/vocabulary changed')
    if count and not (out / 'health_0.json').exists():
        raise ValueError('missing initial health receipt')
    journal_hash = sha(out / 'outputs.jsonl') if (out / 'outputs.jsonl').exists() else None
    summary = out / 'summary.json'
    if summary.exists():
        saved = json.loads(summary.read_text())
        expected = {'complete': True, 'requests': len(rows), 'completed': count,
                    'last_record_sha256': previous, 'journal_sha256': journal_hash}
        if count != len(rows) or any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('complete summary differs from journal')
    return {'status': ('HALTED' if (out / 'HALTED').exists() else
                       'VERIFIED_COMPLETE' if count == len(rows) else 'VERIFIED_PARTIAL'),
            'requests': len(rows), 'completed': count, 'last_record_sha256': previous,
            'journal_sha256': journal_hash, 'halted': (out / 'HALTED').exists()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    runner = sub.add_parser('run')
    for name in ('binary', 'model', 'pool', 'table', 'out'):
        runner.add_argument(name)
    runner.add_argument('--max-new-requests', type=int)
    runner.add_argument('--port', type=int, default=8093)
    auditor = sub.add_parser('analyze')
    auditor.add_argument('out')
    auditor.add_argument('--input-map', help='JSON of input name to exact-hash local file')
    a = p.parse_args()
    if a.action == 'analyze':
        print(json.dumps(analyze(a.out, json.loads(Path(a.input_map).read_text()) if a.input_map else None)))
    else:
        if a.max_new_requests is not None and a.max_new_requests < 1:
            p.error('--max-new-requests must be positive')
        print(json.dumps(run(a.binary, a.model, a.pool, a.table, a.out, a.max_new_requests, a.port)))
