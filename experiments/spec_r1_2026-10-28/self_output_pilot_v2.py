"""Timing-only E10.2 pilot; resume never exceeds first100 frozen requests."""
import argparse
import json
from pathlib import Path
import time

import self_output_pool as pool


def run(binary, model, prompts, table, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out.parent / (out.name.replace('_pool', '') + '_pilot.json')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        audit = pool.analyze(out)
        assert audit['status'] == 'VERIFIED_PARTIAL'
        assert audit['completed'] == result['completed'] == 100
        assert audit['journal_sha256'] == result['journal_sha256']
        return result
    rows = [json.loads(line) for line in Path(prompts).read_text().splitlines() if line.strip()]
    count, _ = pool.recover(out / 'outputs.jsonl', rows, 16384)
    if not 0 <= count <= 100 or len(rows) < 100:
        raise ValueError('pilot journal exceeds frozen first100 scope')
    started = out / 'pilot_started.json'
    if not started.exists():
        if count:
            raise ValueError('missing start receipt for resumed pilot')
        pool.atomic_json(started, {'wall_time': time.time()})
    start = json.loads(started.read_text())['wall_time']
    if count < 100:
        pool.run(binary, model, prompts, table, out, max_new_requests=100-count)
    audit = pool.analyze(out)
    assert audit['completed'] == 100 and audit['status'] == 'VERIFIED_PARTIAL'
    result = {'complete': False, 'requests': len(rows), 'completed': 100,
              'new_requests': 100, 'elapsed_seconds': time.time()-start,
              'timing_scope': 'first100 including startup and any interrupted downtime',
              'journal_sha256': audit['journal_sha256']}
    pool.atomic_json(result_path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('binary', 'model', 'prompts', 'table', 'out'):
        parser.add_argument(name)
    args = parser.parse_args()
    print(json.dumps(run(args.binary, args.model, args.prompts, args.table, args.out)))
