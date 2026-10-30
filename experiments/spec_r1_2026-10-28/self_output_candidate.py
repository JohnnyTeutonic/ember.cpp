"""Build the frozen corpus-first E10.2 candidate from an audited complete journal."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import decode_pick_gate as common
import r1_sim
import self_output_pool as pool


def build(pool_dir, corpus, out):
    pool_dir, corpus, out = map(lambda p: Path(p).resolve(), (pool_dir, corpus, out))
    audit = pool.analyze(pool_dir)
    if audit['status'] != 'VERIFIED_COMPLETE':
        raise ValueError('complete audited pool required')
    manifest = json.loads((pool_dir / 'manifest.json').read_text())
    if common.sha(corpus) != manifest['sha256']['table']:
        raise ValueError('corpus differs from generation table')
    journal = pool_dir / 'outputs.jsonl'
    frozen = {'orders': 6, 'min_count': 2, 'min_conf': 0.10, 'corpus_first': True,
              'pool_status': audit['status'], 'requests': audit['requests'],
              'sha256': {'model': manifest['sha256']['model'], 'corpus': common.sha(corpus),
                         'journal': common.sha(journal), 'builder': common.sha(__file__),
                         'r1_sim': common.sha(r1_sim.__file__)}}
    out.mkdir(exist_ok=True)
    recipe = out / 'recipe.json'
    if recipe.exists():
        saved = json.loads(recipe.read_text())
        expected = dict(frozen)
        expected['sha256'] = dict(frozen['sha256'], merged=common.sha(out / 'merged.bin'),
                                 self=common.sha(out / 'self.npz'))
        if saved != expected:
            raise ValueError('completed candidate/source/input drift')
        return saved
    temporary = out / 'self.partial.npz'
    r1_sim.cmd_build(SimpleNamespace(out=str(temporary), corpus=None, pool=str(journal),
                                     orders=6, min_count=2, min_conf=0.10))
    temporary.replace(out / 'self.npz')
    merged = out / 'merged.partial.bin'
    r1_sim.cmd_merge(SimpleNamespace(out=str(merged), tables=str(corpus) + ',' + str(out / 'self.npz'),
                                     orders=6, vocab_size=16384))
    merged.replace(out / 'merged.bin')
    frozen['sha256'].update(merged=common.sha(out / 'merged.bin'), self=common.sha(out / 'self.npz'))
    temporary = out / 'recipe.partial.json'
    temporary.write_text(json.dumps(frozen, indent=2) + '\n')
    temporary.replace(recipe)
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pool_dir')
    parser.add_argument('corpus')
    parser.add_argument('out')
    args = parser.parse_args()
    print(json.dumps(build(args.pool_dir, args.corpus, args.out), indent=2))
