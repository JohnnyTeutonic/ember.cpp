from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import r1_sim
import self_output_pool


def test_merge_preserves_empty_high_orders_and_corpus_priority(tmp_path):
    corpus = tmp_path / 'corpus.bin'
    r1_sim.dr_sim.dr_table.write(str(corpus), 16384, [
        {'keys': np.array([2], dtype=np.uint64), 'next': np.array([9], dtype=np.uint16),
         'conf': np.array([255], dtype=np.uint8)}])
    self_table = tmp_path / 'self.npz'
    values = {'orders': 6}
    for order in range(6):
        values.update({f'keys{order}': np.array([2] if order == 0 else [], dtype=np.uint64),
                       f'next{order}': np.array([4] if order == 0 else [], dtype=np.uint16),
                       f'conf{order}': np.array([1.] if order == 0 else [], dtype=np.float64)})
    np.savez(self_table, **values)
    merged = tmp_path / 'merged.bin'
    r1_sim.cmd_merge(SimpleNamespace(out=str(merged), tables=str(corpus) + ',' + str(self_table),
                                     orders=6, vocab_size=16384))
    vocab, tables = r1_sim.dr_sim.dr_table.read(str(merged))
    assert vocab == 16384 and len(tables) == 6
    assert tables[0]['next'].tolist() == [9]
    assert all(t['keys'].size == 0 for t in tables[1:])


def test_canonical_checkpoint_journal_builds_existing_self_table(tmp_path):
    journal = tmp_path / 'outputs.jsonl'
    response = {'reply': 'hello', 'ids': [4], 'prompt_ids': [1, 2], 'stop_id': 3, 'tokens': 1}
    previous = '0' * 64
    for i in range(2):
        previous = self_output_pool.append(journal, i, {'user_input': str(i)}, response, previous, 16384)
    built = tmp_path / 'self.npz'
    r1_sim.cmd_build(SimpleNamespace(out=str(built), corpus=None, pool=str(journal),
                                     orders=6, min_count=2, min_conf=0.10))
    table = r1_sim.Table(str(built))
    assert table.orders == 6
    assert table.lookup([1, 2]) == (4, 1.0, 2)
    assert table.lookup([2, 4]) == (3, 1.0, 2)
