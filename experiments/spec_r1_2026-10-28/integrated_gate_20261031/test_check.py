"""Offline adversarial receipts; these are NOT actual CUDA evidence."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import check


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        root = cls.root
        kernels = []
        for v, n, c in check.kernel_inventory():
            stem = f'kernel_{v}_{n}_{c}'
            rec = {'vocab': v, 'rows': n, 'case': c, 'input': stem + '.input.bin',
                   'original': stem + '.original.bin', 'partition4': stem + '.partition4.bin'}
            (root / rec['input']).write_bytes(check.kernel_input(v, n, c))
            result = b''.join(struct.pack('<ffiii', *check.oracle(check.kernel_row(c, v, r))) for r in range(n))
            (root / rec['original']).write_bytes(result)
            (root / rec['partition4']).write_bytes(result)
            kernels.append(rec)
        row = struct.pack('<' + str(check.V) + 'f', *([1.0] * check.V))
        raw = row * 8
        graphs = []
        for mode, repeat, prefix in check.graph_inventory():
            name = f'{mode}_{repeat}_{prefix}.rows.bin'
            (root / name).write_bytes(raw)
            if mode == 'partition4':
                (root / f'{mode}_{repeat}_{prefix}.transitions.bin').write_bytes(row * 3)
            samples = []
            for j in range(8):
                for temp in (0.01, 0.05, 0.7, 1.0):
                    token = 0 if temp < 0.05 else check.V - 1
                    samples.append({'step': j, 'temperature': temp, 'full_id': token,
                                    'compact_id': token, 'full_rng': 'R' * 200,
                                    'compact_rng': 'R' * 200})
            graphs.append({'mode': mode, 'repeat': repeat, 'prefix': prefix,
                           'teacher_ids': check.graph_ids(prefix), 'successful_graph_calls': 8,
                           'graph_has_pick': mode != 'rows', 'graph_pick_parts': 4 if mode == 'partition4' else 0,
                           'scratch_nonnull': mode == 'partition4', 'scratch_stable': True,
                           'row_file': name, 'row_sha256': check.sha(root / name), 'sampler': samples,
                           'transitions': [{'has_pick': False, 'parts': 0}, {'has_pick': True, 'parts': 4},
                                           {'has_pick': False, 'parts': 0}] if mode == 'partition4' else []})
        fallback = root / 'fallback'
        fallback.mkdir()
        for graph in (0, 1):
            for pick in (0, 1):
                rows = [{'config': i, 'prompt': p, 'reply': 'fixture', 'ids': [4],
                         'prompt_ids': [2, 5, 3], 'stop_id': -1, 'tokens': 1}
                        for i in range(4) for p in ['Repeat the word blue three times.', 'What is two plus two?', 'Tell me about a garden.']]
                (fallback / f'graph{graph}_pick{pick}.json').write_text(json.dumps({
                    'rows': rows, 'health': {'deterministic': True},
                    'env': {'EMBER_DECODE_GRAPH': str(graph), 'EMBER_DECODE_PICK': str(pick)}}))
        (root / 'logs').mkdir()
        (root / 'logs/cpu_source_contracts.log').write_text('12 passed in 2.53s\n')
        cls.base = {'schema': 'ember.integrated-pick-correctness.v1', 'source_commit': check.SOURCE,
                    'model_sha256': check.MODEL, 'protocol_sha256': check.sha(Path(check.__file__).with_name('protocol.json')),
                    'complete': True, 'server_sha256': 'test-only', 'fallback_parts': 4,
                    'fallback_summary': {'model_sha256': check.MODEL, 'binary_sha256': 'test-only'}, 'device': {'name': 'NVIDIA L4', 'cuda_device_count': 1, 'major': 8, 'minor': 9},
                    'all_layers_gpu': True, 'ir': False, 'deterministic': True, 'kernels': kernels,
                    'graphs': graphs, 'cpu_source_contracts': {'passed': True, 'count': 12, 'log_sha256': check.sha(root / 'logs/cpu_source_contracts.log')},
                    'adoption': 'NOT_EVALUATED_HISTORICAL_GATE_NOT_PASSED'}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.report = copy.deepcopy(self.base)

    def replay(self):
        (self.root / 'report.json').write_text(json.dumps(self.report))
        return check.audit(self.root)

    def reject(self, mutation):
        mutation(self.report)
        with self.assertRaises((ValueError, KeyError)):
            self.replay()

    def test_registered_inventory_acceptance(self):
        self.assertEqual(self.replay()['full_rows'], 240)

    def test_missing_inventory(self):
        self.reject(lambda r: r['graphs'].pop())

    def test_substituted_teacher_ids(self):
        self.reject(lambda r: r['graphs'][0]['teacher_ids'].__setitem__(0, 7))

    def test_cpu_fallback(self):
        self.reject(lambda r: r.__setitem__('all_layers_gpu', False))

    def test_wrong_gpu(self):
        self.reject(lambda r: r['device'].__setitem__('name', 'CPU'))

    def test_ir_fallback(self):
        self.reject(lambda r: r.__setitem__('ir', True))

    def test_wrong_model(self):
        self.reject(lambda r: r.__setitem__('model_sha256', '0' * 64))

    def test_wrong_source(self):
        self.reject(lambda r: r.__setitem__('source_commit', '0' * 40))

    def test_wrong_protocol(self):
        self.reject(lambda r: r.__setitem__('protocol_sha256', '0' * 64))

    def test_incomplete(self):
        self.reject(lambda r: r.__setitem__('complete', False))

    def test_graph_fallback(self):
        self.reject(lambda r: r['graphs'][0].__setitem__('successful_graph_calls', 7))

    def test_wrong_capture_key(self):
        self.reject(lambda r: r['graphs'][-1].__setitem__('graph_pick_parts', 0))

    def test_unstable_scratch(self):
        self.reject(lambda r: r['graphs'][-1].__setitem__('scratch_stable', False))

    def test_missing_transition(self):
        self.reject(lambda r: r['graphs'][-1]['transitions'].pop())

    def test_wrong_compact_id(self):
        self.reject(lambda r: r['graphs'][0]['sampler'][0].__setitem__('compact_id', 2))

    def test_wrong_rng(self):
        self.reject(lambda r: r['graphs'][0]['sampler'][0].__setitem__('compact_rng', 'S' * 200))

    def test_adoption_claim(self):
        self.reject(lambda r: r.__setitem__('adoption', 'PASSED'))

    def test_escaped_raw_file(self):
        self.reject(lambda r: r['graphs'][0].__setitem__('row_file', '../escaped.bin'))

    def test_kernel_input_tamper(self):
        path = self.root / self.report['kernels'][0]['input']
        original = path.read_bytes()
        try:
            path.write_bytes(b'\0' * len(original))
            with self.assertRaises(ValueError): self.replay()
        finally: path.write_bytes(original)

    def test_kernel_field_tamper(self):
        path = self.root / self.report['kernels'][0]['partition4']
        original = path.read_bytes()
        try:
            path.write_bytes(b'\0' * len(original))
            with self.assertRaises(ValueError): self.replay()
        finally: path.write_bytes(original)

    def test_row_tamper_even_rebound_hash(self):
        path = self.root / self.report['graphs'][-1]['row_file']
        original = path.read_bytes()
        try:
            path.write_bytes(struct.pack('<f', 2.0) + original[4:])
            self.report['graphs'][-1]['row_sha256'] = check.sha(path)
            with self.assertRaises(ValueError): self.replay()
        finally: path.write_bytes(original)

    def test_wrong_fallback_model(self):
        self.reject(lambda r: r['fallback_summary'].__setitem__('model_sha256', '0' * 64))

    def test_missing_fallback_mode(self):
        self.reject(lambda r: r.__setitem__('fallback_parts', 0))

    def test_incomplete_cpu_controls(self):
        self.reject(lambda r: r['cpu_source_contracts'].__setitem__('count', 0))
    def test_all_nan_and_negative_inf_oracles_differ(self):
        self.assertEqual(check.oracle([float('nan')])[2:], (2147483647, -1, 1))
        self.assertEqual(check.oracle([-float('inf')])[2:], (0, 0, 0))

if __name__ == '__main__': unittest.main()
