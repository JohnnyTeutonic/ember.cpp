"""Public compact receipt controls; no GPU or private raw archive required."""
import copy
import json
from pathlib import Path
import unittest
import replay_kernel_receipt as compact

RECEIPT=Path(__file__).resolve().parents[1]/'receipts/integrated_correctness_20261031/kernel_records.json'
class PublicReceiptTests(unittest.TestCase):
    def setUp(self):self.receipt=json.loads(RECEIPT.read_text())
    def test_actual_kernel_records(self):self.assertEqual(compact.replay(self.receipt)['kernel_cases'],96)
    def test_missing_case(self):
        self.receipt['fixtures'].pop()
        with self.assertRaises(ValueError):compact.replay(self.receipt)
    def test_altered_input(self):
        self.receipt['fixtures'][0]['input_sha256']='0'*64
        with self.assertRaises(ValueError):compact.replay(self.receipt)
    def test_altered_raw_field(self):
        self.receipt['fixtures'][0]['partition4_hex']='00'*20
        with self.assertRaises(ValueError):compact.replay(self.receipt)
    def test_wrong_source(self):
        self.receipt['source_packet_commit']='0'*40
        with self.assertRaises(ValueError):compact.replay(self.receipt)
    def test_wrong_archive(self):
        self.receipt['canonical_zip_sha256']='0'*64
        with self.assertRaises(ValueError):compact.replay(self.receipt)
if __name__=='__main__':unittest.main()
