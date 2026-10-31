"""CPU-only checks of the rounding mechanism and preserved historical verdict."""
from fractions import Fraction
import json
from pathlib import Path
import unittest
import audit_default_repeatability as audit

class DefaultRepeatabilityTests(unittest.TestCase):
    def test_exact_sum_and_two_allowed_binary32_orders(self):
        example=audit.order_example()
        exact=sum(Fraction(x) for x in example['positive_warp_partials'])
        self.assertEqual(exact,Fraction(1)+Fraction(1,2**23))
        self.assertEqual(example['sum_bits'],['0x3f800000','0x3f800001'])
        self.assertEqual(Fraction(example['sum_f32'][1]),exact)
        self.assertNotEqual(Fraction(example['sum_f32'][0]),exact)

    def test_nonnegative_partials_are_realizable_at_registered_hidden_width(self):
        x=[0.0]*320;x[0]=1.0;x[32]=x[64]=2**-12
        thread=[]
        for lane in range(256):
            value=0.0
            for i in range(lane,len(x),256):value=audit.f32(value+audit.f32(x[i]*x[i]))
            thread.append(value)
        partials=[]
        for warp in range(8):
            # Only lane0 is nonzero here; this exactly matches the production
            # warp shuffle tree's leader result without relying on associativity.
            self.assertTrue(all(v==0 for v in thread[warp*32+1:(warp+1)*32]))
            partials.append(thread[warp*32])
        self.assertEqual(partials,audit.order_example()['positive_warp_partials'])

    def test_historical_gate_is_not_relabelled(self):
        root=Path(__file__).resolve().parent/'receipts'
        original=json.loads((root/'decode_pick_20261030_final.json').read_text())
        diagnosis=json.loads((root/'default_repeatability_20261101.json').read_text())
        self.assertEqual(original['strict_checker']['status'],'BASELINE_UNSTABLE')
        self.assertFalse(original['strict_checker']['modes']['det']['throughput_pass'])
        self.assertEqual(diagnosis['status'],'KNOWN_DEFAULT_NONDETERMINISM_NOT_FIXED')
        self.assertEqual(diagnosis['archive']['sha256'],original['archive_sha256'])
        self.assertEqual(diagnosis['raw_recomparison']['ids_prompt_stop_row_mapping_equal'],True)
        self.assertEqual(diagnosis['raw_recomparison']['turn_zero_first_row_equal'],60)

if __name__=='__main__':unittest.main()
