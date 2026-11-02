"""Keep activation roundoff handling separate from the strict public-output gates."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('native_parity',Path(__file__).resolve().parents[1]/'tools/run-parity.py')
parity=importlib.util.module_from_spec(spec);spec.loader.exec_module(parity)

class Metrics(unittest.TestCase):
    def test_large_activation_roundoff_does_not_relax_logits_or_cache(self):
        reference=np.array([4500],np.float32)
        actual=reference+np.float32(4*np.spacing(reference[0]))
        self.assertTrue(parity.metric(reference,actual,activation=True)['passed'])
        self.assertFalse(parity.metric(reference,actual)['passed'])
        self.assertFalse(parity.metric(reference,actual,activation=True)['flat_absolute_gate_passed'])

    def test_small_activation_drift_still_fails(self):
        self.assertFalse(parity.metric(np.array([1],np.float32),np.array([1.002],np.float32),activation=True)['passed'])

    def test_nonfinite_or_wrong_shape_cannot_pass(self):
        expected=np.array([1],np.float32)
        self.assertFalse(parity.metric(expected,np.array([np.nan],np.float32),activation=True)['passed'])
        self.assertFalse(parity.metric(expected,np.zeros(2,np.float32),activation=True)['passed'])

    def test_relative_drift_is_also_gated(self):
        self.assertFalse(parity.metric(np.array([1e-7],np.float32),np.array([1e-4],np.float32),activation=True)['passed'])

    def test_distribution_metrics_are_stable_and_separate_from_fp32_gate(self):
        expected=np.linspace(-1000,1000,8197,dtype=np.float32)
        actual=expected+np.float32(20)
        drift=parity.logit_drift(expected,actual,8196)
        self.assertAlmostEqual(drift['kl_reference_to_native'],0,places=12)
        self.assertEqual(drift['top1_agreement'],1)
        self.assertEqual(drift['top32_overlap'],1)
        self.assertAlmostEqual(drift['teacher_log_probability_delta_max_abs'],0,places=12)
        self.assertFalse(parity.metric(expected,actual)['passed'])

if __name__=='__main__':unittest.main()
