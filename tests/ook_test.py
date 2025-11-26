import unittest
import numpy as np
from opticomlib.ook import THRESHOLD_EST, DSP, BER_analizer, theory_BER
from opticomlib.typing import eye, binary_sequence, electrical_signal, gv
from opticomlib.utils import Q
from opticomlib.devices import DAC

class TestOOK(unittest.TestCase):
    def setUp(self):
        # Reset global variables for each test
        gv.default()

    def test_THRESHOLD_EST(self):
        # Case 1: Symmetric noise
        # mu0=0, mu1=1, s0=0.1, s1=0.1
        # Threshold should be approx 0.5
        class MockEye:
            def __init__(self, mu0, mu1, s0, s1):
                self.mu0 = mu0
                self.mu1 = mu1
                self.s0 = s0
                self.s1 = s1
        
        eye_obj = MockEye(0, 1, 0.1, 0.1)
        th = THRESHOLD_EST(eye_obj)
        self.assertAlmostEqual(th, 0.5, delta=0.01)

        # Case 2: Asymmetric noise
        # mu0=0, mu1=1, s0=0.1, s1=0.2
        # Threshold should be shifted towards the smaller noise (0)
        eye_obj_asym = MockEye(0, 1, 0.1, 0.2)
        th_asym = THRESHOLD_EST(eye_obj_asym)
        self.assertTrue(th_asym < 0.5)
        self.assertTrue(th_asym > 0.0)

    def test_DSP(self):
        # Setup a random signal with enough bits for good statistics
        gv(sps=32, R=1e9, N=1000)
        
        # Generate random bits
        from opticomlib.devices import PRBS
        bits = PRBS(order=11, len=1000) # 2047 bits sequence truncated to 1000
        
        # Use DAC with NRZ pulse
        sig = DAC(bits, pulse_shape='nrz', Vpp=1.0)
        
        # Add noise
        sig.noise = np.random.normal(0, 0.02, sig.size) # SNR ~ 50 (linear) ~ 17 dB
        
        # Run DSP with a filter
        rx_bits, eye_obj, th = DSP(sig, BW=0.75*gv.R)
        
        # Check types
        self.assertIsInstance(rx_bits, binary_sequence)
        self.assertIsInstance(eye_obj, eye)
        self.assertIsInstance(th, float)
        
        # Check received bits match transmitted bits
        # With low noise and proper filtering, BER should be 0
        self.assertEqual(len(rx_bits), len(bits))
        
        # Allow for a few errors if any (though with this SNR it should be 0)
        errors = np.sum(rx_bits.data != bits.data)
        self.assertTrue(errors < 10, f"Too many errors: {errors}")
        
        # Check threshold is reasonable (around 0.5)
        # With symmetric noise and ISI, it should be close to 0.5
        self.assertAlmostEqual(th, 0.5, delta=0.1)

    def test_BER_analizer_counter(self):
        tx = binary_sequence([0, 1, 0, 1, 0, 1])
        rx = binary_sequence([0, 1, 0, 0, 0, 1]) # One error at index 3
        
        ber = BER_analizer('counter', Tx=tx, Rx=rx)
        self.assertEqual(ber, 1/6)
        
        # Test with arrays
        ber2 = BER_analizer('counter', Tx=[0,1], Rx=[1,0])
        self.assertEqual(ber2, 1.0)

    def test_BER_analizer_estimator(self):
        class MockEye:
            def __init__(self, mu0, mu1, s0, s1):
                self.mu0 = mu0
                self.mu1 = mu1
                self.s0 = s0
                self.s1 = s1
        
        # Symmetric case
        eye_obj = MockEye(0, 1, 0.1, 0.1)
        # Threshold est will be ~0.5
        # BER = 0.5 * (Q((1-0.5)/0.1) + Q((0.5-0)/0.1)) = Q(5)
        
        ber = BER_analizer('estimator', eye_obj=eye_obj)
        expected_ber = Q(5)
        self.assertAlmostEqual(ber, expected_ber, delta=1e-5)

    def test_theory_BER(self):
        # mu1=1, s0=0.1, s1=0.1
        # Optimal threshold ~ 0.5
        # BER ~ Q(5)
        ber = theory_BER(1, 0.1, 0.1)
        self.assertAlmostEqual(ber, Q(5), delta=1e-5)
        
        # Test vectorization
        ber_vec = theory_BER(np.array([1, 1]), 0.1, 0.1)
        self.assertEqual(len(ber_vec), 2)
        self.assertAlmostEqual(ber_vec[0], Q(5), delta=1e-5)

if __name__ == '__main__':
    unittest.main()
