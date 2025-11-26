import unittest
import numpy as np
from scipy.constants import pi, c
from opticomlib.utils import (
    dec2bin, str2array, db, dbm, idb, idbm, gaus, Q, phase, tau_g, dispersion,
    rcos, si, norm, nearest, nearest_index, p_ase, average_voltages,
    noise_variances, optimum_threshold, theory_BER, shortest_int,
    rcos_pulse, gauss_pulse, nrz_pulse, upfir, phase_estimator, get_psd
)

class TestUtils(unittest.TestCase):
    def test_get_psd(self):
        fs = 10e9 # 10 GHz
        t = np.arange(0, 1000e-9, 1/fs) # 1000 ns duration for good resolution
        f_sig = 1e9 # 1 GHz signal
        
        # Test with array
        sig = np.sin(2*pi*f_sig*t)
        f, psd = get_psd(sig, fs)
        
        # Check peak frequency
        # f is in Hz. Expected peak at 1 GHz (1e9 Hz).
        peak_idx = np.argmax(psd)
        peak_freq = np.abs(f[peak_idx])
        
        # Resolution is fs / nperseg. nperseg is 2048 or len(sig).
        # Here len(sig) = 10000. nperseg=2048.
        # fs = 10e9. df = 10e9/2048 ~ 5 MHz.
        self.assertAlmostEqual(peak_freq, 1e9, delta=50e6)
        
        # Check peak power
        # Power of sin(t) is 0.5 W.
        # Since return_onesided=False, power is split between +f and -f.
        # So expected peak is 0.25 W.
        self.assertAlmostEqual(psd[peak_idx], 0.25, delta=0.05)

        # Test with object having .signal
        class MockSignal:
            def __init__(self, s): self.signal = s
        
        mock_sig = MockSignal(sig)
        f2, psd2 = get_psd(mock_sig, fs)
        np.testing.assert_array_equal(psd, psd2)

    def test_dec2bin(self):
        # Test basic conversion
        np.testing.assert_array_equal(dec2bin(5, 4), np.array([0, 1, 0, 1], dtype=np.uint8))
        np.testing.assert_array_equal(dec2bin(10, 4), np.array([1, 0, 1, 0], dtype=np.uint8))
        
        # Test error handling
        with self.assertRaises(ValueError):
            dec2bin(16, 4) # Too large
        with self.assertRaises(ValueError):
            dec2bin(1.5, 4) # Not integer

    def test_str2array(self):
        # Test binary string
        np.testing.assert_array_equal(str2array('101'), np.array([True, False, True]))
        np.testing.assert_array_equal(str2array('1 0 1'), np.array([True, False, True]))
        np.testing.assert_array_equal(str2array('1,0,1'), np.array([True, False, True]))
        
        # Test numeric string
        np.testing.assert_array_equal(str2array('1 2 3'), np.array([1, 2, 3]))
        np.testing.assert_array_equal(str2array('1.1 2.2'), np.array([1.1, 2.2]))
        
        # Test complex string
        np.testing.assert_array_equal(str2array('1+2j 3-4j'), np.array([1+2j, 3-4j]))
        np.testing.assert_array_equal(str2array('1+2j 3-4j', np.float32), np.array([1.0, 3.0]))
        
        # Test 2D array
        np.testing.assert_array_equal(str2array('1 2; 3 4'), np.array([[1, 2], [3, 4]]))

    def test_db_conversions(self):
        # Test db
        self.assertAlmostEqual(db(1), 0.0)
        self.assertAlmostEqual(db(10), 10.0)
        np.testing.assert_allclose(db([1, 10]), [0.0, 10.0])
        
        # Test dbm
        self.assertAlmostEqual(dbm(1e-3), 0.0)
        self.assertAlmostEqual(dbm(1), 30.0)
        
        # Test idb
        self.assertAlmostEqual(idb(10), 10.0)
        self.assertAlmostEqual(idb(0), 1.0)
        
        # Test idbm
        self.assertAlmostEqual(idbm(0), 1e-3)
        self.assertAlmostEqual(idbm(30), 1.0)

    def test_gaus(self):
        self.assertAlmostEqual(gaus(0, 0, 1), 1/(2*pi)**0.5)
        self.assertAlmostEqual(gaus(1, 1, 2), 1/(2*(2*pi)**0.5))

        self.assertAlmostEqual(np.trapz(gaus(np.linspace(-10,10,1000), 0, 1), np.linspace(-10,10,1000)), 1.0, places=3) # Check integral equals 1

    def test_Q(self):
        self.assertAlmostEqual(Q(0), 0.5)
        self.assertTrue(Q(10) < 1e-6)

    def test_phase(self):
        t = np.linspace(0, 10, 100)
        x = np.exp(1j * t)
        p = phase(x)
        # Phase should be linear with slope 1
        np.testing.assert_allclose(np.diff(p), np.diff(t), atol=1e-5)

    def test_si(self):
        self.assertEqual(si(1e-3, 's'), '1.0 ms')
        self.assertEqual(si(1e9, 'Hz'), '1.0 GHz')
        self.assertEqual(si(0.002, 's'), '2.0 ms')

    def test_norm(self):
        x = np.array([1, 2, 4])
        np.testing.assert_array_equal(norm(x), np.array([0.25, 0.5, 1.0]))

    def test_nearest(self):
        x = np.array([1, 2, 3, 4, 5])
        self.assertEqual(nearest(x, 2.2), 2)
        self.assertEqual(nearest(x, 4.8), 5)
        np.testing.assert_array_equal(nearest(x, [1.1, 3.9]), np.array([1, 4]))
        
        self.assertEqual(nearest_index(x, 2.2), 1)

    def test_p_ase(self):
        # Test with known values
        # G=20dB (100), NF=3dB (2), BW=1GHz
        # P_ase = NF * h * f0 * (G-1) * BW
        # f0 = c/1550nm
        from scipy.constants import h
        f0 = c/1550e-9
        expected = idb(3) * h * f0 * (100-1) * 1e9
        self.assertAlmostEqual(p_ase(True, 1550e-9, 20, 3, 1e9), expected, delta=expected*1e-3)
        self.assertEqual(p_ase(False), 0)

    def test_average_voltages(self):
        # Simple OOK case
        mu, mu_ase = average_voltages(P_avg=0, modulation='ook', amplify=False, r=1, R_L=1)
        # P_avg=0dBm = 1mW. OOK: P_ON = 2*P_avg = 2mW, P_OFF = 0 (ER=inf)
        # mu_ON = r*P_ON*R_L = 2e-3 V
        # mu_OFF = 0
        np.testing.assert_allclose(mu, [0, 2e-3])
        self.assertEqual(mu_ase, 0)

    def test_noise_variances(self):
        # Thermal noise only
        # T=300K, R_L=50, BW_el=1Hz
        # S_th = 4 * kB * T * BW_el * R_L
        kB = 1.38e-23
        expected_th = 4 * kB * 300 * 1 * 50
        
        S = noise_variances(P_avg=-100, modulation='ook', amplify=False, BW_el=1, R_L=50, T=300, NF_el=0)
        # Shot noise should be negligible at -100dBm
        np.testing.assert_allclose(S, [expected_th, expected_th], rtol=1e-1)

    def test_optimum_threshold(self):
        mu0, mu1 = 0, 1
        S0, S1 = 0.1, 0.1
        # Symmetric noise, threshold should be 0.5
        th = optimum_threshold(mu0, mu1, S0, S1, 'ook')
        self.assertAlmostEqual(th, 0.5)

    def test_theory_BER(self):
        # Test OOK BER
        # High power -> low BER
        ber_low = theory_BER(P_avg=0, modulation='ook', amplify=False)
        self.assertTrue(ber_low < 1e-9)
        
        # Low power -> high BER
        ber_high = theory_BER(P_avg=-60, modulation='ook', amplify=False)
        self.assertTrue(ber_high > 1e-3)

    def test_shortest_int(self):
        x = np.random.normal(0, 1, 10000)
        interval = shortest_int(x, 50)
        # For normal distribution, 50% interval is approx [-0.67, 0.67]
        self.assertTrue(interval[0] < 0 < interval[1])
        self.assertAlmostEqual(interval[1] - interval[0], 1.35, delta=0.2)

    def test_pulses(self):
        # Test rcos_pulse
        h_rcos = rcos_pulse(beta=0.5, span=4, sps=8)
        self.assertEqual(len(h_rcos), 4*8+1)
        
        # Test gauss_pulse
        h_gauss = gauss_pulse(span=4, sps=8)
        self.assertEqual(len(h_gauss), 4*8+1)
        
        # Test nrz_pulse
        h_nrz = nrz_pulse(span=4, sps=8, T=1)
        self.assertEqual(len(h_nrz), 4*8+1)

    def test_upfir(self):
        x = np.array([1, 1])
        h = np.array([1, 1])
        # up=1 -> conv([1,1], [1,1]) = [1, 2, 1] -> mode='same' -> [1, 2] or [2, 1] depending on centering
        # scipy.signal.fftconvolve with mode='same' centers the output
        y = upfir(x, h, up=1)
        self.assertEqual(len(y), len(x))

    def test_phase_estimator(self):
        t = np.linspace(0, 1, 1000)
        f = 10
        phi0 = 0.5
        amp0 = 2.0
        # x = A * cos(2*pi*f*t + phi)
        y = amp0 * np.cos(2*pi*f*t + phi0)
        
        est_phi, est_amp = phase_estimator(t, y, f)
        
        # Check phase
        diff = np.angle(np.exp(1j*(est_phi - phi0)))
        self.assertTrue(np.abs(diff) < 1e-2)
        
        # Check amplitude
        self.assertAlmostEqual(est_amp, amp0, delta=1e-2)

if __name__ == '__main__':
    unittest.main()