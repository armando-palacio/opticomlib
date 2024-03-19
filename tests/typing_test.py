import unittest
import numpy as np
from scipy.constants import c, pi
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from opticomlib.typing import (
    global_variables,
    binary_sequence,
    electrical_signal,
    optical_signal,
    eye,
)

class TestGlobalVariables(unittest.TestCase):
    def test_global_variables(self):
        gv = global_variables()

        ## Test default attributes
        assert_(gv.sps == 16)
        assert_(gv.R == 1e9)
        assert_(gv.fs == 16e9)
        assert_(gv.dt == 1/16e9)
        assert_(gv.wavelength == 1550e-9)
        assert_(gv.f0 == c/1550e-9)
        assert_(gv.N is None)
        assert_(gv.t is None)
        assert_(gv.dw is None)
        assert_(gv.w is None)
        assert_raises(AttributeError, lambda: gv.alpha)

        ## Test __call__ method
        gv(sps=8, R=10e9, wavelength=1549e-9, N=10, alpha=0.2)

        assert_(gv.sps == 8)
        assert_(gv.R == 10e9)
        assert_(gv.fs == 80e9)
        assert_(gv.dt == 1/80e9)
        assert_(gv.wavelength == 1549e-9)
        assert_(gv.f0 == c/1549e-9)
        assert_(gv.N == 10)
        assert_equal(gv.t, np.linspace(0, gv.N*gv.sps*gv.dt, gv.N*gv.sps, endpoint=True))
        assert_(gv.dw is not None)
        assert_allclose(gv.w, np.fft.fftshift(np.fft.fftfreq(gv.N*gv.sps, gv.dt))*2*pi, atol=0)
        assert_(gv.alpha == 0.2)

        ## Test print
        try:
            gv.print()
        except Exception as e:
            self.fail(f"gv.print() raised {type(e).__name__} unexpectedly!")

class TestBinarySequence(unittest.TestCase):
    inputs = ['000011110000',
                [0,0,0,0,1,1,1,1,0,0,0,0],
                (0,0,0,0,1,1,1,1,0,0,0,0),
                np.array([0,0,0,0,1,1,1,1,0,0,0,0])]
    
    def test_init(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_((bits == [0,0,0,0,1,1,1,1,0,0,0,0]).data.all()) # Check if the data is correct, and test the __eq__ method
                assert_(bits.len() == 12) # Check that the length is correct, and test the len method
                assert_(bits.execution_time is None) # Check that the execution time is None

    def test_print(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                try:
                    bits.print()
                except Exception as e:
                    self.fail(f"bits.print() raised {type(e).__name__} unexpectedly!")
    
    def test_getitem(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_(bits[0] == 0)
                assert_((bits[:] == [0,0,0,0,1,1,1,1,0,0,0,0]).data.all())
                assert_((bits[4:-1] == [1,1,1,1,0,0,0]).data.all())
                assert_raises(IndexError, lambda: bits[12])

    def test_add_and_radd(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                
                assert_((bits + bits == [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0]).data.all())
                assert_((bits + '0101' == [0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,1]).data.all())
                assert_(('0101' + bits == [0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0]).data.all())
                assert_raises(TypeError, lambda: bits + 1)
                assert_raises(TypeError, lambda: 1 + bits)
                assert_raises(ValueError, lambda: '020' + bits)
                assert_raises(ValueError, lambda: bits + [0,3,0])
                
    def test_invert(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_((~bits == [1,1,1,1,0,0,0,0,1,1,1,1]).data.all())
    
    def test_ones_and_zeros(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_(bits.ones() == 4)
                assert_(bits.zeros() == 8)

class TestElectricalSignal(unittest.TestCase):
    inputs = [np.arange(100).tolist(), np.arange(100)] 

    def test_init(self):
        assert_raises(KeyError, lambda: electrical_signal())
        assert_raises(ValueError, lambda: electrical_signal([0,1,2], [0,1,2,3]))
        assert_raises(ValueError, lambda: electrical_signal([0,1,2,3], [0,1,2]))
        assert_raises(TypeError, lambda: electrical_signal(signal='0,0.1,0.2,0.3'))
        assert_raises(TypeError, lambda: electrical_signal(noise='0,0.1,0.2,0.3'))

        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal(signal.signal, np.arange(100))
                assert_equal(signal.noise, np.zeros(100))
                assert_equal(signal.execution_time, None)
                assert_equal(signal.len(), 100)

    def test_print(self):
        x = np.linspace(0, 1, 100)
        y = np.sin(2*pi*5*x)
        signal = electrical_signal(x, y)
        try:
            signal.print("something")
        except Exception as e:
            self.fail(f"signal.print() raised {type(e).__name__} unexpectedly!")

    def test_getitem(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                assert_equal(signal[0][0], 0)
                assert_equal(signal[:].signal, np.arange(100))
                assert_equal(signal[4:-1].signal, np.arange(4,99))
                assert_equal(signal.noise, np.zeros(100))
                assert_raises(IndexError, lambda: signal[100])

    def test_add_and_radd(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal((signal + signal).signal, np.arange(100)*2)
                assert_equal((signal + 1).signal, np.arange(100)+1)
                assert_equal((1 + signal).signal, np.arange(100)+1)
                assert_raises(ValueError, lambda: signal + [0,3,0]) # Different length
                
    def test_sub_and_rsub(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal((signal - signal).signal, np.zeros(100))
                assert_equal((signal - 1).signal, np.arange(100)-1)
                assert_equal((1 - signal).signal, 1-np.arange(100))
                assert_raises(ValueError, lambda: signal - [0,3,0]) # Different length

    def test_mul_and_rmul(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal((signal * signal).signal, np.arange(100)**2)
                assert_equal((signal * 2).signal, np.arange(100)*2)
                assert_equal((2 * signal).signal, np.arange(100)*2)
                assert_raises(ValueError, lambda: signal * [0,3,0]) # Different length
    
    def test_gt_and_lt(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal((signal + 1 > 0),  np.ones(100))
                assert_equal((signal < 100), np.ones(100))

    def test_call(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                assert_equal(signal('t').signal, np.fft.ifft(input))
                assert_equal(signal('f').signal, np.fft.fft(input))
                assert_equal(signal('w').signal, np.fft.fft(input))
                assert_raises(ValueError, lambda: signal('z'))

                assert_equal(signal('t', shift=True).signal, np.fft.ifftshift(np.fft.ifft(input)))
    
    def test_power(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input, noise=np.ones(100))
                assert_equal(signal.power(), np.sum(np.abs(input + signal.noise)**2)/len(input))
                assert_equal(signal.power('signal'), np.sum(np.abs(input)**2)/len(input))
                assert_equal(signal.power('noise'), 1)
                assert_raises(ValueError, lambda: signal.power('z'))

    def test_abs(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input, noise=np.ones(100))
                assert_equal(signal.abs(), np.abs(input + signal.noise))
                assert_equal(signal.abs('signal'), np.abs(input))
                assert_equal(signal.abs('noise'), np.ones(100))
                assert_raises(ValueError, lambda: signal.abs('z'))

    def test_phase(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input, noise=np.ones(100))
                assert_equal(signal.phase(), np.unwrap(np.angle(input + signal.noise)))
    
    def test_apply(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input, noise=np.ones(100))
                assert_equal(signal.apply(np.abs).signal, np.abs(input))
                assert_equal(signal.apply(np.abs).noise, np.abs(signal.noise))

if __name__ == '__main__':
    unittest.main()