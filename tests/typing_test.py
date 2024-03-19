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

if __name__ == '__main__':
    unittest.main()