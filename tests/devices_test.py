import unittest
import numpy as np

from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from opticomlib import (
    binary_sequence
)

from opticomlib.devices import (
    PRBS,
)
class TestDevices(unittest.TestCase):
    def test_PRBS(self):
        assert_raises(TypeError, PRBS, order=15, len='20') # len must be an integer
        assert_raises(ValueError, PRBS, order=8) # order must be one of [7, 9, 11, 15, 20, 23, 31]
        assert_raises(ValueError, PRBS, order=7, len=0) # len must be greater than 0

        assert_equal(PRBS(7, len=10, seed=0), [1,0,0,0,0,0,1,1,0,0]) # if seed=0 it will be set to 1

        # for default seed
        data_out = [[1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1], # first 20 bits of PRBS7
                    [1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1], # first 20 bits of PRBS9
                    [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1], # first 20 bits of PRBS11
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # first 20 bits of PRBS15
                    [1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0], # first 20 bits of PRBS20
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], # first 20 bits of PRBS23
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] # first 20 bits of PRBS31

        for i, order in enumerate([7, 9, 11, 15, 20, 23, 31]):
            with self.subTest(order=order):
                prbs = PRBS(order=order, len=20)
                
                assert_equal(len(prbs), 20)
                assert_(prbs.type() == binary_sequence)
                assert_equal(prbs.data, data_out[i])

        assert_equal(PRBS(7, len=2*127), PRBS(7, len=127).data.tolist()*2) # checking lengths longer than 2**order-1

            

if __name__ == '__main__':
    unittest.main()