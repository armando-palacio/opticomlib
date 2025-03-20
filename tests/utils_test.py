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

from opticomlib.utils import (
    str2array,
)

class TestUtils(unittest.TestCase):
    def test_str2array(self):
        # test bool values
        assert_equal(str2array('10101'), [1,0,1,0,1])
        assert_equal(str2array('10101', dtype=bool), [1,0,1,0,1])
        assert_equal(str2array('100;101'), [[1,0,0],[1,0,1]])
        assert_equal(str2array('10 100 1000'), [1,0,1,0,0,1,0,0,0]) 
        assert_equal(str2array('1,0,1,0,1'), [1,0,1,0,1])
        assert_equal(str2array('1 0 1 0 1', dtype=bool), [1,0,1,0,1])

        # test int values
        assert_equal(str2array('1 0 1 10', dtype=int), [1,0,1,10])
        assert_equal(str2array('1 -2 1; 4,5,6', dtype=int), [[1, -2, 1], [4, 5, 6]])

        # test float values
        assert_equal(str2array('1 -2 1; 4.1,5,6.4', dtype=float), [[1.0, -2.0, 1.0], [4.1, 5.0, 6.4]])

        # test complex values
        assert_equal(str2array('1+j -2-j 1; 4.1,5,6.4', dtype=complex), [[1+1j, -2-1j, 1+0j], [4.1+0j, 5+0j, 6.4+0j]])




if __name__ == "__main__":
    unittest.main()