import unittest
import numpy as np

from opticomlib.devices import (
    PRBS,
)
class TestDevices(unittest.TestCase):
    def test_PRBS(self):
        with self.assertRaises(TypeError):
            PRBS(order=15, len='20') # len must be an integer
        with self.assertRaises(ValueError):
            PRBS(order=8) # order must be one of [7, 9, 11, 15, 20, 23, 31]

        prbs = PRBS(order=15, len=20)
        self.assertEqual(len(prbs), 20) 

if __name__ == '__main__':
    unittest.main()