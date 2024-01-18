import sys; sys.path.append(sys.path[0]+"\..") 

import unittest
import components as cp
import numpy as np

class Test_PRBS_component(unittest.TestCase):
    def test_PRBS(self):
        self.assertEqual(cp.PRBS(n=10).type(), cp.binary_sequence)
        self.assertNotEqual(cp.PRBS(n=10).data, [])
        self.assertTrue(np.array_equal( np.unique(cp.PRBS(n=10).data), [0,1] ))
        

class Test_PPM_ENCODER_component(unittest.TestCase):
    def test_PPM_ENCODE(self):
        cases = ['1001110', 
                 cp.str_to_list('100111'), 
                 tuple(cp.str_to_list('100111')), 
                 np.array(cp.str_to_list('100111')), 
                 cp.binary_sequence('100111')]
        
        M = 4
        k = np.log2(M)

        for bs in cases:
            self.assertEqual(cp.PPM_ENCODER(bs, M).type(), cp.binary_sequence)
            self.assertNotEqual(cp.PPM_ENCODER(bs, M).data, [])
            self.assertTrue(np.array_equal( np.unique(cp.PPM_ENCODER(bs, M).data), [0,1] ))
            self.assertEqual(cp.PPM_ENCODER(bs, M).len(), len(bs)//k*M)
            self.assertTrue(np.array_equal(cp.PPM_ENCODER(bs, M).data, [0,0,1,0,0,1,0,0,0,0,0,1]))


class Test_DAC(unittest.TestCase):
    def test_DAC(self):
        cases = ['10011101', 
                 cp.str_to_list('100111'), 
                 tuple(cp.str_to_list('100111')), 
                 np.array(cp.str_to_list('100111')), 
                 cp.binary_sequence('100111')]

        sps = cp.global_vars.sps
        fs = cp.global_vars.fs

        for bs in cases:
            self.assertEqual(cp.DAC(bs, sps).type(), cp.electrical_signal)
            self.assertNotEqual(cp.DAC(bs, sps).signal, [])
            self.assertNotEqual(cp.DAC(bs, sps).noise, [])

            self.assertEqual(cp.DAC(bs, sps).len(), len(bs)*sps)
            self.assertEqual(cp.DAC(bs, sps).fs(), fs)

            


if __name__=='__main__':
    unittest.main()