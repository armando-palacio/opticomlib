import unittest
import numpy as np

from opticomlib.ppm import *

class TestPPM(unittest.TestCase):
    def test_ppm_encoder(self):
        inputs = [
            '00011011',
            [0,0,0,1,1,0,1,1],
            (0,0,0,1,1,0,1,1),
            np.array([0,0,0,1,1,0,1,1]),
            binary_sequence('00011011'),
        ]
        M = [4, 8, 16, 32, 64, 128, 256]
        outp = [[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                [1,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,0],
                [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                np.insert(np.zeros(31), 3, 1),
                np.insert(np.zeros(63), 6, 1),
                np.insert(np.zeros(127), 13, 1),
                np.insert(np.zeros(255), 27, 1)]

        for inp in inputs:
            for m, out in zip(M,outp):
                with self.subTest(inp=inp, M=m):
                    self.assertTrue(np.array_equal(PPM_ENCODER(inp, m).data, out))

    def test_ppm_decoder(self):
        imputs = ['1000010000100001',
                [1,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,0],
                (0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0),
                np.insert(np.zeros(31), 3, 1),
                binary_sequence(np.insert(np.zeros(63), 6, 1)),
                np.insert(np.zeros(127), 13, 1),
                np.insert(np.zeros(255), 27, 1)]
        
        M = [4, 8, 16, 32, 64, 128, 256]

        outp = [[0,0,0,1,1,0,1,1],
                [0,0,0,1,1,0],
                [0,0,0,1,1,0,1,1],
                [0,0,0,1,1],
                [0,0,0,1,1,0],
                [0,0,0,1,1,0,1],
                [0,0,0,1,1,0,1,1]]
        
        for i, inp in enumerate(imputs):
            with self.subTest(inp=inp, M=M[i]):
                self.assertTrue(np.array_equal(PPM_DECODER(inp, M[i]).data, outp[i]))

    def test_hdd(self):
        inputs = ['1010 0110 0000 0001',
                 [1,0,1,0, 0,1,1,0, 0,0,0,0, 0,0,0,1],
                 (1,0,1,0, 0,1,1,0, 0,0,0,0, 0,0,0,1),
                 np.array([1,0,1,0, 0,1,1,0, 0,0,0,0, 0,0,0,1]),
                 binary_sequence('1010 0110 0000 0001')]
        M = [4, 8, 16]
        outs_sum = [4, 2, 1]

        for i, m in enumerate(M):
            with self.subTest(M=m):
                self.assertTrue(HDD(inputs[i], m).data.sum() == outs_sum[i]) # check that every output symbol contains only one ON slot

        self.assertRaises(ValueError, HDD, '1010 0110', 3) # check that M is a power of 2
        self.assertRaises(ValueError, HDD, '1010 0110 1', 4) # check that input length is multiple of M

    def test_sdd(self):
        inputs = np.kron([0.1,1.2,0.1,0.2,  0.1,0.9,1.0,1.1,  0.1,0.1,0.1,0.2], np.ones(16))
        M = 4
        out = [0,1,0,0, 0,0,0,1, 0,0,0,1]

        self.assertTrue(np.array_equal(SDD(inputs, M).data, out)) # check that output coincides with maximum value of each slot

        self.assertRaises(ValueError, SDD, inputs, 5) # check that M is a power of 2

    def test_threshold_est(self):
        mu0 = [0.1, 0.2, 0.3, 0.4]
        mu1 = [0.9, 1.0, 1.1, 1.2]
        s0 = s1 = [0.1, 0.2, 0.3, 0.4]

        inputs = [eye({'mu0':mu0[i], 'mu1':mu1[i], 's0':s0[i], 's1':s1[i]}) for i in range(4)]

        M = 4
        out = [0.514014014014014, 0.6532532532532533, 0.8085085085085084, 0.9693693693693693]

        for i, inp in enumerate(inputs):
            with self.subTest(i=i):
                self.assertTrue(THRESHOLD_EST(inp, M) == out[i])   

        self.assertRaises(ValueError, THRESHOLD_EST, inputs[0], 5) # check that M is a power of 2

    def test_dsp(self):
        from opticomlib.devices import DAC
        inputs = DAC(binary_sequence('0100 0010 1000 0001'), pulse_shape='gaussian')
        inputs.noise = np.random.normal(0, 0.1, inputs.len())

        M = 4

        out = [0,1, 1,0, 0,0, 1,1]

        with self.subTest(decision='hard', BW=None):
            self.assertTrue(np.array_equal(DSP(inputs, M, decision='hard')[0].data, out)) # check hard decision is working properly
        with self.subTest(decision='hard', BW=gv.R):
            self.assertTrue(np.array_equal(DSP(inputs, M, decision='hard', BW=gv.R)[0].data, out)) # check hard decision is working properly

        with self.subTest(decision='soft', BW=None):
            self.assertTrue(np.array_equal(DSP(inputs, M, decision='soft').data, out)) # check soft decision is working properly
        with self.subTest(decision='soft', BW=gv.R):
            self.assertTrue(np.array_equal(DSP(inputs, M, decision='soft', BW=gv.R).data, out)) # check soft decision is working properly

        self.assertRaises(ValueError, DSP, inputs, 5) # check that M is a power of 2
        self.assertRaises(ValueError, DSP, inputs, 8, decision='hola') # check that decision is either 'hard' or 'soft'


if __name__ == '__main__':
    unittest.main()