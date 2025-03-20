import unittest
import numpy as np

from opticomlib.ppm import (
    binary_sequence, 
    PPM_ENCODER, 
    PPM_DECODER, 
    HDD, 
    SDD, 
    THRESHOLD_EST, 
    DSP, 
    BER_analizer,
    theory_BER,
    eye, 
)

from opticomlib import global_variables
gv = global_variables()

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

        inputs = [eye(**{'mu0':mu0[i], 'mu1':mu1[i], 's0':s0[i], 's1':s1[i]}) for i in range(4)]

        M = 4
        out = [0.514014014014014, 0.6532532532532533, 0.8085085085085084, 0.9693693693693693]

        for i, inp in enumerate(inputs):
            with self.subTest(i=i):
                self.assertTrue(THRESHOLD_EST(inp, M) == out[i])   

        self.assertRaises(ValueError, THRESHOLD_EST, inputs[0], 5) # check that M is a power of 2

    def test_ber_analizer(self):
        bits = np.random.randint(0, 2, 2**11) # binary sequence of 2^11 bits
        eye_obj = eye(**{'mu0':0.0, 'mu1':1.0, 's0':0.1, 's1':0.1})
        M = 4

        # First, test raises conditions work properly
        with self.subTest(mode='hi'):
            self.assertRaises(ValueError, BER_analizer, mode='hi') # check that mode is either 'counter' or 'estimator'
        with self.subTest(Tx=None, Rx=None):
            self.assertRaises(KeyError, BER_analizer, 'counter') # check that a key error is raised when mode='counter', and 'Tx' and 'Rx' are not provided
        with self.subTest(eye_obj=None, M=None):
            self.assertRaises(KeyError, BER_analizer, 'estimator') # check that a key error is raised when mode='estimator', and 'eye_obj' and 'M' are not provided
        with self.subTest(M=5):
            self.assertRaises(ValueError, BER_analizer, 'estimator', eye_obj=eye(), M=5) # check that M is a power of 2
        with self.subTest(decision='hi'):
            self.assertRaises(ValueError, BER_analizer, 'estimator', eye_obj=eye(), M=M, decision='hi') # check that decision is either 'hard' or 'soft'

        # Second, test that BER is 0.0 when the input and output are the same
        with self.subTest(mode='counter', Tx=bits, Rx=bits):
            self.assertTrue(BER_analizer('counter', Tx=bits, Rx=bits) == 0.0) # check that BER is 0.0
        with self.subTest(mode='estimator', eye_obj=eye_obj, M=M):
            self.assertTrue(BER_analizer('estimator', eye_obj=eye_obj, M=M) < 1e-11)
        

    def test_dsp(self):
        from opticomlib.devices import DAC, PRBS
        
        M = 4
        bits = PRBS(order=11)[:-1] # decoded sequence
        
        ppm_seq = PPM_ENCODER(bits, M)
        inputs = DAC(ppm_seq, pulse_shape='gaussian')
        inputs.noise = np.random.normal(0, 0.05, inputs.len())

        rx_soft = DSP(inputs, M, decision='soft')
        rx_hard = DSP(inputs, M, decision='hard')

        # First test raises conditions work properly

        ## DSP
        with self.subTest(input=2):
            self.assertRaises(TypeError, DSP, input=2, M=5) # check that input is an Array_Like or electrical_signal
        with self.subTest(input=[1,2,3]):
            self.assertRaises(ValueError, DSP, input=[1,2,3], M=5) # check that input have at least gv.sps samples
        with self.subTest(M=5):
            self.assertRaises(ValueError, DSP, inputs, M=5) # check that M is a power of 2
        with self.subTest(decision='hi'):
            self.assertRaises(ValueError, DSP, inputs, M=8, decision='hi') # check that decision is either 'hard' or 'soft'

        with self.subTest(decision='hard'):
            self.assertTrue(np.array_equal(rx_hard.data, bits.data)) # check hard decision is working properly
        
        with self.subTest(decision='soft'):
            self.assertTrue(np.array_equal(rx_soft.data, bits.data)) # check soft decision is working properly

    def test_theory_ber(self):
        self.assertTrue(theory_BER(1, 0.1, 0.1, 4, 'hard') < 1e-6) # check that BER is 0.0
        self.assertTrue(theory_BER(1, 0.1, 0.1, 4, 'soft') < 1e-11) # check that BER is 0.0
        self.assertRaises(ValueError, theory_BER, 1, 0.1, 0.1, 5, 'hard') # check that M is a power of 2
        self.assertTrue(theory_BER(1, 0.1, 0.1, 4, 'hard') > theory_BER(1, 0.1, 0.1, 4, 'soft')) # check that BER is lower when decision is 'soft' than when it is 'hard'
        self.assertTrue((theory_BER([1,1], [0.1,0.1], [0.1,0.1], 4, 'hard') < 1e-6).all()) # check that BER is 0.0

if __name__ == '__main__':
    unittest.main()