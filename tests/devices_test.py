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
    global_variables,
    binary_sequence,
    optical_signal,
    electrical_signal,
    idb,
    idbm,
    dbm,
    plt
)

from opticomlib.devices import (
    PRBS,
    MZM,
)

gv = global_variables()

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


    def test_MZM(self):
        assert_raises(TypeError, MZM, op_input=electrical_signal(np.ones(5)), el_input=3) # op_input must be an optical_signal
        assert_raises(ValueError, MZM, op_input=optical_signal(np.ones(5)), el_input=[1,2,3]) # el_input must be a scalar or an array_like with the same length as op_input
        assert_raises(ValueError, MZM, op_input=optical_signal(np.ones(5)), el_input=3, pol='z') # pol must be one of ['x', 'y']


        gv(R=1e9, N=20, sps=512, Vpi=5)

        op_input = optical_signal(np.ones(gv.N*gv.sps))*idbm(0)**0.5 # 1 mW of peak power
        el_inputs = [gv.Vpi/2, 
                     (np.sin(2*np.pi*gv.R*gv.t)*gv.Vpi/2).tolist(), 
                     np.sin(2*np.pi*gv.R*gv.t)*gv.Vpi/2, 
                     electrical_signal(np.sin(2*np.pi*gv.R*gv.t)*gv.Vpi/2)]

        for el_input in el_inputs:
            with self.subTest(type = type(el_input)):
                mzm = MZM(op_input, el_input, bias=gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=30, pol='x', BW=None)
                

                assert_(mzm.type() == optical_signal)
                assert_equal(mzm.n_pol, 1)
                assert_equal(mzm.len(), op_input.len())
                assert_allclose(dbm(mzm.abs().min()**2), dbm(op_input.power())-32) # check that min power is Po - ER - Loss
                if not isinstance(el_input, (int, float)):
                    assert_allclose(dbm(mzm.abs().max()**2), dbm(op_input.power())-2) # check that max power is Po - Loss
                
                mzm = MZM(op_input, el_input, bias=gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=30, pol='y', BW=10e9)
                # when set a BW, the ER of the output signal is degraded. 

                assert_(mzm.type() == optical_signal)
                assert_equal(mzm.n_pol, 1)
                assert_equal(mzm.len(), op_input.len())
                assert_(dbm(mzm.abs().min()**2) < dbm(op_input.power())-30) # check that min power is under -30 dB
                if not isinstance(el_input, (int, float)):
                    assert_(dbm(mzm.abs().max()**2) > dbm(op_input.power())-3) # check that max power is upper -3 dB


        op_input = optical_signal(np.ones(gv.N*gv.sps), n_pol=2)*idbm(0)**0.5 # 1 mW of peak power for each polarization mode
        
        for el_input in el_inputs:
            with self.subTest(type = type(el_input)):
                mzm = MZM(op_input, el_input, bias=gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=30, pol='x', BW=None)

                assert_(mzm.type() == optical_signal)
                assert_equal(mzm.n_pol, 2)
                assert_equal(mzm.len(), op_input.len())
                
                assert_allclose(dbm(mzm.abs()[0].min()**2), dbm(op_input.power()[0])-32)
                if not isinstance(el_input, (int, float)):
                    assert_allclose(dbm(mzm.abs()[0].max()**2), dbm(op_input.power()[0])-2)

                assert_equal(mzm.signal[1], 0)

                mzm = MZM(op_input, el_input, bias=gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=30, pol='y', BW=None)

                assert_(mzm.type() == optical_signal)
                assert_equal(mzm.n_pol, 2)
                assert_equal(mzm.len(), op_input.len())

                assert_allclose(dbm(mzm.abs()[1].min()**2), dbm(op_input.power()[1])-32)
                if not isinstance(el_input, (int, float)):
                    assert_allclose(dbm(mzm.abs()[1].max()**2), dbm(op_input.power()[1])-2)

                assert_equal(mzm.signal[0], 0)
                

if __name__ == '__main__':
    unittest.main()