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
    plt,
    RealNumber,
    eye,
)

from opticomlib.devices import (
    PRBS,
    DAC,
    MZM,
    PD,
    LASER,
    PM,
    BPF,
    EDFA,
    DM,
    FIBER,
    DBP,
    LPF,
    ADC,
    GET_EYE,
    SAMPLER,
    FBG,
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
                assert_(prbs.type == binary_sequence)
                assert_equal(prbs.data, data_out[i])

        assert_equal(PRBS(7, len=2*127), PRBS(7, len=127)*2) # checking lengths longer than 2**order-1


    def test_DAC(self):
        assert_raises(ValueError, DAC, '010', pulse_shape='triangle') # pulse_shape must be one of ['gaussian', 'rect', 'nrz', 'rz']
        assert_raises(ValueError, DAC, '010', Vpp=50) # Vpp must be in a range of [-48, 48]
        assert_raises(ValueError, DAC, '010', offset=50) # offset must be in a range of [-48, 48]
        assert_raises(ValueError, DAC, '010', pulse_shape='gaussian', T=0) # T must be greater than 0
        assert_raises(ValueError, DAC, '010', pulse_shape='gaussian', T=3*gv.sps) # T must be less than 2*sps
        assert_raises(ValueError, DAC, '010', pulse_shape='gaussian', T=8, m=0) # m must be greater than 0
        
        assert_raises(TypeError, DAC, '010', Vpp='5') # Vpp must be a real number
        assert_raises(TypeError, DAC, '010', offset=1+1j) # offset must be a real number
        assert_raises(TypeError, DAC, '010', pulse_shape='gaussian', T=8.5) # T must be an integer
        assert_raises(TypeError, DAC, '010', pulse_shape='gaussian', m=1.5) # m must be an integer
        assert_raises(TypeError, DAC, '010', pulse_shape='gaussian', c=1+1j) # c must be a real number

        
        gv(sps=16, R=1e9)
        # test NRZ pulse shape

        dac = DAC('010', pulse_shape='nrz', Vpp=5, offset=0)
        assert_equal(dac.type, electrical_signal)
        assert_equal(dac.size, 3*gv.sps)
        assert_allclose(dac.signal, np.concatenate((np.zeros(gv.sps), 5*np.ones(gv.sps), np.zeros(gv.sps))), atol=1e-14)

        # test gaussian pulse shape
        dac = DAC('010', pulse_shape='gaussian', Vpp=5, offset=1, T=8, m=2)
        assert_equal(dac.type, electrical_signal)
        assert_equal(dac.size, 3*gv.sps)



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
                

                assert_(mzm.type == optical_signal)
                assert_equal(mzm.n_pol, 1)
                assert_equal(mzm.size, op_input.size)
                assert_allclose(dbm(mzm.abs().min()**2), dbm(op_input.power())-32) # check that min power is Po - ER - Loss
                if not isinstance(el_input, RealNumber):
                    assert_allclose(dbm(mzm.abs().max()**2), dbm(op_input.power())-2) # check that max power is Po - Loss
                
                mzm = MZM(op_input, el_input, bias=gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=30, pol='y', BW=10e9)
                # when set a BW, the ER of the output signal is degraded. 

                assert_(mzm.type == optical_signal)
                assert_equal(mzm.n_pol, 1)
                assert_equal(mzm.size, op_input.size)
                assert_(dbm(mzm.abs().min()**2) < dbm(op_input.power())-30) # check that min power is under -30 dB
                if not isinstance(el_input, RealNumber):
                    assert_(dbm(mzm.abs().max()**2) > dbm(op_input.power())-3) # check that max power is upper -3 dB


        op_input = optical_signal(np.ones(gv.N*gv.sps), n_pol=2)*idbm(0)**0.5 # 1 mW of peak power for each polarization mode
        
        for el_input in el_inputs:
            with self.subTest(type = type(el_input)):
                mzm = MZM(op_input, el_input, bias=gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=30, pol='x', BW=None)

                assert_(mzm.type == optical_signal)
                assert_equal(mzm.n_pol, 2)
                assert_equal(mzm.size, op_input.size)
                
                assert_allclose(dbm(mzm.abs()[0].min()**2), dbm(op_input.power()[0])-32)
                if not isinstance(el_input, (int, float)):
                    assert_allclose(dbm(mzm.abs()[0].max()**2), dbm(op_input.power()[0])-2)

                assert_equal(mzm.signal[1], 0)

                mzm = MZM(op_input, el_input, bias=gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=30, pol='y', BW=None)

                assert_(mzm.type == optical_signal)
                assert_equal(mzm.n_pol, 2)
                assert_equal(mzm.size, op_input.size)

                assert_allclose(dbm(mzm.abs()[1].min()**2), dbm(op_input.power()[1])-32)
                if not isinstance(el_input, (int, float)):
                    assert_allclose(dbm(mzm.abs()[1].max()**2), dbm(op_input.power()[1])-2)

                assert_equal(mzm.signal[0], 0)
                

    def test_PD(self):
        input = optical_signal(np.ones(100), np.random.normal(0,0.1,100), n_pol=2)

        assert_raises(TypeError, PD, input=electrical_signal([1,2,3]), BW=5e9) # op_input must be an optical_signal
        assert_raises(ValueError, PD, input, BW=5e9, r=0) # r must be greater than 0
        assert_raises(ValueError, PD, input, BW=5e9, T=-10) # T must be greater than 0
        assert_raises(ValueError, PD, input, BW=5e9, R_load=-50) # R_load must be greater than 0
        assert_raises(TypeError, PD, input, BW=5e9, include_noise=True) # include_noise must be a string
        
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='all')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='ase-only')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='thermal-only')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='shot-only')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='ase-thermal')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='ase-shot')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='thermal-shot')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='none')
        assert_(pd.type == electrical_signal)
        assert_equal(pd.size, input.size)
        assert_allclose(pd.mean(), input.power().sum()*50, rtol=1e-1)

        input = optical_signal(np.ones(100), np.random.normal(0,0.1,100), n_pol=1)
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='all')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='ase-only')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='thermal-only')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='shot-only')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='ase-thermal')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='ase-shot')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='thermal-shot')
        pd = PD(input, BW=5e9, r=1, T=200, R_load=50, include_noise='none')
        assert_(pd.type == electrical_signal)
        assert_equal(pd.size, input.size)
        assert_allclose(pd.mean(), input.power().sum()*50, rtol=1e-1)


    def test_LASER(self):
        gv(sps=16, R=1e9)
        P_dBm = 10
        laser = LASER(P0=P_dBm, lw=0, rin=None, df=0)
        assert_equal(laser.type, optical_signal)
        assert_allclose(np.abs(laser.signal)**2, 10**(P_dBm/10)*1e-3)

        # Test with noise
        laser = LASER(P0=P_dBm, lw=1e6, rin=-150)
        assert_equal(laser.type, optical_signal)

    def test_PM(self):
        gv(sps=16, R=1e9)
        op_input = LASER(P0=10)
        el_input = electrical_signal(np.zeros(op_input.size))
        
        # Test constant phase
        pm = PM(op_input, el_input=0, Vpi=5)
        assert_allclose(pm.signal, op_input.signal)
        
        # Test pi phase shift
        pm = PM(op_input, el_input=5, Vpi=5)
        assert_allclose(pm.signal, op_input.signal * np.exp(1j*np.pi))

    def test_BPF(self):
        gv(sps=16, R=1e9)
        op_input = LASER(P0=10)
        bpf = BPF(op_input, BW=10e9)
        assert_equal(bpf.type, optical_signal)
        assert_equal(bpf.size, op_input.size)

    def test_EDFA(self):
        gv(sps=16, R=1e9)
        op_input = LASER(P0=10)
        G_dB = 20
        edfa = EDFA(op_input, G=G_dB, NF=5)
        
        # Check gain
        # Note: EDFA adds noise, so power won't be exactly G * Pin
        # But signal component should be amplified
        assert_allclose(np.abs(edfa.signal[0]), np.abs(op_input.signal) * 10**(G_dB/20))
        assert_allclose(edfa.signal[1], np.zeros_like(edfa.signal[1]))
        assert_equal(edfa.n_pol, 2) # EDFA adds noise in both polarizations

    def test_DM(self):
        gv(sps=16, R=1e9)
        op_input = LASER(P0=10)
        dm = DM(op_input, D=17)
        assert_equal(dm.type, optical_signal)
        assert_equal(dm.size, op_input.size)
        

    def test_FIBER(self):
        gv(sps=16, R=1e9)
        op_input = LASER(P0=10)
        fiber = FIBER(op_input, length=10, alpha=0.2)
        assert_equal(fiber.type, optical_signal)
        
        # Check attenuation
        # Pout = Pin * exp(-alpha * L)
        # alpha in dB/km -> alpha_lin = alpha_dB / 4.343
        alpha_lin = 0.2 / 4.343
        expected_power = np.mean(np.abs(op_input.signal)**2) * np.exp(-alpha_lin * 10)
        actual_power = np.mean(np.abs(fiber.signal)**2)
        assert_allclose(actual_power, expected_power, rtol=1e-3)

    def test_DBP(self):
        gv(sps=16, R=1e9)
        op_input = LASER(P0=10)
        # Back-to-back DBP should recover signal (ignoring noise/nonlinearities for simple case)
        fiber = FIBER(op_input, length=10, alpha=0, beta_2=0, gamma=0)
        dbp = DBP(fiber, length=10, alpha=0, beta_2=0, gamma=0)
        assert_allclose(dbp.signal, op_input.signal, atol=1e-5)

    def test_LPF(self):
        gv(sps=16, R=1e9)
        el_input = electrical_signal(np.ones(100))
        lpf = LPF(el_input, BW=1e9)
        assert_equal(lpf.type, electrical_signal)
        assert_equal(lpf.size, el_input.size)
        
    def test_ADC(self):
        gv(sps=16, R=1e9)
        t = np.linspace(0, 1, 100)
        sig = np.sin(2*np.pi*t)
        el_input = electrical_signal(sig)
        
        adc = ADC(el_input, n=2, otype='n') # 2 bits -> 4 levels (0, 1, 2, 3)
        assert_equal(np.unique(adc.signal).size <= 4, True)
        assert_equal(adc.signal.min() >= 0, True)
        assert_equal(adc.signal.max() <= 3, True)

    def test_GET_EYE(self):
        gv(sps=16, R=1e9)
        dac = DAC('010101', pulse_shape='nrz', Vpp=1)
        eye_obj = GET_EYE(dac, nslots=10)
        assert_equal(type(eye_obj), eye)
        assert_equal(eye_obj.sps, gv.sps)

    def test_SAMPLER(self):
        gv(sps=4, R=1e9)
        # 0 0 0 0 | 1 1 1 1 | 0 0 0 0
        dac = DAC('010', pulse_shape='nrz', Vpp=1) 
        # Sample at instant 0 (start of bit) -> 0, 1, 0
        # Wait, SAMPLER(input, instant) -> input[instant::sps]
        # if instant=0 -> indices 0, 4, 8 -> values 0, 1, 0
        sampled = SAMPLER(dac, instant=0)
        assert_allclose(sampled.signal, [0, 1, 0], atol=1e-15)
        
        # if instant=2 (middle of bit) -> indices 2, 6, 10 -> values 0, 1, 0
        sampled = SAMPLER(dac, instant=2)
        assert_allclose(sampled.signal, [0, 1, 0], atol=1e-15)

    def test_FBG(self):
        gv(sps=16, R=1e9)
        op_input = LASER(P0=10)
        fbg = FBG(op_input, fc=gv.f0, vdneff=1e-4, kL=2)
        assert_equal(fbg.type, optical_signal)
        

if __name__ == '__main__':
    unittest.main()