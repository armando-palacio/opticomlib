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
    gv
)

from opticomlib.devices import GET_EYE
import matplotlib.pyplot as plt


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
    
    assert_raises(TypeError, lambda: binary_sequence())
    assert_raises(ValueError, lambda: binary_sequence([0,1,2,3]))
    assert_raises(ValueError, lambda: binary_sequence('001201'))
    assert_raises(ValueError, lambda: binary_sequence('001;101'))
    
    def test_init(self):
        assert_(binary_sequence(0).data==0)
        assert_(binary_sequence('1').data==1)

        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_((bits == [0,0,0,0,1,1,1,1,0,0,0,0]).data.all()) # Check if the data is correct, and test the __eq__ method
                assert_(bits.len() == 12) # Check that the length is correct, and test the len method
                assert_(bits.execution_time is None) # Check that the execution time is None

    def test_print(self):
        input = '000011110000'
        bits = binary_sequence(input)
        try:
            bits.print('binary test')
        except Exception as e:
            self.fail(f"bits.print() raised {type(e).__name__} unexpectedly!")
    
    def test_getitem_and_eq(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_((bits[0] == 0).data)
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
                assert_raises(ValueError, lambda: bits + '020')
                assert_raises(ValueError, lambda: bits + [0,3,0])
                assert_raises(ValueError, lambda: [0,3,0] + bits)
                
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
    inputs = [np.arange(100).tolist().__str__().strip('[]'), np.arange(100).tolist(), np.arange(100)] 

    def test_init(self):
        assert_raises(TypeError, lambda: electrical_signal())
        assert_raises(TypeError, lambda: electrical_signal(noise=[1,2,3]))
        assert_raises(ValueError, lambda: electrical_signal([0,1,2], [0,1,2,3]))
        assert_raises(ValueError, lambda: electrical_signal([0,1,2,3], [0,1,2]))
        assert_raises(ValueError, lambda: electrical_signal([[1,2,3]]))

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
            signal.print("electric test")
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

                assert_equal((signal + input).signal, np.arange(100)*2)
                # assert_equal((input + signal).signal, np.arange(100)*2)  ## revisar esta linea

                assert_equal((signal + 1).signal, np.arange(100)+1)
                assert_equal((1 + signal).signal, np.arange(100)+1)
                assert_raises(ValueError, lambda: signal + input[:4]) # Different length
                
    def test_sub_and_rsub(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal((signal - signal).signal, np.zeros(100))
                
                assert_equal((signal - input).signal, np.zeros(100))
                # assert_equal((input - signal).signal, np.zeros(100))  ## revisar esta linea

                assert_equal((signal - 1).signal, np.arange(100)-1)
                assert_equal((1 - signal).signal, 1-np.arange(100))
                assert_raises(ValueError, lambda: signal - input[:4]) # Different length

    def test_mul_and_rmul(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal((signal * signal).signal, np.arange(100)**2)
                
                assert_equal((signal * input).signal, np.arange(100)**2)
                # assert_equal((input * signal).signal, np.arange(100)**2)  ## revisar esta linea
                
                assert_equal((signal * 2).signal, np.arange(100)*2)
                assert_equal((2 * signal).signal, np.arange(100)*2)
                assert_raises(ValueError, lambda: signal * input[:4]) # Different length
    
    def test_gt_and_lt(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                
                assert_equal((signal + 1 > 0).data,  np.ones(100))
                assert_equal((signal < 100).data, np.ones(100))

                assert_equal((signal > input).data,  np.zeros(100))
                assert_equal((signal < input).data,  np.zeros(100))
                assert_raises(ValueError, lambda: signal > input[:4])

    def test_call(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input)
                assert_equal(signal('t').signal, np.fft.ifft(np.arange(100)))
                assert_equal(signal('f').signal, np.fft.fft(np.arange(100)))
                assert_equal(signal('w').signal, np.fft.fft(np.arange(100)))
                assert_raises(ValueError, lambda: signal('z'))

                assert_equal(signal('t', shift=True).signal, np.fft.ifftshift(np.fft.ifft(np.arange(100))))
    
    def test_power(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input, noise=np.ones(100))
                assert_equal(signal.power(), np.sum(np.abs(np.arange(100) + signal.noise)**2)/100)
                assert_equal(signal.power('signal'), np.sum(np.abs(np.arange(100))**2)/100)
                assert_equal(signal.power('noise'), 1)
                assert_raises(ValueError, lambda: signal.power('z'))

    def test_abs(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input, noise=np.ones(100))
                assert_equal(signal.abs(), np.arange(100)+1)
                assert_equal(signal.abs('signal'), np.arange(100))
                assert_equal(signal.abs('noise'), np.ones(100))
                assert_raises(ValueError, lambda: signal.abs('z'))

    def test_phase(self):
        signal = electrical_signal(self.inputs[-1] + self.inputs[-1]*1j)
        assert_equal(signal.phase(), np.concatenate(([0], np.ones(99)*pi/4)))
    
    def test_apply(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                signal = electrical_signal(input, noise=np.ones(100))
                assert_equal(signal.apply(np.abs).signal, np.arange(100))
                assert_equal(signal.apply(np.abs).noise, np.ones(100))
    
    def test_plot(self):
        x = electrical_signal(np.ones(100), np.random.normal(0,0.05,100))
        try:
            assert_(x.plot('-r', n=98, xlabel='Time', ylabel='Intensity', style='light', grid=True, hold=False)==x)
        except Exception as e:
            self.fail(f"x.plot() raised {type(e).__name__} unexpectedly!")

    def test_psd(self):
        x = electrical_signal(np.ones(100), np.random.normal(0,0.05,100))
        try:
            assert_(x.psd('--b', n=98, xlabel='Freq', ylabel='Spectra', style='dark', grid=False, hold=False)==x)
        except Exception as e:
            self.fail(f"x.psd() raised {type(e).__name__} unexpectedly!")




class TestOpticalSignal(unittest.TestCase):
    signals_1p = ['+0,1,2,3,4+0j,5.', 
                  [0,1,2,3,4,5], 
                  np.arange(6)]
    noises_1p = ['+0,-1,-2,-3,-4+0i,-5.',
                [0,-1,-2,-3,-4,-5],
                -np.arange(6)]
    
    signals_2p = ['+0,1,2,3,4+0j,5.; +0,1,2,3,4+0i,5.',
                  [[0,1,2,3,4,5]],
                  np.array([[0,1,2,3,4,5]])]

    noises_2p = ['+0,-1,-2,-3,-4+0j,-5.; +0,-1,-2,-3,-4+0i,-5.',
                 [[0,-1,-2,-3,-4,-5]],
                 -np.array([[0,1,2,3,4,5]])]
    
    def test_init(self):
        assert_raises(TypeError, lambda: optical_signal()) # No input
        assert_raises(TypeError, lambda: optical_signal(noise=[[1,2,3]])) # no signal input
        assert_raises(ValueError, lambda: optical_signal([0,1,2], [0,1,2,3])) # Different length
        assert_raises(ValueError, lambda: optical_signal([0,1,2,3], [0,1,2])) # Different length
        assert_raises(ValueError, lambda: optical_signal([[1,2,3], [5,6,7], [8,9,10]]))
        assert_raises(ValueError, lambda: optical_signal([[[1,2,3]]]))

        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi)
                
                assert_equal(x.signal, np.arange(6))
                assert_equal(x.noise, -np.arange(6))
                assert_equal(x.n_pol, 1)
                assert_equal(x.execution_time, None)
                assert_equal(x.len(), 6)

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)

                assert_equal(x.signal, np.tile(np.arange(6),(2,1)))
                assert_equal(x.noise, -np.tile(np.arange(6),(2,1)))
                assert_equal(x.n_pol, 2)
                assert_equal(x.execution_time, None)
                assert_equal(x.len(), 6)

    def test_print(self):
        x_1p = optical_signal([1,2,3], [1,1,0])
        x_2p = optical_signal([[1,2,3]], [[1,1,0]])
        try:
            x_1p.print("optic test 1 pol")
            x_2p.print("optic test 2 pol")
        except Exception as e:
            self.fail(f"signal.print() raised {type(e).__name__} unexpectedly!")

    def test_getitem(self):
        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi)
                
                assert_(x[0].signal == 0)
                assert_equal(x[:].signal, np.arange(6))
                assert_equal(x[1:-1].signal, np.arange(1,5))
                
                assert_(x[0].noise == 0)
                assert_equal(x[:].noise, -np.arange(6))
                assert_equal(x[1:-1].noise, -np.arange(1,5))
                
                assert_raises(IndexError, lambda: x[10])

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi)
                
                assert_equal(x[0].signal, [[0],[0]])
                assert_equal(x[:].signal, np.tile(np.arange(6),(2,1)))
                assert_equal(x[1:-1].signal, np.array([[1,2,3,4], [1,2,3,4]]))
                
                assert_equal(x[0].noise, [[0],[0]])
                assert_equal(x[:].noise, -np.tile(np.arange(6),(2,1)))
                assert_equal(x[1:-1].noise, -np.array([[1,2,3,4], [1,2,3,4]]))
                
                assert_raises(IndexError, lambda: x[10])

    def test_add_and_radd(self):
        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=1)
                
                # self + self = 2*self
                assert_equal((x + x).signal, np.arange(6)*2) 
                assert_equal((x + x).noise, -np.arange(6)*2) 

                assert_equal((x + sig).signal, np.arange(6)*2) # suma por la derecha
                assert_equal((x + sig).noise, np.zeros(6))

                # assert_equal((sig + x).signal, np.arange(6)*2) # suma por la izquierda
                # assert_equal((sig + x).noise, np.zeros(6))

                assert_equal((x + 1).signal, np.arange(1,7)) # suma por la derecha
                assert_equal((x + 1).noise, -np.arange(-1,5))

                assert_equal((1 + x).signal, np.arange(1,7)) 
                assert_equal((1 + x).noise, -np.arange(-1,5)) 
                
                assert_raises(ValueError, lambda: x + sig[:4]) # Different length

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)
                
                # self + self = 2*self
                assert_equal((x + x).signal, np.tile(np.arange(6),(2,1))*2) 
                assert_equal((x + x).noise, -np.tile(np.arange(6),(2,1))*2) 

                assert_equal((x + sig).signal, np.tile(np.arange(6),(2,1))*2)
                assert_equal((x + sig).noise, np.zeros((2,6)))

                # assert_equal((sig + x).signal, np.tile(np.arange(6),(2,1))*2)
                # assert_equal((sig + x).noise, np.zeros((2,6)))

                assert_equal((x + 1).signal, np.tile(np.arange(6),(2,1))+1)
                assert_equal((x + 1).noise, -np.tile(np.arange(6),(2,1))+1)

                assert_equal((1 + x).signal, np.tile(np.arange(6),(2,1))+1)
                assert_equal((1 + x).noise, -np.tile(np.arange(6),(2,1))+1)

                if type(sig) == str: assert_raises(ValueError, lambda: x + '+0,-1,-2') # Different length
                else: assert_raises(ValueError, lambda: x + sig[0][:3]) # Different length


    def test_sub_and_rsub(self):
        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=1)
                
                # self - self = 0
                assert_equal((x - x).signal, np.zeros(6)) 
                assert_equal((x - x).noise, np.zeros(6)) 

                assert_equal((x - sig).signal, np.zeros(6))
                assert_equal((x - noi).noise, np.zeros(6))

                # assert_equal((sig - x).signal, np.zeros(100))
                # assert_equal((sig - x).noise, np.zeros(100))

                assert_equal((x - 1).signal, np.arange(6)-1)
                assert_equal((x - 1).noise, -np.arange(6)-1)

                assert_equal((1 - x).signal, 1-np.arange(6))
                assert_equal((1 - x).noise, 1+np.arange(6))

                assert_raises(ValueError, lambda: x - sig[:4])

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)
                
                # self - self = 0
                assert_equal((x - x).signal, np.zeros((2,6))) 
                assert_equal((x - x).noise, np.zeros((2,6))) 

                assert_equal((x - sig).signal, np.zeros((2,6)))
                assert_equal((x - noi).noise, np.zeros((2,6)))

                # assert_equal((sig - x).signal, np.zeros(100))
                # assert_equal((sig - x).noise, np.zeros(100))

                assert_equal((x - 1).signal, np.tile(np.arange(6),(2,1))-1)
                assert_equal((x - 1).noise, -np.tile(np.arange(6),(2,1))-1)

                assert_equal((1 - x).signal, 1-np.tile(np.arange(6),(2,1)))
                assert_equal((1 - x).noise, 1+np.tile(np.arange(6),(2,1)))

                if type(sig) == str: assert_raises(ValueError, lambda: x - '+0,-1,-2') # Different length
                else: assert_raises(ValueError, lambda: x - sig[0][:3]) # Different length

    def test_mul_and_rmul(self):
        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=1)
                
                # self * self = self**2
                assert_equal((x * x).signal, np.arange(6)**2) 
                assert_equal((x * x).noise, np.arange(6)**2) 

                assert_equal((x * sig).signal, np.arange(6)**2)
                assert_equal((x * sig).noise, -np.arange(6)**2)

                # assert_equal((sig * x).signal, np.arange(6)**2)
                # assert_equal((sig * x).noise, np.zeros(6))

                assert_equal((x * 2).signal, np.arange(6)*2)
                assert_equal((x * 2).noise, -np.arange(6)*2)

                assert_equal((2 * x).signal, np.arange(6)*2)
                assert_equal((2 * x).noise, -np.arange(6)*2)

                assert_raises(ValueError, lambda: x * sig[:4])

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)
                
                # self * self = self**2
                assert_equal((x * x).signal, np.tile(np.arange(6),(2,1))**2) 
                assert_equal((x * x).noise, np.tile(np.arange(6),(2,1))**2) 

                assert_equal((x * sig).signal, np.tile(np.arange(6),(2,1))**2)
                assert_equal((x * sig).noise, -np.tile(np.arange(6),(2,1))**2)

                # assert_equal((sig * x).signal, np.tile(np.arange(6),(2,1))**2)
                # assert_equal((sig * x).noise, -np.tile(np.arange(6),(2,1))**2)

                assert_equal((x * 2).signal, np.tile(np.arange(6),(2,1))*2)
                assert_equal((x * 2).noise, -np.tile(np.arange(6),(2,1))*2)

                assert_equal((2 * x).signal, np.tile(np.arange(6),(2,1))*2)
                assert_equal((2 * x).noise, -np.tile(np.arange(6),(2,1))*2)

                if type(sig) == str: assert_raises(ValueError, lambda: x * '+0,-1,-2') # Different length
                else: assert_raises(ValueError, lambda: x * sig[0][:3]) # Different length

    def test_call(self):
        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=1)
                
                assert_equal(x('t').signal, np.fft.ifft(np.arange(6)))
                assert_equal(x('t').noise, np.fft.ifft(-np.arange(6)))

                assert_equal(x('f').signal, np.fft.fft(np.arange(6)))
                assert_equal(x('f').noise, np.fft.fft(-np.arange(6)))
                
                assert_equal(x('w').signal, np.fft.fft(np.arange(6)))
                assert_equal(x('w').noise, np.fft.fft(-np.arange(6)))
                
                assert_raises(ValueError, lambda: x('z'))

                assert_equal(x('t', shift=True).signal, np.fft.ifftshift(np.fft.ifft(np.arange(6))))

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)
                
                assert_equal(x('t').signal, np.fft.ifft(np.tile(np.arange(6),(2,1))))
                assert_equal(x('f').signal, np.fft.fft(np.tile(np.arange(6),(2,1))))
                assert_equal(x('w').signal, np.fft.fft(np.tile(np.arange(6),(2,1))))
                assert_raises(ValueError, lambda: x('z'))

                assert_equal(x('t', shift=True).signal, np.fft.ifftshift(np.fft.ifft(np.tile(np.arange(6),(2,1)))))

    def test_power(self):
        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=1)
                
                assert_equal(x.power(), 0)
                assert_equal(x.power('signal'), np.mean(np.arange(6)**2))
                assert_equal(x.power('noise'), np.mean(np.arange(6)**2))
                assert_raises(ValueError, lambda: x.power('z'))

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)
                
                assert_equal(x.power(), 0)
                assert_equal(x.power('signal'), np.tile(np.mean(np.arange(6)**2), 2))
                assert_equal(x.power('noise'), np.tile(np.mean(np.arange(6)**2), 2))
                assert_raises(ValueError, lambda: x.power('z'))

    def test_w(self):
        x = optical_signal(np.arange(6), -np.arange(6), n_pol=1)
        assert_equal(x.w(), 2*pi*np.fft.fftfreq(6)*gv.fs)
        assert_equal(x.w(shift=True), 2*pi*np.fft.fftshift(np.fft.fftfreq(6))*gv.fs)

        x = optical_signal(np.arange(6), -np.arange(6), n_pol=2)
        assert_equal(x.w(), 2*pi*np.fft.fftfreq(6)*gv.fs)
        assert_equal(x.w(shift=True), 2*pi*np.fft.fftshift(np.fft.fftfreq(6))*gv.fs)

    def test_phase(self):
        sig, noi = self.signals_1p[-1][1:], self.noises_1p[-1][1:]

        x = optical_signal(sig+1j*sig, noi-1j*noi, n_pol=1)
        assert_equal(x.phase(), np.ones(5)*pi/2)

        sig, noi = self.signals_2p[-1][:,1:], self.noises_2p[-1][:,1:]

        x = optical_signal(sig+1j*sig, noi-1j*noi, n_pol=2)
        assert_equal(x.phase(), np.tile(np.ones(5)*pi/2,(2,1)))

    def test_plot(self):
        x = optical_signal(np.ones(100), np.random.normal(0,0.05,100), n_pol=1)
        try:
            assert_(x.plot('--r', n=98, mode='x', xlabel='Time', ylabel='Intensity', style='light', grid=True, hold=True)==x)
        except Exception as e:
            self.fail(f"x.plot() raised {type(e).__name__} unexpectedly!")

        x = optical_signal(np.ones(100), np.random.normal(0,0.05,100), n_pol=2)
        try:
            assert_(x.plot(['r','b'], n=98, mode='both', xlabel='Time', ylabel='Intensity', style='dark', grid=True, hold=True)==x)
        except Exception as e:
            self.fail(f"x.plot() raised {type(e).__name__} unexpectedly!")

    def test_psd(self):
        x = optical_signal(np.ones(100), np.random.normal(0,0.05,100), n_pol=1)
        try:
            assert_(x.psd('--r', mode='x', n=98, xlabel='Freq', ylabel='Spectra', style='light', grid=True, hold=True)==x)
        except Exception as e:
            self.fail(f"x.psd() raised {type(e).__name__} unexpectedly!")

        x = optical_signal(np.ones(100), np.random.normal(0,0.05,100), n_pol=2)
        try:
            assert_(x.psd('--b', mode='both', n=98, xlabel='Freq', ylabel='Spectra', style='dark', grid=False, hold=True)==x)
        except Exception as e:
            self.fail(f"x.psd() raised {type(e).__name__} unexpectedly!")


class testEye(unittest.TestCase):
    def test_eye(self):
        x = eye()

        assert_raises(ValueError, lambda: x.print())
        assert_raises(ValueError, lambda: x.plot())

        sig = np.kron(10*[1,0], np.ones(128)) + np.random.normal(0,0.05,128*20)
        
        x = GET_EYE(sig) # this returns an eye object with estimated parameters
        
        assert_(x.plot(medias_=True, legend_=True, style='light', cmap='plasma', label='TEST')==x)


if __name__ == '__main__':
    unittest.main()