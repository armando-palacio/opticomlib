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
    eye
)

gv = global_variables()

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
                assert_(bits == [0,0,0,0,1,1,1,1,0,0,0,0]) # Check if the data is correct, and test the __eq__ method
                assert_(bits.len() == 12) # Check that the length is correct, and test the len method
                assert_(bits.execution_time == 0) # Check that the execution time is None

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
                assert_(bits[0] == 0)
                assert_(bits[:] == [0,0,0,0,1,1,1,1,0,0,0,0])
                assert_(bits[4:-1] == [1,1,1,1,0,0,0])
                assert_raises(IndexError, lambda: bits[12])
    
    def test_add_and_radd(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                
                assert_(bits + bits == [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0])
                assert_(bits + '0101' == [0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,1])
                assert_('0101' + bits == [0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0])
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
                assert_(~bits == [1,1,1,1,0,0,0,0,1,1,1,1])
    
    def test_ones_and_zeros(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_(bits.ones() == 4)
                assert_(bits.zeros() == 8)




class TestElectricalSignal(unittest.TestCase):
    signals = ['+0,1,2,3,4+0j,5.', 
                  [0,1,2,3,4.,5], 
                  np.arange(6)]
    noises = ['+0,-1,-2,-3,-4+0i,-5.',
                [0,-1,-2,-3,-4.,-5],
                -np.arange(6)]

    def test_init(self):
        assert_raises(TypeError, lambda: electrical_signal())
        assert_raises(TypeError, lambda: electrical_signal(noise=[1,2,3]))
        assert_raises(ValueError, lambda: electrical_signal([0,1,2], [0,1,2,3]))
        assert_raises(ValueError, lambda: electrical_signal([0,1,2,3], [0,1,2]))
        assert_raises(ValueError, lambda: electrical_signal([[1,2,3]]))

        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                assert_equal(x.signal, np.arange(6))
                assert_equal(x.noise, -np.arange(6))
                assert_equal(x.execution_time, 0)
                assert_(x.len()==6)

                x = electrical_signal(sig) # No noise

                assert_equal(x.signal, np.arange(6))
                assert_(x.noise is None)
                assert_equal(x.execution_time, 0)
                assert_(x.len()==6)

    def test_print(self):
        x = np.linspace(0, 1, 100)
        y = np.sin(2*pi*5*x)
        signal = electrical_signal(x, y)
        try:
            signal.print("electric test")
        except Exception as e:
            self.fail(f"signal.print() raised {type(e).__name__} unexpectedly!")

    def test_getitem(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                # test int indexing 
                assert_(x[0].type() == electrical_signal)
                assert_equal(x[0].signal, 0)
                assert_equal(x[-1].noise, -5)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type() == electrical_signal)
                assert_equal(x[:].signal, np.arange(6))
                assert_equal(x[:].noise, -np.arange(6))
                assert_equal(x[1:-1].signal, np.arange(1,5))
                assert_equal(x[1:-1].noise, -np.arange(1,5))
                assert_equal(x[:100].signal, np.arange(6))

                x = electrical_signal(sig) # No noise

                # test int indexing
                assert_(x[0].type() == electrical_signal)
                assert_equal(x[0].signal, 0)
                assert_(x[0].noise is None)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type() == electrical_signal)
                assert_equal(x[:].signal, np.arange(6))
                assert_(x[:].noise is None)
                assert_equal(x[1:-1].signal, np.arange(1,5))
                assert_(x[1:-1].noise is None)
                assert_equal(x[:100].signal, np.arange(6))

    def test_add_and_radd(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                y = x + x
                assert_equal(y.signal, np.arange(6)*2)
                assert_equal(y.noise, -np.arange(6)*2)

                y = x + sig
                assert_equal(y.signal, np.arange(6)*2)
                assert_equal(y.noise, x.noise)

                # y = sig + x
                # assert_equal(y.signal, np.arange(6)*2)  ## revisar esta linea
                # assert_equal(y.noise, x.noise)  ## revisar esta linea
                # 

                y = x + 1
                assert_equal(y.signal, np.arange(6)+1)
                assert_equal(y.noise, x.noise)

                y = 1 + x
                assert_equal(y.signal, np.arange(6)+1)
                assert_equal(y.noise, x.noise)
                assert_raises(ValueError, lambda: x + sig[:4]) # Different length

                x = electrical_signal(sig) # No noise

                y = x + x
                assert_equal(y.signal, np.arange(6)*2)
                assert_(y.noise is None)

                y = x + sig
                assert_equal(y.signal, np.arange(6)*2)
                assert_(y.noise is None)

                # y = sig + x
                # assert_equal((sig + x).signal, np.arange(6)*2)  ## revisar esta linea
                # assert_((sig + x).noise is None)  ## revisar esta linea
                # 

                y = x + 1
                assert_equal(y.signal, np.arange(6)+1)
                assert_(y.noise is None)

                y = 1 + x
                assert_equal(y.signal, np.arange(6)+1)
                assert_(y.noise is None)
                assert_raises(ValueError, lambda: x + sig[:4])


    def test_sub_and_rsub(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                y = x - x
                assert_equal(y.signal, np.zeros(6))
                assert_equal(y.noise, np.zeros(6))

                y = x - sig
                assert_equal(y.signal, np.zeros(6))
                assert_equal(y.noise, x.noise)

                # y = sig - y
                # assert_equal(y.signal, np.zeros(6))  ## revisar esta linea
                # assert_equal(y.noise, x.noise)  ## revisar esta linea
                # 

                y = x - 1
                assert_equal(y.signal, np.arange(6)-1)
                assert_equal(y.noise, x.noise)

                y = 1 - x
                assert_equal(y.signal, 1-np.arange(6))
                assert_equal(y.noise, -x.noise)
                assert_raises(ValueError, lambda: x + sig[:4]) # Different length

                x = electrical_signal(sig) # No noise

                y = x - x
                assert_equal(y.signal, np.zeros(6))
                assert_(y.noise is None)

                y = x - sig
                assert_equal(y.signal, np.zeros(6))
                assert_(y.noise is None)

                # y = sig - y
                # assert_equal((sig + x).signal, np.zeros(6))  ## revisar esta linea
                # assert_((sig + x).noise is None)  ## revisar esta linea
                # 

                y = x - 1
                assert_equal(y.signal, np.arange(6)-1)
                assert_(y.noise is None)

                y = 1 - x
                assert_equal(y.signal, 1-np.arange(6))
                assert_(y.noise is None)
                assert_raises(ValueError, lambda: x + sig[:4])

    def test_mul_and_rmul(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                y = x * x
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, x.noise**2)
                

                y = x * sig
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, x.noise)
                

                # y = sig * y
                # assert_equal(y.signal, x.signal**2)  ## revisar esta linea
                # assert_equal(y.noise, x.noise)  ## revisar esta linea
                # 

                y = x * 2
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise)
                

                y = 2 * x
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise)
                assert_raises(ValueError, lambda: x + sig[:4]) # Different length

                x = electrical_signal(sig) # No noise

                y = x * x
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is None)
                

                y = x * sig
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is None)
                

                # y = sig * y
                # assert_equal((sig + x).signal, x.signal**2)  ## revisar esta linea
                # assert_((sig + x).noise is None)  ## revisar esta linea
                # 

                y = x * 2
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is None)
                

                y = 2 * x
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is None)
                
                assert_raises(ValueError, lambda: x + sig[:4])
    
    def test_gt_and_lt(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                z = x.signal + x.noise
                
                y = x > x
                assert_equal(y.data, z>z)
                y = x < x
                assert_equal(y.data, z<z)

                y = x > sig
                assert_equal(y.data, z>np.arange(6))
                y = x < sig
                assert_equal(y.data, z<np.arange(6))

                # y = sig > y
                # assert_equal(y.data, np.ones(6))  ## revisar esta linea
                # y = sig < y
                # assert_equal(y.data, np.zeros(6))  ## revisar esta linea

                y = x > 2
                assert_equal(y.data, z>2)
                y = x < 2
                assert_equal(y.data, z<2)

                x = electrical_signal(sig) # No noise
                z = x.signal

                y = x > x
                assert_equal(y.data, z>z)
                y = x < x
                assert_equal(y.data, z<z)

                y = x > sig
                assert_equal(y.data, z>np.arange(6))
                y = x < sig
                assert_equal(y.data, z<np.arange(6))

                # y = sig > y
                # assert_equal(y.data, np.ones(6))  ## revisar esta linea
                # y = sig < y
                # assert_equal(y.data, np.zeros(6))  ## revisar esta linea

                y = x > 2
                assert_equal(y.data, z>2)
                y = x < 2
                assert_equal(y.data, z<2)

    def test_call(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                y = x('t')
                assert_equal(y.signal, np.fft.ifft(x.signal))
                assert_equal(y.noise, np.fft.ifft(x.noise))

                y = x('f')
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_equal(y.noise, np.fft.fft(x.noise))

                y = x('w')
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_equal(y.noise, np.fft.fft(x.noise))

                assert_raises(ValueError, lambda: x('z'))

                y = x('t', shift=True)
                assert_equal(y.signal, np.fft.ifftshift(np.fft.ifft(x.signal)))
                assert_equal(y.noise, np.fft.ifftshift(np.fft.ifft(x.noise)))
    
                y = x('f', shift=True)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_equal(y.noise, np.fft.fftshift(np.fft.fft(x.noise)))

                y = x('w', shift=True)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_equal(y.noise, np.fft.fftshift(np.fft.fft(x.noise)))

                x = electrical_signal(sig) # No noise

                y = x('t')
                assert_equal(y.signal, np.fft.ifft(x.signal))
                assert_(y.noise is None)

                y = x('f')
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is None)

                y = x('w')
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is None)

                assert_raises(ValueError, lambda: x('z'))

                y = x('t', shift=True)
                assert_equal(y.signal, np.fft.ifftshift(np.fft.ifft(x.signal)))
                assert_(y.noise is None)

                y = x('f', shift=True)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is None)

                y = x('w', shift=True)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is None)

    
    def test_abs(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                x.noise=-x.noise
                z = x.signal + x.noise

                y = x.abs()  # same that abs('all')
                assert_equal(y, np.abs(z))

                y = x.abs('all')  # same that abs('all')
                assert_equal(y, np.abs(z))

                y = x.abs('signal')
                assert_equal(y, np.abs(x.signal))

                y = x.abs('noise')
                assert_equal(y, np.abs(x.noise))


                x = electrical_signal(sig) # No noise
                z = x.signal

                y = x.abs()
                assert_equal(y, np.abs(z))
                y = x.abs('all')
                assert_equal(y, np.abs(z))

                y = x.abs('signal')
                assert_equal(y, np.abs(z))

                y = x.abs('noise')
                assert_equal(y, np.zeros(6))

                assert_raises(ValueError, lambda: x.abs('z'))

    def test_power(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                x.noise = -x.noise
                z = x.signal + x.noise

                y = x.power()  # same that power('all')
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('all')  # same that power()
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('signal')
                assert_equal(y, np.mean(np.abs(x.signal)**2))

                y = x.power('noise')
                assert_equal(y, np.mean(np.abs(x.noise)**2))

                x = electrical_signal(sig) # No noise
                z = x.signal

                y = x.power()
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('all')
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('signal')
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('noise')
                assert_equal(y, 0)


    def test_phase(self):
        x = electrical_signal(self.signals[-1] + self.signals[-1]*1j, -self.noises[-1] - self.noises[-1]*1j)
        z = x.signal + x.noise

        assert_equal(x.phase(), np.unwrap(np.angle(z)))
        
        x = electrical_signal(self.signals[-1] + self.signals[-1]*1j)
        z = x.signal

        assert_equal(x.phase(), np.unwrap(np.angle(z)))
    

    def test_apply(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                y = x.apply(np.abs)
                assert_(y.type(), electrical_signal)
                assert_equal(y.signal, np.abs(x.signal))
                assert_equal(y.noise, np.abs(x.noise))

                x = electrical_signal(sig) # No noise

                y = x.apply(np.abs)
                assert_(y.type(), electrical_signal)
                assert_equal(y.signal, np.abs(x.signal))
                assert_(y.noise is None)


    def test_copy(self):
        x = electrical_signal(self.signals[-1], self.noises[-1])
        y = x.copy()
        assert_(x is not y)
        assert_equal(x.signal, y.signal)
        assert_equal(x.noise, y.noise)

        y = x.copy(n=3)
        assert_(x is not y)
        assert_(y.len() == 3)
        assert_equal(y.signal, x.signal[:3])
        assert_equal(y.noise, x.noise[:3])

    
    def test_plot(self):
        x = electrical_signal(np.ones(100), np.random.normal(0,0.05,100))
        try:
            assert_(x.plot('-r', n=98, xlabel='Time', ylabel='Intensity', style='light', grid=True, hold=True)==x)
        except Exception as e:
            self.fail(f"x.plot() raised {type(e).__name__} unexpectedly!")

    def test_psd(self):
        x = electrical_signal(np.ones(100), np.random.normal(0,0.05,100))
        try:
            assert_(x.psd('--b', n=98, xlabel='Freq', ylabel='Spectra', style='dark', grid=False, hold=True)==x)
        except Exception as e:
            self.fail(f"x.psd() raised {type(e).__name__} unexpectedly!")




class TestOpticalSignal(unittest.TestCase):
    signals_1p = ['+0,1,2,3,4+0j,5.', 
                  [0,1,2,3,4.,5], 
                  np.arange(6)]
    noises_1p = ['+0,-1,-2,-3,-4+0i,-5.',
                [0,-1,-2,-3,-4.,-5],
                -np.arange(6)]
    
    signals_2p = ['+0,1,2,3,4+0j,5.; +0,1,2,3,4+0i,5.',
                  [[0,1,2,3,4.,5]],
                  np.array([[0,1,2,3,4,5]])]

    noises_2p = ['+0,-1,-2,-3,-4+0j,-5.; +0,-1,-2,-3,-4+0i,-5.',
                 [[0,-1,-2,-3,-4.,-5]],
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
                x = optical_signal(sig, noi, n_pol=1)
                
                assert_equal(x.signal, np.arange(6))
                assert_equal(x.noise, -np.arange(6))
                assert_equal(x.n_pol, 1)
                assert_equal(x.execution_time, 0)
                assert_equal(x.len(), 6)

                x = optical_signal(sig) # No noise

                assert_equal(x.signal, np.arange(6))
                assert_(x.noise is None)
                assert_equal(x.n_pol, 1)
                assert_(x.execution_time == 0)
                assert_(x.len() == 6)


        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)

                assert_equal(x.signal, np.tile(np.arange(6),(2,1)))
                assert_equal(x.noise, -np.tile(np.arange(6),(2,1)))
                assert_equal(x.n_pol, 2)
                assert_equal(x.execution_time, 0)
                assert_equal(x.len(), 6)

                x = optical_signal(sig) # No noise

                assert_equal(x.signal, np.tile(np.arange(6),(2,1)))
                assert_(x.noise is None)
                assert_equal(x.n_pol, 2)
                assert_(x.execution_time == 0)
                assert_(x.len() == 6)

    def test_copy(self):
        x = optical_signal(self.signals_1p[-1], self.noises_1p[-1])
        y = x.copy()
        assert_(x is not y)
        assert_(type(y) == optical_signal)
        assert_equal(x.signal, y.signal)
        assert_equal(x.noise, y.noise)

        y = x.copy(n=3)
        assert_(x is not y)
        assert_(type(y) == optical_signal)
        assert_(y.len() == 3)
        assert_equal(y.signal, x.signal[:3])
        assert_equal(y.noise, x.noise[:3])
    
    def test_print(self):
        # with noise
        x_1p = optical_signal([1,2,3], [1,1,0])
        x_2p = optical_signal([[1,2,3]], [[1,1,0]])
        try:
            x_1p.print("optic test 1 pol")
            x_2p.print("optic test 2 pol")
        except Exception as e:
            self.fail(f"signal.print() raised {type(e).__name__} unexpectedly!")

        # without noise
        x_1p = optical_signal([1,2,3])
        x_2p = optical_signal([[1,2,3]])
        try:
            x_1p.print("optic test 1 pol")
            x_2p.print("optic test 2 pol")
        except Exception as e:
            self.fail(f"signal.print() raised {type(e).__name__} unexpectedly!")

    def test_getitem(self):
        for sig, noi in zip(self.signals_1p, self.noises_1p):
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig, noi)
                
                # test int indexing 
                assert_(x[0].type() == optical_signal)
                assert_equal(x[0].signal, 0)
                assert_equal(x[-1].noise, -5)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type() == optical_signal)
                assert_equal(x[:].signal, np.arange(6))
                assert_equal(x[:].noise, -np.arange(6))
                assert_equal(x[1:-1].signal, np.arange(1,5))
                assert_equal(x[1:-1].noise, -np.arange(1,5))
                assert_equal(x[:100].signal, np.arange(6))

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi)
                
                # test int indexing 
                assert_(x[0].type() == optical_signal)
                assert_equal(x[0].signal, [[0],[0]])
                assert_equal(x[-1].noise, [[-5],[-5]])
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type() == optical_signal)
                assert_equal(x[:].signal, x.signal)
                assert_equal(x[:].noise, x.noise)
                assert_equal(x[1:-1].signal, x.signal[:,1:-1])
                assert_equal(x[1:-1].noise, x.noise[:,1:-1])
                assert_equal(x[:100].signal, x.signal)

        for sig in self.signals_1p:
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig)
                
                # test int indexing 
                assert_(x[0].type() == optical_signal)
                assert_equal(x[0].signal, 0)
                assert_(x[-1].noise is None)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type() == optical_signal)
                assert_equal(x[:].signal, x.signal)
                assert_(x[:].noise is None)
                assert_equal(x[1:-1].signal, x.signal[1:-1])
                assert_(x[1:-1].noise is None)
                assert_equal(x[:100].signal, x.signal)

        for sig in self.signals_2p:
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig)
                
                # test int indexing 
                assert_(x[0].type() == optical_signal)
                assert_equal(x[0].signal, [[0],[0]])
                assert_(x[-1].noise is None)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type() == optical_signal)
                assert_equal(x[:].signal, x.signal)
                assert_(x[:].noise is None)
                assert_equal(x[1:-1].signal, x.signal[:,1:-1])
                assert_(x[1:-1].noise is None)
                assert_equal(x[:100].signal, x.signal)

    def test_add_and_radd(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2) # signal and noise
                
                y = x + x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise*2)
                

                y = x + sig
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise)
                

                # y = sig + x
                # assert_(y.type()==optical_signal)
                # assert_equal(y.signal, x.signal*2)  ## revisar esta linea
                # assert_equal(y.noise, x.noise)  ## revisar esta linea
                # 

                y = x + 1
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_equal(y.noise, x.noise)
                

                y = 1 + x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_equal(y.noise, x.noise)
                

                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length

                x = optical_signal(sig) # No noise

                y = x + x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is None)
                

                y = x + sig
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is None)
                

                # y = sig + x
                # assert_(y.type()==optical_signal)
                # assert_equal((sig + x).signal, x.signal*2)  ## revisar esta linea
                # assert_((sig + x).noise is None)  ## revisar esta linea
                # 

                y = x + 1
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_(y.noise is None)
                

                y = 1 + x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_(y.noise is None)
                
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length


    def test_sub_and_rsub(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                
                y = x - x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_equal(y.noise, np.zeros((2,6)))
                

                y = x - sig
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_equal(y.noise, x.noise)
                

                # y = sig - y
                # assert_(y.type()==optical_signal)
                # assert_equal(y.signal, np.zeros(6))  ## revisar esta linea
                # assert_equal(y.noise, x.noise)  ## revisar esta linea
                # 

                y = x - 1
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal-1)
                assert_equal(y.noise, x.noise)
                

                y = 1 - x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, 1-x.signal)
                assert_equal(y.noise, -x.noise)

                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x - '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x - sig[0][:4]) # Different length

                x = optical_signal(sig) # No noise

                y = x - x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_(y.noise is None)
                

                y = x - sig
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_(y.noise is None)
                

                # y = sig - y
                # assert_(y.type()==optical_signal)
                # assert_equal((sig + x).signal, np.zeros(6))  ## revisar esta linea
                # assert_((sig + x).noise is None)  ## revisar esta linea
                # 

                y = x - 1
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal-1)
                assert_(y.noise is None)
                

                y = 1 - x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, 1-x.signal)
                assert_(y.noise is None)
                
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x - '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x - sig[0][:4]) # Different length

    def test_mul_and_rmul(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                
                y = x * x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, x.noise**2)
                

                y = x * sig
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, x.noise)
                

                # y = sig * y
                # assert_(y.type()==optical_signal)
                # assert_equal(y.signal, x.signal**2)  ## revisar esta linea
                # assert_equal(y.noise, x.noise)  ## revisar esta linea
                # 

                y = x * 2
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise)
                

                y = 2 * x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise)
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length

                x = optical_signal(sig) # No noise

                y = x * x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is None)
                

                y = x * sig
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is None)
                

                # y = sig * y
                # assert_(y.type()==optical_signal)
                # assert_equal((sig + x).signal, x.signal**2)  ## revisar esta linea
                # assert_((sig + x).noise is None)  ## revisar esta linea
                # 

                y = x * 2
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is None)
                

                y = 2 * x
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is None)
                
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length

    def test_call(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                
                y = x('t')
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.ifft(x.signal))
                assert_equal(y.noise, np.fft.ifft(x.noise))

                y = x('f')
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_equal(y.noise, np.fft.fft(x.noise))

                y = x('w')
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_equal(y.noise, np.fft.fft(x.noise))

                assert_raises(ValueError, lambda: x('z'))

                y = x('t', shift=True)
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.ifftshift(np.fft.ifft(x.signal)))
                assert_equal(y.noise, np.fft.ifftshift(np.fft.ifft(x.noise)))
    
                y = x('f', shift=True)
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_equal(y.noise, np.fft.fftshift(np.fft.fft(x.noise)))

                y = x('w', shift=True)
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_equal(y.noise, np.fft.fftshift(np.fft.fft(x.noise)))

                x = optical_signal(sig) # No noise

                y = x('t')
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.ifft(x.signal))
                assert_(y.noise is None)

                y = x('f')
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is None)

                y = x('w')
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is None)

                assert_raises(ValueError, lambda: x('z'))

                y = x('t', shift=True)
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.ifftshift(np.fft.ifft(x.signal)))
                assert_(y.noise is None)

                y = x('f', shift=True)
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is None)

                y = x('w', shift=True)
                assert_(y.type()==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is None)
    
    def test_power(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                x.noise = -x.noise
                z = x.signal + x.noise

                y = x.power()  # same that power('all')
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('all')  # same that power()
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('signal')
                assert_equal(y, np.mean(np.abs(x.signal)**2))

                y = x.power('noise')
                assert_equal(y, np.mean(np.abs(x.noise)**2))

                x = optical_signal(sig) # No noise
                z = x.signal

                y = x.power()
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('all')
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('signal')
                assert_equal(y, np.mean(np.abs(z)**2))

                y = x.power('noise')
                assert_equal(y, 0)

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
        
        assert_(x.plot(style='light', cmap='plasma', title='TEST')==x)


if __name__ == '__main__':
    unittest.main()