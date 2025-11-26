import unittest
import numpy as np
from scipy.constants import c, pi
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_array_equal
)

from opticomlib.typing import (
    NULL,
    binary_sequence,
    electrical_signal,
    optical_signal,
    gv
)

from opticomlib.utils import ComplexNumber, RealNumber, IntegerNumber


class TestGlobalVariables(unittest.TestCase):
    def setUp(self):
        gv.default()

    def test_global_variables(self):
        ## Test default attributes
        assert_(gv.sps == 16)
        assert_(gv.R == 1e9)
        assert_(gv.fs == 16e9)
        assert_(gv.dt == 1/16e9)
        assert_(gv.wavelength == 1550e-9)
        assert_(gv.f0 == c/1550e-9)
        assert_(gv.N == 128)
        assert_(gv.plt_style == 'fast')
        assert_(gv.verbose == None)
        assert_raises(AttributeError, lambda: gv.alpha)

        ## Test __call__ method
        gv(sps=8, R=10e9, wavelength=1549e-9, N=10, alpha=0.2, verbose=10)

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
        assert_(gv.verbose == 10)

        ## Test print
        try:
            gv.print()
        except Exception as e:
            self.fail(f"gv.print() raised {type(e).__name__} unexpectedly!")

    def test_default(self):
        gv(sps=32, R=20e9) # Change values
        assert_(gv.sps == 32)
        
        gv.default() # Reset
        assert_(gv.sps == 16)
        assert_(gv.R == 1e9)
        assert_(gv.fs == 16e9)
        # Check that custom attributes are removed
        if hasattr(gv, 'alpha'):
            assert_raises(AttributeError, lambda: gv.alpha)


class TestBinarySequence(unittest.TestCase):
    inputs = ['000011110000',
            [0,0,0,0,1,1,1,1,0,0,0,0],
            (0,0,0,0,1,1,1,1,0,0,0,0),
            np.array([0,0,0,0,1,1,1,1,0,0,0,0])]
    
    def setUp(self):
        gv.default()

    def test_init_errors(self):
        assert_raises(TypeError, lambda: binary_sequence())
        assert_raises(ValueError, lambda: binary_sequence([0,1,2,3]))
        assert_raises(ValueError, lambda: binary_sequence('001201'))
        assert_raises(ValueError, lambda: binary_sequence('001;101'))
    
    def test_init(self):
        assert_(np.array_equal(binary_sequence([]) , []))
        assert_(np.array_equal(binary_sequence(binary_sequence([1,0])) , [1,0]))
        assert_(binary_sequence(0)[0]==0)
        assert_(binary_sequence('1')[0]==1)

        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_(bits == [0,0,0,0,1,1,1,1,0,0,0,0]) # Check if the data is correct, and test the __eq__ method
                assert_(bits.size == 12) # Check that the length is correct, and test the len method
                assert_(bits.execution_time == 0) # Check that the execution time is 0

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
                
                assert_(bits + bits == 2*[0,0,0,0,1,1,1,1,0,0,0,0])
                assert_(bits + '0101' == [0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,1])
                assert_('0101' + bits == [0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0])
                assert_(bits + 1 == [0,0,0,0,1,1,1,1,0,0,0,0,1])
                assert_(1+bits == [1,0,0,0,0,1,1,1,1,0,0,0,0])
                assert_raises(ValueError, lambda: '020' + bits)
                assert_raises(ValueError, lambda: bits + '020')
                assert_raises(ValueError, lambda: bits + [0,3,0])
                assert_raises(ValueError, lambda: [0,3,0] + bits)
                
    def test_invert(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_(~bits == [1,1,1,1,0,0,0,0,1,1,1,1])
                assert_(bits.flip() == [1,1,1,1,0,0,0,0,1,1,1,1])
    
    def test_ones_and_zeros(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                assert_(bits.ones == 4)
                assert_(bits.zeros == 8)

    def test_hamming_distance(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                other = binary_sequence('000111000111')
                
                assert_(isinstance(bits.hamming_distance(other), IntegerNumber))
                assert_(bits.hamming_distance(other) == 6)

    def test_bitwise(self):
        for input in self.inputs:
            with self.subTest(input_type=type(input)):
                bits = binary_sequence(input)
                other = binary_sequence('000111000111')
                
                assert_(np.array_equal(bits & other, [0,0,0,0,1,1,0,0,0,0,0,0]))
                assert_(np.array_equal(other & bits, [0,0,0,0,1,1,0,0,0,0,0,0]))

                assert_(np.array_equal(bits | other, [0,0,0,1,1,1,1,1,0,1,1,1]))
                assert_(np.array_equal(other | bits, [0,0,0,1,1,1,1,1,0,1,1,1]))

                assert_(np.array_equal(bits ^ other, [0,0,0,1,0,0,1,1,0,1,1,1]))
                assert_(np.array_equal(other ^ bits, [0,0,0,1,0,0,1,1,0,1,1,1]))

    def test_mul(self):
        bits = binary_sequence('10')
        # Repetition
        assert_equal(bits * 3, binary_sequence('101010'))
        # AND operation
        assert_equal(bits * binary_sequence('11'), binary_sequence('10'))

    def test_prbs(self):
        # Test PRBS generation
        prbs7 = binary_sequence.prbs(order=7)
        assert_equal(prbs7.size, 2**7 - 1)
        assert_(np.all((prbs7.data == 0) | (prbs7.data == 1)))
        
        # Test seed consistency
        prbs7_1 = binary_sequence.prbs(order=7, seed=123)
        prbs7_2 = binary_sequence.prbs(order=7, seed=123)
        assert_equal(prbs7_1, prbs7_2)

        # Test return_seed
        seq, last_seed = binary_sequence.prbs(order=7, len=10, seed=1, return_seed=True)
        assert_equal(seq.size, 10)
        assert_(isinstance(last_seed, (int, np.integer)))

    def test_dac(self):
        bits = binary_sequence('101')
        gv(sps=4)
        h = np.ones(4)
        sig = bits.dac(h)
        assert_(isinstance(sig, electrical_signal))
        # Expected size is len(bits)*sps due to upfir implementation
        assert_equal(sig.size, 3*4)

    def test_numpy_interop(self):
        bits = binary_sequence('1010')
        assert_equal(np.sum(bits), 2)
        assert_equal(np.mean(bits), 0.5)
        assert_equal(np.max(bits), 1)
        assert_equal(np.min(bits), 0)


class TestElectricalSignal(unittest.TestCase):
    signals = ['+0,1,2,3,4+0j,5.', 
                  [0,1,2,3,4.,5], 
                  np.arange(6)]
    noises = ['+0,-1,-2,-3,-4+0i,-5.',
                [0,-1,-2,-3,-4.,-5],
                -np.arange(6)]

    def setUp(self):
        gv.default()

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
                assert_(x.size==6)

                x = electrical_signal(sig) # No noise

                assert_equal(x.signal, np.arange(6))
                assert_(x.noise is NULL)
                assert_equal(x.execution_time, 0)
                assert_(x.size==6)

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
                assert_(x[0].type == electrical_signal)
                assert_equal(x[0].signal, 0)
                assert_equal(x[-1].noise, -5)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type == electrical_signal)
                assert_equal(x[:].signal, np.arange(6))
                assert_equal(x[:].noise, -np.arange(6))
                assert_equal(x[1:-1].signal, np.arange(1,5))
                assert_equal(x[1:-1].noise, -np.arange(1,5))
                assert_equal(x[:100].signal, np.arange(6))

                x = electrical_signal(sig) # No noise

                # test int indexing
                assert_(isinstance(x[0], ComplexNumber))
                assert_equal(x[0], 0)
                assert_(x.noise is NULL)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type == electrical_signal)
                assert_equal(x[:].signal, np.arange(6))
                assert_(x[:].noise is NULL)
                assert_equal(x[1:-1].signal, np.arange(1,5))
                assert_(x[1:-1].noise is NULL)
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

                y = sig + x
                assert_equal(y.signal, np.arange(6)*2)  ## revisar esta linea
                assert_equal(y.noise, x.noise)  ## revisar esta linea
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
                assert_(y.noise is NULL)

                y = x + sig
                assert_equal(y.signal, np.arange(6)*2)
                assert_(y.noise is NULL)

                y = sig + x
                assert_equal((sig + x).signal, np.arange(6)*2)  ## revisar esta linea
                assert_((sig + x).noise is NULL)  ## revisar esta linea
                # 

                y = x + 1
                assert_equal(y.signal, np.arange(6)+1)
                assert_(y.noise is NULL)

                y = 1 + x
                assert_equal(y.signal, np.arange(6)+1)
                assert_(y.noise is NULL)
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

                y = sig - x
                assert_equal(y.signal, np.zeros(6))  ## revisar esta linea
                assert_equal(y.noise, -x.noise)  ## revisar esta linea
                

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
                assert_(y.noise is NULL)

                y = x - sig
                assert_equal(y.signal, np.zeros(6))
                assert_(y.noise is NULL)

                y = sig - x
                assert_equal(y.signal, np.zeros(6))  ## revisar esta linea
                assert_(y.noise is NULL)  ## revisar esta linea
                # 

                y = x - 1
                assert_equal(y.signal, np.arange(6)-1)
                assert_(y.noise is NULL)

                y = 1 - x
                assert_equal(y.signal, 1-np.arange(6))
                assert_(y.noise is NULL)
                assert_raises(ValueError, lambda: x + sig[:4])

    def test_mul_and_rmul(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                
                y = x * x
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, 2*x.signal*x.noise + x.noise**2)
                
                y = x**2
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, 2*x.signal*x.noise + x.noise**2)
                
                y = x**3
                assert_equal(y.signal, (x.signal + x.noise)**3)
                assert_(y.noise is NULL)
                
                y = x * sig
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, x.noise*x.signal)
                

                y = sig * x
                assert_equal(y.signal, x.signal**2)  
                assert_equal(y.noise, x.noise*x.signal)  

                y = x * 2
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise*2)
                

                y = 2 * x
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise*2)
                assert_raises(ValueError, lambda: x + sig[:4]) # Different length

                x = electrical_signal(sig) # No noise

                y = x * x
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is NULL)
                

                y = x * sig
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is NULL)
                

                y = sig * x
                assert_equal(y.signal, x.signal**2)  
                assert_(y.noise is NULL)  
                # 

                y = x * 2
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is NULL)
                

                y = 2 * x
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is NULL)
                
                assert_raises(ValueError, lambda: x + sig[:4])
    
    def test_truediv_and_floordiv(self):
        x = electrical_signal([2, 4, 6], [1, 2, 3])
        
        # True div
        y = x / 2
        assert_equal(y.signal, [1, 2, 3])
        assert_equal(y.noise, [0.5, 1, 1.5])
        
        # Floor div
        y = x // 2
        assert_equal(y.signal, [1, 2, 3])
        assert_equal(y.noise, [0, 1, 1]) # floor(0.5)=0, floor(1)=1, floor(1.5)=1

        assert_raises(ZeroDivisionError, lambda: x / 0)

    def test_pow(self):
        x = electrical_signal([2, 3], [1, 1])
        
        # Power 0
        y = x ** 0
        assert_equal(y.signal, [1, 1])
        assert_(y.noise is NULL)
        
        # Power 1
        y = x ** 1
        assert_equal(y.signal, x.signal)
        assert_equal(y.noise, x.noise)
        
        # Power 2
        y = x ** 2
        assert_equal(y.signal, x.signal**2)
        # (s+n)^2 = s^2 + 2sn + n^2. Signal part is s^2, noise part is 2sn + n^2
        assert_equal(y.noise, 2*x.signal*x.noise + x.noise**2)

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

                y = sig > x
                assert_equal(y.data, np.arange(6)>z) 
                y = sig < x
                assert_equal(y.data, np.arange(6)<z) 

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

                y = sig > x
                assert_equal(y.data, np.arange(6)>z) 
                y = sig < x
                assert_equal(y.data, np.arange(6)<z) 

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
                assert_(y.noise is NULL)

                y = x('f')
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is NULL)

                y = x('w')
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is NULL)

                assert_raises(ValueError, lambda: x('z'))

                y = x('t', shift=True)
                assert_equal(y.signal, np.fft.ifftshift(np.fft.ifft(x.signal)))
                assert_(y.noise is NULL)

                y = x('f', shift=True)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is NULL)

                y = x('w', shift=True)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is NULL)

    
    def test_abs(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                z = np.abs(x.signal + x.noise)

                y = x.abs()  # same that abs('all')
                assert_equal(y, z)

                y = x.abs('all')  # same that abs('all')
                assert_equal(y, z)
                assert_(y.type == electrical_signal)

                y = x.abs('signal')
                assert_equal(y, np.abs(x.signal))
                assert_(y.type == electrical_signal)

                y = x.abs('noise')
                assert_equal(y, np.abs(x.noise))
                assert_(y.type == electrical_signal)


                x = electrical_signal(sig) # No noise
                z = x.signal

                y = x.abs()
                assert_equal(y, z)
                assert_(y.type == electrical_signal)

                y = x.abs('all')
                assert_equal(y, z)
                assert_(y.type == electrical_signal)

                y = x.abs('signal')
                assert_equal(y, z)
                assert_(y.type == electrical_signal)

                y = x.abs('noise')
                assert_equal(y, np.zeros_like(z))
                assert_(y.type == electrical_signal)

                assert_raises(ValueError, lambda: x.abs('z'))

    def test_power(self):
        for sig, noi in zip(self.signals, self.noises):
            with self.subTest(type=type(sig)):
                x = electrical_signal(sig, noi) # signal and noise
                z = np.mean(np.abs(x.signal + x.noise)**2)

                y = x.power()  # same that power('all')
                assert_equal(y, z)
                assert_(isinstance(y, RealNumber))

                y = x.power(of='all')  # same that power(of=)
                assert_equal(y, z)
                assert_(isinstance(y, RealNumber))

                y = x.power(of='signal')
                assert_equal(y, np.mean(np.abs(x.signal)**2))
                assert_(isinstance(y, RealNumber))

                y = x.power(of='noise')
                assert_equal(y, np.mean(np.abs(x.noise)**2))
                assert_(isinstance(y, RealNumber))

                x = electrical_signal(sig) # No noise
                z = x.signal

                y = x.power()
                assert_equal(y, np.mean(np.abs(z)**2))
                assert_(isinstance(y, RealNumber))

                y = x.power(of='all')
                assert_equal(y, np.mean(np.abs(z)**2))
                assert_(isinstance(y, RealNumber))

                y = x.power(of='signal')
                assert_equal(y, np.mean(np.abs(z)**2))
                assert_(isinstance(y, RealNumber))

                y = x.power(of='noise')
                assert_(y == 0)
                assert_(isinstance(y, RealNumber))

    def test_normalize(self):
        x = electrical_signal([1, 2, 3])
        
        # Power normalization
        y = x.normalize('power')
        assert_almost_equal(y.power('W', 'signal'), 1.0)
        
        # Amplitude normalization
        y = x.normalize('amplitude')
        assert_equal(np.max(np.abs(y.signal)), 1.0)

    def test_phase(self):
        x = electrical_signal(self.signals[-1] + self.signals[-1]*1j, -self.noises[-1] - self.noises[-1]*1j)
        z = x.signal + x.noise

        assert_equal(x.phase(), np.unwrap(np.angle(z)))
        
        x = electrical_signal(self.signals[-1] + self.signals[-1]*1j)
        z = x.signal

        assert_equal(x.phase(), np.unwrap(np.angle(z)))

    def test_filter(self):
        x = electrical_signal([1, 0, 0, 0])
        h = [0.5, 0.5]
        y = x.filter(h)
        # fftconvolve mode='same'
        # [1, 0, 0, 0] * [0.5, 0.5] -> [0.5, 0.5, 0, 0, 0]
        # mode='same' centered: [0.5, 0.5, 0, 0]
        assert_equal(y.size, 4)
        assert_(isinstance(y, electrical_signal))

    def test_sum(self):
        x = electrical_signal([1, 2, 3], [0.1, 0.2, 0.3])
        y = x.sum()
        assert_equal(y.signal, 6)
        assert_almost_equal(y.noise, 0.6)

    def test_conj(self):
        x = electrical_signal([1+1j], [2-2j])
        y = x.conj()
        assert_equal(y.signal, [1-1j])
        assert_equal(y.noise, [2+2j])

    def test_plot(self):
        x = electrical_signal(np.ones(100), np.random.normal(0,0.05,100))
        try:
            assert_equal(x.plot('-r', n=98, xlabel='Time', ylabel='Intensity', grid=True, hold=True), x)
        except Exception as e:
            self.fail(f"x.plot() raised {type(e).__name__} unexpectedly!")

    def test_psd(self):
        x = electrical_signal(np.ones(100), np.random.normal(0,0.05,100))
        try:
            assert_equal(x.psd('--b', n=98, xlabel='Freq', ylabel='Spectra', grid=False, hold=True), x)
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
    
    def setUp(self):
        gv.default()
        
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
                assert_equal(x.size, 6)

                x = optical_signal(sig) 

                assert_equal(x.signal, np.arange(6))
                assert_(x.noise is NULL)
                assert_equal(x.n_pol, 1)
                assert_(x.execution_time == 0)
                assert_(x.size == 6)


        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2)

                assert_equal(x.signal, np.tile(np.arange(6),(2,1)))
                assert_equal(x.noise, -np.tile(np.arange(6),(2,1)))
                assert_equal(x.n_pol, 2)
                assert_equal(x.execution_time, 0)
                assert_equal(x.size, 6)
                assert_equal(len(x), 6)

                x = optical_signal(sig) # No noise

                assert_equal(x.signal, np.tile(np.arange(6),(2,1)))
                assert_(x.noise is NULL)
                assert_equal(x.n_pol, 2)
                assert_(x.execution_time == 0)
                assert_(x.size == 6)
                assert_(len(x) == 6)

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
                assert_(x[0].type == optical_signal)
                assert_equal(x[0].signal, 0)
                assert_equal(x[-1].noise, -5)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type == optical_signal)
                assert_equal(x[:].signal, np.arange(6))
                assert_equal(x[:].noise, -np.arange(6))
                assert_equal(x[1:-1].signal, np.arange(1,5))
                assert_equal(x[1:-1].noise, -np.arange(1,5))
                assert_equal(x[:100].signal, np.arange(6))

        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig, noi)
                
                # test int indexing 
                assert_(x[0].type == optical_signal)
                assert_equal(x[0].signal, np.arange(6))
                assert_equal(x[-1].noise, -np.arange(6))
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type == optical_signal)
                assert_equal(x[:].signal, x.signal)
                assert_equal(x[:].noise, x.noise)
                assert_equal(x[1:-1].signal, x.signal[:,1:-1])
                assert_equal(x[1:-1].noise, x.noise[:,1:-1])
                assert_equal(x[:100].signal, x.signal)

        for sig in self.signals_1p:
            with self.subTest(pol=1, type=type(sig)):
                x = optical_signal(sig)
                
                # test int indexing 
                assert_(isinstance(x[0], ComplexNumber))
                assert_equal(x[0], 0)
                assert_(x.noise is NULL)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type == optical_signal)
                assert_equal(x[:].signal, x.signal)
                assert_(x[:].noise is NULL)
                assert_equal(x[1:-1].signal, x.signal[1:-1])
                assert_(x[1:-1].noise is NULL)
                assert_equal(x[:100].signal, x.signal)

        for sig in self.signals_2p:
            with self.subTest(pol=2, type=type(sig)):
                x = optical_signal(sig)
                
                # test int indexing 
                assert_(x[0].type == optical_signal)
                assert_equal(x[0].signal, np.arange(6))
                assert_(x[-1].noise is NULL)
                assert_raises(IndexError, lambda: x[10])

                # test slice indexing
                assert_(x[:].type == optical_signal)
                assert_equal(x[:].signal, x.signal)
                assert_(x[:].noise is NULL)
                assert_equal(x[1:-1].signal, x.signal[:,1:-1])
                assert_(x[1:-1].noise is NULL)
                assert_equal(x[:100].signal, x.signal)

    def test_add_and_radd(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi, n_pol=2) # signal and noise
                
                y = x + x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise*2)
                

                y = x + sig
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, x.noise)
                

                y = sig + x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)  ## revisar esta linea
                assert_equal(y.noise, x.noise)  ## revisar esta linea
                # 

                y = x + 1
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_equal(y.noise, x.noise)
                

                y = 1 + x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_equal(y.noise, x.noise)
                

                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length

                x = optical_signal(sig) # No noise

                y = x + x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is NULL)
                

                y = x + sig
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is NULL)
                

                y = sig + x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)  ## revisar esta linea
                assert_(y.noise is NULL)  ## revisar esta linea
                # 

                y = x + 1
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_(y.noise is NULL)
                

                y = 1 + x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal+1)
                assert_(y.noise is NULL)
                
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length


    def test_sub_and_rsub(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                
                y = x - x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_equal(y.noise, np.zeros((2,6)))
                

                y = x - sig
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_equal(y.noise, x.noise)
                

                y = sig - x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))  ## revisar esta linea
                assert_equal(y.noise, -x.noise)  ## revisar esta linea
                # 

                y = x - 1
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal-1)
                assert_equal(y.noise, x.noise)
                

                y = 1 - x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, 1-x.signal)
                assert_equal(y.noise, -x.noise)

                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x - '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x - sig[0][:4]) # Different length

                x = optical_signal(sig) # No noise

                y = x - x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_(y.noise is NULL)
                

                y = x - sig
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))
                assert_(y.noise is NULL)
                

                y = sig - x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.zeros((2,6)))  ## revisar esta linea
                assert_(y.noise is NULL)  ## revisar esta linea
                # 

                y = x - 1
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal-1)
                assert_(y.noise is NULL)
                

                y = 1 - x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, 1-x.signal)
                assert_(y.noise is NULL)
                
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x - '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x - sig[0][:4]) # Different length

    def test_mul_and_rmul(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                
                y = x * x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, x.noise**2 + 2*x.signal*x.noise)
                

                y = x * sig
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_equal(y.noise, x.noise*x.signal)
                

                y = sig * x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal**2)  ## revisar esta linea
                assert_equal(y.noise, x.noise*x.signal)  ## revisar esta linea
                # 

                y = x * 2
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, 2*x.noise)
                

                y = 2 * x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_equal(y.noise, 2*x.noise)
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length

                x = optical_signal(sig) # No noise

                y = x * x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is NULL)
                

                y = x * sig
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal**2)
                assert_(y.noise is NULL)
                

                y = sig * x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal**2)  ## revisar esta linea
                assert_(y.noise is NULL)  ## revisar esta linea
                # 

                y = x * 2
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is NULL)
                

                y = 2 * x
                assert_(y.type==optical_signal)
                assert_equal(y.signal, x.signal*2)
                assert_(y.noise is NULL)
                
                
                if isinstance(sig, str): 
                    assert_raises(ValueError, lambda: x + '+0,-1,-2')
                else:
                    assert_raises(ValueError, lambda: x + sig[0][:4]) # Different length

    def test_gt_lt_error(self):
        x = optical_signal([1,2], [0,0])
        assert_raises(NotImplementedError, lambda: x > x)
        assert_raises(NotImplementedError, lambda: x < x)

    def test_call(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                
                y = x('t')
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.ifft(x.signal))
                assert_equal(y.noise, np.fft.ifft(x.noise))

                y = x('f')
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_equal(y.noise, np.fft.fft(x.noise))

                y = x('w')
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_equal(y.noise, np.fft.fft(x.noise))

                assert_raises(ValueError, lambda: x('z'))

                y = x('t', shift=True)
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.ifftshift(np.fft.ifft(x.signal)))
                assert_equal(y.noise, np.fft.ifftshift(np.fft.ifft(x.noise)))
    
                y = x('f', shift=True)
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_equal(y.noise, np.fft.fftshift(np.fft.fft(x.noise)))

                y = x('w', shift=True)
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_equal(y.noise, np.fft.fftshift(np.fft.fft(x.noise)))

                x = optical_signal(sig) # No noise

                y = x('t')
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.ifft(x.signal))
                assert_(y.noise is NULL)

                y = x('f')
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is NULL)

                y = x('w')
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fft(x.signal))
                assert_(y.noise is NULL)

                assert_raises(ValueError, lambda: x('z'))

                y = x('t', shift=True)
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.ifftshift(np.fft.ifft(x.signal)))
                assert_(y.noise is NULL)

                y = x('f', shift=True)
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is NULL)

                y = x('w', shift=True)
                assert_(y.type==optical_signal)
                assert_equal(y.signal, np.fft.fftshift(np.fft.fft(x.signal)))
                assert_(y.noise is NULL)
    
    def test_power(self):
        for sig, noi in zip(self.signals_2p, self.noises_2p):
            with self.subTest(type=type(sig)):
                x = optical_signal(sig, noi) # signal and noise
                z = np.mean(np.abs(x.signal + x.noise)**2, axis=-1)

                y = x.power()  # same that power('all')
                assert_equal(y, z)

                y = x.power(of='all')  # same that power()
                assert_equal(y, z)

                y = x.power(of='signal')
                assert_equal(y, np.mean(np.abs(x.signal)**2, axis=-1))

                y = x.power(of='noise')
                assert_equal(y, np.mean(np.abs(x.noise)**2, axis=-1))

                x = optical_signal(sig) # No noise
                z = np.mean(np.abs(x.signal)**2, axis=-1)

                y = x.power()
                assert_equal(y, z)

                y = x.power(of='all')
                assert_equal(y, z)

                y = x.power(of='signal')
                assert_equal(y, z)

                y = x.power(of='noise')
                assert_(np.array_equal(y, [0,0]))

    def test_w(self):
        x = optical_signal(np.arange(6), -np.arange(6), n_pol=1)
        assert_allclose(x.w(), 2*pi*np.fft.fftfreq(6)*gv.fs)
        assert_allclose(x.w(shift=True), 2*pi*np.fft.fftshift(np.fft.fftfreq(6))*gv.fs)

        x = optical_signal(np.arange(6), -np.arange(6), n_pol=2)
        assert_allclose(x.w(), 2*pi*np.fft.fftfreq(6)*gv.fs)
        assert_allclose(x.w(shift=True), 2*pi*np.fft.fftshift(np.fft.fftfreq(6))*gv.fs)

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
            assert_equal(x.plot('--r', n=98, mode='field', xlabel='Time', ylabel='Intensity', grid=True, hold=True), x)
        except Exception as e:
            self.fail(f"x.plot() raised {type(e).__name__} unexpectedly!")

        x = optical_signal(np.ones(100), np.random.normal(0,0.05,100), n_pol=2)
        try:
            assert_equal(x.plot('-', n=98, mode='power', xlabel='Time', ylabel='Intensity', grid=True, hold=True), x)
        except Exception as e:
            self.fail(f"x.plot() raised {type(e).__name__} unexpectedly!")

    def test_psd(self):
        x = optical_signal(np.ones(100), np.random.normal(0,0.05,100), n_pol=1)
        try:
            assert_equal(x.psd('--r', mode='x', n=98, xlabel='Freq', ylabel='Spectra', grid=True, hold=True), x)
        except Exception as e:
            self.fail(f"x.psd() raised {type(e).__name__} unexpectedly!")

        x = optical_signal(np.ones(100), np.random.normal(0,0.05,100), n_pol=2)
        try:
            assert_equal(x.psd('--b', mode='both', n=98, xlabel='Freq', ylabel='Spectra', grid=False, hold=True), x)
        except Exception as e:
            self.fail(f"x.psd() raised {type(e).__name__} unexpectedly!")


if __name__ == '__main__':
    unittest.main()
