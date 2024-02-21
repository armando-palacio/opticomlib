"""
==============================================================
Models for Opto-Electronic devices (:mod:`opticomlib.devices`)
==============================================================

.. autosummary::
   :toctree: generated/

   PRBS                  -- Pseudorandom binary sequence generator
   DAC                   -- Digital-to-analog converter (DAC) model
   PM                    -- Optical phase modulator (PM) model
   MZM                   -- Mach-Zehnder modulator (MZM) model
   BPF                   -- Optical band-pass filter (BPF) bessel model
   EDFA                  -- Erbium-doped fiber amplifier (EDFA) simple model
   DM                    -- Dispersion medium model
   FIBER                 -- Optical fiber model (dispersion, attenuation and non-linearities, Split-Step Fourier Method)
   LPF                   -- Electrical low-pass filter (LPF) bessel model
   PD                    -- Photodetector (PD) model
   ADC                   -- Analog-to-digital converter (ADC) model
   GET_EYE               -- Eye diagram parameters and metrics estimator
   SAMPLER               -- Sampler device
   FBG                   -- Fiber Bragg Grating (FBG) model
"""


"""Basic physical models for optical/electronic components."""
import numpy as np
import scipy.signal as sg
from scipy.integrate import solve_ivp
from typing import Literal, Union
from numpy import ndarray
from scipy.constants import pi, k as kB, e, h, c
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import sklearn.cluster as sk
from tqdm.auto import tqdm # barra de progreso
from numpy.lib.scimath import sqrt as csqrt

import warnings


from .typing import (
    electrical_signal,
    binary_sequence,
    optical_signal,
    gv,
    eye,
)

from .utils import (
    generate_prbs,
    idbm,
    idb,
    db,
    tic,
    toc,
    rcos,
    si
)



def PRBS(n=2**8, 
         user=[], 
         order=None):
    """
    Pseudorandom binary sequence generator.

    Args:
        n (int) : lenght of random binary sequence (default: `n=2**8`)
        user (str | list | ndarray) : binary sequence user pattern (default: `user=[]`)
        order (int, optional) : degree of the generating pseudorandom polynomial (default: `order=None`)

    Returns:
        binary_sequence: generated binary sequence

    Examples:
        Using parameter **n**, this function generate a random sequence of lenght `n`. Internally it use ``numpy.random.randint`` function.
        
        >>> from opticomlib.devices import PRBS
        >>> PRBS(10).data
        array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1], dtype=uint8)

        On the other hand, the **user** parameter can be used for a custom sequence.
        We can input it in *str* format separating the values by spaces ``' '`` or by commas ``','``. 

        >>> PRBS(user='1 0 1 0   0 1 1 1   0,1,0,0   1,1,0,1').data
        array([1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=uint8)

        The last way in which the function can be used is by passing the **order** of the generating polynomial
        as an argument, which will return a pseudo-random binary sequence of lenght 2^order-1, using an internal algorithm.

        >>> PRBS(order=7).data 
        array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,
            0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
            0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=uint8)
    """
    tic()

    if user:
        output = binary_sequence( user )
    elif order:
        output = binary_sequence( generate_prbs(order) )
    else:
        output = binary_sequence( np.random.randint(0, 2, n) )
    output.ejecution_time = toc()
    return output



def DAC(input: Union[str, list, tuple, ndarray, binary_sequence], 
        Vout: float=None,
        pulse_shape: Literal['rect','gaussian']='rect', 
        **kargs):  
    """
    Digital-to-Analog Converter. Converts a binary sequence into an electrical signal, sampled at a frequency ``fs``.

    Args:
        input (str | list | tuple | ndarray | binary_sequence): Input binary sequence.
        Vout (float, Optional): Output signal amplitude [-15 to 15 Volts]. (default: `Vout=1.0`)
        pulse_shape (str, Optional): Pulse shape at the output, can be "rect" or "gaussian". (default: `pulse_shape="rect"`)

    Keyword Args:
        c (float): Chirp of the Gaussian pulse. Only if `pulse_shape=gaussian`. (default: `c=0.0`)
        m (int): Order of the super-Gaussian pulse. Only if `pulse_shape=gaussian`. (default: `m=1`)
        T (int): Pulse width at half maximum in number of samples. Only if `pulse_shape=gaussian`. (default: `T=sps`)

    Returns:
        electrical_signal: The converted electrical signal.

    Raises:
        TypeError: If `input` type is not in [str, list, tuple, ndarray, binary_sequence].
        NameError: If `pulse_shape` is not "rect" or "gaussian".
        ValueError: If `Vout` is not between -15 and 15 Volts.

    Example:
        >>> from opticomlib.devices import DAC
        >>> from opticomlib import gv
        >>> 
        >>> gv(sps=8) # set samples per bit
        >>>
        >>> DAC('0 0 1 0 0', Vout=5, pulse_shape='gaussian', m=2).plot('r.-').grid().show()

        .. image:: /_images/DAC_example1.svg
            :alt: DAC example 1
            :align: center
    """
    tic()
    if not isinstance(input, binary_sequence):
        input = binary_sequence(input)
    
    sps = kargs['sps'] if 'sps' in kargs.keys() else gv.sps

    if pulse_shape == 'rect':
        x = np.kron(input.data, np.ones(sps))
    
    elif pulse_shape == 'gaussian':
        c = kargs['c'] if 'c' in kargs.keys() else 0.0
        m = kargs['m'] if 'm' in kargs.keys() else 1
        T = kargs['T'] if 'T' in kargs.keys() else sps

        p = lambda t, T: np.exp(-(1+1j*c)/2 * (t/T)**(2*m))

        t = np.linspace(-2*sps, 2*sps, 4*sps) # vector de tiempo del pulso gaussiano
        k = 2*(2*np.log(2))**(1/(2*m)) # factor de escala entre el ancho de un slot y la desviación estándar del pulso gaussiano
        pulse = p(t, T/k) # pulso gaussiano

        s = np.zeros(input.len()*sps)
        s[int(sps//2)::sps]=input.data
        s[int(sps//2-1)::sps]=input.data

        x = sg.fftconvolve(s, pulse, mode='same')/2
    else:
        raise NameError('El parámetro `type` debe ser uno de los siguientes valores ("rect","gaussian").')

    if Vout:
        if np.abs(Vout)>=15:
            raise ValueError('El parámetro `Vout` debe ser un valor entre -15 y 15 Volts.')
        x = x * Vout / x.max()

    output = electrical_signal( x )

    output.ejecution_time = toc()
    return output



def PM(op_input: optical_signal, 
       el_input: Union[float, ndarray, electrical_signal], 
       Vpi: float=5.0):
    """
    Optical Phase Modulator (PM) model. Modulate de phase of the input optical signal through input electrical signal.

    .. math:: E_{out} = E_{in} \\cdot e^{j\\pi \\frac{V_{in}}{V_{\\pi}}} 

    Args:
        op_input (optical_signal): optical signal to be modulated
        el_input (float | ndarray | electrical_signal): driver voltage. It can be an integer value, in which case the phase modulation is constant, or an electrical signal of the same length as the optical signal.
        Vpi (float, Optional): voltage at which the device achieves a phase shift of π (default: ``Vpi=5.0`` [V])

    Returns:
        optical_signal: modulated optical signal

    Raises:
        TypeError: If ``op_input`` type is not `optical_signal`.
        TypeError: If ``el_input`` type is not in [`float`, `ndarray`, `electrical_signal`].
        ValueError: If ``el_input`` is ndarray or `electrical_signal` but, length is not equal to ``op_input`` length.

    Example:
        .. code-block:: python
            :linenos:
            :emphasize-lines: 13, 23, 32
            
            from opticomlib.devices import PM
            from opicomlib import optical_signal, gv
            import matplotlib.pyplot as plt
            import numpy as np
           
            gv(sps=8, R=1e9) # set samples per bit and bitrate
           
            op_input = optical_signal(np.exp(1j*np.linspace(0,4*np.pi, 1000))) # input optical signal ( exp(j*w*t) )
            t = op_input.t()*1e9
            w = 4*np.pi/t[-1]
           
            # Constant phase
            output = PM(op_input, el_input=2.5, Vpi=5)
           
            plt.subplot(311)
            plt.plot(t, op_input.phase()[0] - w*t, 'r', t, output.phase()[0] - w*t, 'b', lw=3)
            plt.xticks([])
            plt.ylabel('Fase [rad]')
            plt.legend(['input', 'output'], bbox_to_anchor=(1, 1), loc='upper left')
            plt.title(r'Constant phase change ($\Delta f=0$)')
           
            # Lineal phase
            output = PM(op_input, el_input=np.linspace(0,5*np.pi,op_input.len()), Vpi=5)
           
            plt.subplot(312)
            plt.plot(t, op_input.phase()[0] - w*t, 'r-', t, output.phase()[0] - w*t, 'b', lw=3)
            plt.xticks([])
            plt.ylabel('Fase [rad]')
            plt.title(r'Linear phase change  ($\Delta f \\rightarrow cte.$)')
           
            # Quadratic phase
            output = PM(op_input, el_input=np.linspace(0,(5*np.pi)**0.5,op_input.len())**2, Vpi=5)
            
            plt.subplot(313)
            plt.plot(t, op_input.phase()[0] - w*t, 'r-', t, output.phase()[0] - w*t, 'b', lw=3)
            plt.xlabel('Tiempo [ns]')
            plt.ylabel('Fase [rad]')
            plt.title(r'Quadratic phase change ($\Delta f \\rightarrow linear$)')
            plt.tight_layout()
            plt.show()

    .. image:: /_images/PM_example1.svg
        :alt: result of PM example 1
        :align: center
    """
    tic()

    if not isinstance(op_input, optical_signal):
        raise TypeError("`op_input` must be of type (optical_signal).")

    if isinstance(el_input, (float, int)):
        el_input = np.ones(op_input.len()) * el_input
    elif isinstance(el_input, electrical_signal):
        el_input = el_input.signal
        if el_input.size != op_input.signal.len():
            raise ValueError("The length of `el_input` must be equal to the length of `op_input`.")
    elif isinstance(el_input, ndarray):
        if len(el_input) != op_input.len():
            raise ValueError("The length of `el_input` must be equal to the length of `op_input`.")
    else:
        raise TypeError("`el_input` must be of type (int or electrical_signal).")
    
    output = optical_signal(np.zeros_like(op_input.signal))

    output.signal = op_input.signal * np.exp(1j * el_input * pi / Vpi)

    if np.sum(op_input.noise):
        output.noise = op_input.noise * np.exp(1j * el_input * pi / Vpi)
    
    output.ejecution_time = toc()
    return output



def MZM(op_input: optical_signal, 
        el_input: Union[float, ndarray, electrical_signal], 
        bias: float=0.0, 
        Vpi: float=5.0, 
        loss_dB: float=0.0, 
        eta: float=0.1, 
        BW: float=40e9):
    """
    Mach-Zehnder modulator (MZM) model. Asymmetric coupler and opposite driving voltages (V1=-V2 Push-Pull config). 

    .. math:: 
        E_{out} = E_{in} \\cdot \\sqrt{l} \\cdot \\left( \\cos(\\frac{\\pi}{2V_{\\pi}}(V_{in}+V_{bias})) + j \\frac{\\eta}{2} \\sin(\\frac{\\pi}{2V_{\\pi}}(V_{in}+V_{bias})) \\right) 
    
    See model theory in `Tetsuya Kawanishi - Electro-optic Modulation for Photonic Networks (Textbooks in Telecommunication Engineering)-Springer (2022)` Chapter 4.3.
    
    Args:
        op_input (optical_signal): Optical signal to be modulated.
        el_input (float | ndarray | electrical_signal): Driver voltage.
        bias (float, Optional): Modulator bias voltage (default: 0.0 [V]).
        Vpi (float, Optional): Voltage at which the device switches from on-state to off-state (default: 5.0 [V]).
        loss_dB (float, Optional): Propagation or insertion losses in the modulator, value in dB (default: 0.0).
        eta (float, Optional): Imbalance ratio of light intensity between the two arms of the modulator (default: 0.1). ER = -20*log10(eta/2) (=26 dB by default).
        BW (float, Optional): Modulator bandwidth in [Hz] (default: 40e9).

    Returns:
        optical_signal: Modulated optical signal.

    Raises:
        TypeError: If ``op_input`` type is not `optical_signal`.
        TypeError: If ``el_input`` type is not in [`float`, `ndarray`, `electrical_signal`].
        ValueError: If ``el_input`` is ndarray or `electrical_signal` but, length is not equal to ``op_input`` length.
    """

    tic()
    if not isinstance(op_input, optical_signal): 
        raise TypeError("`op_input` debe ser del tipo (optical_signal).")
    
    if isinstance(el_input, (int, float)):
        el_input = np.ones(op_input.len()) * el_input
    elif isinstance(el_input, electrical_signal):
        el_input = el_input.signal
        if el_input.size != op_input.signal.len():
            raise ValueError("La longitud de `el_input` debe ser igual a la longitud de `op_input`.")
    elif isinstance(el_input, ndarray):
        if len(el_input) != op_input.len():
            raise ValueError("La longitud de `el_input` debe ser igual a la longitud de `op_input`.")
    else:
        raise TypeError("`el_input` debe ser del tipo (int, float, ndarray ó electrical_signal).")
    
    loss = idb(-loss_dB)

    output = op_input.copy()
    g_t = pi/2/Vpi * (el_input + bias)
    output.signal = op_input.signal * loss**0.5 * (np.cos(g_t) + 1j*eta/2*np.sin(g_t))

    if np.sum(op_input.noise):
        output.noise = op_input.noise * loss**0.5 * (np.cos(g_t) + 1j*eta/2*np.sin(g_t))

    t_ = toc()
    output = LPF(output, BW)

    output.ejecution_time += t_ 
    return output



def BPF(input: optical_signal, 
        BW: float, 
        n: int=4, 
        fs: float=None):
    """
    Optical Band-Pass Filter (BPF). Filters the input optical signal, allowing only the desired frequency band to pass.
    Bessel filter model.

    Args:
        input (optical_signal): The optical signal to be filtered.
        BW (float): The bandwidth of the filter in Hz.
        n (int, Optional): The order of the filter (default: ``n=4``).
        fs (float, Optional): The sampling frequency of the input signal (default: ``fs=gv.fs``).
    
    Returns:
        optical_signal: The filtered optical signal.
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")
    if not fs:
        fs = gv.fs

    sos_band = sg.bessel(N=n, Wn=BW/2, btype='low', fs=fs, output='sos', norm='mag')

    output = optical_signal(np.zeros((2, input.len())))

    output.signal = sg.sosfiltfilt(sos_band, input.signal, axis=-1)

    if np.sum(input.noise):
        output.noise = sg.sosfiltfilt(sos_band, input.noise, axis=-1)

    output.ejecution_time = toc()
    return output


def EDFA(input: optical_signal, 
         G: float, 
         NF: float, 
         BW: float):
    """
    Erbium Doped Fiber (EDFA). Amplifies the optical signal at the input, adding amplified spontaneous emission (ASE) noise. 
    Simplest model (no saturation output power).
    
    Args:
        input (optical_signal): The optical signal to be amplified.
        G (float): The gain of the amplifier, in [dB].
        NF (float): The noise figure of the amplifier, in [dB].
        BW (float): The bandwidth of the amplifier, in [Hz].
    
    Returns:
        optical_signal: The amplified optical signal.

    Raises:
        TypeError: if ``input`` is not an optical signal.    
    """
    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")
     
    output = BPF( input * idb(G)**0.5, BW )
    # ase = BPF( optical_signal( np.zeros_like(input.signal), np.exp(-1j*np.random.uniform(0, 2*pi, input.noise.shape)) ), BW )
    ase = BPF( optical_signal( noise=np.exp(-1j*np.random.uniform(0, 2*pi, input.signal.shape)) ), BW )
    t_ = output.ejecution_time + ase.ejecution_time
    
    tic()
    P_ase = idb(NF) * h * gv.f0 * (idb(G)-1) * BW

    norm_x, norm_y = ase.power('noise') # power of ASE noise in [W] for each polarization

    ase.noise[0] /= norm_x**0.5 / (P_ase/2)**0.5
    ase.noise[1] /= norm_y**0.5 / (P_ase/2)**0.5

    output += ase

    output.ejecution_time = t_ + toc()
    return output


def DM(input: optical_signal, D: float):
    """
    Dispersive Medium. Emulates a medium with only the dispersion property, i.e., only β2 different from zero.

    .. math:: H(\\omega) = e^{-j \\frac{D}{2} \\omega^2}

    .. math :: E_{out}(t) = \\mathcal{F}^{-1} \\left\\{ H(\\omega) \\cdot \\mathcal{F} \\left\\{ E_{in}(t) \\right\\} \\right\\}

    Args:
        input (optical_signal): The input optical signal.
        D (float): The dispersion coefficient of the medium (β2·z), in [ps^2].
    
    Returns:
        optical_signal: The output optical signal.
    
    Raises:
        TypeError: If ``input`` is not an optical signal.

    Example:
        .. code-block:: python
            :linenos:
            :emphasize-lines: 12

            from opticomlib.devices import DM, DAC
            from opticomlib import optical_signal, gv, idbm

            import matplotlib.pyplot as plt
            import numpy as np

            gv(N=7, sps=32, R=10e9)

            signal = DAC('0,0,0,1,0,0,0', pulse_shape='gaussian')
            input = optical_signal( signal.signal/signal.power()**0.5*idbm(20)**0.5 )

            output = DM(input, D=4000)

            plt.subplot(211)
            input.plot('r-', label='input', lw=3)
            output.plot('b-', label='output', lw=3).grid()

            plt.subplot(212)
            plt.plot(gv.t[:-1]*1e9, np.diff(input.phase()[0])/gv.dt*1e-9, 'r-', lw=3)
            plt.plot(gv.t[:-1]*1e9, np.diff(output.phase()[0])/gv.dt*1e-9, 'b-', lw=3)
            input.grid()
            plt.xlabel('Time (ns)')
            plt.ylabel(r'$f_i(t)$ (GHz)')
            plt.ylim(-150, 150)
            plt.show()

        .. image:: /_images/DM_example1.svg
            :alt: result of DM example 1
            :align: center
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("The input must be an optical signal!") 

    # Convert units of D:
    D *= 1e-12**2
    
    output = (input('w') * np.exp(- 1j * input.w()**2 * D/2 ))('t')
    
    output.ejecution_time = toc()
    return output


def FIBER(input: optical_signal, 
          length: float, 
          alpha: float=0.0, 
          beta_2: float=0.0, 
          beta_3: float=0.0, 
          gamma: float=0.0, 
          phi_max:float=0.05, 
          show_progress=False):
    """Optical Fiber.

    Simulates the transmission through an optical fiber, solving Schrödinger's equation numerically,
    by using split-step Fourier method with adaptative step (method based on limmiting the nonlilear phase rotation). 
    Polarization mode dispersion (PMD) is not considered in this model.

    paper source: https://ieeexplore.ieee.org/document/1190149

    Args:
        input (optical_signal): Input optical signal.
        length (float): Length of the fiber, in [km].
        alpha (float, Optional): Attenuation coefficient of the fiber, in [dB/km] (default: 0.0).
        beta_2 (float, Optional): Second-order dispersion coefficient of the fiber, in [ps^2/km] (default: 0.0).
        beta_3 (float, Optional): Third-order dispersion coefficient of the fiber, in [ps^3/km] (default: 0.0).
        gamma (float, Optional): Nonlinearity coefficient of the fiber, in [(W·km)^-1] (default: 0.0).
        phi_max (float, Optional): Upper bound of the nonlinear phase rotation, in [rad] (default: 0.05).
        show_progress (bool, Optional): Show progress bar (default: False).

    Returns:
        optical_signal: Output optical signal.

    Raises:
        TypeError: If ``input`` is not an optical signal.

    Example:
        >>> from opticomlib.devices import FIBER, DAC
        >>> from opticomlib import optical_signal, gv, idbm
        >>>
        >>> gv(sps=32, R=10e9)
        >>>
        >>> signal = DAC('0,0,0,1,0,0,0', pulse_shape='gaussian')
        >>> input = optical_signal( signal.signal/signal.power()**0.5*idbm(20) )
        >>>
        >>> output = FIBER(input, length=50, alpha=0.01, beta_2=-20, gamma=2, show_progress=True)
        100%|█████████████████████████████████████████████| 100.0/100 [00:00<00:00, 12591.73it/s]
        >>>
        >>> input.plot('r-', label='input', lw=3)
        >>> output.plot('b-', label='output', lw=3).grid().show()

    .. image:: /_images/FIBER_example1.svg
        :alt: result of FIBER example 1
        :align: center
    """

    tic()
    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")

    alpha  = alpha/4.343 # [1/km]

    w = input.w()*1e-12 # [rad/ps]
    D_op = -alpha/2 - 1j/2 * beta_2 * w**2 - 1j/6 * beta_3 * w**3

    A = input.signal

    h = length if (beta_2==0 and beta_3==0) or gamma==0 else phi_max / (gamma * (np.abs(A[0])**2 + np.abs(A[1])**2)).max()

    x_length = h

    if show_progress:
        barra_progreso = tqdm(total=100)

    while True:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) ) # Symmetric Split-Step Fourier Method 

        if show_progress:
            barra_progreso.update( 100 * h / length )

        h = phi_max / (gamma * (np.abs(A[0])**2 + np.abs(A[1])**2)).max() if gamma != 0 else length 

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length
    
    if h != 0:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        
        if show_progress:
            barra_progreso.update( 100 * h / length )

    output = optical_signal( A, input.noise )
    output.ejecution_time = toc()
    return output



def LPF(input: Union[ndarray, electrical_signal], 
        BW: float, 
        n: int=4, 
        fs: float=None):
    """
    Low Pass Filter (LPF) for electrical signals. Filters the input electrical signal, allowing only the desired frequency band to pass.
    Bessel filter model.
    
    Args:
        input (ndarray | electrical_signal): Electrical signal to be filtered.
        BW (float): Filter bandwidth or cutoff frecuency, in [Hz].
        n (int, optional): Filter order (default: 4).
        fs (float, optional): Sampling frequency of the input signal (default: ``fs=gv.fs``).
    
    Returns:
        electrical_signal: Filtered electrical signal.
    
    Raises:
        TypeError: If ``input`` is not of type ndarray or electrical_signal.

    Example:
        .. code-block:: python
            :linenos:
            :emphasize-lines: 11

            from opticomlib.devices import LPF, electrical_signal
            from opticomlib import gv
            import matplotlib.pyplot as plt
            import numpy as np
           
            gv(N = 10, sps=128, R=1e9)
           
            t = gv.t
            c = 20e9/t[-1]   # frequency chirp from 0 to 20 GHz
           
            input = electrical_signal( np.sin( np.pi*c*t**2) )
            output = LPF(input, 10e9)
           
            input.psd('r', label='input', lw=2)
            output.psd('b', label='output', lw=2)
            
            plt.xlim(-30,30)
            plt.ylim(-20, 5)
            plt.annotate('-6 dB', xy=(10, -5), xytext=(10, 2), c='r', arrowprops=dict(arrowstyle='<->'), fontsize=12, ha='center', va='center')
            plt.show()
    
    .. image:: /_images/LPF_example1.svg
        :alt: result of LPF example 1
        :align: center
    """
    tic()

    if not isinstance(input, (ndarray, electrical_signal)):
        raise TypeError("`input` must be of type (ndarray or electrical_signal).")
    elif isinstance(input, electrical_signal):
            signal = input.signal
            noise = input.noise
    else:
        signal = input
        noise = np.zeros_like(input)

    if not fs:
        fs = gv.fs

    sos_band = sg.bessel(N = n, Wn = BW, btype = 'low', fs=fs, output='sos', norm='mag')

    output = electrical_signal( np.zeros_like(signal) )

    output.signal = sg.sosfiltfilt(sos_band, signal)

    if np.sum(noise):
        output.noise = sg.sosfiltfilt(sos_band, noise)
    
    output.ejecution_time = toc()
    return output



def PD(input: optical_signal, 
       BW: float, 
       responsivity: float=1.0, 
       T: float=300.0, 
       R_load: float=50.0, 
       noise: Literal['ase-only','thermal-only','shot-only','ase-thermal','ase-shot','thermal-shot','all']='all'):
    """
    Photodetector. 
    
    Simulates the detection of an optical signal by a photodetector.
    
    Args:
        input (optical_signal): optical signal to be detected
        BW (float): detector bandwidth in [Hz]
        responsivity (float, Optional): detector responsivity in [A/W] (default: 1.0)
        T (float, Optional): detector temperature in [K] (default: 300.0)
        R_load (float, Optional): detector load resistance in [Ohm] (default: 50.0)
        noise (str, Optional): type of noise to include in the simulation (default: 'all')
            
            - ``'ase-only'``: only include ASE noise
            - ``'thermal-only'``: only include thermal noise
            - ``'shot-only'``: only include shot noise
            - ``'ase-thermal'``: include ASE and thermal noise
            - ``'ase-shot'``: include ASE and shot noise
            - ``'thermal-shot'``: include thermal and shot noise
            - ``'all'``: include all types of noise
    
    Returns:
        electrical_signal: the detected electrical signal
    
    Raises:
        ValueError: if the ``noise`` argument is not one of the valid options
    """
    tic()
    if not isinstance(input, optical_signal):
        raise TypeError('')

    i_sig = responsivity * np.sum(input.abs('signal')**2, axis=0) # se suman las dos polarizaciones

    if 'thermal' in noise or 'all' in noise:
        S_T = 4 * kB * T * BW / R_load # Density of thermal noise in [A^2]
        i_T = np.random.normal(0, S_T**0.5, input.len())
    
    if 'shot' in noise or 'all' in noise:
        S_N = 2 * e * i_sig * BW # Density of shot noise in [A^2]
        i_N = np.vectorize(lambda s: np.random.normal(0,s))(S_N**0.5)

    if 'ase' in noise or 'all' in noise:
        i_sig_sp = responsivity * np.abs(input.signal[0]*input.noise[0].conjugate() + \
                            input.signal[0].conjugate()*input.noise[0] + \
                            input.signal[1]*input.noise[1].conjugate() + \
                            input.signal[1].conjugate()*input.noise[1])
        i_sp_sp = responsivity * np.sum(input.abs('noise')**2, axis=0) # se suman las dos polarizaciones

    if noise == 'ase-only':
        noise = i_sig_sp  + i_sp_sp
    elif noise == 'thermal-only':
        noise = i_T 
    elif noise == 'shot-only':
        noise = i_N
    elif noise == 'ase-shot':
        noise = i_sig_sp  + i_sp_sp + i_N 
    elif noise == 'ase-thermal':
        noise = i_sig_sp  + i_sp_sp + i_T 
    elif noise == 'thermal-shot':
        noise = i_T + i_N 
    elif noise == 'all':
        noise = i_sig_sp  + i_sp_sp + i_N + i_T 
    else:
        raise ValueError("The argument `noise` must be one of the following: 'ase-only','thermal-only','shot-only','ase-thermal','ase-shot','thermal-shot','all'.")
    
    t_ = toc()
    filt = LPF(i_sig, BW, n=4)
    output = electrical_signal(filt.signal, noise)

    output.ejecution_time = filt.ejecution_time + t_
    return output



def ADC(input: electrical_signal, fs: float=None, BW: float=None, nbits: int=8) -> binary_sequence:
    """
    Analog-to-Digital Converter. 
    
    Converts an analog electrical signal into a quantized digital signal, sampled at a frequency `fs`
    and filtered with a bandwidth BW.

    Args:
        input (electrical_signal): Electrical signal to be quantized.
        fs (float, Optional): Sampling frequency of the output signal. (default: None).
        BW (float, Optional): ADC bandwidth in Hz. (default: None).
        nbits (int, Optional): Vertical resolution of the ADC, in bits. (default: 8 bits).

    Returns:
        electrical_signal: Quantized digital signal.

    Raises:
        TypeError: If the ``input`` is not of type `electrical_signal`.
    """
    tic()

    if not isinstance(input, electrical_signal):
        raise TypeError("`input` debe ser del tipo (electrical_signal).")
    
    if not fs:
        fs = gv.fs

    if BW:
        filt = sg.bessel(N = 4, Wn = BW, btype = 'low', fs=input.fs(), output='sos', norm='mag')
        signal = sg.sosfiltfilt(filt, input.signal)
    else: 
        signal = input.signal

    signal = sg.resample(signal, int(input.len()*fs/input.fs()))

    V_min = signal.min()
    V_max = signal.max()
    dig_signal = np.round( (signal - V_min) / (V_max - V_min) * (2**nbits - 1) ) # normalizo la señal entre 0 y 2**nbits-1
    dig_signal = dig_signal / (2**nbits - 1) * (V_max - V_min) + V_min # vuelvo a la amplitud original

    if np.sum(input.noise):
        noise = sg.sosfiltfilt(filt, input.noise)
        noise = sg.resample(noise, int(input.len()*fs//input.fs()))
        V_min = noise.min()
        V_max = noise.max()
        dig_noise = np.round( (noise - V_min) / (V_max - V_min) * (2**nbits - 1) ) # normalizo la señal entre 0 y 2**nbits-1
        dig_noise = dig_noise / (2**nbits - 1) * (V_max - V_min) + V_min # vuelvo a la amplitud original

        output = electrical_signal( dig_signal, dig_noise )
    else:
        output = electrical_signal( dig_signal )

    output.ejecution_time = toc()
    return output


def GET_EYE(input: Union[electrical_signal, optical_signal, ndarray], nslots: int=4096, sps_resamp: int=None):
    """Get Eye Params.
    
    Estimates all the fundamental parameters and metrics of the eye diagram of the input electrical signal.

    Args:
        input (electrical_signal | optical_signal): Electrical or optical signal from which the eye diagram will be estimated.
        nslots (int, Optional): Number of slots to consider for eye reconstruction (default: 4096).
        sps_resamp (int, Optional): Number of samples per slot to interpolate de original signal (default: None).
    
    Returns:
        eye: Object of the Eye class with all the parameters and metrics of the eye diagram.

    Example:
        .. code-block:: python
            :linenos:
            :emphasize-lines: 10

            from opticomlib.devices import PRBS, DAC, GET_EYE
            from opticomlib import gv
            import numpy as np
           
            gv(sps=64, R=1e9)
           
            y = DAC( PRBS(), pulse_shape='gaussian')
            y.noise = np.random.normal(0, 0.05, y.len())
           
            GET_EYE(y, sps_resamp=512).plot() # with interpolation

        .. image:: /_images/GET_EYE_example1.png
            :alt: result of GET_EYE example 1
            :align: center
    """
    tic()


    def shorth_int(data: np.ndarray) -> tuple[float, float]:
        """
        Estimation of the shortest interval containing 50% of the samples in 'data'.
        
        Args:
            data (np.ndarray): Array of data.

        Returns:
            tuple[float, float]: The shortest interval containing 50% of the samples in 'data'.
        """
        diff_lag = lambda data,lag: data[lag:]-data[:-lag]  # Difference between two elements of an array separated by a distance 'lag'
        
        data = np.sort(data)
        lag = len(data)//2
        diff = diff_lag(data,lag)
        i = np.where(np.abs(diff - np.min(diff))<1e-10)[0]
        if len(i)>1:
            i = int(np.mean(i))
        return (data[i], data[i+lag])
    
    def find_nearest(levels: np.ndarray, data: Union[np.ndarray, float]) -> Union[np.ndarray, float]: 
        """
        Find the element in 'levels' that is closest to each value in 'data'.
        
        Args:
            levels (np.ndarray): Reference levels.
            data (Union[np.ndarray, float]): Values to compare.
        
        Returns:
            Union[np.ndarray, float]: Vector or float with the values from 'levels' corresponding to each value in 'data'.
        """

        if type(data) == float or type(data) == np.float64:
            return levels[np.argmin(np.abs(levels - data))]
        else:
            return levels[np.argmin( np.abs( np.repeat([levels],len(data),axis=0) - np.reshape(data,(-1,1)) ),axis=1 )]

    eye_dict = {}

    if isinstance(input, ndarray):
        if input.ndim == 2:
            input = optical_signal(input)
        elif input.ndim == 1:
            input = electrical_signal(input)
        else:
            raise ValueError("The `input` must be a 1D or 2D array.")

    sps = input.sps(); eye_dict['sps'] = sps
    dt = input.dt(); eye_dict['dt'] = dt

    
    n = input[sps:].len()%(2*input.sps())
    if n: input = input[sps:-n]
    
    nslots = min(input.len()//sps//2*2, nslots)
    input = input[:nslots*sps]

    if isinstance(input, optical_signal):
        s = input.abs()
        input = (s[0]**2 + s[1]**2)**0.5
    elif isinstance(input, electrical_signal):
        input = (input.signal+input.noise).real
    else:
        raise TypeError("The argument 'input' must be 'optical_signal' o 'electrical_signal'.")

    input = np.roll(input, -sps//2+1) # To focus the eye on the chart
    y_set = np.unique(input)

    # resampled the signal to obtain a higher resolution in both axes
    if sps_resamp:
        input = sg.resample(input, nslots*sps_resamp); eye_dict['y'] = input
        t = np.kron(np.ones(nslots//2), np.linspace(-1, 1-dt, 2*sps_resamp)); eye_dict['t'] = t
    else:
        eye_dict['y'] = input
        t = np.kron(np.ones((len(input)//sps)//2), np.linspace(-1, 1-dt, 2*sps)); eye_dict['t'] = t

    # We obtain the centroid of the samples on the Y axis
    vm = np.mean(sk.KMeans(n_clusters=2, n_init=10).fit(input.reshape(-1,1)).cluster_centers_)

    # we obtain the shortest interval of the upper half that contains 50% of the samples
    top_int = shorth_int(input[input>vm]) 
    # We obtain the LMS of level 1
    state_1 = np.mean(top_int)
    # we obtain the shortest interval of the lower half that contains 50% of the samples
    bot_int = shorth_int(input[input<vm])
    # We obtain the LMS of level 0
    state_0 = np.mean(bot_int)

    # We obtain the amplitude between the two levels 0 and 1
    d01 = state_1 - state_0

    # We take 75% threshold level
    v75 = state_1 - 0.25*d01

    # We take 25% threshold level
    v25 = state_0 + 0.25*d01

    t_set = np.array(list(set(t)))

    # The following vector will be used only to determine the crossing times
    tt = t[(input>v25)&(input<v75)]

    # We get the centroid of the time data
    tm = np.mean(sk.KMeans(n_clusters=2, n_init=10).fit(tt.reshape(-1,1)).cluster_centers_)

    # We obtain the left crossing time
    t_left = find_nearest(t_set, np.mean(tt[tt<tm])); eye_dict['t_left'] = t_left

    # We obtain the crossing time from the right
    t_right = find_nearest(t_set, np.mean(tt[tt>tm])); eye_dict['t_right'] = t_right

    # Determine the center of the eye
    t_center = find_nearest(t_set, (t_left + t_right)/2); eye_dict['t_opt'] = t_center

    # For 20% of the center of the eye diagram
    t_dist = t_right - t_left; eye_dict['t_dist'] = t_dist
    t_span0 = t_center - 0.05*t_dist; eye_dict['t_span0'] = t_span0
    t_span1 = t_center + 0.05*t_dist; eye_dict['t_span1'] = t_span1

    # Within the 20% of the data in the center of the eye diagram, we separate into two clusters top and bottom
    y_center = find_nearest(y_set, (state_0 + state_1)/2)

    # We obtain the optimum time for down sampling
    if sps_resamp:
        instant = np.abs(t-t_center).argmin() - sps_resamp//2 + 1
        instant = int(instant/sps_resamp*sps)
    else:
        instant = np.abs(t-t_center).argmin() - sps//2 + 1
    eye_dict['i'] = instant

    # We obtain the upper cluster
    y_top = input[(input > y_center) & ((t_span0 < t) & (t < t_span1))]; eye_dict['y_top'] = y_top

    # We obtain the lower cluster
    y_bot = input[(input < y_center) & ((t_span0 < t) & (t < t_span1))]; eye_dict['y_bot'] = y_bot

    # For each cluster we calculated the means and standard deviations
    mu1 = np.mean(y_top); eye_dict['mu1'] = mu1
    s1 = np.std(y_top); eye_dict['s1'] = s1
    mu0 = np.mean(y_bot); eye_dict['mu0'] = mu0
    s0 = np.std(y_bot); eye_dict['s0'] = s0

    # We obtain the extinction ratio
    er = 10*np.log10(mu1/mu0) if mu0>0 else np.nan; eye_dict['er'] = er

    # We obtain the eye opening
    eye_h = mu1 - 3 * s1 - mu0 - 3 * s0; eye_dict['eye_h'] = eye_h

    eye_dict['ejecution_time'] = toc()
    return eye(eye_dict)


def SAMPLER(input: electrical_signal, _eye_: eye):
    """
    Receives an electrical signal and an eye object and performs the sampling of the signal
    at the optimal instant determined by the eye object.

    Args:
        input: The electrical signal to be sampled.
        _eye_: The eye object that contains the eye diagram information.

    Returns:
        electrical_signal: The sampled electrical signal at one sample per slot.
    """
    tic()
    output = input[_eye_.i::_eye_.sps]

    output.execution_time = toc()
    return output


def FBG(input: optical_signal, 
        neff: float=1.45,
        v: float=1.0,
        landa_D: float=None,
        fc: float=None,
        kL: float=None,
        L: float=None,
        N: int=None,
        dneff: float=None,
        vdneff: float=None,
        apodization: Literal['uniform', 'rcos', 'gaussian', 'parabolic', 'theory'] = 'uniform',
        F: float=0,
        print_params: bool=True,
        filtfilt: bool=True,
        retH: bool=False):
    """**Fiber Bragg Grating**.

    This function numerically calculates the reflection (transfer function H in reflection) of the grating.
    We have used a method where we reduce the pair of coupled-wave equations to a single Riccati differential equation, which
    can then be solved by tested numerical techniques such as the Runge-Kutta method.
    
    In order to design the grating, combination of the following parameters can be used:
    
        1. **neff**, **v**, **fc**, (**dneff** | **vdneff**), (**N** | **kL** | **L**)
        2. **neff**, **v**, **lambdaD**, (**dneff** | **vdneff**), (**N** | **kL** | **L**)
        3. **neff**, **v**, **lambdaD**, **kL**, (**N** | **L**)

    Bandwidth is governed essentially by three parameters:

        1. Bragg wavelength (:math:`\\lambda_D`). Bandwidth is proportional to :math:`\\lambda_D`.
        2. Product of visibility and effective index change (:math:`v\\delta n_{eff}`). If :math:`v\\delta n_{eff}` is small, the bandwidth is small.
        3. Length of the grating (:math:`L`). Bandwidth is inversely proportional to :math:`L`.

        On the other hand, chirp parameter :math:`F` can increase the bandwidth of the grating as well.

    Parameters
    ----------
    input : optical_signal
        The input optical signal.
    neff : float, optional
        Effective refractive index of core fiber.
    v : float, optional
        Visibility of the grating.
    landa_D : float, optional
        Bragg wavelength (resonance wavelength).
    fc : float, optional
        Center frequency of the grating. Default is global variable f0.
    kL : float, optional
        Product of the coupling coefficient and the length of the grating.
    L : float, optional
        Length of the grating. 
    N : int, optional
        Number of period along grating length.
    dneff : float, optional
        Effective index change.
    vdneff : float, optional
        Effective index change multiplied by visibility (case of approximation σ->0).
    apodization : str, optional, default: 'uniform'
        Apodization function.
    F : float, optional, default: 0
        Chirp parameter. 
    filtfilt : bool, optional, default: True
        If True, group delay will be corrected in output signal.
    retH : bool, optional, default: False
        If True, the function will return the reflectivity (H(w)) of the grating.

    Returns
    -------
    output: optical_signal
        The reflected optical signal
    H: ndarray, optional
        Frequency response of grating fiber H(w), only returned if ``retH=True`` 

    Raises
    ------
    TypeError
        If ``input`` is not an ``optical_signal``.
    ValueError
        If the parameters are not correctly specified.

    Warns
    -----
    UserWarning
        If the apodization function is not recognized, a warning will be issued and the function will use uniform apodization.
    UserWarning
        If bandwith is too large, the function will issue a warning and will use a default bandwidth of `fs`.
        
    Notes
    -----
    Following coupled-wave theory, we assume a periodic, single-mode
    waveguide with an electromagnetic field which can be represented by
    two contradirectional coupled waves in the form [1]_:

    .. math:: E(z) = R(z)e^{-j\\beta_0 z} + S(z)e^{j\\beta_0 z}

    where R and S are the complex amplitudes of the forward- and backward-running mode. These amplitudes are linked by the standard
    coupled-wave equations [2]_:

    .. math::
        \\begin{align*}
            R' &= -j\\hat{\\sigma} R - j\\kappa S \\\\
            S' &= j\\hat{\\sigma} S + j\\kappa R
        \\end{align*}
    
    The coupling coefficient :math:`\\kappa` is related to the amplitude of the waveguide perturbation, also known as AC coupling coefficient and is given by:

    .. math:: \\kappa = \\frac{\\pi}{\\lambda}v\\delta n_{eff}

    :math:`\\hat{\\sigma}` is defined as:

    .. math:: \\hat{\\sigma} = \\delta + \\sigma - \\frac{1}{2}\\phi'

    where :math:`\\delta` indicates the frequency desviation from the Bragg condition, :math:`\\sigma` is the DC coupling coefficient and :math:`\\phi` is the phase-shift 
    of the periodicity:

    .. math::
        \\begin{align*}
            \\delta &= 2\\pi n_{eff} \\left( \\frac{1}{\\lambda} - \\frac{1}{\\lambda_{D}} \\right) \\\\
            \\sigma &= \\frac{2}{v}\\kappa = \\frac{2\\pi}{\\lambda}\\delta n_{eff} \\\\
            \\phi' &= 2Fz/L^2
        \\end{align*}

    :math:`F` parameter is the dimensionless chirp parameter, which is defined as:
    
    .. math:: F = 2\\pi N \\Delta L/L

    We consider, now, structures in which the coupling coefficients :math:`\\sigma(z),\\: \\kappa(z)` and the grating phase :math:`\\phi(z)` are slowly varying functions of z,
    indicating the nonuniformity in the grating parameters. We assume that the structure has a length :math:`L` and extends from :math:`z = —L/2` to
    :math:`z = L/2`. The boundary conditions for our scattering problem are then:

    .. math:: R(-L/2) = 1, \\: S(L/2) = 0

    The key to the reduction of the coupled-wave equations to a single differential equation is the definition of a local reflection coefficient :math:`\\rho(z)`,
    
    .. math:: \\rho(z) = \\frac{S}{R}
    
    The z-derivative of this is
    
    .. math:: \\rho' = \\frac{S'}{R} - \\frac{S}{R^2}R' = \\frac{S'}{R} - \\rho R'

    Combining the above expressions, we obtain a Riccati differential equation for :math:`\\rho` which is of the form
    
    .. math:: \\rho' = j\\hat{\\sigma}\\rho + j\\kappa(1+\\rho^2)

    The boundary condition for this equation is :math:`\\rho(L/2) = 0`.  

    References
    ----------
    .. [1] Turan Erdogan, "Fiber Grating Spectra," VOL. 15, NO. 8, AUGUST 1997. doi: https://doi.org/10.1109/50.618322
    .. [2] H. KOGELNIK, "Filter Response of Nonuniform Almost-Periodic Structures" Vol. 55, No. 1, January 1976. doi: https://doi.org/10.1002/j.1538-7305.1976.tb02062.x 
    
    Example:
        .. code-block:: python
            :linenos:

            from opticomlib import optical_signal, gv, pi, db, plt, np
            from opticomlib.devices import FBG

            gv(fs=100e9)

            x = optical_signal(np.ones(2**12))
            f = x.w(shift=True)/2/pi*1e-9

            for apo in ['uniform', 'parabolic', 'rcos', 'gaussian']:
                _,H = FBG(x, fc=gv.f0, vdneff=1e-4, kL=16, apodization=apo, retH=True)
                plt.plot(f, db(np.abs(H)**2), lw=2, label=apo)

            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.ylim(-100,)
            plt.xlim(-20, 20)
            plt.show()
    
        .. image:: /_images/FBG_example1.svg
            :alt: result of FBG example 1
            :align: center
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")
    
    if fc:
        if dneff:
            if not (L or kL or N):
                raise ValueError("If `fc` and `dneff` are specified, `L`, `kL` or `N` must be specified.")
            
            landa_D = 1/(1 + dneff/neff)*c/fc
            vdneff = dneff*v

            if kL:
                L = kL / (pi*dneff*v / landa_D)
            elif N:
                L = N * landa_D / (2*neff) 
        
        if vdneff:
            if not (L or kL or N):
                raise ValueError("If `fc` and `vdneff` are specified, `L`, `kL` or `N` must be specified.")
            
            landa_D = c/fc
            dneff = 0

            if kL:
                L = kL / (pi*vdneff / landa_D)
            elif N:
                L = N * landa_D / (2*neff)
        else:
            raise ValueError("If `fc` is specified, `dneff` or `vdneff` must be specified.")
    
    elif landa_D:
        if dneff:
            if not (L or kL or N):
                raise ValueError("If `landa_D` and `dneff` are specified, `L`, `kL` or `N` must be specified.")
            
            vdneff = dneff*v

            if kL:
                L = kL / (pi*dneff*v / landa_D)
            elif N:
                L = N * landa_D / (2*neff) 

        if vdneff:
            if not (L or kL or N):
                raise ValueError("If `landa_D` and `vdneff` are specified, `L`, `kL` or `N` must be specified.")
            
            dneff = 0

            if kL:
                L = kL / (pi*vdneff / landa_D)
            elif N:
                L = N * landa_D / (2*neff) 

        elif kL:
            if not (L or N):
                raise ValueError("If `landa_D` and `kL` are specified, `L` or `N` must be specified.")
            if N:
                L = N * landa_D / (2*neff)
            
            dneff = kL*landa_D / (pi*v*L)
            vdneff = dneff*v

        else: 
            raise ValueError("If `landa_D` is specified, `dneff`, 'vdneff' or `kL` must be specified.")
    
    else:
        raise ValueError("Either `fc` or `landa_D` must be specified.")

 
    λ_D = landa_D  # Bragg wavelength
    Λ = λ_D / (2*neff)  # period of the grating
    
    λc = (1+dneff/neff)*λ_D # center wavelength of the grating
    fc = c/λc # center frequency of the grating
    
    λ =  2*pi*c / (input.w(shift=True) + 2*pi*gv.f0) # wavelength vector, centered at global variable f0
    δλ = λ[0] - λ[1] # wavelength resolution

    N = int(L/Λ) # number of periods of the grating

    kL = pi/λ_D*vdneff*L

    δ = 2*pi*neff * (1/λ - 1/λ_D)
    s = 2*pi*dneff / λ    # self-coupling coefficient DC
    k = pi*vdneff / λ  # self-coupling coefficient AC
    
    if apodization == 'theory':
        sigma_ = δ + s

        q = csqrt(k**2 - sigma_**2)

        ## reflection coefficient (frequency response H(w))
        H = (-k * np.sinh(q*L)) / (sigma_ * np.sinh(q*L) + 1j*q* np.cosh(q*L))

    else:
        def ode_func(z, rho, delta, sigma, k, F, apodization): # ODE function, normalized to L (z/L, δL, σL, kL, φ'L)
            if apodization == 'rcos':
                p = rcos(z, alpha=1, T=2)
            elif apodization == 'gaussian':
                p = np.exp(-4*np.log(2)*z**2/(1/3)**2)
            elif apodization == 'parabolic':
                p = 1 - (2*z)**2
            elif apodization == 'uniform':
                p = 1
            else:
                warnings.warn("Apodization function not recognized. Using uniform apodization.")
                p = 1

            k = k*p
            sigma = sigma*p
            sigma_ = delta + sigma - F*z

            return -2j*sigma_*rho - 1j*k*(1+rho**2)
        
        δ = δ[:, np.newaxis] * L
        s = s[:, np.newaxis] * L
        k = k[:, np.newaxis] * L

        rho0 = np.zeros_like(λ, dtype=complex)  # Initial value 
        sol = solve_ivp(ode_func, [0.5, -0.5], rho0, method='RK45', args=(δ, s, k, F, apodization), vectorized=True)
        H = sol.y[:,-1]

    
    y = np.abs(H)

    peaks,_ = sg.find_peaks(y)
    H_max = y[peaks].max()
    
    if (y>0.5).all():
        warnings.warn("Bandwidth of the grating is too large for current sampling rate (`fs`). Consider increasing `fs`.")
        bandwith_str = f' - Δf = >{si(gv.fs,"Hz")} (Δλ = >{si(gv.fs*c/fc**2,"m")})'
    else:
        r = sg.peak_widths(y, peaks)

        BW_λ = r[0].max()*δλ
        BW_f = fc**2*BW_λ/c

        bandwith_str = f' - Δf = {si(BW_f,"Hz")} (Δλ = {si(BW_λ,"m")})'

    ## Print parameters of the grating
    if print_params:
        print('\n*** Fiber Bragg Grating Features ***')
        print(f' - Λ = {si(Λ,"m")}')
        print(f' - N = {N}')
        print(f' - L = {si(L,"m")}')
        print(f' - λo = {si(c/fc,"m",4)}')
        print(bandwith_str)
        print(f' - ρo = {y.max():.2f}')
        print(f' - loss = {-db(H_max**2):.1f} dB')
        print(f' - vδneff = {vdneff:.0e}')
        print(f' - kL = {kL:.1f}')
        if F:
            print(f' - F = {F:.1f}')
            print(f' - ΔΛ = {si(Λ*F/(2*pi*N),"m")}')
        print('************************************\n')


    if filtfilt: ## correct H(w)
        phase = np.unwrap(np.angle(H)) # phase of H(w)

        dw = 2*pi*gv.fs/input.len() # frequency resolution
        tau_g = np.diff(phase, append=phase[-1]) / dw # group delay
        i = np.argmin( np.abs(λ - c/fc) ) # index of the center wavelength

        H = H * np.exp(-1j * input.w(shift=True) * tau_g[i]) # corrected H(w)

    ## apply to input optical signal
    output = ifft(fft(input.signal)*ifftshift(H))

    output = optical_signal(output)
    output.ejecution_time = toc()

    if retH:
        return output, H
    return output








### algunas funciones de prueba
def animated_fiber_propagation(input: optical_signal, M :int, length_: float, alpha_: float=0.0, beta_2_: float=0.0, beta_3_: float=0.0, gamma_: float=0.0, phi_max:float=0.05):
    from matplotlib.animation import FuncAnimation

    # cambio las unidades
    length = length_ * 1e3 
    alpha  = alpha_ * 1/(4.343*1e3)
    beta_2 = beta_2_ * 1e-12**2/1e3
    beta_3 = beta_3_ * 1e-12**3/1e3
    gamma  = gamma_ * 1/1e3

    w = input.w()
    D_op = -alpha/2 - 1j/2 * beta_2 * w**2 - 1j/6 * beta_3 * w**3

    A = input.signal[0]

    h = length if (beta_2==0 and beta_3==0) or gamma==0 else phi_max / (gamma * np.abs(A)**2).max()

    x_length = h
    A_z = [A]
    hs = [0]

    while True:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A)
        hs.append(h)

        h = phi_max / (gamma * np.abs(A)**2).max() if gamma != 0 else length 

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length
    
    if h > 0:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A)
        hs.append(h)

    t = input.t()*gv.slot_rate

    fig, ax = plt.subplots()

    line, = ax.plot(t, np.abs(A_z[0]), lw=2, color = 'red', ls='--')
    line, = ax.plot([], [], lw=2, color = 'k')

    plt.suptitle(r'Fiber: $\alpha = {:.2f}$ dB/km, $\beta_2 = {}$ ps^2/km, $\gamma = {}$ (W·km)^-1'.format(alpha_, beta_2_, gamma_))
    ax.set_xlabel(r'$t/T_{slot}$')
    ax.set_ylabel('|A(z,t)|')
    ax.set_xlim((0, t.max()))
    ax.set_ylim((abs(A_z[0]).min()*0.95, np.abs(A_z).max()*1.05))

    time_text = ax.text(0.05,0.9, '', transform = ax.transAxes)


    def init():
        line.set_data([],[])
        time_text.set_text('z = 0.0 Km')
        for i in t[::M*gv.sps]:
           plt.axvline(i, color='k', ls='--')
        for i in t[::gv.sps]:
           plt.axvline(i, color='k', ls='--', alpha=0.3,lw=1)
        return [line, time_text] 

    def animate(i):
        y = np.abs(A_z[i])
        line.set_data(t, y)
        time_text.set_text('z = {:.2f} Km'.format(np.cumsum(hs)[i]/1e3))
        return [line, time_text] 

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(A_z), interval=100, blit=True, repeat=False)
    plt.show()


def animated_fiber_propagation_with_psd(input: optical_signal, M :int, length_: float, alpha_: float=0.0, beta_2_: float=0.0, beta_3_: float=0.0, gamma_: float=0.0, phi_max:float=0.05, n:int=None):
    from matplotlib.animation import FuncAnimation

    n = input.len() if n is None else n*M*input.sps() 
    
    length = length_
    alpha  = alpha_/4.343
    beta_2 = beta_2_
    beta_3 = beta_3_
    gamma  = gamma_

    w = input.w() * 1e-12 # rad/ps
    D_op = -alpha/2 - 1j/2 * beta_2 * w**2 - 1j/6 * beta_3 * w**3

    A = input.signal[0]

    h = length if (beta_2==0 and beta_3==0) or gamma==0 else phi_max / (gamma * np.abs(A)**2).max()

    x_length = h
    A_z = [A]
    A_z_w = [fft(A)]
    hs = [0]

    while True:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A*np.exp(alpha*x_length/2))
        A_z_w.append(fft(A*np.exp(alpha*x_length/2)))
        hs.append(h)

        h = phi_max / (gamma * np.abs(A)**2).max() if gamma != 0 else length 

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length
    
    if h > 0:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A*np.exp(alpha*length/2))
        A_z_w.append(fft(A*np.exp(alpha*length/2)))
        hs.append(h)

    t = input.t()*gv.slot_rate

    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(6,6))

    line1, = ax1.plot(t[:n], np.abs(A_z[0])[:n], lw=2, color = 'red', ls='--')
    line1, = ax1.plot([], [], lw=2, color = 'k')

    plt.suptitle(r'Fiber: $\alpha = {:.2f}$ dB/km, $\beta_2 = {}$ ps^2/km, $\gamma = {}$ (W·km)^-1'.format(alpha_, beta_2_, gamma_))
    ax1.set_xlabel('t/T')
    ax1.set_ylabel('|A(z,t)|')
    ax1.set_xlim((0, t[:n].max()))
    ax1.set_ylim((0, np.abs(A_z[:n]).max()))

    z_text = ax2.text(0.05,0.9, '', transform = ax2.transAxes)

    f = fftshift(w/2/np.pi)*1e3 # GHz

    line2, = ax2.plot(f, fftshift(np.abs(A_z_w[0])**2)/input.len(), '--g', lw=2)
    line2, = ax2.plot([], [], 'k', lw=2)

    ax2.set_xlabel('f [GHz]')
    ax2.set_ylabel(r'$|A(z,w)|^2$')
    sigma = -f[np.cumsum(np.abs(fftshift(A_z_w[0]))**2)<0.001*np.sum(np.abs(A_z_w[0])**2)][-1]
    ax2.set_xlim((-2*sigma, 2*sigma))
    ax2.set_ylim((0, np.abs(A_z_w).max()**2*1.05/input.len()))
    ax2.grid()

    plt.tight_layout()


    def init():
        line1.set_data([],[])
        z_text.set_text('z = 0.0 Km')
        for i in t[:n:M*gv.sps]:
            ax1.axvline(i, color='k', ls='--')
        for i in t[:n:gv.sps]:
            ax1.axvline(i, color='k', ls='--', alpha=0.3,lw=1)
        return [line1, z_text] 

    def animate(i):
        y = np.abs(A_z[i])
        line1.set_data(t[:n], y[:n])
        z_text.set_text('z = {:.2f} Km'.format(np.cumsum(hs)[i]))

        y = fftshift(np.abs(A_z_w[i])**2)/input.len()
        line2.set_data(f, y)
        return [line1, line2, z_text] 

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(A_z), interval=100, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':
    pass



