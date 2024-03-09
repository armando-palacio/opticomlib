"""
.. rubric:: Devices
.. autosummary::

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
from typing import Literal, Union, Callable
from numpy import ndarray
from scipy.constants import pi, k as kB, e, h, c
from numpy.fft import fft, ifft, fftshift, ifftshift

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif' 

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
    si,
    tau_g,
    bode,
    dispersion,
)



def PRBS(n=2**8, 
         user=[], 
         order=None):
    r"""**Pseudorandom binary sequence generator**

    Parameters
    ----------
    n : int, optional, default: 2**8
        lenght of random binary sequence
    user : str or array_like, optional, default: []
        binary sequence user pattern
    order : int, optional, default: None
        degree of the generating pseudorandom polynomial

    Returns
    -------
    seq : binary_sequence
        generated binary sequence

    Examples
    --------
    Using parameter **n**, this function generate a random sequence of lenght `n`. Internally it use ``numpy.random.randint`` function.
    
    >>> from opticomlib.devices import PRBS
    >>> PRBS(10).data
    array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1], dtype=uint8)  #random

    On the other hand, the **user** parameter can be used for a custom sequence.
    We can input it in *str* format separating the values by spaces ``' '`` or by commas ``','``. 

    >>> PRBS(user='1 0 1 0   0 1 1 1   0,1,0,0   1,1,0,1').data
    array([1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=uint8)

    The last way in which the function can be used is by passing the **order** of the generating polynomial
    as an argument, which will return a pseudo-random binary sequence of lenght :math:`2^{order}-1`, using an internal algorithm.

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
    output.execution_time = toc()
    return output



def DAC(input: Union[str, list, tuple, ndarray, binary_sequence], 
        Vout: float=None,
        pulse_shape: Literal['rect','gaussian']='rect', 
        **kargs):  
    r"""
    **Digital-to-Analog Converter**

    Converts a binary sequence into an electrical signal, sampled at a frequency ``gv.fs``.

    Parameters
    ----------
    input : str, list, tuple, ndarray, or binary_sequence
        Input binary sequence.
    Vout : float, default: 1.0
        Output signal amplitude. Should be in the range [-15, 15] Volts.
    pulse_shape : str, default: "rect"
        Pulse shape at the output. Can be ``'rect'`` or ``'gaussian'``.

    Other Parameters
    ----------------
    c : float, default: 0.0
        Chirp of the Gaussian pulse. Only applicable if ``pulse_shape='gaussian'``.
    m : int, default: 1
        Order of the super-Gaussian pulse. Only applicable if ``pulse_shape='gaussian'``.
    T : int, default: ``gv.sps``
        Pulse width at half maximum in number of samples. Only applicable if ``pulse_shape='gaussian'``.

    Returns
    -------
    electrical_signal
        The converted electrical signal.

    Raises
    ------
    TypeError
        If ``input`` type is not in [str, list, tuple, ndarray, binary_sequence].
    NameError
        If ``pulse_shape`` is not 'rect' or 'gaussian'.
    ValueError
        If ``Vout`` is not between -15 and 15 Volts.

    Examples
    --------
    .. plot::
        :include-source:
        :alt: DAC example 1
        :align: center

        from opticomlib.devices import DAC
        from opticomlib import gv

        gv(sps=32) # set samples per bit

        DAC('0 0 1 0 0', Vout=5, pulse_shape='gaussian', m=2).plot('r', lw=3).show()
    """
    tic()
    if not isinstance(input, binary_sequence):
        input = binary_sequence(input)
    
    sps = gv.sps

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

    output.execution_time = toc()
    return output



def PM(op_input: optical_signal, 
       el_input: Union[float, ndarray, electrical_signal], 
       Vpi: float=5.0):
    r"""
    **Optical Phase Modulator**

    Modulate the phase of the input optical signal through input electrical signal.

    Parameters
    ----------
    op_input : :obj:`optical_signal`
        Optical signal to be modulated.
    el_input : :obj:`float`, :obj:`ndarray`, or :obj:`electrical_signal`
        Driver voltage. It can be an integer value, in which case the phase modulation is constant, or an electrical signal of the same length as the optical signal.
    Vpi : :obj:`float`
        Voltage at which the device achieves a phase shift of :math:`\pi`. Default value is 5.0.

    Returns
    -------
    op_output: :obj:`optical_signal`
        Modulated optical signal.

    Raises
    ------
    TypeError
        If ``op_input`` type is not [:obj:`optical_signal`].
        If ``el_input`` type is not in [:obj:`float`, :obj:`ndarray`, :obj:`electrical_signal`].
    ValueError
        If ``el_input`` is [:obj:`ndarray`] or [:obj:`electrical_signal`] but, length is not equal to ``op_input`` length.

    Notes
    -----
    The output signal is given by:

    .. figure:: _images/PMv2.png
        :width: 50%
        :align: center
        :alt: MZM
    
    .. math:: E_{out} = E_{in} \cdot e^{\left(j\pi \frac{u(t)}{V_{\pi}}\right)}

    Examples
    --------
    .. code-block:: python
        :linenos:

        from opticomlib.devices import PM
        from opticomlib import optical_signal, gv
        import matplotlib.pyplot as plt
        import numpy as np

        gv(sps=16, R=1e9) # set samples per bit and bitrate

        op_input = optical_signal(np.exp(1j*np.linspace(0,4*np.pi, 1000))) # input optical signal ( exp(j*w*t) )
        t = op_input.t()*1e9

        fig, axs = plt.subplots(3,1, sharex=True, tight_layout=True)

        # Constant phase
        output = PM(op_input, el_input=2.5, Vpi=5)

        axs[0].set_title(r'Constant phase change ($\Delta f=0$)')
        axs[0].plot(t, op_input.signal[0].real, 'r-', label='input', lw=3)
        axs[0].plot(t, output.signal[0].real, 'b-', label='output', lw=3)
        axs[0].grid()

        # Lineal phase
        output = PM(op_input, el_input=np.linspace(0,5*np.pi,op_input.len()), Vpi=5)

        axs[1].set_title(r'Linear phase change  ($\Delta f \rightarrow cte.$)')
        axs[1].plot(t, op_input.signal[0].real, 'r-', label='input', lw=3)
        axs[1].plot(t, output.signal[0].real, 'b-', label='output', lw=3)
        axs[1].grid()

        # Quadratic phase
        output = PM(op_input, el_input=np.linspace(0,(5*np.pi)**0.5,op_input.len())**2, Vpi=5)

        plt.title(r'Quadratic phase change ($\Delta f \rightarrow linear$)')
        axs[2].plot(t, op_input.signal[0].real, 'r-', label='input', lw=3)
        axs[2].plot(t, output.signal[0].real, 'b-', label='output', lw=3)
        axs[2].grid()

        plt.xlabel('Tiempo [ns]')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.show()
    
    .. image:: _images/PM_example1.svg
        :width: 100%
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
    
    output.execution_time = toc()
    return output



def MZM(op_input: optical_signal, 
        el_input: Union[float, ndarray, electrical_signal], 
        bias: float=0.0, 
        Vpi: float=5.0, 
        loss_dB: float=0.0, 
        eta: float=0.1, 
        BW: float=40e9):
    r"""
    **Mach-Zehnder modulator**

    Asymmetric coupler and opposite driving voltages model (:math:`V_1=-V_2` Push-Pull config). 

    Parameters
    ----------
    op_input : :obj:`optical_signal`
        Optical signal to be modulated.
    el_input : :obj:`float`, :obj:`ndarray`, or :obj:`electrical_signal`
        Driver voltage, with zero bias. 
    bias : :obj:`float`, default: 0.0
        Modulator bias voltage.
    Vpi : :obj:`float`, default: 5.0
        Voltage at which the device switches from on-state to off-state.
    loss_dB : :obj:`float`, default: 0.0
        Propagation or insertion losses in the modulator, value in dB.
    eta : :obj:`float`, default: 0.1
        Imbalance ratio of light intensity between the two arms of the modulator. :math:`ER = -20\log_{10}(\eta/2)` (:math:`=26` dB by default).
    BW : :obj:`float`, default: 40e9
        Modulator bandwidth in Hz.

    Returns
    -------
    :obj:`optical_signal`
        Modulated optical signal.

    Raises
    ------
    TypeError
        If ``op_input`` type is not [:obj:`optical_signal`].
        If ``el_input`` type is not in [:obj:`float`, :obj:`ndarray`, :obj:`electrical_signal`].
    ValueError
        If ``el_input`` is [:obj:`ndarray`] or [:obj:`electrical_signal`] but, length is not equal to ``op_input`` length.
    
    Notes
    -----
    .. figure:: _images/MZMv2.png
        :width: 50%
        :align: center
        :alt: MZM

    The output signal is given by [1]_:


    .. math:: 
        E_{out} = E_{in} \cdot \sqrt{l} \cdot \left[ \cos\left(\frac{\pi}{2V_{\pi}}(u(t)+V_{bias})\right) + j \frac{\eta}{2} \sin\left(\frac{\pi}{2V_{\pi}}(u(t)+V_{bias})\right) \right] 

    References
    ----------
    .. [1] Tetsuya Kawanishi, "Electro-optic Modulation for Photonic Networks", Chapter 4.3 (2022). doi: https://doi.org/10.1007/978-3-030-86720-1

    Examples
    --------
    .. code-block:: python
        :linenos:

        from opticomlib import idbm, dbm, optical_signal, gv
        from opticomlib.devices import MZM

        import numpy as np
        import matplotlib.pyplot as plt

        gv(sps=128, R=10e9) # set samples per bit and bitrate

        Vpi = 5
        tx_seq = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0], bool); not_tx_seq = ~tx_seq
        V = 2*(np.kron(not_tx_seq, np.ones(gv.sps)) - 0.5 )*Vpi/2 

        input = optical_signal( np.ones_like(V)*idbm(10)**0.5 )
        t = input.t()*1e9

        mod_sig = MZM(input, el_input=V, bias=Vpi/2, Vpi=Vpi, loss_dB=3, eta=0.1, BW=2*gv.R)

        fig, axs = plt.subplots(3,1, sharex=True, tight_layout=True)

        # Plot input and output power
        axs[0].plot(t, dbm(input.signal[0].real**2), 'r-', label='input', lw=3)
        axs[0].plot(t, dbm(mod_sig.abs('signal')[0]**2), 'C1-', label='output', lw=3)
        axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
        axs[0].set_ylabel('Potencia [dBm]')
        for i in t[::gv.sps]:
            axs[0].axvline(i, color='k', linestyle='--', alpha=0.5)

        # Plot fase
        phi_in = input.phase()[0]
        phi_out = mod_sig.phase()[0]

        axs[1].plot(t, phi_in, 'b-', label='Fase in', lw=3)
        axs[1].plot(t, phi_out, 'C0-', label='Fase out', lw=3)
        axs[1].set_ylabel('Fase [rad]')
        axs[1].legend(bbox_to_anchor=(1, 1), loc='upper left')
        for i in t[::gv.sps]:
            axs[1].axvline(i, color='k', linestyle='--', alpha=0.5)

        # Frecuency chirp
        freq_in = 1/2/np.pi*np.diff(phi_in)/np.diff(t)
        freq_out = 1/2/np.pi*np.diff(phi_out)/np.diff(t)

        axs[2].plot(t[:-1], freq_in, 'k', label='Frequency in', lw=3)
        axs[2].plot(t[:-1], freq_out, 'C7', label='Frequency out', lw=3)
        axs[2].set_xlabel('Tiempo [ns]')
        axs[2].set_ylabel('Frequency Chirp [Hz]')
        axs[2].legend(bbox_to_anchor=(1, 1), loc='upper left')
        for i in t[::gv.sps]:
            axs[2].axvline(i, color='k', linestyle='--', alpha=0.5)
        plt.show()

    .. image:: _images/MZM_example1.svg
        :width: 100%
        :align: center
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

    output.execution_time += t_ 
    return output



def BPF(input: optical_signal, 
        BW: float, 
        n: int=4):
    r"""
    **Optical Band-Pass Filter**

    Filters the input optical signal, allowing only the desired frequency band to pass.
    Bessel filter model.

    Parameters
    ----------
    input : optical_signal
        The optical signal to be filtered.
    BW : float
        The bandwidth of the filter in Hz.
    n : int, default: 4
        The order of the filter.

    Returns
    -------
    optical_signal
        The filtered optical signal.
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")

    sos_band = sg.bessel(N=n, Wn=BW/2, btype='low', fs=gv.fs, output='sos', norm='mag')

    output = optical_signal(np.zeros((2, input.len())))

    output.signal = sg.sosfiltfilt(sos_band, input.signal, axis=-1)

    if np.sum(input.noise):
        output.noise = sg.sosfiltfilt(sos_band, input.noise, axis=-1)

    output.execution_time = toc()
    return output


def EDFA(input: optical_signal, 
         G: float, 
         NF: float, 
         BW: float):
    r"""
    **Erbium Doped Fiber**

    Amplifies the optical signal at the input, adding amplified spontaneous emission (ASE) noise. 
    Simplest model (no saturation output power).

    Parameters
    ----------
    input : optical_signal
        The optical signal to be amplified.
    G : float
        The gain of the amplifier, in dB.
    NF : float
        The noise figure of the amplifier, in dB.
    BW : float
        The bandwidth of the amplifier, in Hz.

    Returns
    -------
    optical_signal
        The amplified optical signal.

    Raises
    ------
    TypeError
        If ``input`` is not an optical_signal.    

    """
    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")
     
    output = BPF( input * idb(G)**0.5, BW )
    # ase = BPF( optical_signal( np.zeros_like(input.signal), np.exp(-1j*np.random.uniform(0, 2*pi, input.noise.shape)) ), BW )
    ase = BPF( optical_signal( noise=np.exp(-1j*np.random.uniform(0, 2*pi, input.signal.shape)) ), BW )
    t_ = output.execution_time + ase.execution_time
    
    tic()
    P_ase = idb(NF) * h * gv.f0 * (idb(G)-1) * BW

    norm_x, norm_y = ase.power('noise') # power of ASE noise in [W] for each polarization

    ase.noise[0] /= norm_x**0.5 / (P_ase/2)**0.5
    ase.noise[1] /= norm_y**0.5 / (P_ase/2)**0.5

    output += ase

    output.execution_time = t_ + toc()
    return output


def DM(input: optical_signal, D: float, retH: bool=False):
    r"""
    **Dispersive Medium**

    Emulates a medium with only the dispersion property, i.e., only :math:`\beta_2` different from zero.

    Parameters
    ----------
    input : optical_signal
        The input optical signal.
    D : float
        The dispersion coefficient of the medium (:math:`\beta_2z`), in [ps^2].
    retH : bool, default: False
        If True, the frequency response of the medium is also returned.

    Returns
    -------
    optical_signal
        The output optical signal.
    H : ndarray
        The frequency response of the medium. If ``retH=True``.

    Raises
    ------
    TypeError
        If ``input`` is not an optical signal.

    Notes
    -----
    Frequency response of the medium is given by:

    .. math:: H(\omega) = e^{-j \frac{D}{2} \omega^2}

    The output signal is simply a fase modulation in the frequency domain of the input signal:

    .. math :: E_{out}(t) = \mathcal{F}^{-1} \left\{ H(\omega) \cdot \mathcal{F} \left\{ E_{in}(t) \right\} \right\}

    Example
    -------
    .. plot::
        :include-source:
        :alt: DM example 1
        :align: center

        from opticomlib.devices import DM, DAC
        from opticomlib import optical_signal, gv, idbm, bode 

        import matplotlib.pyplot as plt
        import numpy as np

        gv(N=7, sps=32, R=10e9)

        signal = DAC('0,0,0,1,0,0,0', pulse_shape='gaussian')
        input = optical_signal( signal.signal/signal.power()**0.5*idbm(20)**0.5 )

        output, H = DM(input, D=4000, retH=True)

        t = gv.t*1e9

        plt.style.use('dark_background')
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.05})
        
        ax[0].plot(t, input.abs()[0], 'r-', lw=3, label='input')
        ax[0].plot(t, output.abs()[0], 'b-', lw=3, label='output')

        ax[0].set_ylabel(r'$|E(t)|$')

        ax[1].plot(t[:-1], np.diff(input.phase()[0])/gv.dt*1e-9, 'r-', lw=3)
        ax[1].plot(t[:-1], np.diff(output.phase()[0])/gv.dt*1e-9, 'b-', lw=3)

        plt.xlabel('Time (ns)')
        plt.ylabel(r'$f_i(t)$ (GHz)')
        plt.ylim(-150, 150)
        plt.show()
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("The input must be an optical signal!") 

    # Convert units of D:
    D *= 1e-12**2

    H = np.exp(- 1j * input.w()**2 * D/2 )
    
    output = (input('w') * H)('t')
    
    output.execution_time = toc()
    
    if retH:
        H = np.exp(- 1j * input.w()**2 * D/2 )
        return output, fftshift(H)
    return output


def FIBER(input: optical_signal, 
          length: float, 
          alpha: float=0.0, 
          beta_2: float=0.0, 
          beta_3: float=0.0, 
          gamma: float=0.0, 
          phi_max:float=0.05, 
          show_progress=False):
    r"""
    **Optical Fiber**

    Simulates the transmission through an optical fiber, solving Schrödinger's equation numerically,
    by using split-step Fourier method with adaptive step (method based on limiting the nonlinear phase rotation) [#first]_. 
    Polarization mode dispersion (PMD) is not considered in this model.

    Parameters
    ----------
    input : optical_signal
        Input optical signal.
    length : float
        Length of the fiber, in [km].
    alpha : float, default: 0.0
        Attenuation coefficient of the fiber, in [dB/km].
    beta_2 : float, default: 0.0
        Second-order dispersion coefficient of the fiber, in [ps^2/km].
    beta_3 : float, default: 0.0
        Third-order dispersion coefficient of the fiber, in [ps^3/km].
    gamma : float, default: 0.0
        Nonlinearity coefficient of the fiber, in [(W·km)^-1].
    phi_max : float, default: 0.05
        Upper bound of the nonlinear phase rotation, in [rad].
    show_progress : bool, default: False
        Show algorithm progress bar.

    Returns
    -------
    optical_signal
        Output optical signal.

    Raises
    ------
    TypeError
        If ``input`` is not an optical signal.

    References
    ----------
    .. [#] O.V. Sinkin; R. Holzlohner; J. Zweck; C.R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber communications systems," vol. 21, no. 1, pp. 61-68, Jan. 2003, doi: https://doi.org/10.1109/JLT.2003.808628
    
    Example
    -------
    .. plot:: 
        :include-source:
        :alt: FIBER example 1
        :align: center
        :caption: The input signal is a 10 Gbps NRZ signal with 20 dBm of power. The fiber has a length of 50 km, an attenuation of 0.01 dB/km, 
                    a second-order dispersion of -20 ps^2/km, and a nonlinearity coefficient of 0.1 (W·km)^-1. The output signal is shown in blue.

        from opticomlib.devices import FIBER, DAC
        from opticomlib import optical_signal, gv, idbm

        gv(sps=32, R=10e9)

        signal = DAC('0,0,0,1,0,0,0', pulse_shape='gaussian')
        input = optical_signal( signal.signal/signal.power()**0.5*idbm(20) )

        output = FIBER(input, length=50, alpha=0.01, beta_2=-20, gamma=0.1, show_progress=True)

        input.plot('r-', label='input', lw=3)
        output.plot('b-', label='output', lw=3).show()
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
    output.execution_time = toc()
    return output



def LPF(input: Union[ndarray, electrical_signal], 
        BW: float, 
        n: int=4, 
        fs: float=None,
        retH: bool=False):
    r"""
    **Low Pass Filter**

    Filters the input electrical signal, allowing only the desired frequency band to pass.
    Bessel filter model.

    Parameters
    ----------
    input : ndarray or electrical_signal
        Electrical signal to be filtered.
    BW : float
        Filter bandwidth or cutoff frequency, in [Hz].
    n : int, default: 4
        Filter order.
    fs : float, default: gv.fs
        Sampling frequency of the input signal.
    retH : bool, default: False
        If True, the frequency response of the filter is also returned.

    Returns
    -------
    electrical_signal
        Filtered electrical signal.

    Raises
    ------
    TypeError
        If ``input`` is not of type ndarray or electrical_signal.

    Example
    -------
    .. plot::
        :include-source:
        :alt: LPF example 1
        :align: center
        
        from opticomlib.devices import LPF
        from opticomlib import gv, electrical_signal
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
    
    output.execution_time = toc()
    if retH:
        _,H = sg.sosfreqz(sos_band, worN=signal.size, fs=fs, whole=True)
        return output, fftshift(H) 
    return output



def PD(input: optical_signal, 
       BW: float, 
       responsivity: float=1.0, 
       T: float=300.0, 
       R_load: float=50.0, 
       noise: Literal['ase-only','thermal-only','shot-only','ase-thermal','ase-shot','thermal-shot','all']='all'):
    r"""
    **Photodetector**

    Simulates the detection of an optical signal by a photodetector.

    Parameters
    ----------
    input : optical_signal
        Optical signal to be detected.
    BW : float
        Detector bandwidth in [Hz].
    responsivity : float, default: 1.0
        Detector responsivity in [A/W].
    T : float, default: 300.0
        Detector temperature in [K].
    R_load : float, default: 50.0
        Detector load resistance in [:math:`\Omega`].
    noise : str, default: 'all'
        Type of noise to include in the simulation.
        Options include:
        
        - ``'ase-only'``: only include ASE noise
        - ``'thermal-only'``: only include thermal noise
        - ``'shot-only'``: only include shot noise
        - ``'ase-thermal'``: include ASE and thermal noise
        - ``'ase-shot'``: include ASE and shot noise
        - ``'thermal-shot'``: include thermal and shot noise
        - ``'all'``: include all types of noise

    Returns
    -------
    electrical_signal
        The detected electrical signal.

    Raises
    ------
    TypeError
        If ``input`` is not of type optical_signal.
    ValueError
        If the ``noise`` argument is not one of the valid options.
    """
    tic()
    if not isinstance(input, optical_signal):
        raise TypeError('`input` must be of type (optical_signal).')

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

    output.execution_time = filt.execution_time + t_
    return output



def ADC(input: electrical_signal, fs: float=None, BW: float=None, nbits: int=8) -> binary_sequence:
    r"""
    **Analog-to-Digital Converter**

    Converts an analog electrical signal into a quantized digital signal, sampled at a frequency `fs`
    and filtered with a bandwidth BW.

    Parameters
    ----------
    input : electrical_signal
        Electrical signal to be quantized.
    fs : float, default: None
        Sampling frequency of the output signal.
    BW : float, default: None
        ADC bandwidth in Hz.
    nbits : int, default: 8
        Vertical resolution of the ADC, in bits.

    Returns
    -------
    electrical_signal
        Quantized digital signal.

    Raises
    ------
    TypeError
        If the ``input`` is not of type `electrical_signal`.
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

    output.execution_time = toc()
    return output


def GET_EYE(input: Union[electrical_signal, optical_signal, ndarray], nslots: int=4096, sps_resamp: int=None):
    r"""
    **Get Eye Parameters Estimator**

    Estimates all the fundamental parameters and metrics of the eye diagram of the input electrical signal.

    Parameters
    ----------
    input : electrical_signal | optical_signal
        Electrical or optical signal from which the eye diagram will be estimated.
    nslots : int, default: 4096
        Number of slots to consider for eye reconstruction.
    sps_resamp : int, default: None
        Number of samples per slot to interpolate the original signal.

    Returns
    -------
    eye
        Object of the eye class with all the parameters and metrics of the eye diagram.

    Raises
    ------
    ValueError
        If the ``input`` is a ndarray but dimention is >2.
    TypeError
        If the ``input`` is not of type `electrical_signal`, `optical_signal` or `np.ndarray`.

    Example
    -------
    .. code-block:: python
        :linenos:

        from opticomlib.devices import PRBS, DAC, GET_EYE
        from opticomlib import gv
        import numpy as np

        gv(sps=64, R=1e9)

        y = DAC( PRBS(), pulse_shape='gaussian')
        y.noise = np.random.normal(0, 0.05, y.len())

        GET_EYE(y, sps_resamp=512).plot() # with interpolation

    .. image:: /_images/GET_EYE_example1.png
    """
    tic()


    def shorth_int(data: np.ndarray) -> tuple[float, float]:
        r"""
        Estimation of the shortest interval containing 50% of the samples in 'data'.

        Parameters
        ----------
        data : ndarray
            Array of data.

        Returns
        -------
        tuple[float, float]
            The shortest interval containing 50% of the samples in 'data'.
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
        r"""
        Find the element in 'levels' that is closest to each value in 'data'.

        Parameters
        ----------
        levels : np.ndarray
            Reference levels.
        data : Union[np.ndarray, float]
            Values to compare.

        Returns
        -------
        Union[np.ndarray, float]
            Vector or float with the values from 'levels' corresponding to each value in 'data'.
        """

        if type(data) == float or type(data) == np.float64:
            return levels[np.argmin(np.abs(levels - data))]
        else:
            return levels[np.argmin( np.abs( np.repeat([levels],len(data),axis=0) - np.reshape(data,(-1,1)) ),axis=1 )]

    eye_dict = {}

    if isinstance(input, ndarray):
        if input.ndim == 2:
            if input.shape[0] != 2 and input.shape[1] == 2:
                input = input.T
            elif input.shape[0] == 2 and input.shape[1] != 2:
                pass
            else:
                raise ValueError("2D arrays must have a shape (2,N) or (N,2).")
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
        raise TypeError("The argument 'input' must be 'optical_signal', 'electrical_signal' or 'np.ndarray'.")

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

    eye_dict['execution_time'] = toc()
    return eye(eye_dict)


def SAMPLER(input: electrical_signal, _eye_: eye):
    """**Digital sampler**

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
        apodization: Union[Literal['uniform', 'rcos', 'gaussian', 'parabolic'], Callable] = 'uniform',
        F: float=0,
        print_params: bool=True,
        filtfilt: bool=True,
        retH: bool=False):
    r"""**Fiber Bragg Grating**.

    This function numerically calculates the reflectivity (transfer function :math:`H(f)` in reflection) of the grating by
    solving the coupled-wave equations using `Runge-Kutta` method with help of ``signal.integrate.solve_ivp()`` function. See Notes_ 
    section for more details.
    
    In order to design the grating, combination of the following parameters can be used:
    
    1. ``neff``, ``v``, ``fc``, (``dneff`` or ``vdneff``), (``N`` or ``kL`` or ``L``)
    2. ``neff``, ``v``, ``landaD``, (``dneff`` or ``vdneff``), (``N`` or ``kL`` or ``L``)
    3. ``neff``, ``v``, ``landaD``, ``kL``, (``N`` or ``L``)

    Bandwidth is governed essentially by three parameters:

    1. Bragg wavelength (:math:`\lambda_D`). Bandwidth is proportional to :math:`\lambda_D`.
    2. Product of visibility and effective index change (:math:`v\delta n_{eff}`). If :math:`v\delta n_{eff}` is small, the bandwidth is small.
    3. Length of the grating (:math:`L`). Bandwidth is inversely proportional to :math:`L`.

    On the other hand, chirp parameter :math:`F` can increase the bandwidth of the grating as well.

    Parameters
    ----------
    input : :obj:`optical_signal`
        The input optical signal.
    neff : :obj:`float`, optional
        Effective refractive index of core fiber. Default is 1.45.
    v : :obj:`float`, optional
        Visibility of the grating. Default is 1.
    landa_D : :obj:`float`, optional
        Bragg wavelength (resonance wavelength). Default is None.
    fc : :obj:`float`, optional
        Center frequency of the grating. Default is None. 
    kL : :obj:`float`, optional
        Product of the coupling coefficient and the length of the grating. Default is None.
    L : :obj:`float`, optional
        Length of the grating.  Default is None.
    N : :obj:`int`, optional
        Number of period along grating length. Default is None.
    dneff : :obj:`float`, optional
        Effective index change. Default is None.
    vdneff : :obj:`float`, optional 
        Effective index change multiplied by visibility (case of approximation σ->0). Default is None.
    apodization : :obj:`str` or :obj:`callable`
        Apodization function. Can be an string with the name of the apodization function or a custom function. Default is ``'uniform'``.
        
        The following apodization functions are available:

        * ``'uniform'``: Uniform apodization, ``f(z) = 1``.
        * ``'rcos'``: Raised cosine apodization, ``f(z) = 1/2*(1 + np.cos(pi*z))``.
        * ``'gaussian'``: Gaussian apodization, ``f(z) = np.exp(-4*np.log(2)*(3*z)**2)``.
        * ``'parabolic'``: Parabolic apodization, ``f(z) = 1 - (2*z)**2``.
        
        If a custom function is used, it must be a function of the form ``f(z)`` 
        where ``z`` is the position along the grating length normalized by ``L`` (i.e. ``z = z/L``),
        and the function must be defined in the range ``-0.5 <= z <= 0.5``.  

    F : :obj:`float`, optional
        Chirp parameter. Default is 0.
    filtfilt : :obj:`bool`, optional
        If True, group delay will be corrected in output signal. Default is True.
    retH : :obj:`bool`, optional
        If True, the function will return the reflectivity (H(w)) of the grating. Default is False.

    Returns
    -------
    output: :obj:`optical_signal`
        The reflected optical signal
    H: :obj:`np.ndarray`, optional
        Frequency response of grating fiber H(w), only returned if ``retH=True`` 

    Raises
    ------
    TypeError
        If ``input`` is not an :obj:`optical_signal`.
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
    .. _Notes:   

    Following coupled-wave theory, we assume a periodic, single-mode
    waveguide with an electromagnetic field which can be represented by
    two contradirectional coupled waves in the form [#first]_:

    .. math:: E(z) = A(z)e^{-j\beta_0 z} + B(z)e^{j\beta_0 z}

    where A and B are slowly varying amplitudes of mode traveling in :math:`+z` and math:`-z` directions, respectively
    These amplitudes are linked by the standard coupled-wave equations:

    .. math::
        R' &= j\hat{\sigma} R + j\kappa S \\
        S' &= -j\hat{\sigma} S - j\kappa R

    where :math:`R` and :math:`S` are :math:`R(z) = A(z)e^{j\delta z - \phi/2}` and :math:`S(z) = B(z)e^{-j\delta z + \phi/2}`. 
    In these equations :math:`\kappa` is the “AC” coupling coefficient and :math:`\hat{\sigma}` is a general “dc” self-coupling coefficient defined as
    
    .. math:: \hat{\sigma} = \delta + \sigma - \frac{1}{2}\phi'

    The detuning :math:`\delta`, which is independent of :math:`z` for all gratings, is defined to be

    .. math:: \delta = 2\pi n_{eff} \left( \frac{1}{\lambda} - \frac{1}{\lambda_{D}} \right)

    where :math:`\lambda_D = 2n_{eff}\Lambda` is the “design wavelength” for Bragg scattering by an infinitesimally weak grating :math:`(\delta n_{eff}\rightarrow 0)` with
    a period :math:`\Lambda`.

    For a single-mode Bragg reflection grating:

    .. math::
        \sigma &= \frac{2\pi}{\lambda}\delta n_{eff} \\
        \kappa &= \frac{v}{2}\sigma = \frac{\pi}{\lambda}v\delta n_{eff}
        
    If the grating is uniform along :math:`z`, then :math:`\delta n_{eff}` is a constant and :math:`\phi' = 0`, 
    and thus :math:`\kappa`, :math:`\sigma`, and :math:`\hat{\sigma}` constants.

    For apodized gratings, :math:`\delta n_{eff}` is a function of :math:`z`, and therefore :math:`\kappa`, :math:`\sigma`, and :math:`\hat{\sigma}` are also functions of :math:`z`.

    If phase chirp is present, :math:`\phi` and :math:`\phi'` are also a function of :math:`z`. This implementation considers only linear chirp, so:

    .. math:: \phi'(z) = 2Fz/L^2

    where :math:`F` is a dimensionless "chirp parameter", given by [#second]_:
    
    .. math:: F = \pi N \Delta \Lambda/\Lambda 

    or 

    .. math:: F = \pi N \Delta \lambda_D/\lambda_D = 2\pi n_{eff} \frac{\Delta \lambda_D}{\lambda_D^2}L

    where 
    
    .. math:: \Delta \lambda_D = \lambda_D(z=-L/2) - \lambda_D(z=L/2)

    ODE resolution is performed using `scipy.integrate.solve_ivp` function:

    - Dimensionless variables are used: :math:`z = z/L`, :math:`\delta = \delta L`, :math:`\kappa = \kappa L`, :math:`\sigma = \sigma L`, :math:`\phi' = \phi' L`.
    - The integration is performed from :math:`z = 1/2` to :math:`z = -1/2`.
    - The initial conditions are :math:`R(1/2) = 1` and :math:`S(1/2) = 0`.
    - The output is the relation :math:`\rho = S(-1/2)/R(-1/2)`. 

    References
    ----------
    .. [#] Turan Erdogan, "Fiber Grating Spectra," VOL. 15, NO. 8, AUGUST 1997. doi: https://doi.org/10.1109/50.618322
    .. [#] H. KOGELNIK, "Filter Response of Nonuniform Almost-Periodic Structures" Vol. 55, No. 1, January 1976. doi: https://doi.org/10.1002/j.1538-7305.1976.tb02062.x 
    
    Examples
    --------

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
        
        elif vdneff:
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
                L = kL / (pi*vdneff / landa_D)
            elif N:
                L = N * landa_D / (2*neff) 

        elif vdneff:
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
            
            vdneff = kL*landa_D / (pi*L)
            dneff = vdneff/v

        else: 
            raise ValueError("If `landa_D` is specified, `dneff`, 'vdneff' or `kL` must be specified.")
    
    else:
        raise ValueError("Either `fc` or `landa_D` must be specified.")

 
    λ_D = landa_D  # Bragg wavelength
    Λ = λ_D / (2*neff)  # period of the grating
    
    λc = (1+dneff/neff)*λ_D # center wavelength of the grating
    fc = c/λc # center frequency of the grating
    
    λ =  2*pi*c / (input.w(shift=True) + 2*pi*gv.f0) # wavelength vector, centered at global variable f0
    δλ = λ[1] - λ[0] # wavelength resolution

    N = int(L/Λ) # number of periods of the grating

    kL = pi/λ_D*vdneff*L

    δ = 2*pi*neff * (1/λ - 1/λ_D) * L
    s = 2*pi*dneff / λ * L   # self-coupling coefficient DC
    k = pi*vdneff / λ * L # self-coupling coefficient AC
    
    def ode_system(z, rho, δ, s, k, F=0, apo_func=None): # ODE function, normalized to L (z/L, δL, σL, kL)
        R = rho[:len(rho)//2]
        S = rho[len(rho)//2:]

        if apo_func:
            p = apo_func(z)
            s = s * p
            k = k * p 

        s_ = δ + s - F*z

        dRdz =  1j * (s_ * R + k * S)
        dSdz = -1j * (s_ * S + k * R)
        return [dRdz, dSdz]
        
    δ = δ[:, np.newaxis]
    s = s[:, np.newaxis]
    k = k[:, np.newaxis]

    # initial conditions
    S0 = np.zeros(input.len(), dtype=complex)
    R0 = np.ones(input.len(), dtype=complex)
    y0 = np.concatenate([R0, S0])

    if apodization == 'rcos':
        apo_func = lambda z: rcos(z, alpha=1, T=2)
    elif apodization == 'gaussian':
        apo_func = lambda z: np.exp(-4*np.log(2)*(3*z)**2)
    elif apodization == 'parabolic':
        apo_func = lambda z: 1 - (2*z)**2
    elif apodization == 'uniform':
        apo_func = None
    elif callable(apodization): # custom apodization function
        apo_func = apodization
    elif isinstance(apodization, str):
        warnings.warn("Apodization function not recognized. Using uniform apodization.")
        apo_func = None
    else:
        raise ValueError("Apodization must be a string or a function.")

    sol = solve_ivp(ode_system, 
                    t_span = [0.5, -0.5], 
                    y0 = y0, 
                    method='RK45', 
                    args=(δ, s, k, F, apo_func), 
                    vectorized=True,
    )
    
    y = sol.y[:, -1]
    R = y[:len(y)//2]
    S = y[len(y)//2:]

    H = S/R   

    y = np.abs(H) # reflectivity of the grating

    ic = np.argmin(np.abs(λ - c/fc))

    peaks,_ = sg.find_peaks(y)
    H_max = y[ic]

    
    if (y>0.5).all():
        warnings.warn("Bandwidth of the grating is too large for current sampling rate (`fs`). Consider increasing `fs`.")
        bandwith_str = f' - Δf = >{si(gv.fs,"Hz")} (Δλ = >{si(gv.fs*c/fc**2,"m")})'
    # elif (y<0.01).all():
    #     raise ValueError("Maximum reflectivity is less than 1%.") 
    elif len(peaks):
        r = sg.peak_widths(y, peaks)

        BW_λ = r[0].max()*δλ
        BW_f = fc**2*BW_λ/c

        bandwith_str = f' - Δf = {si(BW_f,"Hz")} (Δλ = {si(BW_λ,"m")})'
    else:
        warnings.warn("No peaks found in the reflectivity of the grating.")
        bandwith_str = f' - Δf = -- GHz (Δλ = -- nm)'

    D = dispersion(H, gv.fs, fc)[ic] # dispersion in ps/nm

    ## Print parameters of the grating
    if print_params:
        print('\n*** Fiber Bragg Grating Features ***')
        print(f' - Λ = {si(Λ,"m")}')
        print(f' - N = {N}')
        print(f' - L = {si(L,"m")}')
        print(f' - λc = {si(c/fc,"m",4)}')
        print(bandwith_str)
        print(f' - ρo = {y.max():.2f}')
        print(f' - loss = {-db(H_max**2):.1f} dB')
        print(f' - vδneff = {vdneff:.1e}')
        print(f' - kL = {kL:.1f}')
        print(f' - D(λc) = {D:.1f} ps/nm')
        if F:
            print(f' - F = {F:.1f}')
            print(f' - ΔΛ = {si(np.abs(Λ*F/(2*pi*N)),"m")}')
        print('************************************\n')


    if filtfilt: ## correct H(w)
        H = H * np.exp(-1j * input.w(shift=True) * tau_g(H, gv.fs)[ic]*1e-12) # corrected H(w)

    ## apply to input optical signal
    output = ifft(fft(input.signal)*ifftshift(H))

    output = optical_signal(output)
    output.execution_time = toc()

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



