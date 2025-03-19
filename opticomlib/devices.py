"""
.. rubric:: Devices
.. autosummary::

   PRBS
   DAC
   LASER
   PM
   MZM
   BPF
   EDFA
   DM
   FIBER
   LPF
   PD
   ADC
   GET_EYE
   SAMPLER
   FBG
"""

import numpy as np
import scipy.signal as sg
from typing import Literal, Union, Callable
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from scipy.constants import pi, k as kB, e, h, c
from numpy.fft import fft, ifft, fftshift, ifftshift
import sklearn.cluster as sk
from tqdm.auto import tqdm  # progress bar
import warnings
import matplotlib.pyplot as plt

from .typing import (
    electrical_signal,
    binary_sequence,
    optical_signal,
    gv,
    eye,
)

from .utils import (
    idb,
    db,
    idbm,
    dbm,
    tic,
    toc,
    rcos,
    si,
    tau_g,
    dispersion,
    shortest_int,
)

plt.rcParams["font.family"] = "serif"


def PRBS(
    order: Literal[7, 9, 11, 15, 20, 23, 31],
    len: int = None,
    seed: int = None,
    return_seed: bool = False,
):
    r"""**Pseudorandom binary sequence generator**

    Parameters
    ----------
    order : :obj:`int`, {7, 9, 11, 15, 20, 23, 31}
        degree of the generating pseudorandom polynomial
    len : :obj:`int`, optional
        lenght of output binary sequence
    seed : :obj:`int`, optional
        seed of the generator (initial state of the LFSR).
        It must be provided if you want to continue the sequence.
        Default is 2**order-1.
    return_seed : :obj:`bool`, optional
        If True, the last state of LFSR is returned. Default is False.

    Returns
    -------
    out : :obj:`binary_sequence`
        generated pseudorandom binary sequence

    Raises
    ------
    ValueError
        If ``order`` is not in [7, 9, 11, 15, 20, 23, 31].
    TypeError
        If ``len`` is not an integer.

    Warns
    -----
    UserWarning
        If the seed is 0 or a multiple of 2**order.

    Examples
    --------
    You can generate a PRBS sequence using the following code:

    >>> from opticomlib.devices import PRBS
    >>> PRBS(order=7, len=10)
    binary_sequence([1 0 0 0 0 0 0 1 0 0])
    >>> PRBS(order=31, len=20)
    binary_sequence([1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0])

    You can fix the LFSR iniitial state of generator by using the following code:

    >>> PRBS(order=7, len=10, seed=124)
    binary_sequence([0 0 0 0 0 1 0 0 0 0])


    Notes
    -----
    For more details, see [prbs]_.

    - :math:`2^7-1` bits. Polynomial :math:`= X^7 + X^6 + 1`
    - :math:`2^9-1` bits. Polynomial :math:`= X^9 + X^5 + 1`
    - :math:`2^{11}-1` bits. Polynomial :math:`= X^{11} + X^9 + 1`
    - :math:`2^{15}-1` bits. Polynomial :math:`= X^{15} + X^{14} + 1`
    - :math:`2^{20}-1` bits. Polynomial :math:`= X^{20} + X^3 + 1`
    - :math:`2^{23}-1` bits. Polynomial :math:`= X^{23} + X^{18} + 1`
    - :math:`2^{31}-1` bits. Polynomial :math:`= X^{31} + X^{28} + 1`

    References
    ----------
    .. [prbs] "Pseudorandom binary sequence" https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
    """
    tic()
    taps = {
        7: [7, 6],
        9: [9, 5],
        11: [11, 9],
        15: [15, 14],
        20: [20, 3],
        23: [23, 18],
        31: [31, 28],
    }
    seed = seed % (2**order) if seed is not None else (1 << order) - 1
    if seed == 0:
        seed = 1
        warnings.warn(
            "The seed can't be 0 or a multiple of 2**order. It has been changed to 1.",
            UserWarning,
        )

    if len is not None:
        if not isinstance(len, int):
            raise TypeError("The parameter `len` must be an integer.")
        elif len <= 0:
            raise ValueError(
                "The parameter `len` must be an integer greater than cero."
            )
    else:
        len = 2**order - 1

    if order not in taps.keys():
        raise ValueError(
            "The parameter `order` must be one of the following values (7, 9, 11, 15, 20, 23, 31)."
        )

    prbs = np.empty((len,), dtype=np.uint8)  # Preallocate memory for the PRBS
    lfsr = seed  # initial state of the LFSR
    tap1, tap2 = np.array(taps[order]) - 1

    index = 0
    while index < len:
        prbs[index] = lfsr & 1
        new = ((lfsr >> tap1) ^ (lfsr >> tap2)) & 1
        lfsr = ((lfsr << 1) | new) & (1 << order) - 1
        index += 1
        # if lfsr == seed:
        #     break

    output = binary_sequence(prbs)
    output.execution_time = toc()

    if not return_seed:
        return output
    return output, lfsr


def DAC(
    input: str | list | tuple | np.ndarray | binary_sequence,
    bias: float = 0.0,
    Vout: float = 1.0,
    pulse_shape: Literal["nrz", "rz", "rect", "gaussian"] = "nrz",
    BW: float = None,
    **kwargs,
):
    r"""**Digital-to-Analog Converter**

    Converts a binary sequence into an electrical signal, sampled at a frequency ``gv.fs``.

    Warning
    -------
    Parameter value ``pulse_shape='rect'`` is equivalent to ``pulse_shape='nrz'``. It is recommended to use ``pulse_shape='nrz'``, ``'rect'`` will be removed in futures versions.

    Parameters
    ----------
    input : :obj:`str`, :obj:`list`, :obj:`tuple`, :obj:`np.ndarray`, or :obj:`binary_sequence`
        Input binary sequence.
    bias : :obj:`float`
        DC bias of the output signal. Default: 0.0
    Vout : :obj:`float`
        Output signal amplitude. Should be in the range [-48, 48] Volts. Default: 1.0
    pulse_shape : :obj:`str`, {'nrz', 'rect', 'gaussian'}
        Pulse shape at the output. Default: 'nrz'
    BW : :obj:`float`
        Bandwidth of DAC. If ``None`` bandwidth is not limited. Default: None

    Other Parameters
    ----------------
    c : :obj:`float`
        Chirp of the Gaussian pulse. Only applicable if ``pulse_shape='gaussian'``. Default: 0.0
    m : :obj:`int`
        Order of the super-Gaussian pulse. Only applicable if ``pulse_shape='gaussian'``. Default: 1
    T : :obj:`int`
        Pulse width at half maximum in number of samples. Only applicable if ``pulse_shape='gaussian'``. Default: ``gv.sps``

    Returns
    -------
    :obj:`electrical_signal`
        The converted electrical signal.

    Raises
    ------
    ValueError
        If ``pulse_shape`` is not ``'rect'`` or ``'gaussian'``.
        If ``Vout`` is not between -48 and 48 Volts.
        If ``bias`` is not between -48 and 48 Volts.
        If ``T`` is <=0 or greater than 2 times the samples per bit.
        If ``m`` is not a positive integer.
    TypeError
        If ``Vout`` is not a scalar value.
        If ``bias`` is not a scalar value.
        If ``c`` is not a scalar value.
        If ``m`` is not an integer value.
        If ``T`` is not an integer value.

    Examples
    --------
    .. plot::
        :include-source:
        :alt: DAC example 1
        :align: center

        from opticomlib.devices import DAC
        from opticomlib import gv

        gv(sps=32) # set samples per bit

        DAC('0 0 1 0 0', Vout=5, pulse_shape='gaussian', m=2).plot('r', lw=3, grid=True).show()
    """
    tic()
    if not isinstance(input, binary_sequence):
        input = binary_sequence(input)

    sps = gv.sps

    if pulse_shape in ["rect", "nrz", "NRZ"]:
        x = np.kron(input.data, np.ones(sps))

    elif pulse_shape in ["rz", "RZ"]:
        rz_pulse = np.zeros(sps)
        rz_pulse[: sps // 2] = 1

        mask = np.tile(rz_pulse, input.len())

        x = np.kron(input.data, np.ones(sps)) * mask

    elif pulse_shape in ["gaussian", "GAUSSIAN"]:
        c = kwargs.get("c", 0.0)
        m = kwargs.get("m", 1)
        T = kwargs.get("T", sps)

        if not isinstance(c, (int, float)):
            raise TypeError("The parameter `c` must be a scalar value.")

        if not isinstance(m, int):
            raise TypeError("The parameter `m` must be an integer value.")
        else:
            if m <= 0:
                raise ValueError("The parameter `m` must be a positive integer value.")

        if not isinstance(T, int):
            raise TypeError("The parameter `T` must be an integer value.")
        else:
            if T > 2 * sps or T <= 0:
                raise ValueError("The parameter `T` must be in the range [0, 2*sps].")

        def p(t, T):
            return np.exp(-(1 + 1j * c) / 2 * (t / T) ** (2 * m))

        t = np.linspace(-4 * sps, 4 * sps, 8 * sps)  # time vector of the Gaussian pulse
        k = 2 * (2 * np.log(2)) ** (
            1 / (2 * m)
        )  # scaling factor between the width of a slot and the standard deviation of a Gaussian pulse
        pulse = p(t, T / k)  # gaussian pulse

        s = np.zeros(input.len() * sps)
        s[int(sps // 2) :: sps] = input.data
        s[int(sps // 2 - 1) :: sps] = input.data

        x = sg.fftconvolve(s, pulse, mode="same") / 2
    else:
        raise ValueError(
            'The parameter `pulse_shape` must be one of the following values ("rect", "gaussian")'
        )

    if Vout is not None:
        if not isinstance(Vout, (int, float)):
            raise TypeError("The parameter `Vout` must be a scalar value.")
        if np.abs(Vout) >= 48:
            raise ValueError(
                "The parameter `Vout` must be in the range [-48, 48] Volts."
            )
        x = x * Vout

    if bias is not None:
        if not isinstance(bias, (int, float)):
            raise TypeError("The parameter `bias` must be a scalar value.")
        if np.abs(bias) >= 48:
            raise ValueError(
                "The parameter `bias` must be in the range [-48, 48] Volts."
            )
        x = x + bias

    output = electrical_signal(x)

    if BW is not None:
        output = LPF(output, BW)

    output.execution_time = toc()
    return output


def LASER(t, p, lw=None, rin=None,  df=None):
    r"""
    **Continuous Wave Laser**

    Simple model of Laser with phase and RIN noises. Baseband equivalent (complex envelope).

    Parameters
    ----------
    t : :obj:`ndarray`
        Time vector.
    p : :obj:`float`
        Optical Power of laser, in dBm.
    lw : :obj:`float`
        LineWidth of laser, in Hz.
    rin : :obj:`float`
        Relative Intensity Noise power density, in dB/Hz.
    df : :obj:`float`
        Frequency offset of the laser, in Hz.

    Returns
    -------
    op_output : :obj:`optical_signal`
        Complex envolve of laser optical signal.

    Notes
    -----
    
    *Base-band equivalent (Complex Envelope)*

    Simulate a laser in band-pass is impossible in practice due to THz frequencies. So that base-band equivalent is used to simulate the complex envelop
    of optical signal:

    .. math:: E_{BB}(t) = \sqrt{P_0[1+\text{rin}(t)]} \cdot e^{j\phi_N(t)}e^{j\Delta\omega t}

    where :math:`P_0` is the optical power of laser, :math:`rin(t)` is the relative intensity noise, :math:`\phi_N(t)` is the phase noise, :math:`\Delta\omega` is the optical frequency offset respect of central frequency of simulation ``gv.f0``. It can be converted to a band-pass signal by making:

    .. math:: E(t) = \sqrt{2} \cdot Re\{E_{BB}(t)e^{j\omega_0 t}\}


    *Phase Noise as Wiener Random Process*
    
    In the active laser medium (atoms, molecules or carriers), photons can be emitted spontaneously when electrons decay from an excited to a fundamental state. These photons have random phases, which introduces phase noise in the emitted light.
    The phase noise is modeled as a Wiener process, where the phase at time :math:`t_{n+1}` is given by:

    .. math:: W(t_{n+1}) = W(t_n) + \Delta W
    
    where:

    1. :math:`\Delta W \sim \mathcal{N}(0,\sigma^2)`, is a gaussian increment of zero mean and variance :math:`\sigma^2`.
    2. :math:`\sigma^2` depend of Laser Linewidth :math:`\Delta \nu`:

    .. math:: \sigma^2 = 2\pi\Delta \nu \cdot \delta t
    
    where :math:`\delta t` is the sampling interval. Phase noise :math:`\phi_N(t)` is proportional to :math:`W(t)`.

    
    *Relative Intensity Noise (RIN)*

    The RIN is the relative fluctuations in laser intensity or optical power due to variations in the output number of photons. It is modeled as gaussian noise :math:`\mathcal{N}(0, \sigma_{RIN}^2)`, where:

    .. math:: \sigma_{RIN}^2 = 10^{\text{RIN}_\text{dB}/10} \cdot f_s 

    where :math:`f_s` is the sampling frequency and :math:`\text{RIN}_\text{dB}` is the RIN spectral density in dB/Hz.

    
    Examples
    --------
    For an ideal Laser with linewidth zero (without phase noise), we set parameter ``lw=None`` or just ignore it when call the laser function. Lets try a little offset frequency of 1 GHz as well.
    We can see tha optical signal is a continuous wave with 1000 mW of power (1 W), and the spectrum is a delta at frequency 1 GHz with a floor noise due to RIN, as expected. 
    
    .. code-block:: python
        :linenos:

        from opticomlib.devices import LASER, gv
        from opticomlib import gv, np, plt

        t = np.arange(0, 100e-9, gv.dt)

        P = 30       # 30 dBm (1 W)
        RIN = -140   # -140 dB/Hz Spectral density of RIN
        df = 1e9     # 1 GHz frequency offset   
        
        l = LASER(t, p=P, rin=RIN, df=df) 

        plt.subplot(211)
        l.plot('b',  style='light').grid()
        plt.title('Time Domain')
        plt.ylim(0, 2000)
        plt.xlim(0, 100)

        plt.subplot(212)
        l.psd('r', style='light').grid()
        plt.title('Frequency domain')
        plt.ylim(-50, 40)
        plt.tight_layout()
        plt.show()

    .. image:: _images/LASER_example1.svg
        :width: 100%
        :align: center

    In a more practical situation, active medium of laser have spontaneous emissions that cause phase noise and therefore an spread in bandwidth. 
    The follow example show this spread.  

    .. code-block:: python
        :linenos:

        from opticomlib.devices import LASER, gv
        from opticomlib import gv, np, plt

        t = np.arange(0, 100e-9, gv.dt)

        P = 30       # 30 dBm (1 W)
        LW = 10e6    # 10 MHz laser linewidth
        RIN = -140   # -140 dB/Hz Spectral density of RIN
        
        l = LASER(t, p=P, lw=LW, rin=RIN) 

        plt.subplot(211)
        l.plot('b',  style='light').grid()
        plt.title('Time Domain')
        plt.ylim(0, 2000)
        plt.xlim(0, 100)

        plt.subplot(212)
        l.psd('r', style='light').grid()
        plt.title('Frequency domain')
        plt.ylim(-50, 40)
        plt.tight_layout()
        plt.show()

    .. image:: _images/LASER_example2.svg
        :width: 100%
        :align: center
    """
    tic()
    op_output = np.ones_like(t) * np.sqrt( idbm(p) )

    if lw is not None: 
        # generate phase noise (random walk - wiener)
        phase_noise = np.cumsum( np.random.normal(0, np.sqrt(2*pi * lw * gv.dt), t.size) )

        # add the phase noise to the signal
        op_output = op_output * np.exp( 1j * phase_noise ) 

    if rin is not None:  
        # generate rin noise
        rin_noise = np.random.normal(0, np.sqrt( idb(rin) * gv.fs ) , t.size)
        
        if rin_noise.min() < -1:
            raise ValueError('Noise power is to high, try decrease RIN parameter.')
        
        # add rin noise to the signal
        op_output = op_output * np.sqrt(1 + rin_noise)
    
    if df is not None:
        if np.abs(df) > gv.fs/2:
            raise ValueError('The laser frequency is out of the Nyquist range. Try increase the sampling frequency.')

        op_output = op_output * np.exp(1j * 2*pi*df * t)

    op_output = optical_signal(op_output)
    op_output.execution_time = toc()
    return op_output


def PM(
    op_input: optical_signal,
    el_input: Union[float, np.ndarray, electrical_signal],
    Vpi: float = 5.0,
):
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
            raise ValueError(
                "The length of `el_input` must be equal to the length of `op_input`."
            )
    elif isinstance(el_input, np.ndarray):
        if len(el_input) != op_input.len():
            raise ValueError(
                "The length of `el_input` must be equal to the length of `op_input`."
            )
    else:
        raise TypeError("`el_input` must be of type (int or electrical_signal).")

    output = optical_signal(np.zeros_like(op_input.signal))

    output.signal = op_input.signal * np.exp(1j * el_input * pi / Vpi)

    if np.sum(op_input.noise):
        output.noise = op_input.noise * np.exp(1j * el_input * pi / Vpi)

    output.execution_time = toc()
    return output


def MZM(
    op_input: optical_signal,
    el_input: float | np.ndarray | electrical_signal,
    bias: float = 0.0,
    Vpi: float = 5.0,
    loss_dB: float = 0.0,
    ER_dB: float = 26.0,
    pol: Literal["x", "y"] = "x",
    BW: float = None,
):
    r"""
    **Mach-Zehnder modulator**

    Asymmetric coupler and opposite driving voltages model (:math:`u_1(t)=-u_2(t)=u(t)` Push-Pull configuration).
    The input and output are polarization maintained. Internally, the modulator can select the polarization
    to be modulated by setting the parameter ``pol`` to ``'x'`` or ``'y'``. If one of them is selected, the other is
    strongly attenuated (set to zeros).

    Parameters
    ----------
    op_input : :obj:`optical_signal`
        Optical signal to be modulated. This optical signal must contain only one polarization ``op_input.n_pol=1``. Otherwise
        it remove the second polarization.
    el_input : Number, :obj:`ndarray`, or :obj:`electrical_signal`
        Driver voltage, with zero bias.
    bias : :obj:`float`, optional
        Modulator bias voltage. Default is 0.0.
    Vpi : :obj:`float`, optional
        Voltage at which the device switches from on-state to off-state. Default is 5.0 V.
    loss_dB : :obj:`float`, optional
        Propagation or insertion losses in the modulator, value in dB. Default is 0.0 dB.
    ER_dB: :obj:`float`, optional
        Extinction ratio of the modulator, in dB. Default is 26 dB.
    pol : :obj:`str`, {'x', 'y'} optional
        Polarization of the modulator. Default is ``'x'``.
    BW : :obj:`float`, optional
        Modulator bandwidth in Hz. If not provided, the bandwidth is not limited.

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
        \vec{E}_{out} = \vec{E}_{in} \cdot \sqrt{l} \cdot \left[ \cos\left(\frac{\pi}{2V_{\pi}}(u(t)+V_{bias})\right) + j \frac{\eta}{2} \sin\left(\frac{\pi}{2V_{\pi}}(u(t)+V_{bias})\right) \right]

    where :math:`\eta = 2\times 10^{-ER_{dB}/10}` and :math:`l = 10^{-loss_{dB}/10}`.

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
        tx_seq = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0], bool)

        V = DAC(~tx_seq, Vout=Vpi, pulse_shape='rect') - Vpi/2

        input = optical_signal( np.ones_like(V.signal)*idbm(10)**0.5 )
        input.noise = np.random.normal(0, 0.01, input.len())
        t = input.t()*1e9

        mod_sig = MZM(input, el_input=V, bias=Vpi/2, Vpi=Vpi, loss_dB=2, ER_dB=40, BW=40e9)

        fig, axs = plt.subplots(3,1, sharex=True, tight_layout=True)


        # Plot input and output power
        axs[0].plot(t, dbm(input.abs()**2), 'r-', label='input', lw=3)
        axs[0].plot(t, dbm(mod_sig.abs()**2), 'C1-', label='output', lw=3)
        axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
        axs[0].set_ylabel('Potencia [dBm]')
        for i in t[::gv.sps]:
            axs[0].axvline(i, color='k', linestyle='--', alpha=0.5)

        # # Plot fase
        phi_in = input.phase()
        phi_out = mod_sig.phase()

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

    .. image:: _images/MZM_example.svg
        :width: 100%
        :align: center
    """

    tic()
    if not isinstance(op_input, optical_signal):
        raise TypeError("`op_input` must be of type (optical_signal).")

    if not isinstance(el_input, electrical_signal):
        el_input = electrical_signal(el_input)

    if op_input.len() != el_input.len() and el_input.len() != 1:
        raise ValueError(
            "Length of `op_input` and `el_input` must be equal or `el_input` must be an scalar value. Current lengths are {} and {}.".format(
                op_input.len(), el_input.len()
            )
        )

    if pol not in ["x", "y"]:
        raise ValueError(
            "The parameter `pol` must be one of the following values ('x', 'y')."
        )

    loss = idb(-loss_dB)  # Propagation losses
    eta = 2 * idb(-ER_dB) ** 0.5  # arms desbalance factor

    output = op_input[:]

    g_t = pi / 2 / Vpi * (el_input.signal + bias)
    h_t = loss**0.5 * (np.cos(g_t) + 1j * eta / 2 * np.sin(g_t))

    output.signal = output.signal * h_t
    if output.noise is not None:
        output.noise = output.noise * h_t

    if pol == "x" and output.n_pol == 2:
        output.signal[1] = 0
        if output.noise is not None:
            output.noise[1] = 0
    elif pol == "y" and output.n_pol == 2:
        output.signal[0] = 0
        if output.noise is not None:
            output.noise[0] = 0

    if BW is not None:
        output = BPF(
            output, BW
        )  # Filter the modulated optical signal and add the execution time of the filter

    output.execution_time = toc()
    return output


def BPF(input: optical_signal, BW: float, n: int = 4):
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

    sos_band = sg.bessel(
        N=n, Wn=BW / 2, btype="low", fs=gv.fs, output="sos", norm="mag"
    )

    output = input[:]  # copy the input signal

    output.signal = sg.sosfiltfilt(sos_band, input.signal, axis=-1)

    if output.noise is not None:
        output.noise = sg.sosfiltfilt(sos_band, input.noise, axis=-1)

    output.execution_time = toc()
    return output


def EDFA(input: optical_signal, G: float, NF: float, BW: float=None):
    r"""
    **Erbium Doped Fiber**

    Amplifies the optical signal at the input, adding amplified spontaneous emission (ASE) noise in two polarizations at the output.
    Simplest model (no saturation output power).

    Parameters
    ----------
    input : optical_signal
        The optical signal to be amplified.
    G : float
        The gain of the amplifier, in dB.
    NF : float
        The noise figure of the amplifier, in dB.
    BW : float, optional
        The bandwidth of the amplifier, in Hz. If ``None`` bandwidth will be ``gv.fs``.

    Returns
    -------
    optical_signal
        The amplified optical signal.

    Raises
    ------
    TypeError
        If ``input`` is not an optical_signal.

    Notes
    -----
    ASE noise power must be theoretically:

    .. math:: 
        P_\text{ase} = \text{NF} h f_0 (G-1) BW

    where :math:`h` is the Planck constant and :math:`f_0` is the central frequency of communication 
    (by default ``gv.f0`` is taken, if you wish change this value, you can change ``wavelength`` 
    parameter in ``gv()``). Noise is generated for two polarizations xy as a complex signal, with a
    distribution :math:`\mathcal{N}(0, P_\text{ase}/4)` for real and imaginary parts.  

    Examples
    --------
    Following picture show the input-output of EDFA from a sinusoidal signal. For values of example, the output noise power 
    must be :math:`P_{ase} \approx -27` dBm.

    .. plot::
        :include-source:
        :alt: DM example 1
        :align: center

        from opticomlib.devices import EDFA
        from opticomlib import optical_signal, gv, np, plt

        gv(sps=256, R=1e9, N=5, G=20, NF=5, BW=50e9)

        x = optical_signal(
            signal=[
                (1e-3)*np.sin(2*np.pi*gv.R*gv.t), 
                np.zeros_like(gv.t)
            ], 
            n_pol=2
        )

        y = EDFA(x, G=gv.G, NF=gv.NF, BW=gv.BW)

        fig, axs = plt.subplots(2,1, sharex=True, figsize=(8,6))
        plt.suptitle(f"EDFA input-output (G={gv.G} dB, NF={gv.NF} dB, BW={gv.BW*1e-9} GHz)")

        axs[0].set_title('Input')
        axs[0].plot(gv.t*1e9, x.signal.T)
        axs[0].set_ylim(-0.015, 0.015)

        axs[1].set_title('Output')
        axs[1].plot(gv.t*1e9, y.signal.T + y.noise.T.real)
        axs[1].set_ylim(-0.015, 0.015)

        plt.legend(['x-pol', 'y-pol'])
        plt.xlabel('t [ns]')
        plt.show()

    >>> from opticomlib import dbm
    >>> print(dbm(y.power('noise').sum())) # print sum of power of two polarizations
    -28.068263005828555

    We can see that noise power is a little less than theoretical prediction, this is because 
    the filter used in the EDFA is not a rectangular response filter (it's a 4th order Bessel filter).
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")

    output = optical_signal(signal=input.signal, noise=input.noise, n_pol=2) * np.sqrt( idb(G) )
    
    if input.n_pol == 1:
        output.signal[1] = np.zeros_like(output.signal[0])  # y-polarization of signal is set to zeros.

    # generate ASE noise (2-polarizations with real and imaginary parts)
    # gv.fs is taken as initial bandwidth of noise 
    P_ase = idb(NF) * h * gv.f0 * (idb(G) - 1) * gv.fs

    # generate 4 vectors for, x-polarization real and imaginary parts and y-polarization real and imaginary parts
    ase = np.sqrt(P_ase/4) * np.random.randn(4, input.len())
    ase = ase[:2] + 1j*ase[2:]

    if output.noise is not None:
        output.noise += ase
    else:
        output.noise = ase

    if BW is not None:
        output = BPF(output, BW)

    output.execution_time = toc()
    return output


def DM(input: optical_signal, D: float, retH: bool = False):
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
        input = optical_signal( signal.signal/signal.power()**0.5*idbm(20)**0.5, n_pol=2 )

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

    H = np.exp(-1j * input.w() ** 2 * D / 2)

    output = (input("w") * H)("t")

    if retH:
        H = np.exp(-1j * input.w() ** 2 * D / 2)
        return output, fftshift(H)
    
    output.execution_time = toc()
    return output


def FIBER(
    input: optical_signal,
    length: float,
    alpha: float = 0.0,
    beta_2: float = 0.0,
    beta_3: float = 0.0,
    gamma: float = 0.0,
    phi_max: float = 0.05,
    show_progress=False,
):
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
        input = optical_signal( signal.signal/signal.power()**0.5*idbm(20)**0.5, n_pol=2)

        output = FIBER(input, length=50, alpha=0.01, beta_2=-20, gamma=0.1, show_progress=True)

        input.plot('r-', label='input', lw=3)
        output.plot('b-', label='output', lw=3).show()
    """

    tic()
    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")

    alpha = alpha / 4.343  # [1/km]

    w = input.w() * 1e-12  # [rad/ps]
    D_op = -alpha / 2 - 1j / 2 * beta_2 * w**2 - 1j / 6 * beta_3 * w**3

    A = input.signal

    h = (
        length
        if (beta_2 == 0 and beta_3 == 0) or gamma == 0
        else phi_max / (gamma * (np.abs(A[0]) ** 2 + np.abs(A[1]) ** 2)).max()
    )

    x_length = h

    if show_progress:
        barra_progreso = tqdm(total=100)

    while True:
        exp_NL = np.exp(1j * gamma * (h / 2) * np.abs(A) ** 2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft(
            exp_L * fft(exp_NL * A)
        )  # Symmetric Split-Step Fourier Method

        if show_progress:
            barra_progreso.update(100 * h / length)

        h = (
            phi_max / (gamma * (np.abs(A[0]) ** 2 + np.abs(A[1]) ** 2)).max()
            if gamma != 0
            else length
        )

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length

    if h != 0:
        exp_NL = np.exp(1j * gamma * (h / 2) * np.abs(A) ** 2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft(exp_L * fft(exp_NL * A))

        if show_progress:
            barra_progreso.update(100 * h / length)

    output = optical_signal(A, input.noise)
    output.execution_time = toc()
    return output


def LPF(
    input: Union[np.ndarray, electrical_signal],
    BW: float,
    n: int = 4,
    fs: float = None,
    retH: bool = False,
):
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

    if not isinstance(input, (np.ndarray, electrical_signal)):
        raise TypeError("`input` must be of type (ndarray or electrical_signal).")

    elif isinstance(input, electrical_signal):
        signal = input.signal
        noise = input.noise
    else:
        input = electrical_signal(input)
        signal = input.signal
        noise = None

    if not fs:
        fs = gv.fs

    sos_band = sg.bessel(N=n, Wn=BW, btype="low", fs=fs, output="sos", norm="mag")

    output = input[:]

    output.signal = sg.sosfiltfilt(sos_band, signal).real

    if noise is not None:
        output.noise = sg.sosfiltfilt(sos_band, noise).real

    if retH:
        _, H = sg.sosfreqz(sos_band, worN=signal.size, fs=fs, whole=True)
        return output, fftshift(H)
    
    output.execution_time = toc()
    return output


def PD(
    input: optical_signal,
    BW: float,
    r: float = 1.0,
    T: float = 300.0,
    R_load: float = 50.0,
    include_noise: Literal[
        "ase-only",
        "thermal-only",
        "shot-only",
        "ase-thermal",
        "ase-shot",
        "thermal-shot",
        "all",
    ] = "all",
    i_dark: float = 10e-9,
    Fn=0,
):
    r"""
    **P-I-N Photodetector**

    Simulates the detection of an optical signal by a P-I-N photodetector.

    Parameters
    ----------
    input : :obj:`optical_signal`
        Optical signal to be photodetected.
    BW : :obj:`float`
        Detector bandwidth in [Hz].
    r : :obj:`float`, optional
        Detector responsivity in [A/W]. Default: 1.0.
    T : :obj:`float`, optional
        Detector temperature in [K]. Default: 300.0.
    R_load : :obj:`float`, optional
        Detector load resistance in [:math:`\Omega`]. Default: 50.0.
    include_noise : :obj:`str`, optional
        Type of noise to include in the simulation. Default: 'all'.
        Options include:

        - ``'ase-only'``: only include ASE noise
        - ``'thermal-only'``: only include thermal noise
        - ``'shot-only'``: only include shot noise
        - ``'ase-thermal'``: include ASE and thermal noise
        - ``'ase-shot'``: include ASE and shot noise
        - ``'thermal-shot'``: include thermal and shot noise
        - ``'all'``: include all types of noise

    i_dark : :obj:`float`, optional
        Dark current of the photodetector in [A]. Default: 10e-9 [10 nA].
    Fn : :obj:`float`, optional
        Noise figure of Photodetector amplifiers states, in [dB]. Default: 0 dB

    Returns
    -------
    electrical_signal
        The detected electrical signal, in [v].

    Raises
    ------
    TypeError
        If ``input`` is not of type optical_signal.
        If ``r``, ``T``, or ``R_load`` are not scalar values.
        If ``include_noise`` is not a string.
    ValueError
        If ``r`` is not between (0, 1].
        If ``T`` or ``R_load`` are negative values.
        If ``include_noise`` argument is not one of the valid options.

    Notes
    -----
    The total photodetected current is given by:

    .. math::
        i_{ph} = \mathcal{R}P_{in} + i_{th} + i_{sh} + i_{dark}

    where :math:`\mathcal{R}` is the responsivity of the photodetector, :math:`P_{in}` is the input power, :math:`i_{th}` and :math:`i_{sh}` are thermal and shot noise
    respectively and :math:`i_{dark}` is the dark current of photodetector.

    The input power :math:`P_{in}` is determinated as:

    .. math::
        P_{in} &= |E_x + n_x|^2 + |E_y + n_y|^2 \\
        P_{in} &= |E_x|^2 + |E_y|^2 + E_x n_x^* + E_x^* n_x + E_y n_y^*+E_y^* n_y + |n_x|^2 + |n_y|^2 \\
        P_{in} &= P_\text{sig} + P_\text{sig-ase} + P_\text{ase-ase} \\

    where :math:`E_x` and :math:`E_y` are the amplitudes of x-polarization and y-polarization modes respectively
    and :math:`n_x` and :math:`n_y` are the noise of x-polarization and y-polarization modes respectively.

    The thermal and shot noises are random variables with normal distribution and variance given by [Agrawal]_:

    .. math::
        \sigma_{th}^2 &= \frac{4k_B T}{R_L}F_n \Delta f \\
        \sigma_{sh}^2 &= 2e\left[ r(P_\text{sig} + P_\text{ase-ase}) + i_{dark} \right]\Delta f

    where :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature of the photodetector, :math:`R_L` is the load resistance,
    :math:`F_n` is the noise figure of the photodetector, :math:`\Delta f` is the bandwidth of the photodetector, :math:`e` is the electron charge.

    .. math::
        i_{ph} &= \mathcal{R}P_{sig} + \mathcal{R}P_{sig-ase} + \mathcal{R}P_{ase-ase} + i_{th} + i_{sh} + i_{dark} \\
        i_{ph} &= i_\text{sig} + i_\text{sig-ase} + i_\text{ase-ase} + i_{th} + i_{sh} + i_{dark}
    
    Finally, the output voltage is given by:

    .. math::
        v_{ph} = i_{ph}R_L

    References
    ----------
    .. [Agrawal] Agrawal, G.P., "Fiber-Optic Communication Systems". Chapter 4.4 (1997).

    """
    tic()
    # check inputs
    if not isinstance(input, optical_signal):
        raise TypeError("`input` must be of type (optical_signal).")

    if not isinstance(r, (int, float)):
        raise TypeError("`r` must be a scalar value.")
    elif r <= 0 or r > 1:
        raise ValueError("`r` must be in the range (0,1]")

    if not isinstance(T, (int, float)):
        raise TypeError("`T` must be a scalar value.")
    elif T < 0:
        raise ValueError("`T` must be a positive value.")

    if not isinstance(R_load, (int, float)):
        raise TypeError("`R_load` must be a scalar value.")
    elif R_load < 0:
        raise ValueError("`R_load` must be a positive value.")

    if not isinstance(include_noise, str):
        raise TypeError("`include_noise` must be a string.")

    # function body
    i_sig = r * input.abs("signal") ** 2

    if input.n_pol == 2:
        i_sig = i_sig.sum(axis=0)

    include_noise = include_noise.lower()  # This allow write in upper or lower case

    if "thermal" in include_noise or "all" in include_noise:
        S_T = 4 * kB * T * gv.fs/2 * idb(Fn) / R_load  # thermal noise variance, in [A^2]
        i_T = np.random.normal(0, S_T**0.5, input.len())  # thermal noise current, in [A]

    if "shot" in include_noise or "all" in include_noise:
        if input.noise is not None:
            i_ase = r * input.power("noise").sum()
        else:
            i_ase = 0

        S_N = 2 * e * (i_sig.mean() + i_ase + i_dark) * gv.fs/2  # shot noise variance, in [A^2]
        i_N = np.random.normal(0, S_N**0.5, input.len())  # shot noise current, in [A]

    if "ase" in include_noise or "all" in include_noise:
        if input.noise is not None:
            i_s_n = r * (input.signal * input.noise.conj() + input.noise * input.signal.conj()).real  # SIG-Noise term
            i_n_n = r * input.abs("noise") ** 2  # Noise-Noise term

            if input.n_pol == 2:
                i_s_n = i_s_n.sum(axis=0)
                i_n_n = i_n_n.sum(axis=0)
        else:
            i_s_n = np.zeros(input.len())
            i_n_n = np.zeros(input.len())

    if include_noise == "ase-only":
        i_noise = i_s_n + i_n_n + i_dark
    elif include_noise == "thermal-only":
        i_noise = i_T + i_dark
    elif include_noise == "shot-only":
        i_noise = i_N + i_dark
    elif include_noise == "ase-shot":
        i_noise = i_s_n + i_n_n + i_N + i_dark
    elif include_noise == "ase-thermal":
        i_noise = i_s_n + i_n_n + i_T + i_dark
    elif include_noise == "thermal-shot":
        i_noise = i_T + i_N + i_dark
    elif include_noise == "all":
        i_noise = i_s_n + i_n_n + i_N + i_T + i_dark
    else:
        raise ValueError(
            "The argument `include_noise` must be one of the following: 'ase-only','thermal-only','shot-only','ase-thermal','ase-shot','thermal-shot','all'."
        )

    output = electrical_signal(signal=i_sig*R_load, noise=i_noise*R_load)
    
    output = LPF(output, BW)

    output.execution_time = toc()
    return output


def ADC(
    input: electrical_signal | np.ndarray, 
    fs: float = None,
    n: int = 8,
    otype: Literal['v', 'n'] = 'v'
) -> binary_sequence:
    r"""
    **Analog-to-Digital Converter**

    Converts an analog electrical signal into a quantized :math:`2^n` bits digital signal, sampled at a frequency `fs`.

    Parameters
    ----------
    input : electrical_signal | np.array
        Electrical signal to be quantized.
    fs : float, default: None
        Sampling frequency of the output signal. If None, signal is not sampled.
    n : int, default: 8
        Bits of quantization. Default is 8 bits.
    otype : str, default: 'v'
        Signal output type. If 'v' discrete amplitudes are return, if 'n' integer amplitudes between 0 and 2**n-1 are return. 

    Returns
    -------
    electrical_signal
        Quantized digital signal.

    Example
    -------
    .. plot::
        :include-source:
        :alt: ADC
        :align: center

        from opticomlib.devices import ADC
        from opticomlib import gv, electrical_signal
        import numpy as np

        gv(sps=64, R=1e9, N=2)

        y = electrical_signal( np.sin(2*np.pi*gv.R*gv.t) )

        yn = ADC(y, n=2)

        y.plot(
            style='light', 
            grid=True, 
            lw=5,
            label = 'analog signal'
        )
        yn.plot('.-', style='light', lw=2, label=' 2 bits quantized signal').show()
    """
    tic()

    if isinstance(input, electrical_signal):
        if input.noise is not None:
            signal = input.signal + input.noise
        else:
            signal = input.signal
    else:
        signal = input

    if fs is not None:
        signal = sg.resample(signal, int(input.len() * fs / input.fs()))

    V_min, V_max = shortest_int(signal, 99.99)
    
    dig_signal = np.round(
        (signal - V_min) / (V_max - V_min) * (2**n - 1)
    ).astype(int)  # quantize signal between 0 and 2**n-1
    
    if otype == 'v':
        dig_signal = (
            dig_signal / (2**n - 1) * (V_max - V_min) + V_min
        )  # back to discrete amplitude 
    elif otype != 'n':
        raise ValueError("`otype` must be 'v' or 'n'.")

    output = electrical_signal(dig_signal)

    output.execution_time = toc()
    return output


def GET_EYE(
    input: electrical_signal | np.ndarray,
    nslots: int = 4096,
    sps_resamp: int = None,
):
    r"""
    **Get Eye Parameters Estimator**

    Estimates all fundamental parameters and metrics of the eye diagram 
    of the input electrical signal.

    Parameters
    ----------
    input : :obj:`electrical_signal` | :obj:`np.ndarray`
        Electrical or optical signal from which the eye diagram will be estimated.
    nslots : :obj:`int`, default: 4096
        Number of slots to consider for eye reconstruction.
    sps_resamp : :obj:`int`, default: None
        Number of samples per slot to interpolate the original signal. If None the signal is not interpolated.

    Returns
    -------
    :obj:`eye`
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

        y = DAC( PRBS(order=7), pulse_shape='gaussian')
        y.noise = np.random.normal(0, 0.05, y.len())

        GET_EYE(y, sps_resamp=512).plot().show() # with interpolation

    .. image:: /_images/GET_EYE_example1.png
    """
    tic()

    def find_nearest(
        levels: np.ndarray, data: np.ndarray | float
    ) -> np.ndarray | float:
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

        if isinstance(data, (float, np.float64)):
            return levels[np.argmin(np.abs(levels - data))]
        else:
            return levels[
                np.argmin(
                    np.abs(
                        np.repeat([levels], len(data), axis=0) - np.reshape(data, (-1, 1))
                    ),
                    axis=1,
                )
            ]

    #########################
    ## Preprocessing input ##
    #########################

    eye_dict = {}

    if not isinstance(input, electrical_signal):
        input = electrical_signal(input)

    eye_dict["sps"] = sps = input.sps()
    eye_dict["dt"] = dt = input.dt()

    # truncate
    n = input.len() % (2 * sps)  # we obtain the rest %(2*sps)
    if n: # if rest is not zero
        input = input[:-n] # ignore last 'n' samples
                              
    nslots = min( int(input.len() // sps), nslots) # determine the minimum between slots of signal and 'nslots' parameter
    input = input[: nslots * sps] # truncate signal

    input = (
        (input.signal + input.noise).real
        if input.noise is not None
        else input.signal.real
    ) # add noise to signal, if there is noise

    input = np.roll(input, -sps // 2 + 1)  # roll (-sps/2) to focus the eye in center of figure
    y_set = np.unique(input) # take a set of signal values

    # resampled the signal to obtain a higher resolution in both axes
    if sps_resamp:
        input = sg.resample(input, nslots * sps_resamp)
        eye_dict["y"] = input
        eye_dict["sps_resamp"] = sps_resamp
        eye_dict["t"] = t = np.kron(np.ones(nslots // 2), np.linspace(-1, 1 - 1/sps_resamp, 2 * sps_resamp))
    else:
        eye_dict["y"] = input
        eye_dict["t"] = t = np.kron(np.ones(nslots // 2), np.linspace(-1, 1 - 1/sps, 2 * sps))

    ###############
    ## Algorithm ##
    ###############

    kmeans = sk.KMeans(n_clusters=2, n_init=10) # A model of sklearn to separete clusters

    # Obtain centroide of data (y)
    vm = np.mean(kmeans.fit(input.reshape(-1,1)).cluster_centers_)

    # we obtain the shortest interval of the upper half that contains 50% of the samples
    eye_dict["top_int"] = top_int = shortest_int(input[input > vm], percent=50)
    # We obtain the LMS of level 1
    state_1 = np.mean(top_int)
    # we obtain the shortest interval of the lower half that contains 50% of the samples
    eye_dict["bot_int"] = bot_int = shortest_int(input[input < vm], percent=50)
    # We obtain the LMS of level 0
    state_0 = np.mean(bot_int)

    # We obtain the amplitude between the two levels 0 and 1
    d01 = state_1 - state_0

    # We take 75% threshold level
    v75 = state_1 - 0.25 * d01

    # We take 25% threshold level
    v25 = state_0 + 0.25 * d01

    t_set = np.unique(t)

    try:
        # The following vectors will be used only to determine the crossing times
        # and crossing amplitude
        cond = (input > v25) & (input < v75)

        ty = np.vstack([t[cond], input[cond]]).T

        # We get centroids of 2 clusters for t,y
        kmeans.fit(ty)
        ty_c = kmeans.cluster_centers_

        left = np.argmin(ty_c[:,0])
        right = np.argmax(ty_c[:,0])

        eye_dict["t_left"] = t_left = find_nearest(t_set, ty_c[left,0])
        eye_dict["t_right"] = t_right = find_nearest(t_set, ty_c[right,0])
        eye_dict["t_opt"] = t_center = find_nearest(t_set, ty_c[:,0].mean())
        
        eye_dict["y_left"] = find_nearest(y_set, ty_c[left,1])
        eye_dict["y_right"] = find_nearest(y_set, ty_c[right,1])

        eye_dict["y_25_75"] = y_25_75 = input.copy()
        y_25_75[~cond] = np.nan

    except ValueError:
        eye_dict["t_left"] = t_left = -0.5
        eye_dict["t_right"] = t_right = 0.5
        eye_dict["t_opt"] = t_center = 0.0

        eye_dict["y_left"] = None
        eye_dict["y_right"] = None

    except Exception as e:
        raise e

    # For 10% of the center of the eye diagram
    eye_dict["t_dist"] = t_dist = t_right - t_left
    eye_dict["t_span0"] = t_span0 = t_center - 0.05 * t_dist
    eye_dict["t_span1"] = t_span1 = t_center + 0.05 * t_dist

    # Within the 10% of the data in the center of the eye diagram, we separate into two clusters top and bottom
    y_center = find_nearest(y_set, (state_0 + state_1) / 2)

    # We obtain the optimum time for down sampling
    if sps_resamp:
        instant = np.abs(t - t_center).argmin() - sps_resamp // 2 + 1
        instant = int(instant / sps_resamp * sps)
    else:
        instant = np.abs(t - t_center).argmin() - sps // 2 + 1
    eye_dict["i"] = instant

    # We obtain the upper cluster
    cond = (input > y_center) & ((t_span0 < t) & (t < t_span1))
    y_top = input.copy()
    y_top[~cond]=np.nan
    eye_dict["y_top"] = y_top

    # We obtain the lower cluster
    cond = (input < y_center) & ((t_span0 < t) & (t < t_span1))
    y_bot = input.copy()
    y_bot[~cond]=np.nan
    eye_dict["y_bot"] = y_bot

    # For each cluster we calculated the means and standard deviations
    eye_dict["mu1"] = mu1 = np.mean(y_top, where=~np.isnan(y_top))
    eye_dict["s1"] = s1 = np.std(y_top, where=~np.isnan(y_top))
    eye_dict["mu0"] = mu0 = np.mean(y_bot, where=~np.isnan(y_bot))
    eye_dict["s0"] = s0 = np.std(y_bot, where=~np.isnan(y_bot))

    # compute umbral
    x = np.linspace(mu0, mu1, 500)
    y = input[ ((t_span0 < t) & (t < t_span1)) ]
    
    try:
        pdf = gaussian_kde(y).evaluate(x)
        eye_dict["threshold"] = x[np.argmin(pdf)]
    except:
        eye_dict["threshold"] = None

    # We obtain the extinction ratio
    eye_dict["er"] = 10 * np.log10(mu1 / mu0) if mu0 > 0 else np.inf if mu0 == 0 else np.nan

    # We obtain the eye opening
    eye_dict["eye_h"] = mu1 - 3 * s1 - mu0 - 3 * s0

    eye_dict["execution_time"] = toc()
    return eye(**eye_dict)


def SAMPLER(input: electrical_signal, instant: int):
    """**Digital sampler**

    Receives an electrical signal and an eye object and performs the sampling of the signal
    at the optimal instant determined by the eye object.

    Args:
        input: The electrical signal to be sampled.
        instant: slot instant to take the sample [0, gv.sps].

    Returns:
        electrical_signal: The sampled electrical signal at one sample per slot.
    """
    tic()
    output = input[instant :: gv.sps]

    output.execution_time = toc()
    return output


def FBG(
    input: optical_signal,
    neff: float = 1.45,
    v: float = 1.0,
    landa_D: float = None,
    fc: float = None,
    kL: float = None,
    L: float = None,
    N: int = None,
    dneff: float = None,
    vdneff: float = None,
    apodization: Union[
        Literal["uniform", "rcos", "gaussian", "parabolic"], Callable
    ] = "uniform",
    F: float = 0,
    print_params: bool = True,
    filtfilt: bool = True,
    retH: bool = False,
):
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
                raise ValueError(
                    "If `fc` and `dneff` are specified, `L`, `kL` or `N` must be specified."
                )

            landa_D = 1 / (1 + dneff / neff) * c / fc
            vdneff = dneff * v

            if kL:
                L = kL / (pi * dneff * v / landa_D)
            elif N:
                L = N * landa_D / (2 * neff)

        elif vdneff:
            if not (L or kL or N):
                raise ValueError(
                    "If `fc` and `vdneff` are specified, `L`, `kL` or `N` must be specified."
                )

            landa_D = c / fc
            dneff = 0

            if kL:
                L = kL / (pi * vdneff / landa_D)
            elif N:
                L = N * landa_D / (2 * neff)
        else:
            raise ValueError(
                "If `fc` is specified, `dneff` or `vdneff` must be specified."
            )

    elif landa_D:
        if dneff:
            if not (L or kL or N):
                raise ValueError(
                    "If `landa_D` and `dneff` are specified, `L`, `kL` or `N` must be specified."
                )

            vdneff = dneff * v

            if kL:
                L = kL / (pi * vdneff / landa_D)
            elif N:
                L = N * landa_D / (2 * neff)

        elif vdneff:
            if not (L or kL or N):
                raise ValueError(
                    "If `landa_D` and `vdneff` are specified, `L`, `kL` or `N` must be specified."
                )

            dneff = 0

            if kL:
                L = kL / (pi * vdneff / landa_D)
            elif N:
                L = N * landa_D / (2 * neff)

        elif kL:
            if not (L or N):
                raise ValueError(
                    "If `landa_D` and `kL` are specified, `L` or `N` must be specified."
                )
            if N:
                L = N * landa_D / (2 * neff)

            vdneff = kL * landa_D / (pi * L)
            dneff = vdneff / v

        else:
            raise ValueError(
                "If `landa_D` is specified, `dneff`, 'vdneff' or `kL` must be specified."
            )

    else:
        raise ValueError("Either `fc` or `landa_D` must be specified.")

    λ_D = landa_D  # Bragg wavelength
    Λ = λ_D / (2 * neff)  # period of the grating

    λc = (1 + dneff / neff) * λ_D  # center wavelength of the grating
    fc = c / λc  # center frequency of the grating

    λ = (
        2 * pi * c / (input.w(shift=True) + 2 * pi * gv.f0)
    )  # wavelength vector, centered at global variable f0
    δλ = λ[1] - λ[0]  # wavelength resolution

    N = int(L / Λ)  # number of periods of the grating

    kL = pi / λ_D * vdneff * L

    δ = 2 * pi * neff * (1 / λ - 1 / λ_D) * L
    s = 2 * pi * dneff / λ * L  # self-coupling coefficient DC
    k = pi * vdneff / λ * L  # self-coupling coefficient AC

    def ode_system(
        z, rho, δ, s, k, F=0, apo_func=None
    ):  # ODE function, normalized to L (z/L, δL, σL, kL)
        R = rho[: len(rho) // 2]
        S = rho[len(rho) // 2 :]

        if apo_func:
            p = apo_func(z)
            s = s * p
            k = k * p

        s_ = δ + s - F * z

        dRdz = 1j * (s_ * R + k * S)
        dSdz = -1j * (s_ * S + k * R)
        return [dRdz, dSdz]

    δ = δ[:, np.newaxis]
    s = s[:, np.newaxis]
    k = k[:, np.newaxis]

    # initial conditions
    S0 = np.zeros(input.len(), dtype=complex)
    R0 = np.ones(input.len(), dtype=complex)
    y0 = np.concatenate([R0, S0])

    if apodization == "rcos":

        def apo_func(z):
            return rcos(z, alpha=1, T=2)

    elif apodization == "gaussian":

        def apo_func(z):
            return np.exp(-4 * np.log(2) * (3 * z) ** 2)

    elif apodization == "parabolic":

        def apo_func(z):
            return 1 - (2 * z) ** 2

    elif apodization == "uniform":
        apo_func = None
    elif callable(apodization):  # custom apodization function
        apo_func = apodization
    elif isinstance(apodization, str):
        warnings.warn("Apodization function not recognized. Using uniform apodization.")
        apo_func = None
    else:
        raise ValueError("Apodization must be a string or a function.")

    sol = solve_ivp(
        ode_system,
        t_span=[0.5, -0.5],
        y0=y0,
        method="RK45",
        args=(δ, s, k, F, apo_func),
        vectorized=True,
    )

    y = sol.y[:, -1]
    R = y[: len(y) // 2]
    S = y[len(y) // 2 :]

    H = S / R

    y = np.abs(H)  # reflectivity of the grating

    ic = np.argmin(np.abs(λ - c / fc))

    peaks, _ = sg.find_peaks(y)
    H_max = y[ic]

    if (y > 0.5).all():
        warnings.warn(
            "Bandwidth of the grating is too large for current sampling rate (`fs`). Consider increasing `fs`."
        )
        bandwith_str = f' - Δf = >{si(gv.fs, "Hz")} (Δλ = >{si(gv.fs*c/fc**2, "m")})'
    # elif (y<0.01).all():
    #     raise ValueError("Maximum reflectivity is less than 1%.")
    elif len(peaks):
        r = sg.peak_widths(y, peaks)

        BW_λ = r[0].max() * δλ
        BW_f = fc**2 * BW_λ / c

        bandwith_str = f' - Δf = {si(BW_f, "Hz")} (Δλ = {si(BW_λ, "m")})'
    else:
        warnings.warn("No peaks found in the reflectivity of the grating.")
        bandwith_str = " - Δf = -- GHz (Δλ = -- nm)"

    D = dispersion(H, gv.fs, fc)[ic]  # dispersion in ps/nm

    # Print parameters of the grating
    if print_params:
        print("\n*** Fiber Bragg Grating Features ***")
        print(f' - Λ = {si(Λ, "m")}')
        print(f" - N = {N}")
        print(f' - L = {si(L, "m")}')
        print(f' - λc = {si(c/fc, "m", 4)}')
        print(bandwith_str)
        print(f" - ρo = {y.max():.2f}")
        print(f" - loss = {-db(H_max**2):.1f} dB")
        print(f" - vδneff = {vdneff:.1e}")
        print(f" - kL = {kL:.1f}")
        print(f" - D(λc) = {D:.1f} ps/nm")
        if F:
            print(f" - F = {F:.1f}")
            print(f' - ΔΛ = {si(np.abs(Λ*F/(2*pi*N)), "m")}')
        print("************************************\n")

    if filtfilt:  # correct H(w)
        H = H * np.exp(
            -1j * input.w(shift=True) * tau_g(H, gv.fs)[ic] * 1e-12
        )  # corrected H(w)

    # apply to input optical signal
    output = ifft(fft(input.signal) * ifftshift(H))
    output = optical_signal(output)

    if retH:
        return output, H
    
    output.execution_time = toc()
    return output


# algunas funciones de prueba
def animated_fiber_propagation(
    input: optical_signal,
    M: int,
    length_: float,
    alpha_: float = 0.0,
    beta_2_: float = 0.0,
    beta_3_: float = 0.0,
    gamma_: float = 0.0,
    phi_max: float = 0.05,
):
    from matplotlib.animation import FuncAnimation

    # cambio las unidades
    length = length_ * 1e3
    alpha = alpha_ * 1 / (4.343 * 1e3)
    beta_2 = beta_2_ * 1e-12**2 / 1e3
    beta_3 = beta_3_ * 1e-12**3 / 1e3
    gamma = gamma_ * 1 / 1e3

    w = input.w()
    D_op = -alpha / 2 - 1j / 2 * beta_2 * w**2 - 1j / 6 * beta_3 * w**3

    A = input.signal[0]

    h = (
        length
        if (beta_2 == 0 and beta_3 == 0) or gamma == 0
        else phi_max / (gamma * np.abs(A) ** 2).max()
    )

    x_length = h
    A_z = [A]
    hs = [0]

    while True:
        exp_NL = np.exp(1j * gamma * (h / 2) * np.abs(A) ** 2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft(exp_L * fft(exp_NL * A))
        A_z.append(A)
        hs.append(h)

        h = phi_max / (gamma * np.abs(A) ** 2).max() if gamma != 0 else length

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length

    if h > 0:
        exp_NL = np.exp(1j * gamma * (h / 2) * np.abs(A) ** 2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft(exp_L * fft(exp_NL * A))
        A_z.append(A)
        hs.append(h)

    t = input.t() * gv.slot_rate

    fig, ax = plt.subplots()

    (line,) = ax.plot(t, np.abs(A_z[0]), lw=2, color="red", ls="--")
    (line,) = ax.plot([], [], lw=2, color="k")

    plt.suptitle(
        r"Fiber: $\alpha = {:.2f}$ dB/km, $\beta_2 = {}$ ps^2/km, $\gamma = {}$ (W·km)^-1".format(
            alpha_, beta_2_, gamma_
        )
    )
    ax.set_xlabel(r"$t/T_{slot}$")
    ax.set_ylabel("|A(z,t)|")
    ax.set_xlim((0, t.max()))
    ax.set_ylim((abs(A_z[0]).min() * 0.95, np.abs(A_z).max() * 1.05))

    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text("z = 0.0 Km")
        for i in t[:: M * gv.sps]:
            plt.axvline(i, color="k", ls="--")
        for i in t[:: gv.sps]:
            plt.axvline(i, color="k", ls="--", alpha=0.3, lw=1)
        return [line, time_text]

    def animate(i):
        y = np.abs(A_z[i])
        line.set_data(t, y)
        time_text.set_text("z = {:.2f} Km".format(np.cumsum(hs)[i] / 1e3))
        return [line, time_text]

    FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(A_z),
        interval=100,
        blit=True,
        repeat=False,
    )
    plt.show()


def animated_fiber_propagation_with_psd(
    input: optical_signal,
    M: int,
    length_: float,
    alpha_: float = 0.0,
    beta_2_: float = 0.0,
    beta_3_: float = 0.0,
    gamma_: float = 0.0,
    phi_max: float = 0.05,
    n: int = None,
):
    from matplotlib.animation import FuncAnimation

    n = input.len() if n is None else n * M * input.sps()

    length = length_
    alpha = alpha_ / 4.343
    beta_2 = beta_2_
    beta_3 = beta_3_
    gamma = gamma_

    w = input.w() * 1e-12  # rad/ps
    D_op = -alpha / 2 - 1j / 2 * beta_2 * w**2 - 1j / 6 * beta_3 * w**3

    A = input.signal[0]

    h = (
        length
        if (beta_2 == 0 and beta_3 == 0) or gamma == 0
        else phi_max / (gamma * np.abs(A) ** 2).max()
    )

    x_length = h
    A_z = [A]
    A_z_w = [fft(A)]
    hs = [0]

    while True:
        exp_NL = np.exp(1j * gamma * (h / 2) * np.abs(A) ** 2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft(exp_L * fft(exp_NL * A))
        A_z.append(A * np.exp(alpha * x_length / 2))
        A_z_w.append(fft(A * np.exp(alpha * x_length / 2)))
        hs.append(h)

        h = phi_max / (gamma * np.abs(A) ** 2).max() if gamma != 0 else length

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length

    if h > 0:
        exp_NL = np.exp(1j * gamma * (h / 2) * np.abs(A) ** 2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft(exp_L * fft(exp_NL * A))
        A_z.append(A * np.exp(alpha * length / 2))
        A_z_w.append(fft(A * np.exp(alpha * length / 2)))
        hs.append(h)

    t = input.t() * gv.slot_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

    (line1,) = ax1.plot(t[:n], np.abs(A_z[0])[:n], lw=2, color="red", ls="--")
    (line1,) = ax1.plot([], [], lw=2, color="k")

    plt.suptitle(
        r"Fiber: $\alpha = {:.2f}$ dB/km, $\beta_2 = {}$ ps^2/km, $\gamma = {}$ (W·km)^-1".format(
            alpha_, beta_2_, gamma_
        )
    )
    ax1.set_xlabel("t/T")
    ax1.set_ylabel("|A(z,t)|")
    ax1.set_xlim((0, t[:n].max()))
    ax1.set_ylim((0, np.abs(A_z[:n]).max()))

    z_text = ax2.text(0.05, 0.9, "", transform=ax2.transAxes)

    f = fftshift(w / 2 / np.pi) * 1e3  # GHz
    y = fftshift(np.abs(A_z_w[0]) ** 2)

    (line2,) = ax2.plot(f, y / input.len(), "--g", lw=2)
    (line2,) = ax2.plot([], [], "k", lw=2)

    ax2.set_xlabel("f [GHz]")
    ax2.set_ylabel(r"$|A(z,w)|^2$")
    sigma = -f[
        np.cumsum(np.abs(fftshift(A_z_w[0])) ** 2)
        < 0.001 * np.sum(np.abs(A_z_w[0]) ** 2)
    ][-1]
    ax2.set_xlim((-2 * sigma, 2 * sigma))
    ax2.set_ylim((0, np.abs(A_z_w).max() ** 2 * 1.05 / input.len()))
    ax2.grid()

    plt.tight_layout()

    def init():
        line1.set_data([], [])
        z_text.set_text("z = 0.0 Km")
        for i in t[: n : M * gv.sps]:
            ax1.axvline(i, color="k", ls="--")
        for i in t[: n : gv.sps]:
            ax1.axvline(i, color="k", ls="--", alpha=0.3, lw=1)
        return [line1, z_text]

    def animate(i):
        y = np.abs(A_z[i])
        line1.set_data(t[:n], y[:n])
        z_text.set_text("z = {:.2f} Km".format(np.cumsum(hs)[i]))

        y = fftshift(np.abs(A_z_w[i]) ** 2) / input.len()
        line2.set_data(f, y)
        return [line1, line2, z_text]

    FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(A_z),
        interval=100,
        blit=True,
        repeat=False,
    )
    plt.show()


if __name__ == "__main__":
    pass
