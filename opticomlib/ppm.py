"""
.. rubric:: Functions
.. autosummary::

   PPM_ENCODER           
   PPM_DECODER           
   HDD                   
   SDD                   
   THRESHOLD_EST         
   DSP                   
   BER_analizer          
   theory_BER           
"""

import numpy as np
from typing import Literal, Union
from numpy import ndarray
from scipy.integrate import quad
from scipy.constants import pi

from .devices import GET_EYE, SAMPLER, LPF
from .typing import binary_sequence, electrical_signal, eye, gv, Array_Like
from .utils import tic, toc, str2array, dec2bin, Q



def PPM_ENCODER(input: Union[str, list, tuple, ndarray, binary_sequence], M: int) -> binary_sequence:
    r"""PPM Encoder

    Converts an input binary sequence into a binary sequence PPM encoded.

    Parameters
    ----------
    input : :obj:`binary_sequence`
        Input binary sequence.
    M : :obj:`int`
        Number of slots that a symbol contains.

    Returns
    -------
    ppm_seq : :obj:`binary_sequence`
        Encoded binary sequence in PPM.

    Notes
    -----
    The input binary sequence is converted into a PPM sequence by grouping each :math:`\log_2{M}` bits 
    and converting them into decimal. Then, the decimal values are the positions of ON slots into the PPM symbols of
    length :math:`M`.

    Examples
    --------
    >>> from opticomlib.ppm import PPM_ENCODER
    >>> PPM_ENCODER('01111000', 4).data.astype(int)
    array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])

    """
    tic()

    if isinstance(input, binary_sequence):
        input = input.data
    elif isinstance(input, str):
        input = str2array(input, bool)
    elif isinstance(input, Array_Like):
        input = np.array(input, dtype=bool)
    else:
        raise TypeError("`input` must be of type (str, list, tuple, ndarray, binary_sequence)")

    k = int(np.log2(M))

    input = input[:len(input)//k*k] 

    decimal = np.sum(input.reshape(-1,k)*2**np.arange(k)[::-1], axis=-1) # convert bits to decimal
    ppm_s = np.zeros(decimal.size*M, dtype=bool)

    ppm_s[np.arange(decimal.size)*M + decimal] = 1 # coded the symbols
   
    output = binary_sequence(ppm_s) 
    output.execution_time = toc()
    return output



def PPM_DECODER(input: Union[str, list, tuple, np.ndarray, binary_sequence], M: int) -> binary_sequence:
    """PPM Decoder

    Receives a binary sequence encoded in PPM and decodes it.

    Parameters
    ----------
    input : binary sequence in form of a string, list, tuple, ndarray or binary_sequence
        Binary sequence encoded in PPM.
    M : :obj:`int`
        Order of PPM modulation.

    Returns
    -------
    :obj:`binary_sequence`
        Decoded binary sequence.

    Examples
    --------
    >>> from opticomlib.ppm import PPM_DECODER
    >>> PPM_DECODER('0100000100101000', 4).data.astype(int)
    array([0, 1, 1, 1, 1, 0, 0, 0])
    """
    tic()

    if isinstance(input, binary_sequence):
        input = input.data
    elif isinstance(input, str):
        input = str2array(input, bool)
    elif isinstance(input, Array_Like):
        input = np.array(input, dtype=bool)
    else:
        raise TypeError("`input` must be of type (str, list, tuple, ndarray, binary_sequence)")
    
    k = int(np.log2(M))

    decimal = np.where(input==1)[0]%M # get decimals

    output = np.array(list(map(lambda x: dec2bin(x,k), decimal))).ravel() # convert decimals to bits
    output= binary_sequence(output)

    output.execution_time = toc()
    return output


def HDD(input: Union[str, list, tuple, np.ndarray, binary_sequence], M: int):
    """Hard Decision Decoder

    Estimates the most probable PPM symbols from the given binary sequence.

    - If there is any symbol without ON slots, then one of them is raised randomly
    - If there is any symbol with more tan one ON slots, then one of them is selected randomly
    - Other case algorithm do nothing.   

    Parameters
    ----------
    input : binary sequence in form of a string, list, tuple, ndarray or binary_sequence
        Binary sequence to estimate.

    Returns
    -------
    :obj:`binary_sequence`
        Sequence of estimated symbols ready to decode.

    Raises
    ------
    ValueError
        If `M` is not a power of 2.
    ValueError
        If the length of `input` is not a multiple of `M`.

    Examples
    --------
    >>> from opticomlib.ppm import HDD, binary_sequence
    >>> 
    >>> HDD(binary_sequence('0100 0111 0000'), 4).data.astype(int)
    array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    """
    tic()

    if isinstance(input, binary_sequence):
        input = input.data
    elif isinstance(input, str):
        input = str2array(input, bool)
    elif isinstance(input, Array_Like):
        input = np.array(input, dtype=bool)
    else:
        raise TypeError("`input` must be of type (str, list, tuple, ndarray, binary_sequence)")

    if not M & (M-1) == 0:
        raise ValueError("`M` must be a power of 2.")

    if input.size % M != 0:
        raise ValueError("The length of `input` must be a multiple of `M`.")

    n_simb = int(input.size/M) # number of symbols

    s = np.sum(input.reshape(n_simb, M), axis=-1) # number of ON slots per symbol

    output = input.copy() 

    for i in np.where(s==0)[0]: 
        output[i*M + np.random.randint(M)] = 1  # raise one slot randomly for each symbol without ON slots

    for i in np.where(s>1)[0]: 
        j = np.where(output[i*M:(i+1)*M]==1)[0]
        output[i*M:(i+1)*M] = 0
        output[i*M + np.random.choice(j)]=1  # select one ON slot randomly for each symbol with more than one ON slots

    output = binary_sequence(output)
    output.execution_time = toc()
    return output



def SDD(input: electrical_signal, M: int) -> binary_sequence:
    """Soft Decision Decoder

    Estimates the most probable PPM symbols from the given electrical signal without sampling.
    It integrate the signal in slots and then, it selects the slot with the highest energy.

    Parameters
    ----------
    input : :obj`electrical_signal`
        Unsampled electrical signal.

    Returns
    -------
    :obj`binary_sequence`
        Sequence of estimated symbols ready to decode.

    Raises
    ------
    ValueError
        If `M` is not a power of 2.
    ValueError
        If the length of `input` is not a multiple of `M*sps`.

    Examples
    --------
    >>> from opticomlib.ppm import SDD, electrical_signal, gv
    >>> import numpy as np
    >>>
    >>> x = np.kron([0.1,1.2,0.1,0.2,  0.1,0.9,1.0,1.1,  0.1,0.1,0.1,0.2], np.ones(gv.sps))
    >>> SDD(electrical_signal(x), M=4).data.astype(int)
    array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    """
    tic()

    if not M & (M-1) == 0:
        raise ValueError("`M` must be a power of 2.")
    
    if isinstance(input, electrical_signal):
        input = input.signal + input.noise

    elif isinstance(input, Array_Like):
        input = np.array(input)
    
    if input.size % (M*gv.sps) != 0:
        raise ValueError("The length of `input` must be a multiple of `M*sps`.")

    signal = np.sum( input.reshape(-1, gv.sps), axis=-1)

    i = np.argmax( signal.reshape(-1, M), axis=-1)

    output = np.zeros_like(signal, dtype=np.uint8)
    output[np.arange(i.shape[0])*M+i] = 1

    output = binary_sequence(output)
    output.execution_time = toc()
    return output



def THRESHOLD_EST(eye_obj: eye, M: int):
    """Threshold Estimator
    
    Estimates the decision threshold for M-PPM from means and standard deviations of ``eye_obj``.

    Parameters
    ----------
    eye_obj : :obj:`eye`
        `eye` object with the parameters of the eye diagram.
    M : :obj:`int`
        Order of PPM.

    Returns
    -------
    :obj:`float`
        Estimated threshold.

    Raises
    ------
    ValueError
        If `M` is not a power of 2.
    TypeError
        If `eye_obj` is not of type `eye`.
    
    Examples
    --------
    >>> from opticomlib.ppm import THRESHOLD_EST, eye
    >>>
    >>> eye_obj = eye({'mu0':0.1, 'mu1':1.1, 's0':0.1, 's1':0.1})
    >>> THRESHOLD_EST(eye_obj, M=4)
    """
    if not M & (M-1) == 0:
        raise ValueError("`M` must be a power of 2.")
    
    if not isinstance(eye_obj, eye):
        raise TypeError("`eye_obj` must be of type `eye`.")

    mu0 = eye_obj.mu0
    mu1 = eye_obj.mu1
    s0 = eye_obj.s0
    s1 = eye_obj.s1

    r = np.linspace(mu0, mu1, 1000)
    umbral = r[np.argmin(1 - Q((r-mu1)/s1) * (1-Q((r-mu0)/s0))**(M-1))]
    return umbral



def DSP(input: electrical_signal, M :int, decision: Literal['hard','soft']='hard', BW: float=None):
    """PPM Digital Signal Processor
    
    Performs the decision task of the photodetected electrical signal. 

    1. If ``BW`` is provided bessel filter will be applied to the signal (:func:`opticomlib.devices.LPF`)
    2. eye diagram parameters are estimated from the input electrical signal with function :func:`opticomlib.devices.GET_EYE`.
    3. it subsamples the electrical signal to 1 sample per slot using function :func:`opticomlib.devices.SAMPLER`. 
    4. if ``decision='hard'`` it compares the amplitude of the subsampled signal with optimal threshold. The optimal threshold is obtained from function :func:`opticomlib.ppm.THRESHOLD_EST`. 
    5. then, it make the decision (:func:`opticomlib.ppm.HDD` if ``decision='hard'`` or :func:`opticomlib.ppm.SDD` if ``decision='soft'``).
    6. Finally, it returns the received binary sequence, eye object and optimal threshold.

    Parameters
    ----------
    input : :obj:`electrical_signal`
        Photodetected electrical signal.
    M : :obj:`int`
        Order of PPM modulation.
    decision : :obj:`str`, optional
        Type of decision to make. Default is 'hard'.
    BW : :obj:`float`, optional
        Bandwidth of DSP filter. If not specified, signal won't be filtered.

    Returns
    -------
    output : :obj:`binary_sequence`
        Received bits.
    eye_obj : :obj:`eye`, optional
        Eye diagram parameters, only if ``decision='hard'``.
    rth : :obj:`float`, optional
        Decision threshold for PPM, only if ``decision='hard'``.

    Raises
    ------
    TypeError
        If `input` is not of type `electrical_signal` or `Array_Like`.
    ValueError
        If `input` has less samples than `sps`.
    ValueError
        If `M` is not a power of 2.
    ValueError
        If `decision` is not 'hard' or 'soft'.
    
    Examples
    --------
    .. plot::
        :include-source:
        :alt: DSP PPM
        :align: center
        :width: 720

        from opticomlib.devices import DAC, gv
        from opticomlib.ppm import DSP

        import numpy as np
        import matplotlib.pyplot as plt

        gv(sps=64, R=1e9)

        x = DAC('0100 1010 0000', pulse_shape='gaussian')
        x.noise = np.random.normal(0, 0.1, x.len())

        y = DSP(x, M=4, decision='soft')

        DAC(y).plot(c='r', lw=3, label='Received sequence').show()
    """
    if not isinstance(input, (electrical_signal,) + Array_Like):
        raise TypeError("`input` must be of type `electrical_signal` or `Array_Like`.")
    
    if not isinstance(input, electrical_signal):
        input = electrical_signal(input)
    
    if input.len() < gv.sps:
        raise ValueError("`input` must have at least `sps` samples.")
    
    if not M & (M-1) == 0:
        raise ValueError("`M` must be a power of 2.")

    if BW is not None:
        x = LPF(input, BW)
    else:
        x = input
        x.execution_time = 0

    if decision.lower() == 'hard':
        eye_obj = GET_EYE(x, nslots=8192, sps_resamp=128); time = eye_obj.execution_time + x.execution_time
        rth = THRESHOLD_EST(eye_obj, M)
        x = SAMPLER(x, eye_obj); time += x.execution_time

        tic()
        output = x > rth
        simbols = HDD(output, M); simbols.execution_time += toc() + time

        output = PPM_DECODER(simbols, M)
        output.execution_time += simbols.execution_time

        return output, eye_obj, rth
    
    elif decision.lower() == 'soft':
        tic()
        simbols = SDD(input, M); simbols.execution_time += toc() + x.execution_time
        output = PPM_DECODER(simbols, M)
        output.execution_time += simbols.execution_time
        return output
    
    else:
        raise ValueError('`decision` must be "hard" or "soft"')



def BER_analizer(mode: Literal['counter', 'estimator'], **kwargs):
    """BER Analizer
    
    Calculates the bit error rate (BER), either by error counting (comparing the received sequence with the transmitted one) 
    or by estimation (using estimated means and variances from the eye diagram and substituting those values into the theoretical expressions).

    Parameters
    ----------
    mode : :obj:`str`
        Mode in which the Bit Error Rate (BER) will be determined.

    Other Parameters
    ----------------
    Tx : :obj:`binary_sequence`, optional
        Transmitted binary sequence. Required if `mode='counter'`.
    Rx : :obj:`binary_sequence`, optional
        Received binary sequence. Required if `mode='counter'`.
    eye_obj : :obj:`eye`, optional
        `eye` object with the estimated parameters of the eye diagram. Required if `mode='estimator'`.
    M : :obj:`int`, optional
        Order of PPM modulation. Required if `mode='estimator'`.
    decision : :obj:`str`, optional
        Type of decision to make, 'hard' or 'soft'. Default is 'soft'. Required if `mode='estimator'`.

    Returns
    -------
    :obj:`float`
        BER.

    Raises
    ------
    ValueError
        If `mode` is not 'counter' or 'estimator'.
    ValueError
        If `decision` is not 'hard' or 'soft'.
    KeyError
        If `Tx` or `Rx` are not provided when `mode='counter'`.
    KeyError
        If `eye_obj` or `M` are not provided when `mode='estimator'`.
    ValueError
        If `M` is not a power of 2.
    """
        
    if mode.lower() == 'counter':
        Tx = kwargs.get('Tx', None)
        Rx = kwargs.get('Rx', None)

        if Tx is None or Rx is None:
            raise KeyError("`Tx` and `Rx` are required arguments for `mode='counter'`.")

        if not isinstance(Rx, binary_sequence):
            Rx = binary_sequence( Rx )
        if not isinstance(Tx, binary_sequence):
            Tx = binary_sequence( Tx )

        Tx = Tx[:Rx.len()]
        assert Tx.len() == Rx.len(), "Error: `Tx` and `Rx` must have the same length."

        return np.sum(Tx.data != Rx.data)/Tx.len()

    elif mode.lower() == 'estimator':
        eye_obj = kwargs.get('eye_obj', None)
        M = kwargs.get('M', None)
        decision = kwargs.get('decision', 'soft')

        if eye_obj is None or M is None:
            raise KeyError("`eye_obj` and `M` are required arguments for `mode='estimator'`.")

        if not M & (M-1) == 0:
            raise ValueError("`M` must be a power of 2.")

        if decision.lower() not in ['hard', 'soft']:
            raise ValueError("`decision` must be 'hard' or 'soft'.")

        I1 = eye_obj.mu1
        I0 = eye_obj.mu0
        s1 = eye_obj.s1
        s0 = eye_obj.s0
        um = THRESHOLD_EST(eye_obj, M)

        if decision == 'hard':
            Pe_sym = 1 - Q((um-I1)/s1) * (1-Q((um-I0)/s0))**(M-1)
        elif decision == 'soft':
            Pe_sym = 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((I1-I0+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0]
        return M/2/(M-1)*Pe_sym

    else:
        raise ValueError('Invalid mode. Use `counter` or `estimator`.')


def theory_BER(mu1: Union[float, ndarray], s0: Union[float, ndarray], s1: Union[float, ndarray], M: int, decision: Literal['soft','hard']='soft'):
    r"""
    Calculates the theoretical bit error probability for a PPM system.

    Parameters
    ----------
    mu1 : :obj:`float` or :obj:`ndarray`
        Average current (or voltage) value of the signal corresponding to a bit 1.
    s0 : :obj:`float` or :obj:`ndarray`
        Standard deviation of current (or voltage) of the signal corresponding to a bit 0.
    s1 : :obj:`float` or :obj:`ndarray`
        Standard deviation of current (or voltage) of the signal corresponding to a bit 1.
    M : :obj:`int`
        Order of PPM modulation.
    decision : :obj:`str`, optional
        Type of PPM decoding. Default is 'soft'.

    Returns
    -------
    :obj:`float`
        Theoretical bit error probability (BER).

    Raises
    ------
    ValueError
        If `M` is not a power of 2.
    ValueError
        If `decision` is not 'hard' or 'soft'.

    Notes
    -----
    The theoretical bit error probability is calculated using the following expression:

    .. math::
        P_e = \frac{M/2}{(M-1)}P_{e_{sym}}

    where :math:`P_{e_{sym}}` is the symbol error probability, and is calculated as follows for ``decision='soft'``:

    .. math::
        P_{e_{sym}} = 1 - \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \left( 1-Q\left( \frac{\mu_1+s_1x}{s_0} \right) \right) ^{M-1} e^{-x^2/2}dx

    and for ``decision='hard'``:

    .. math::
        P_{e_{sym}} = 1 - Q\left( \frac{r_{th}-\mu_1}{s_1} \right) \left( 1-Q\left( \frac{r_{th}}{s_0} \right) \right)^{M-1}

    Examples
    --------
    >>> from opticomlib.ppm import theory_BER
    >>> theory_BER(mu1=1, s0=0.1, s1=0.1, M=8, decision='hard')
    8.515885763544466e-07
    >>> theory_BER(mu1=1, s0=0.1, s1=0.1, M=8, decision='soft')
    3.074810247686141e-12

    """
    if not M & (M-1) == 0:
        raise ValueError("`M` must be a power of 2.")

    if decision == 'soft':
        fun = np.vectorize( lambda mu1,s0,s1,M: 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((mu1+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0] )
    elif decision == 'hard':
        @np.vectorize
        def fun(mu1_,s0_,s1_,M_):
            r = np.linspace(0,mu1_,1000)
            return np.min(1 - Q((r-mu1_)/s1_) * (1-Q(r/s0_))**(M_-1))
    else:
        raise ValueError('`decision` must be `soft` or `hard`.')
    return fun(mu1,s0,s1,M)*0.5*M/(M-1)

