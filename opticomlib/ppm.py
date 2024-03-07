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
from .typing import binary_sequence, electrical_signal, eye, gv
from .utils import tic, toc, str2array, dec2bin, Q



def PPM_ENCODER(input: Union[str, list, tuple, ndarray, binary_sequence], M: int) -> binary_sequence:
    """PPM Encoder

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
    elif isinstance(input, (list, tuple)):
        input = np.array(input, dtype=bool)
    else:
        raise TypeError("`input` must be of type (str, list, tuple, ndarray, binary_sequence)")

    k = int(np.log2(M))

    input = input[:len(input)//k*k] 

    decimal = np.sum(input.reshape(-1,k)*2**np.arange(k)[::-1], axis=-1) # convert bits to decimal
    ppm_s = np.zeros(decimal.size*M, dtype=bool)

    ppm_s[np.arange(decimal.size)*M + decimal] = 1 # coded the symbols
   
    output = binary_sequence(ppm_s) 
    output.ejecution_time = toc()
    return output



def PPM_DECODER(input: Union[str, list, tuple, np.ndarray, binary_sequence], M: int) -> binary_sequence:
    """PPM Decoder

    Receives a binary sequence encoded in PPM and decodes it.

    Parameters
    ----------
    input : :obj:`binary_sequence`
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
    elif isinstance(input, (list, tuple)):
        input = np.array(input, dtype=bool)
    else:
        raise TypeError("`input` must be of type (str, list, tuple, ndarray, binary_sequence)")
    
    k = int(np.log2(M))

    decimal = np.where(input==1)[0]%M # get decimal

    output = np.array(list(map(lambda x: dec2bin(x,k), decimal))).ravel() # convert to binary again
    output= binary_sequence(output)

    output.ejecution_time = toc()
    return output


def HDD(input: binary_sequence, M: int) -> binary_sequence:
    """Hard Decision Decoder

    Estimates the most probable PPM symbols from the given binary sequence.

    - If there is any symbol without ON slots, then one of them is raised randomly
    - If there is any symbol with more tan one ON slots, then one of them is selected randomly
    - Other case algorithm do nothing.   

    Parameters
    ----------
    input : :obj:`binary_sequence`
        Binary sequence to estimate.

    Returns
    -------
    :obj:`binary_sequence`
        Sequence of estimated symbols ready to decode.

    Examples
    --------
    >>> from opticomlib.ppm import HDD, binary_sequence
    >>> 
    >>> HDD(binary_sequence('0100 0111 0000'), 4).data.astype(int)
    array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    """
    tic()

    n_simb = int(input.len()/M)

    s = np.sum(input.data.reshape(n_simb, M), axis=-1)

    output = np.array(input.data, dtype=np.uint8)

    for i in np.where(s==0)[0]: # si existe algún símbolo sin ningún slot encendido, se prende uno al azar
        output[i*M + np.random.randint(M)] = 1

    for i in np.where(s>1)[0]: # si existe algún símbolo con más de 1 slot encendido, se elige uno de ellos al azar)
        j = np.where(output[i*M:(i+1)*M]==1)[0]
        output[i*M:(i+1)*M] = 0
        output[i*M + np.random.choice(j)]=1

    output = binary_sequence(output)
    output.ejecution_time = toc()
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

    signal = np.sum( (input.signal + input.noise).reshape(-1, input.sps()), axis=-1)

    i = np.argmax( signal.reshape(-1, M), axis=-1)

    output = np.zeros_like(signal, dtype=np.uint8)
    output[np.arange(i.shape[0])*M+i] = 1

    output = binary_sequence(output)
    output.ejecution_time = toc()
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
    """

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
    if BW is not None:
        x = LPF(input, BW)
    else:
        x = input
        x.ejecution_time = 0

    if decision == 'hard':
        eye_obj = GET_EYE(x, nslots=8192, sps_resamp=128); time = eye_obj.ejecution_time + x.ejecution_time
        rth = THRESHOLD_EST(eye_obj, M)
        x = SAMPLER(x, eye_obj); time += x.ejecution_time

        tic()
        output = x > rth
        simbols = HDD(output, M); simbols.ejecution_time += toc() + time

        output = PPM_DECODER(simbols, M)
        output.ejecution_time += simbols.ejecution_time

        return output, eye_obj, rth
    
    elif decision == 'soft':
        tic()
        simbols = SDD(input, M); simbols.ejecution_time += toc() + x.ejecution_time
        output = PPM_DECODER(simbols, M)
        output.ejecution_time += simbols.ejecution_time
        return output
    
    else:
        raise TypeError('`decision` must be "hard" or "soft"')



def BER_analizer(mode: Literal['counter', 'estimator'], **kargs):
    """
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
    """
        
    if mode == 'counter':
        assert 'Rx' in kargs.keys() and 'Tx' in kargs.keys(), "`Tx` and `Rx` are required arguments for `mode='counter'`."
        Rx = kargs['Rx']
        Tx = kargs['Tx']

        if not isinstance(Rx, binary_sequence) and not isinstance(Tx, binary_sequence):
            Rx = binary_sequence( Rx )
            Tx = binary_sequence( Tx )

        Tx = Tx[:Rx.len()]
        assert Tx.len() == Rx.len(), "Error: `Tx` and `Rx` must have the same length."

        return np.sum(Tx.data != Rx.data)/Tx.len()

    elif mode == 'estimator':
        assert 'eye_obj' in kargs.keys(), "`eye_obj` is a required argument for `mode='estimator'`."
        assert 'M' in kargs.keys(), "`M` is a required argument for `mode='estimator'`"

        eye_obj = kargs['eye_obj']
        M = kargs['M']
        decision = kargs['decision'] if 'decision' in kargs.keys() else 'soft'

        assert decision.lower() in ('hard', 'soft'), "Error: el argumento decision debe tomar los valores `hard` o `soft`"

        I1 = eye_obj.mu1
        I0 = eye_obj.mu0
        s1 = eye_obj.s1
        s0 = eye_obj.s0
        um = THRESHOLD_EST(eye_obj, M)

        if decision == 'hard':
            Pe_sym = 1 - Q((um-I1)/s1) * (1-Q((um-I0)/s0))**(M-1)
        elif decision == 'soft':
            Pe_sym = 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((I1-I0+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0]
        else:
            raise TypeError('`decision` must be "hard" or "soft"')
        return M/2/(M-1)*Pe_sym

    else:
        raise TypeError('Invalid mode. Use `counter` or `estimator`.')


def theory_BER(mu1: Union[int, ndarray], s0: Union[int, ndarray], s1: Union[int, ndarray], M: int, decision: Literal['soft','hard']='soft'):
    """
    Calculates the theoretical bit error probability for a PPM system.

    Parameters
    ----------
    mu1 : :obj:`float`
        Average current (or voltage) value of the signal corresponding to a bit 1.
    s0 : :obj:`float`
        Standard deviation of current (or voltage) of the signal corresponding to a bit 0.
    s1 : :obj:`float`
        Standard deviation of current (or voltage) of the signal corresponding to a bit 1.
    M : :obj:`int`
        Order of PPM modulation.
    decision : :obj:`str`, optional
        Type of PPM decoding. Default is 'soft'.

    Returns
    -------
    :obj:`float`
        Theoretical bit error probability (BER).
    """

    if decision == 'soft':
        fun = np.vectorize( lambda mu1,s0,s1,M: 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((mu1+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0] )
    elif decision == 'hard':
        def fun(mu1_,s0_,s1_,M_):
            r = np.linspace(0,mu1_,1000)
            return np.min(1 - Q((r-mu1_)/s1_) * (1-Q((r)/s0_))**(M_-1))
        fun = np.vectorize( fun )
    else:
        raise ValueError('`decision` must be `soft` or `hard`.')
    return fun(mu1,s0,s1,M)*0.5*M/(M-1)

