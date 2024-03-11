"""
.. rubric:: Functions
.. autosummary::

   THRESHOLD_EST       
   DSP                  
   BER_analizer          
   theory_BER           
"""

from numpy import ndarray
from typing import Literal, Union

import numpy as np

from .devices import GET_EYE, SAMPLER, LPF
from .typing import binary_sequence, electrical_signal, eye, gv
from .utils import Q, tic, toc



def THRESHOLD_EST(eye_obj: eye):
    """Threshold estimator

    Estimates the decision threshold for OOK from the means and standard deviations of the eye diagram.

    Parameters
    ----------
    eye_obj : :obj:`eye`
        Object with the parameters of the eye diagram.

    Returns
    -------
    :obj:`float`
        Decision threshold for OOK.

    Notes
    -----
    The decision threshold is estimated as the value of amplitud that minimizes the probability of error
    given the means and standard deviations of the eye diagram. This is done by minimizing the probability function [th]_:

    .. math::
        f(r) = \\frac{1}{2} Q\\left(\\frac{\\mu_1 - r}{\\sigma_1}\\right) + \\frac{1}{2} Q\\left(\\frac{r - \\mu_0}{\\sigma_0}\\right)

    where :math:`\\mu_0` and :math:`\\mu_1` are the means of the eye diagram, :math:`\\sigma_0` and :math:`\\sigma_1` are the standard deviations
    and :func:`~opticomlib.utils.Q` is the Q-function.

    References
    ----------
    .. [th] Armando Palacio Romeu, "Comunicaciones ópticas entre satélites LEO y GEO", chapter 2.4. link: https://ricabib.cab.cnea.gov.ar/1143/1/1Palacio_Romeu.pdf
    """

    mu0 = eye_obj.mu0
    mu1 = eye_obj.mu1
    s0 = eye_obj.s0
    s1 = eye_obj.s1

    r = np.linspace(mu0, mu1, 1000)
    umbral = r[np.argmin( 0.5*(Q((mu1-r)/s1) + Q((r-mu0)/s0)) )]
    return umbral


def DSP(input: electrical_signal, BW: float = None):
    """On-Off Keying Digital Signal Processing
    
    Performs the decision task of the photodetected electrical signal. 

    1. If ``BW`` is provided bessel filter will be applied to the signal (:func:`opticomlib.devices.LPF`)
    2. eye diagram parameters are estimated from the input electrical signal with function :func:`opticomlib.devices.GET_EYE`.
    3. it subsamples the electrical signal to 1 sample per bit using function :func:`opticomlib.devices.SAMPLER`. 
    4. Then, it compares the amplitude of the subsampled signal with optimal threshold. The optimal threshold is obtained from function :func:`opticomlib.ook.THRESHOLD_EST`. 
    5. Finally, it returns the received binary sequence, eye object and optimal threshold.

    Parameters
    ----------
    input : :obj:`electrical_signal`
        Photodetected electrical signal.
    BW : :obj:`float`, optional
        Bandwidth of DSP filter. If not specified, signal won't be filtered.

    Returns
    -------
    output : :obj:`binary_sequence`
        Received bits.
    eye_obj : :obj:`eye`
        Eye diagram parameters.
    rth : :obj:`float`
        Decision threshold for OOK.

    Examples
    --------
    .. plot::
        :include-source:
        :alt: DSP OOK
        :align: center
        :width: 720

        from opticomlib.devices import DAC, gv
        from opticomlib.ook import DSP

        import numpy as np
        import matplotlib.pyplot as plt

        gv(sps=64, R=1e9)

        x = DAC('01000100100000', 1, pulse_shape='gaussian')
        x.noise = np.random.normal(0, 0.1, x.len())

        y, eye_, xth = DSP(x)

        x.plot('y', label='Photodetected signal')
        DAC(y).plot(c='r', lw=2, label='Received sequence')
        plt.axhline(xth, color='b', linestyle='--', label='Threshold')
        plt.legend(loc='upper right')
        plt.show()
    """
    if BW is not None:
        x = LPF(input, BW)
    else:
        x = input
        x.execution_time = 0

    eye_obj = GET_EYE(x, nslots=8192, sps_resamp=128); time = eye_obj.execution_time + x.execution_time
    rth = THRESHOLD_EST(eye_obj)

    x = SAMPLER(x, eye_obj) # one sample per bit 
    
    tic()
    output = x > rth
    
    output.execution_time = toc() + time + x.execution_time
    return output, eye_obj, rth


def BER_analizer(mode: Literal['counter', 'estimator'], **kargs):
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

    Returns
    -------
    :obj:`float`
        BER.
    
    Examples
    --------
    .. code-block:: python
        
        from opticomlib.devices import DAC, gv, binary_sequence
        from opticomlib.ook import DSP, BER_analizer

        import numpy as np

        gv(sps=64, R=1e9)

        tx = binary_sequence('01000100100000')
        x = DAC(tx, pulse_shape='gaussian')
        x.noise = np.random.normal(0, 0.1, x.len())

        rx, eye_, xth = DSP(x)
        BER_count = BER_analizer('counter', Tx=tx, Rx=rx)
        BER_est = BER_analizer('estimator', eye_obj=eye_)

        print(f'BER by counting: {BER_count:.1e}')
        print(f'BER by estimation: {BER_est:.1e}')
    
    Output:
        
    ::
        
        BER by counting: 0.0e+00
        BER by estimation: 3.7e-07
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

        eye_obj = kargs['eye_obj']

        I1 = eye_obj.mu1
        I0 = eye_obj.mu0
        s1 = eye_obj.s1
        s0 = eye_obj.s0
        um = THRESHOLD_EST(eye_obj)

        return 0.5*(Q((I1-um)/s1) + Q((um-I0)/s0))

    else:
        raise TypeError('Invalid mode. Use `counter` or `estimator`.')
    


def theory_BER(mu1: Union[int, ndarray], s0: Union[int, ndarray], s1: Union[int, ndarray]):
    r"""Calculates the theoretical bit error probability for an OOK system.

    Parameters
    ----------
    mu1 : :obj:`float`
        Average current (or voltage) value of the signal corresponding to a bit 1.
    s0 : :obj:`float`
        Standard deviation of current (or voltage) of the signal corresponding to a bit 0.
    s1 : :obj:`float`
        Standard deviation of current (or voltage) of the signal corresponding to a bit 1.

    Returns
    -------
    :obj:`float`
        Theoretical bit error probability (BER).

    Notes
    -----
    The theoretical bit error probability is calculated using the following expression:

    .. math::
        P_e = \frac{1}{2} \left[Q\left(\frac{\mu_1 - r_{th}}{\sigma_1}\right) + Q\left(\frac{r_{th}}{\sigma_0}\right)\right]

    Examples
    --------
    >>> from opticomlib.ook import theory_BER
    >>> theory_BER(mu1=1, s0=0.1, s1=0.1)
    2.8674468224390994e-07
    """
    @np.vectorize
    def fun(mu1_,s0_,s1_):
        r = np.linspace(0,mu1_,1000)
        return 0.5*np.min(Q((mu1_-r)/s1_) + Q(r/s0_))
                     
    return fun(mu1,s0,s1)