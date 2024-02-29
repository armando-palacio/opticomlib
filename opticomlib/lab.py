"""
.. autosummary::
   :toctree: generated/

   SYNC                  -- Sincronizador de inicio de trama
   GET_EYE_v2            -- Eye diagram parameters and metrics estimator v2
"""
import numpy as np
import scipy.signal as sg

from .typing import binary_sequence, electrical_signal, eye

from .utils import tic, toc


def SYNC(signal_rx: electrical_signal, 
         slots_tx: binary_sequence, 
         sps: int):
    r"""
    **Synchronizer**

    Synchronizes the received signal with the transmitted signal to determine the starting position in the received signal for further processing. 
    This is done by performing a correlation between the received signal and the transmitted signal and finding the maximum correlation position.

    Parameters
    ----------
    signal_rx : electrical_signal
        The received digital signal (from the oscilloscope or an ADC).
    slots_tx : binary_sequence
        The transmitted slots sequence.
    sps : int
        Number of samples per slot of the digitalized signal ``signal_rx``.

    Returns
    -------
    tuple[electrical_signal, int]
        A tuple containing the synchronized digital signal and the position in the ``signal_rx`` array from which synchronization was performed.

    Raises
    ------
    TypeError
        The ``sps`` must be an integer to perform synchronization.
    BufferError
        If the number of received slots have to be greater than the transmitted slots.
    ValueError
        If no correlation maximum is found.
    """
    
    tic()
    if not isinstance(sps, int):
        raise TypeError('The "sps" must be an integer to perform synchronization.')

    signal_tx = np.kron(slots_tx.data, np.ones(sps))
    signal_rx = signal_rx.signal

    if len(signal_rx)<len(signal_tx): 
        raise BufferError('The length of the received vector must be greater than the transmitted vector!!')

    l = len(signal_tx)
    corr = sg.fftconvolve(signal_rx[:2*l], signal_tx[l::-1], mode='valid') # Correlation of the transmitted signal with the received signal in a window of 2*l (sufficient to find a maximum)

    if np.max(corr) < 3*np.std(corr): 
        raise ValueError('No correlation maximum found!!') # false positive
    
    i = np.argmax(corr)

    signal_sync = electrical_signal(signal_rx[i:-(l-i)])
    signal_sync.ejecution_time = toc()
    return signal_sync, i


def GET_EYE_v2(sync_signal: electrical_signal, slots_tx: binary_sequence, sps:int, nslots:int=8192):
    r"""
    Calculates the means and standard deviations of levels 0 and 1 in the received signal.

    Parameters
    ----------
    sync_signal : electrical_signal
        Synchronized digital signal in time with the transmitted signal.
    slots_tx : binary_sequence
        Transmitted bit sequence.
    sps : int
        Number of samples per slot of the digitalized signal ``sync_signal``.
    nslots : int, default: 8192
        Number of slots to use for estimation.

    Returns
    -------
    dict
        A dictionary containing the following keys:

            - ``sps``: Samples per slot of the digital signal.
            - ``y``: Synchronized digital signal.
            - ``unos``: Received signal levels corresponding to transmitted level 1.
            - ``zeros``: Received signal levels corresponding to transmitted level 0.
            - ``t0``: Time instants for level 0.
            - ``t1``: Time instants for level 1.
            - ``i``: Position in the 'signal' vector from which synchronization was performed.
            - ``mu0``: Mean of level 0.
            - ``mu1``: Mean of level 1.
            - ``s0``: Standard deviation of level 0.
            - ``s1``: Standard deviation of level 1.
    """
    sps = sync_signal.sps()

    eye_dict = {}

    eye_dict['sps'] = sps


    rx = sync_signal[:nslots*sps].signal + sync_signal[:nslots*sps].noise; eye_dict['y'] = rx
    tx = np.kron(slots_tx.data[:nslots], np.ones(sps))

    unos = rx[tx==1]; eye_dict['unos']=unos
    zeros = rx[tx==0]; eye_dict['zeros']=zeros

    t0 = np.kron(np.ones(zeros.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False)); eye_dict['t0']=t0
    t1 = np.kron(np.ones(unos.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False)); eye_dict['t1']=t1

    eye_dict['i']=sps//2

    unos_ = unos[(t1<0.05) & (t1>-0.05)]
    zeros_ = zeros[(t0<0.05) & (t0>-0.05)]

    mu0 = np.mean(zeros_).real; eye_dict['mu0'] = mu0
    mu1 = np.mean(unos_).real; eye_dict['mu1'] = mu1

    s0 = np.std(zeros_).real; eye_dict['s0'] = s0
    s1 = np.std(unos_).real; eye_dict['s1'] = s1

    return eye(eye_dict)