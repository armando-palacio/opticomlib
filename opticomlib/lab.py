"""
===================================
Laboratory and Practice funcions (:mod:`opticomlib.lab`)
===================================

.. autosummary::
   :toctree: generated/

   SYNC                  -- Sincronizador de inicio de trama
   GET_EYE_v2            -- Eye diagram parameters and metrics estimator v2
"""
import numpy as np
import scipy.signal as sg

from .typing import binary_sequence, electrical_signal, eye

from .utils import tic, toc


## Funciones para el Laboratorio
def SYNC(signal_rx: electrical_signal, slots_tx: binary_sequence, sps: int) -> tuple[electrical_signal, int]:
    """
    ### Descripción:
    Se realiza una sincronización de la señal recibida con la señal transmitida para saber a 
    partir de que posición de la señal recibida se debe comenzar a procesar. Para ello se realiza una correlación 
    entre la señal recibida y la señal transmitida y se busca el máximo de la correlación.
    
    ---

    ### Args:
    - `signal_rx` - señal digital recibida (del osciloscopio).
    - `bits_tx` - secuencia de bits transmitida.

    ### Returns:
    - `signal_sync` - señal digital sincronizada.
    - `i` - posición del vector 'signal' a partir de la cual se realiza la sincronización.  
    """
    
    tic()
    if not isinstance(sps, int):
        raise TypeError('Los sps deben ser un número entero para realizar la sincronización.')

    signal_tx = np.kron(slots_tx.data, np.ones(sps))
    signal_rx = signal_rx.signal

    if len(signal_rx)<len(signal_tx): raise BufferError('La longitud del vector recibido debe ser mayor al vector transmitido!!')

    l = len(signal_tx)
    corr = sg.fftconvolve(signal_rx[:2*l], signal_tx[l::-1], mode='valid') # Correlación de la señal transmitida con la señal recibida en una ventana de 2*l (suficiente para encontrar un máximo)

    if np.max(corr) < 3*np.std(corr): raise ValueError('No se encontró un máximo de correlación!!') # falso positivo
    
    i = np.argmax(corr)

    signal_sync = electrical_signal(signal_rx[i:-(l-i)])
    signal_sync.ejecution_time = toc()
    return signal_sync, i


def GET_EYE_v2(sync_signal: electrical_signal, slots_tx: binary_sequence, sps:int, nslots:int=8192):
    """
    ### Descripción:
    Esta función obtiene las medias y desviaciones estándar de los niveles 0 y 1 de la señal recibida. Utiliza los
    slots de la señal transmitida para determinar los instantes de tiempo en los que se encuentran los niveles 0 y 1.
    
    ---

    ### Args:
    - `sync_signal` - señal digital sincronizada en tiempo con la señal transmitida.
    - `bits_tx` - secuencia de bits transmitida.
    - `sps` - muestras por slot de la señal digital (default: global_vars.sps).
    - `nslots` [Opcional] - cantidad de slots a utilizar para la estimación (default: 8192).

    ### Returns:
    - `signal_sync` - señal digital sincronizada.
    - `i` - posición del vector 'signal' a partir de la cual se realiza la sincronización.  
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