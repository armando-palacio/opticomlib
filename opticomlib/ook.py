"""
===================================
OOK devices (:mod:`opticomlib.ook`)
===================================

.. autosummary::
   :toctree: generated/

   THRESHOLD_EST         -- Threshold for detection
   DSP                   -- Digital signal processing for PPM systems
   BER_analizer          -- Bit error rate analizer
   theory_BER            -- Theoretical bit error rate
"""

from numpy import ndarray
from typing import Literal, Union

import numpy as np

from .typing import binary_sequence, electrical_signal, eye
from .utils import Q, tic, toc


def THRESHOLD_EST(eye_obj: eye):
    """
    ### Descripción: 
    Esta función estima el umbral de decisión para OOK a partir de las medias y desviaciones estándar del diagrama de ojos.

    ---

    ### Args:
    - `eye_obj` - objeto `eye` con los parámetros del diagrama de ojos.

    ---

    ### Returns:
    - `umbral` - umbral de decisión para OOK.
    """

    mu0 = eye_obj.mu0
    mu1 = eye_obj.mu1
    s0 = eye_obj.s0
    s1 = eye_obj.s1

    # obtenemos el umbral de decisión para OOK
    r = np.linspace(mu0, mu1, 1000)
    umbral = r[np.argmin( 0.5*(Q((mu1-r)/s1) + Q((r-mu0)/s0)) )]
    return umbral


def DSP(input: electrical_signal,  eye_obj: eye):
    """
    ### Descripción:
    Este componente realiza la tarea de decisión de la señal eléctrica photodetectada. Primero sub-muestrea la señal eléctrica a 1 muestra por slot
    tomando el instante de decisión óptimo estimado del diagrama de ojos `eye_obj.i`. Luego, compara la amplitud de la señal submuetreada con el umbral
    de decisión óptimo estimado `THRESHOLD_EST(eye_obj)`. Finalmente retorna la secuenci binaria recibida. 

    ---

    ### Args:
    - `input` - objeto `electrical_signal` con los parámetros de la señal eléctrica de entrada.
    - `eye_obj` - objeto `eye` con los parámetros del diagrama de ojos.

    ---

    ### Returns:
    - `output` - objeto `binary_sequence` con los bits recibidos.
    """

    tic()
    output = input[eye_obj.i::eye_obj.sps] > THRESHOLD_EST(eye_obj)
    
    output.ejecution_time = toc()
    return output


def BER_analizer(mode: Literal['counter', 'estimator'], **kargs):
    """
    ### Descripción:
    Calcula la tasa de error de bits (BER), por conteo de errores (comparando la secuencia recibida con la transmitida) 
    o por estimación (utilizando medias y varianzas estimadas del diagrama de ojo y sustituyendo esos valores en las expresiones teóricas)
    
    ---

    ### Args:
    - `mode` - modo en que se determinará el Bit Error Rate (BER)

    ---

    ### Kargs:
    si `mode='counter'`:
    - `Tx` - secuencia binaria transmitida
    - `Rx` - secuencia binaria recibida
    
    si `mode='estimator'`:
    - `eye_obj` - objeto `eye` con los parámetros estimados del diagrama de ojo

    ### Returns:
    - `float` - BER
    """

    if mode == 'counter':
        assert 'Rx' in kargs.keys() and 'Tx' in kargs.keys(), "Introduzca las secuencias binarias enviada `Tx` y recibida `Rx` como argumentos"
        Rx = kargs['Rx']
        Tx = kargs['Tx']

        if not isinstance(Rx, binary_sequence) and not isinstance(Tx, binary_sequence):
            Rx = binary_sequence( Rx )
            Tx = binary_sequence( Tx )

        Tx = Tx[:Rx.len()]
        assert Tx.len() == Rx.len(), "Error: por alguna razón la secuencia recibida es más larga que la transmitida!"

        return np.sum(Tx.data != Rx.data)/Tx.len()

    elif mode == 'estimator':
        assert 'eye_obj' in kargs.keys(), "Introduzca un objeto `eye` como argumento"

        eye_obj = kargs['eye_obj']

        I1 = eye_obj.mu1
        I0 = eye_obj.mu0
        s1 = eye_obj.s1
        s0 = eye_obj.s0
        um = THRESHOLD_EST(eye_obj)

        return 0.5*(Q((um-I1)/s1) + Q((um-I0)/s0))

    else:
        raise TypeError('Elija entre `counter` o `estimator` e introduzca los argumentos correspondientes en cada caso.')
    


def theory_BER(mu0: Union[int, ndarray], mu1: Union[int, ndarray], s0: Union[int, ndarray], s1: Union[int, ndarray]):
    """
    Esta función calcula la probabilidad de error de bit teórica para un sistema OOK.

    Args:
    - `mu0` - valor de corriente (o tensión) medio de la señal correspondiente a un bit 0
    - `mu1` - valor de corriente (o tensión) medio de la señal correspondiente a un bit 1
    - `s0` - deviación estandar de corriente (o tensión) de la señal correspondiente a un bit 0
    - `s1` - deviación estandar de corriente (o tensión) de la señal correspondiente a un bit 1

    Returns:
    - `BER` - probabilidad de error de bit teórica
    """

    @np.vectorize
    def fun(mu0_,mu1_,s0_,s1_):
        r = np.linspace(mu0_,mu1_,1000)
        return 0.5*np.min(Q((mu1_-r)/s1_) + Q((r-mu0_)/s0_))
                     
    return fun(mu0,mu1,s0,s1)