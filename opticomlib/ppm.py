"""
.. rubric:: Functions
.. autosummary::

   PPM_ENCODER           -- Pulse position modulation encoder
   PPM_DECODER           -- Pulse position modulation decoder
   HDD                   -- Hard-decision decoder model 
   SDD                   -- Soft-decision decoder model 
   THRESHOLD_EST         -- Threshold for detection
   DSP                   -- Digital signal processing for PPM systems
   BER_analizer          -- Bit error rate analizer
   theory_BER            -- Theoretical bit error rate
"""

import numpy as np
from typing import Literal, Union
from numpy import ndarray
from scipy.integrate import quad
from scipy.constants import pi

from .typing import binary_sequence, electrical_signal, eye
from .utils import tic, toc, str2array, dec2bin, Q



def PPM_ENCODER(input: Union[str, list, tuple, ndarray, binary_sequence], M: int) -> binary_sequence:
    """
    ### Descripción:
    Codificador digital PPM. Convierte una secuencia binaria de entrada en una secuencia binaria codificada en PPM. 

    ---
    
    ### Args:
    - `input` - secuencia binaria de entrada
    - `M` - cantidad de slots que contiene un símbolo

    ---
    
    ### Returns:
    - `binary_sequence`
    """
    tic()

    if isinstance(input, binary_sequence):
        input = input.data
    elif isinstance(input, str):
        input = str2array(input)
    elif isinstance(input, (list, tuple)):
        input = np.array(input)
    else:
        raise TypeError("El argumento `input` debe ser del tipo (str, list, tuple, ndarray, binary_sequence).")

    k = int(np.log2(M))

    input = input[:len(input)//k*k] # truncamos la secuencia de bits a un múltiplo de k

    decimal = np.sum(input.reshape(-1,k)*2**np.arange(k)[::-1], axis=-1) # convertimos los símbolos a decimal
    ppm_s = np.zeros(decimal.size*M, dtype=np.uint8)

    ppm_s[np.arange(decimal.size)*M + decimal] = 1 # codificamos los símbolos en PPM
   
    output = binary_sequence(ppm_s) 
    output.ejecution_time = toc()
    return output



def PPM_DECODER(input: Union[str, list, tuple, np.ndarray, binary_sequence], M: int) -> binary_sequence:
    """
    ### Descripción:
    Recibe una secuencia de bits codificada en PPM y la decodifica.
    
    ---

    ### Args:
    - `input` - secuencia binaria codificada en PPM
    - `M` - orden de modulación PPM

    ### Returns:
    - `binary_sequence` - secuencia binaria decodificada
    """
    tic()

    if isinstance(input, binary_sequence):
        input = input.data
    elif isinstance(input, str):
        input = str2array(input)
    elif isinstance(input, (list, tuple)):
        input = np.array(input)
    else:
        raise TypeError("El argumento `input` debe ser del tipo (str, list, tuple, ndarray, binary_sequence).")
    
    k = int(np.log2(M))

    decimal = np.where(input==1)[0]%M # obtenemos el decimal de cada símbolo

    output = np.array(list(map(lambda x: dec2bin(x,k), decimal))).ravel() # convertimos a binario cada decimal
    output= binary_sequence(output)

    output.ejecution_time = toc()
    return output


def HDD(input: binary_sequence, M: int) -> binary_sequence:
    """
    ### Descripción:
    Estima los símbolos PPM más probables a partir de la secuencia binaria dada como entrada.
    
    ---

    ### Args:
    - `input` - secuencia binaria a estimar

    ### Returns:
    - `binary_sequence` - secuencia de símbolos estimados
    """
    tic()

    n_simb = int(input.len()/M)

    s = np.sum(input.data.reshape(n_simb, M), axis=-1)

    output = np.array(input.data, dtype=np.uint8)

    for i in np.where(s==0)[0]: # si existe algún símbolo sin ningún slot encendidos, se prende uno al azar
        output[i*M + np.random.randint(M)] = 1

    for i in np.where(s>1)[0]: # si existe algún símbolo con más de 1 slot encendido, se elige uno de ellos al azar)
        j = np.where(output[i*M:(i+1)*M]==1)[0]
        output[i*M:(i+1)*M] = 0
        output[i*M + np.random.choice(j)]=1

    output = binary_sequence(output)
    output.ejecution_time = toc()
    return output



def SDD(input: electrical_signal, M: int) -> binary_sequence:
    """
    ### Descripción:
    Estima los símbolos PPM más probables a partir de la señal eléctrica dada como entrada.
    
    ---

    ### Args:
    - `input` - señal eléctrica sin muestrear

    ### Returns:
    - `binary_sequence` - secuencia de símbolos estimados
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
    """
    ### Descripción: 
    Esta función estima el umbral de decisión para M-PPM a partir de las medias y desviaciones estándar.

    ---

    ### Args:
    - `eye_obj` - objeto `eye` con los parámetros del diagrama de ojos
    - `M` - orden PPM
    """

    mu0 = eye_obj.mu0
    mu1 = eye_obj.mu1
    s0 = eye_obj.s0
    s1 = eye_obj.s1

    # obtenemos el umbral de decisión para PPM
    r = np.linspace(mu0, mu1, 1000)
    umbral = r[np.argmin(1 - Q((r-mu1)/s1) * (1-Q((r-mu0)/s0))**(M-1))]
    return umbral



def DSP(input: electrical_signal, eye_obj: eye, M :int, decision: Literal['hard','soft']='hard') -> binary_sequence:
    """
    ### Descripción:
    Este componente realiza todas las tareas de decisión y decodificación de la señal eléctrica photodetectada. 
    
    Si se selecciona la decisión dura, se realiza el submuestreo de la señal digital a 1 muestra por slot en el instante óptimo 
    determinado por el objeto `eye`, luego se realiza la comparación con un umbral de decisión estimado por `eye`, se decide que 
    símbolo PPM se transmitió y se decodifica la secuencia binaria.

    Si se selecciona la decisión blanda, se realiza la integración de cada slot de la señal digital, se decide que símbolo PPM tiene
    mayor verosimilitud y se decodifica la secuencia binaria.
    
    ---

    ### Args:
    - `input` - secuencia binaria codificada en PPM
    - `eye_obj` [Opcional] - objeto eye con los parámetros del diagrama de ojos (default: `_eye_=eye()`)
    - `decision` [Opcional] - tipo de decisión a realizar (default: `decision='hard'`)

    ### Returns:
    - `binary_sequence` - secuencia binaria decodificada
    """
    
    if decision == 'hard':
        output = input[eye_obj.i::eye_obj.sps] > THRESHOLD_EST(eye_obj, M)
        simbols = HDD(output, M); simbols.ejecution_time += output.ejecution_time
    elif decision == 'soft':
        simbols = SDD(input, M)
    else:
        raise TypeError('No existe el tipo de decisión seleccionada!!')

    output = PPM_DECODER(simbols, M) 

    output.ejecution_time += simbols.ejecution_time 
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
    - `M` - orden de modulación PPM
    - `decision` [Opcional] - tipo de decision `'hard'` o `'soft'` (default: `decision='soft'`)

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
        assert 'M' in kargs.keys(), "Introduzca el orden de modulación `M` como argumento"

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
            raise TypeError('decision debe ser "hard" o "soft"!!') 
        return M/2/(M-1)*Pe_sym

    else:
        raise TypeError('Elija entre `counter` o `estimator` e introduzca los argumentos correspondientes en cada caso.')


def theory_BER(mu1: Union[int, ndarray], s0: Union[int, ndarray], s1: Union[int, ndarray], M: int, kind: Literal['soft','hard']='soft'):
    """
        Esta función calcula la probabilidad de error de bit teórica para un sistema PPM.

    Args:
    - `mu1` - valor de corriente (o tensión) medio de la señal correspondiente a un bit 1
    - `s0` - deviación estandar de corriente (o tensión) de la señal correspondiente a un bit 0
    - `s1` - deviación estandar de corriente (o tensión) de la señal correspondiente a un bit 1
    - `M` - orden de la modulación PPM. 
    - `kind` [Opcional] - tipo de decodificación PPM (default: `kind='soft'`). Se debe especificar si `modulation='PPM'`

    Returns:
    - `BER` - probabilidad de error de bit teórica
    """

    if kind == 'soft':
        fun = np.vectorize( lambda mu1,s0,s1,M: 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((mu1+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0] )
    elif kind == 'hard':
        def fun(mu1_,s0_,s1_,M_):
            r = np.linspace(0,mu1_,1000)
            return np.min(1 - Q((r-mu1_)/s1_) * (1-Q((r)/s0_))**(M_-1))
        fun = np.vectorize( fun )
    else:
        raise ValueError('`kind` must be `soft` or `hard`.')
    return fun(mu1,s0,s1,M)*0.5*M/(M-1)


if __name__ == '__main__':
    print(BER_analizer('counter', Tx='0 1 0 1', Rx='0,1,0,0'))
