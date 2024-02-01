import re
import numpy as np
import timeit, time as tm
import scipy.special as sp

from typing import Literal, Union
from numpy import ndarray

from scipy.constants import pi, k as kB, e
from scipy.integrate import quad


def generate_prbs(order: int=None):
    """
    Esta función genera una secuencia pseudoaleatoria binaria (PRBS) de orden deseado.

    Args:
    - `order` [Opcional] - orden del polinomio generador {7, 9, 11, 15, 20, 23, 31} (default: `order=7`)

    Returns:
    - `prbs` - secuencia PRBS de orden `order` (array de 0s y 1s)
    """
    
    taps = {7: [7,6], 9: [9,5], 11: [11,9], 15: [15,14], 20: [20,3], 23: [23,18], 31: [31,28]}
    if order is None: 
        order = 7
    elif order not in taps.keys():
        raise ValueError(f'`order` must be in {list(taps.keys())}')

    prbs = []
    lfsr = (1<<order)-1
    tap1, tap2 = np.array(taps[order])-1

    while True:
        prbs.append(lfsr&1)
        new = ((lfsr>>tap1)^(lfsr>>tap2))&1
        lfsr = ((lfsr<<1) | new) & (1<<order)-1
        if lfsr == (1<<order)-1:
            break
    return np.array(prbs, np.uint8)



def dec2bin(num: int, digits: int=8) -> np.ndarray:
    """
        Esta función convierte un número entero a su representación binaria.

    Args:
    - `num` - número entero a convertir
    - `digits` [Opcional] - cantidad de bits de la representación binaria (default: `digits=8`)

    Returns:
    - `binary` - representación binaria de `num` (array de 0s y 1s)
    """

    binary = np.zeros(digits, np.uint8)
    if num > 2**digits-1: raise ValueError(f'El número es demasiado grande para ser representado con {digits} bits.')
    i = digits - 1
    while num > 0 and i >= 0:
        binary[i] = num % 2
        num //= 2
        i -= 1
    return binary



def str2array(string: str): 
    """
    Esta función convierte una cadena de caracteres a un array. Usar como separadores de elementos comas o espacios en blancos.
    los elementos pueden ser enteros o de punto flotante.

    Args:
    - `string` - cadena de caracteres a convertir

    Returns:
    - `array` - array numérico
    """
    
    # Definir el patrón regex que permite números, espacios en blanco, comas y puntos
    patron = r'^[0-9, .\s]+$'
    if not re.match(patron, string): 
        raise ValueError('La cadena de caracteres contiene caracteres no permitidos.')
    
    if '.' in string:
        type = float
    else:
        type = int
    
    string = re.split(r'[,\s]+', string)
    # delete empty strings
    string = [x for x in string if x and x!=' ' and x!=',']

    return np.array(string).astype(type)



def get_time(line_of_code: str, n:int): return timeit.timeit(line_of_code, number=n)/n



def tic(): global __; __ = tm.time()
def toc(): global __; return tm.time()-__ 



def db(x):
    """ db = 10*log10(x)

    This function calculates the logarithm in base 10 of the input x and multiplies it by 10.

    Args:
        x (float | list | tuple | ndarray): input value (x>=0)

    Returns:
        float: dB value

    Raise:
        TypeError: if x is not a number, list, tuple or ndarray
        ValueError: if x or any(x) < 0
    """
    if not isinstance(x, (int, float, list, ndarray)):
        raise TypeError('The input value must be a number, list, tuple or ndarray.')
    
    x = np.array(x)
    
    if (x<0).any():
        raise ValueError('Some values of input array are negative.')

    return 10*np.log10(x) 


def dbm(x):
    """ dbm = 10*log10(x*1e3)

    This function calculates dBm from Watts.

    Args:
        x (float | list | tuple | ndarray): input value (x>=0)
    
    Returns:
        float: dBm value
    
    Raise:
        TypeError: if x is not a number, list, tuple or ndarray
        ValueError: if x or any(x) < 0
    """
    if not isinstance(x, (int, float, list, ndarray)):
        raise TypeError('The input value must be a number, list, tuple or ndarray.')
    
    x = np.array(x)

    if (x<0).any():
        raise ValueError('Some values of input array are negative.')
    
    return 10*np.log10(x*1e3)


idb = lambda x: 10**(x/10) # función de conversión a veces

idbm = lambda x: 10**(x/10-3) # función de conversión a Watts

gaus = lambda x,mu,std: 1/std/(2*pi)**0.5*np.exp(-0.5*(x-mu)**2/std**2) # función gaussiana

Q = lambda x: 0.5*sp.erfc(x/2**0.5) # función Q



def theory_BER(mu1: Union[int, ndarray], s0: Union[int, ndarray], s1: Union[int, ndarray], modulation: str='OOK', M: int=None, kind: Literal['soft','hard']='soft'):
    """
        Esta función calcula la probabilidad de error de bit teórica para un sistema de comunicaciones ópticas y una modulación dada.

    Args:
    - `mu1` - valor de corriente (o tensión) medio de la señal correspondiente a un bit 1
    - `s0` - deviación estandar de corriente (o tensión) de la señal correspondiente a un bit 0
    - `s1` - deviación estandar de corriente (o tensión) de la señal correspondiente a un bit 1
    - `modulation` [Opcional] - modulación utilizada (default: `modulation='OOK'`)
    - `M` [Opcional] - orden de la modulación PPM (default: `M=None`). Se debe especificar si `modulation='PPM'`
    - `kind` [Opcional] - tipo de decodificación PPM (default: `kind='soft'`). Se debe especificar si `modulation='PPM'`

    Returns:
    - `BER` - probabilidad de error de bit teórica
    """

    if modulation == 'OOK':
        def fun(mu1_,s0_,s1_):
            r = np.linspace(0,mu1_,1000)
            return 0.5*np.min(Q((mu1_-r)/s1_) + Q(r/s0_))
        fun = np.vectorize( fun )
        return fun(mu1,s0,s1)

    if modulation == 'PPM':
        if M is None: raise ValueError('`M` must be specified for PPM modulation.')

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
    
    # Others modulations
    else:
        raise ValueError('`modulation` must be `OOK` or `PPM`. More modulations will be added in the future.')
    


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    P = np.linspace(-25,-15,1000)
    p = idbm(P)
    
    B = 5e9 # Hz

    mu1 = p
    s0 = (4 * kB * 300 * B / 50)**0.5
    s1 = (4 * kB * 300 * B / 50 + 2 * e * mu1 * B)**0.5

    plt.semilogy(dbm(p/2), theory_BER(mu1,s0,s1, 'OOK'), '-r', lw=2, label='OOK')
    
    for M, c in zip([2,4,8,16], ['C0', 'C1', 'C2', 'C3']):
        plt.semilogy(dbm(p/M), theory_BER(mu1,s0,s1, 'PPM', M, 'soft'), f'--{c}', label=f'{M}-PPM - soft')
        plt.semilogy(dbm(p/M), theory_BER(mu1,s0,s1, 'PPM', M, 'hard'), f'{c}', label=f'{M}-PPM - hard')
    
    plt.legend()
    plt.ylim(1e-9,0.5)
    plt.grid()
    plt.xlabel('Potencia media (dBm)')
    plt.ylabel('BER')
    plt.show()
