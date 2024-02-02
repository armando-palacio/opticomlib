"""
===========================================
Utility functions (:mod:`opticomlib.utils`)
===========================================

.. autosummary::
   :toctree: generated/

    generate_prbs          -- Generate a pseudo-random binary sequence (PRBS) of desired order.
    dec2bin                -- Convert a decimal number to its binary representation.
    str2array              -- Convert a string to a numeric array.
    get_time               -- Get the average time of execution of a line of code.
    tic                    -- Start a timer.
    toc                    -- Stop a timer.
    db                     -- Convert a number value to dB.
    dbm                    -- Convert a power value in W to dBm.
    idb                    -- Convert a dB value to a number.
    idbm                   -- Convert a dBm value to a power value in W.
    gaus                   -- Gaussian function.
    Q                      -- Q(x) = 1/2*erfc(x/sqrt(2)) function.

"""

import re
import numpy as np
import timeit, time as tm
import scipy.special as sp

from numpy import ndarray

from scipy.constants import pi

import warnings


def generate_prbs(order: int=None):
    """
    Generates a pseudo-random binary sequence (PRBS) of desired order.

    Args:
        order (int): order of the generator polynomial {7, 9, 11, 15, 20, 23, 31} (default: ``order=7``).

    Returns:
        ndarray: PRBS sequence of length ``2^order-1``.

    Raises:
        ValueError: if ``order`` is not in {7, 9, 11, 15, 20, 23, 31}.
    
    Example:
        >>> generate_prbs(7)
            array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,
            0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
            0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=uint8)

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



def dec2bin(num: int, digits: int=8):
    """
    Converts an integer to its binary representation.

    Args:
        num (int): integer to convert.
        digits (int, Optional): number of bits of the binary representation (default: ``digits=8``).

    Returns:
        binary_sequence: binary representation of ``num`` of length ``digits``.

    Raises:
        ValueError: if ``num`` is too large to be represented with ``digits`` bits.

    Example:
        >>> dec2bin(5, 4)
        array([0, 1, 0, 1], dtype=uint8)
    """

    binary = np.zeros(digits, np.uint8)
    if num > 2**digits-1: raise ValueError(f'The number is too large to be represented with {digits} bits.')
    i = digits - 1
    while num > 0 and i >= 0:
        binary[i] = num % 2
        num //= 2
        i -= 1
    return binary



def str2array(string: str): 
    """
    Converts a string to array of numbers. Use commas or whitespace as element separators.
    Elements can be integer or floating point. If there is at least one floating point number, then all elements are converted to float.

    Args:
        string (str): string to convert.

    Returns:
        ndarray: numeric array
    
    Raises:
        ValueError: if the string contains invalid characters.
    
    Example:
        >>> str2array('1 2 3 4')
        array([1, 2, 3, 4])
        >>> str2array('1,2,3,4')
        array([1, 2, 3, 4])
        >>> str2array('1.1 2.2 3.3 4.4')
        array([1.1, 2.2, 3.3, 4.4])
    """
    
    # Define regex pattern that allows numbers, whitespace, commas and periods.
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



def get_time(line_of_code: str, n:int): 
    """
    Get the average time of execution of a line of code.

    Args:
        line_of_code (str): line of code to execute.
        n (int): number of iterations.

    Returns:
        float: average time of execution, in seconds.

    Example:
        >>> get_time('for i in range(1000): pass', 1000)
        1.1955300000010993e-05
    """
    return timeit.timeit(line_of_code, number=n)/n



def tic(): 
    """
    Start a timer. Create a global variable with the current time.
    Then you can use toc() to get the elapsed time.

    Example:
        >>> tic() # wait some time
        >>> toc()
        2.687533378601074
    """
    global __
    __ = tm.time()

def toc():
    """
    Stop a timer. Get the elapsed time since the last call to tic().
    Returns:
        float: elapsed time, in seconds.

    Example:
        >>> tic() # wait some time
        >>> toc()
        2.687533378601074
    """
    global __
    return tm.time()-__ 



def db(x):
    """ This function calculates the logarithm in base 10 of the input x and multiplies it by 10.
    
    .. math:: db = 10\\log_{10}{x}

    Args:
        x (float | list | tuple | ndarray): input value (``x>=0``).

    Returns:
        float: dB value

    Raise:
        TypeError: if ``x`` is not a `number`, `list`, `tuple` or `ndarray`.
        ValueError: if ``x`` or ``any(x) < 0``.
    
    Example:
        >>> db(1)
        0.0
        >>> db([1,2,3,4])
        array([0.        , 3.01029996, 4.77121255, 6.02059991])
    """
    if not isinstance(x, (int, float, list, ndarray)):
        raise TypeError('The input value must be a number, list, tuple or ndarray.')
    
    x = np.array(x)
    
    if (x<0).any():
        raise ValueError('Some values of input array are negative.')

    warnings.filterwarnings("ignore", category=RuntimeWarning) # to avoid warning when x=0
    return 10*np.log10(x) 


def dbm(x):
    """ This function calculates dBm from Watts.
    
    .. math:: \\text{dbm} = 10\\log_{10}{x}+30

    Args:
        x (float | list | tuple | ndarray): input value (``x>=0``).
    
    Returns:
        float: dBm value.
    
    Raise:
        TypeError: if ``x`` is not a `number`, `list`, `tuple` or `ndarray`.
        ValueError: if ``x`` or ``any(x) < 0``.
    """
    if not isinstance(x, (int, float, list, ndarray)):
        raise TypeError('The input value must be a number, list, tuple or ndarray.')
    
    x = np.array(x)

    if (x<0).any():
        raise ValueError('Some values of input array are negative.')
    
    return 10*np.log10(x*1e3)


def idb(x):
    """ Calculates the number value from a dB value.
    
    .. math:: y = 10^{\\frac{x}{10}}

    Args:
        x (float | list | tuple | ndarray): input value.
    
    Returns:
        float: number value.
    
    Example:
        >>> idb(3)
        1.9952623149688795
        >>> idb([0,3,6,9])
        array([1.        , 1.99526231, 3.98107171, 7.94328235])
    """
    x = np.array(x)
    return 10**(x/10)


def idbm(x):
    """ Calculates the power value in Watts from a dBm value.
    
    .. math:: y = 10^{(\\frac{x}{10}-3)}

    Args:
        x (float | list | tuple | ndarray): input value.

    Returns:
        float: power value in Watts.

    Example:
        >>> idbm(0)
        0.001
        >>> idbm([0,3,6,9])
        array([0.001     , 0.00199526, 0.00398107, 0.00794328])
    """
    x = np.array(x)
    return 10**(x/10-3)


def gaus(x, mu: float=None, std: float=None):
    """ Gaussian function.
    
    .. math:: \\text{gaus}(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}


    Args:
        x (float | list | tuple | ndarray): input value.
        mu (float): mean (default: 0).
        std (float): standard deviation (default: 1).

    Returns:
        float: Gaussian function value.

    Example:
        >>> gaus(0, 0, 1)
        0.3989422804014327
        >>> gaus([0,1,2,3], 0, 1)
        array([0.39894228, 0.24197072, 0.05399097, 0.00443185])
    """
    x = np.array(x)

    if mu is None: mu = 0
    if std is None: std = 1

    return 1/std/(2*pi)**0.5*np.exp(-0.5*(x-mu)**2/std**2)

def Q(x):
    """ Q-function.
    
    .. math:: Q(x) = \\frac{1}{2}\\text{erfc}\\left( \\frac{x}{\\sqrt{2}} \\right) 
    
    Args:
        x (float | list | tuple | ndarray): input value.

    Returns:
        float: Q(x) value.

    Example:
        >>> Q(0)
        0.5
        >>> Q([0,1,2,3])
        array([0.5       , 0.15865525, 0.02275013, 0.0013499 ])
    """
    x = np.array(x)
    return 0.5*sp.erfc(x/2**0.5) 

