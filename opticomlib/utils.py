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
    bode                   -- Plot the Bode plot of a given transfer function H (magnitud, phase, group delay and dispersion).
    rcos                   -- Raised cosine function.
    si                     -- Unit of measure classifier.

"""

import re
import numpy as np
import timeit, time as tm
import scipy.special as sp
import scipy.signal as sg

from numpy import ndarray
from typing import Literal

from scipy.constants import pi, c

import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift

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


def bode(H: ndarray, 
         fs: float, 
         f0: float=None, 
         xaxis: Literal['f','w','lambda']='f', 
         disp: bool=False,
         ret: bool=False, 
         show_: bool=True, 
         style: Literal['dark', 'light']='dark'):
    """Plot the Bode plot of a given transfer function H (magnitud, phase and group delay).

    Args:
        H (ndarray): The transfer function.
        fs (float): The sampling frequency.
        f0 (float, Optional): The center frequency. Defaults to None. If not None, dispersion are also plotted.
        xaxis (str, Optional): The x-axis (frequecy, angular velocity, wavelength). Defaults to ``'f'``.
        disp (bool, Optional): Whether to plot the dispersion. Defaults to False.
        ret (bool, Optional): Whether to return the plotted data. Defaults to False.
        show_ (bool, Optional): Whether to display the plot. Defaults to True.
        style (str, Optional): The plot style. Defaults to ``'dark'``.

    Returns:
        tuple[ndarray, ndarray, ndarray]: A tuple containing the frequency, magnitude, phase, and group delay if ``ret=True``.

    Raises:
        ValueError: if ``style`` is not "dark" or "light".

    Example:
        >>> from opticomlib bode
        >>> H, phase, tau_g = bode(H, fs, ret=True, show_=False)
    """
    
    f = fftshift(fftfreq(H.size, d=1/fs))
    
    if xaxis == 'f':
        x = f*1e-9
        xlabel = 'Frequency [GHz]'
    elif xaxis == 'w':
        x = 2*pi*f*1e-9
        xlabel = r'$\omega$ [Grad/s]'
    elif xaxis == 'lambda' and f0:
        x = c/(f + f0)*1e9
        xlabel = r'$\lambda$ [nm]'

    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'light':
        plt.style.use('default')
    else:
        raise ValueError('`style` must be "dark" or "light".')

    nplots = 4 if disp and f0 else 3

    _, axs = plt.subplots(nplots, 1, figsize=(8, 6), sharex=True, gridspec_kw={'hspace': 0.02})
    plt.suptitle('Frequency Response')
    
    axs[0].plot(x, np.abs(H), 'r', lw=2)
    axs[0].set_ylabel(r'$|H(\omega)|$', rotation=0, labelpad=20)
    axs[0].grid(alpha=0.3)
    axs[0].yaxis.set_label_position("left")
    axs[0].yaxis.tick_right()
    axs[0].set_ylim(-0.1,1.1)
    

    phase = np.unwrap(np.angle(H))
    axs[1].plot(x, phase, 'b', lw=2)
    axs[1].set_ylabel(r'$\phi$ [rad]', rotation=0, labelpad=20)
    axs[1].grid(alpha=0.3)
    axs[1].yaxis.set_label_position("left")
    axs[1].yaxis.tick_right()

    dw = 2*pi*fs/H.size
    tau_g = sg.medfilt(np.diff(phase)/dw, 5)
    axs[2].plot(x[:-1], tau_g*1e12, 'g', lw=2)
    axs[2].set_ylabel(r'$\tau_g$ [ps]', rotation=0, labelpad=20)
    axs[2].grid(alpha=0.3)
    axs[2].yaxis.set_label_position("left")
    axs[2].yaxis.tick_right()

    if disp and f0: 
        landa = c/(f + c/f0)
        D = sg.medfilt(-2*pi*c/landa**2 * np.diff(phase, 2, append=phase[-2:])/dw**2, 5)
        axs[3].plot(x, D*1e3, 'm', lw=2)
        axs[3].set_ylabel(r'D [ps/nm]', rotation=0, labelpad=28)
        axs[3].set_xlabel(xlabel)
        axs[3].grid(alpha=0.3)
        axs[3].yaxis.set_label_position("left")
        axs[3].yaxis.tick_right()
    else:
        axs[2].set_xlabel(xlabel)

    plt.style.use('default')

    if show_:
        plt.show()
    
    if ret:
        return H, phase, tau_g


def rcos(x, alpha, T):
    """
    Raised cosine function.

    Args:
        x (ndarray): input values.
        alpha (float): roll-off factor.
        T (float): symbol period.

    Returns:
        ndarray: raised cosine function.

    References:
        https://en.wikipedia.org/wiki/Raised-cosine_filter

    Example:
        >>> x = np.linspace(-1, 1, 64)
        >>> H = rcos(x, alpha=1, T=1)
    """
    first_condition = np.abs(x) <= (1-alpha)/(2*T)
    second_condition = (np.abs(x)>(1-alpha)/(2*T)) & (np.abs(x)<=(1+alpha)/(2*T))
    third_condition = np.abs(x) > (1+alpha)/(2*T)

    if not isinstance(x, ndarray):
        return 1 if first_condition else 0 if third_condition else 0.5*(1+np.cos(pi*T/alpha*(np.abs(x)-(1-alpha)/(2*T))))
    
    H = np.zeros(len(x))

    H[ first_condition ] = 1
    H[ second_condition ] = 0.5*(1+np.cos(pi*T/alpha*(np.abs(x[second_condition])-(1-alpha)/(2*T))))

    return H


def si(x, unit: Literal['m','s']='s', k: int=1):
    """ Unit of measure classifier 
    
    Args:
        x (int | float): number

    Return:
        str: string with number and unit

    Example:
        >>> si(0.002, 's')
        >>> 2.0 ms
    """
    if 1e12<= x:
        return f'{x*1e-9:.{k}f} T{unit}' 
    if 1e9<= x <1e12:
        return f'{x*1e-9:.{k}f} G{unit}' 
    if 1e6<= x <1e9:
        return f'{x*1e-6:.{k}f} M{unit}' 
    if 1e3<= x <1e6:
        return f'{x*1e-3:.{k}f} k{unit}' 
    if 1<= x <1e3:
        return f'{x:.{k}f} {unit}' 
    if 1e-3<= x <1:
        return f'{x*1e3:.{k}f} m{unit}' 
    if 1e-6<= x <1e-3:
        return f'{x*1e6:.{k}f} μ{unit}'
    if 1e-9<= x <1e-6:
        return f'{x*1e9:.{k}f} n{unit}'
    if 1e-12<= x <1e-9:
        return f'{x*1e12:.{k}f} p{unit}'
