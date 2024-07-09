"""
.. rubric:: Functions
.. autosummary::
     
    dec2bin                
    str2array              
    get_time              
    tic                    
    toc                    
    db                     
    dbm                    
    idb                    
    idbm                   
    gaus                   
    Q                      
    phase                  
    tau_g                  
    dispersion             
    bode                   
    rcos                   
    si                     
    norm                   
    nearest  
    theory_BER               
"""

import re
import numpy as np
import timeit, time as tm
import scipy.special as sp # type: ignore
import scipy.signal as sg # type: ignore

from typing import Literal, Union

from scipy.constants import pi, c # type: ignore

import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift

import warnings

Array_Like = (list, tuple, np.ndarray)
Number = (int, float)



def dec2bin(num: int, digits: int=8):
    r"""
    Converts an integer to its binary representation.

    Parameters
    ----------
    num : int
        Integer to convert.
    digits : int, default: 8
        Number of bits of the binary representation.

    Returns
    -------
    binary_sequence
        Binary representation of ``num`` of length ``digits``.

    Raises
    ------
    ValueError
        If ``num`` is too large to be represented with ``digits`` bits.

    Example
    -------
    .. code-block:: python

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

def _get_type_array_from_str(string):
    # just 0 and 1 without +/- --> bool
    if re.match(r'^[0-1,;\s]+$', string):
        return bool
    # just numbers with +/- and without dots --> int
    if re.match(r'^[0-9,;\-\+\s]+$', string):
        return int
    # just numbers with +/- and dots --> float
    if re.match(r'^[0-9,;.\+\-\s]+$', string): 
        return float
     # just numbers with +/-, dots and j --> complex
    if re.match(r'^[0-9,;.\+\-\sji]+$', string):
        return complex
    return None

def str2array(string: str, dtype: bool | int | float | complex | None = None): 
    r"""
    Converts a string to array of numbers. Use comma (``,``) or whitespace (`` ``) as element separators and semicolon (``;``) as row separator.
    Also, ``i`` or ``j`` can be used to represent the imaginary unit.

    Parameters
    ----------
    string : :obj:`str`
        String to convert.
    dtype : :obj:`type`, optional
        Data type of the output array. 
        If ``dtype`` is not given, the data type is determined from the input string.
        If ``dtype`` is given, the data output is cast to the given type.
        Allowed values are ``bool``, ``int``, ``float`` and ``complex``.

    Returns
    -------
    arr : :obj:`np.ndarray`
        Numeric array.

    Raises
    ------
    ValueError
        If the string contains invalid characters.

    Example
    -------
    For binary numbers, string must contain only 0 and 1. 
    Only in this case, sequence don't need to be separated by commas or spaces although it is allowed.

    >>> str2array('101')
    array([ True, False, True])
    >>> str2array('1 0 1; 0 1 0')
    array([[ True, False,  True],
           [False,  True, False]])

    Special case 
    >>> str2array('1 0 1 10')
    array([True, False, True, True, False])
    >>> str2array('1 0 1 10', dtype=int)
    array([ 1,  0,  1, 10])
    >>> str2array('1 0 1 10', dtype=float)
    array([ 1.,  0.,  1., 10.])
    >>> str2array('1 0 1 10', dtype=complex)
    array([ 1.+0.j,  0.+0.j,  1.+0.j, 10.+0.j])

    For integer and float numbers
    >>> str2array('1 2 3 4')
    array([1, 2, 3, 4])
    >>> str2array('1.1 2.2 3.3 4.4')
    array([1.1, 2.2, 3.3, 4.4])
    
    For complex numbers
    >>> str2array('1+2j 3-4i')
    array([1.+2.j, 3.-4.j])
    """
    _dtype = _get_type_array_from_str(string)
    
    if _dtype == bool:
        # for special cases when string = '10 100 1000' and dtype = int | float | complex 
        if dtype == int or dtype==float or dtype==complex: 
            strings = string.split(';')
            if len(strings) == 1:
                arr = np.array(re.split(r'[,\s]+', strings[0].strip()), dtype=dtype)
            else:
                arr = np.array([re.split(r'[,\s]+', item.strip()) for item in strings], dtype=dtype)
        else:
            strings = string.replace(' ', '').replace(',', '').split(';')
            if len(strings) == 1:
                arr = np.array(list(strings[0])).astype(_dtype)
            else:
                arr = np.array([list(item) for item in strings]).astype(_dtype)

    elif _dtype == int or _dtype==float:
        strings = string.split(';')
        if len(strings) == 1:
            arr = np.array(re.split(r'[,\s]+', strings[0].strip()), dtype=_dtype)
        else:
            arr = np.array([re.split(r'[,\s]+', item.strip()) for item in strings], dtype=_dtype)

    elif _dtype == complex:
        strings = string.replace('i','j').split(';')
        if len(strings) == 1:
            arr = np.array(re.split(r'[,\s]+', strings[0].strip()), dtype=_dtype)
        else:
            arr = np.array([re.split(r'[,\s]+', item.strip()) for item in strings], dtype=_dtype)

    else:
        raise ValueError('The string contains invalid characters and can\'t be converted to an array.')
    
    return arr.astype(dtype) if dtype else arr



def get_time(line_of_code: str, n:int): 
    r"""
    Get the average time of execution of a line of code.

    Parameters
    ----------
    line_of_code : str
        Line of code to execute.
    n : int
        Number of iterations.

    Returns
    -------
    time : :obj:`float`
        Average time of execution, in seconds.

    Example
    -------
    .. code-block:: python

        >>> get_time('for i in range(1000): pass', 1000)
        1.1955300000010993e-05
    """
    return timeit.timeit(line_of_code, number=n)/n



def tic(): 
    r"""
    Start a timer. Create a global variable with the current time.
    Then you can use toc() to get the elapsed time.

    Example
    -------
    .. code-block:: python

        >>> tic() # wait some time
        >>> toc()
        2.687533378601074
    """
    global __
    __ = tm.time()

def toc():
    r"""Stop a timer. Get the elapsed time since the last call to tic().

    Returns
    -------
    time : :obj:`float`
        Elapsed time, in seconds.

    Example
    -------
    .. code-block:: python

        >>> tic() # wait some time
        >>> toc()
        2.687533378601074
    """
    global __
    return tm.time()-__ 



def db(x):
    r"""Calculates the logarithm in base 10 of the input x and multiplies it by 10.

    .. math:: \text{db} = 10\log_{10}{x}

    Parameters
    ----------
    x : Number or Array_Like
        Input value (``x>=0``).

    Returns
    -------
    out : :obj:`float` or :obj:`np.ndarray`
        dB value.

    Raises
    ------
    TypeError
        If ``x`` is not a `number`, `list`, `tuple` or `ndarray`.
    ValueError
        If ``x`` or ``any(x) < 0``.

    Example
    -------
    .. code-block:: python

        >>> db(1)
        0.0
        >>> db([1,2,3,4])
        array([0.        , 3.01029996, 4.77121255, 6.02059991])
    """
    if not isinstance(x, (Number + Array_Like)):
        raise TypeError('The input value must be a number, list, tuple or ndarray.')
    
    x = np.array(x)
    
    if (x<0).any():
        raise ValueError('Some values of input array are negative.')

    warnings.filterwarnings("ignore", category=RuntimeWarning) # to avoid warning when x=0
    return 10*np.log10(x) 


def dbm(x):
    r"""Calculates dBm from Watts.

    .. math:: \text{dbm} = 10\log_{10}{x}+30

    Parameters
    ----------
    x : Number or Array_Like
        Input value (``x>=0``).

    Returns
    -------
    out : :obj:`float` or :obj:`np.ndarray`
        dBm value. If ``x`` is a number, then the output is a :obj:`float`. If ``x`` is an array_like, then the output is an :obj:`np.ndarray`.

    Raises
    ------
    TypeError
        If ``x`` is not a `number`, `list`, `tuple` or `ndarray`.
    ValueError
        If ``x`` or ``any(x) < 0``.

    Example
    -------
    .. code-block:: python

        >>> dbm(1)
        30.0
        >>> dbm([1,2,3,4])
        array([30.        , 33.01029996, 34.77121255, 36.02059991])
    """
    if not isinstance(x, (Number + Array_Like)):
        raise TypeError('The input value must be a number, list, tuple or ndarray.')
    
    x = np.array(x)

    if (x<0).any():
        raise ValueError('Some values of input array are negative.')
    
    return 10*np.log10(x*1e3)


def idb(x):
    r"""Calculates the number value from a dB value.

    .. math:: y = 10^{\frac{x}{10}}

    Parameters
    ----------
    x : Number or Array_Like
        Input value.

    Returns
    -------
    out : :obj:`float` or :obj:`np.ndarray`
        Number value. If ``x`` is a number, then the output is a :obj:`float`. If ``x`` is an array_like, then the output is an :obj:`np.ndarray`.

    Example
    -------
    .. code-block:: python

        >>> idb(3)
        1.9952623149688795
        >>> idb([0,3,6,9])
        array([1.        , 1.99526231, 3.98107171, 7.94328235])
    """
    x = np.array(x)
    return 10**(x/10)


def idbm(x):
    r"""Calculates the power value in Watts from a dBm value.

    .. math:: y = 10^{(\frac{x}{10}-3)}

    Parameters
    ----------
    x : Number or Array_Like
        Input value.

    Returns
    -------
    out : :obj:`float` or :obj:`np.ndarray`
        Power value in Watts. If ``x`` is a number, then the output is a :obj:`float`. If ``x`` is an array_like, then the output is an :obj:`np.ndarray`.

    Example
    -------
    .. code-block:: python

        >>> idbm(0)
        0.001
        >>> idbm([0,3,6,9])
        array([0.001     , 0.00199526, 0.00398107, 0.00794328])
    """
    x = np.array(x)
    return 10**(x/10-3)


def gaus(x, mu: float=None, std: float=None):
    r"""Gaussian function.

    .. math:: \text{gaus}(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    Parameters
    ----------
    x : Number or Array_Like
        Input value.
    mu : :obj:`float`, default: 0
        Mean.
    std : :obj:`float`, default: 1
        Standard deviation.

    Returns
    -------
    out : :obj:`float` or :obj:`np.ndarray`
        Gaussian function value. If ``x`` is a number, then the output is a :obj:`float`. If ``x`` is an array_like, then the output is an :obj:`np.ndarray`.

    Examples
    --------
    .. code-block:: python

        >>> gaus(0, 0, 1)
        0.3989422804014327
        >>> gaus([0,1,2,3], 0, 1)
        array([0.39894228, 0.24197072, 0.05399097, 0.00443185])
    
    .. plot:: 
        :include-source:
        :alt: Gaussian function
        :align: center

        from opticomlib import gaus
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(-5, 5, 1000)
        y = gaus(x, 0, 1)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'r', lw=2)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.grid(alpha=0.3)
        plt.show()
    """
    x = np.array(x)

    if mu is None: mu = 0
    if std is None: std = 1

    return 1/std/(2*pi)**0.5*np.exp(-0.5*(x-mu)**2/std**2)

def Q(x):
    r"""
    Q-function.

    .. math:: Q(x) = \frac{1}{2}\text{erfc}\left( \frac{x}{\sqrt{2}} \right)

    Parameters
    ----------
    x : Numper or Array_Like
        Input value.

    Returns
    -------
    out : :obj:`float` or :obj:`np.ndarray`
        Q(x) values. If ``x`` is a number, then the output is a :obj:`float`. If ``x`` is an array_like, then the output is an :obj:`np.ndarray`.

    Examples
    --------
    .. code-block:: python

        >>> Q(0)
        0.5
        >>> Q([0,1,2,3])
        array([0.5       , 0.15865525, 0.02275013, 0.0013499 ])
    
    .. plot:: 
        :include-source:
        :alt: Gaussian function
        :align: center

        from opticomlib import Q
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(-5, 5, 1000)

        plt.figure(figsize=(8, 5))
        plt.plot(x, Q(x), 'r', lw=3, label='Q(x)')
        plt.plot(x, Q(-x), 'b', lw=3, label='Q(-x)')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.legend()
        plt.grid()
        plt.show()
    """
    x = np.array(x)
    return 0.5*sp.erfc(x/2**0.5) 


def phase(H: np.ndarray):
    r"""
    Calculate the unwrapped phase of a frequency response.

    Parameters
    ----------
    H : :obj:`np.ndarray`
        Frequency response of a system.

    Returns
    -------
    phase : :obj:`np.ndarray`
        Unwrapped phase in radians.

    Examples
    --------
    .. plot:: 
        :include-source:
        :alt: Gaussian function
        :align: center

        from opticomlib import phase
        import matplotlib.pyplot as plt
        import numpy as np

        t = np.linspace(-5, 5, 1000)
        y = np.exp(1j*t**2) 
        phi = phase(y)

        plt.figure(figsize=(8, 5))
        plt.plot(t, phi, 'r', lw=2)
        plt.ylabel('phase [rad]')
        plt.xlabel('t')
        plt.grid(alpha=0.3)
        plt.show()
    """
    return np.unwrap(np.angle(H))

def tau_g(H: np.ndarray, fs: float):
    r"""
    Calculate the group delay of a frequency response.

    Parameters
    ----------
    H : :obj:`np.ndarray`
        Frequency response of a system.
    fs : :obj:`float`
        Sampling frequency of the system.

    Returns
    -------
    tau: :obj:`np.ndarray`
        Group delay of the system, in [ps].
    
    Examples
    --------
    .. plot:: 
        :include-source:
        :alt: Gaussian function
        :align: center

        from opticomlib import tau_g
        import matplotlib.pyplot as plt
        import numpy as np

        t = np.linspace(-5, 5, 1000)
        y = np.exp(1j*t**2) 
        phi = tau_g(y, 1e2)

        plt.figure(figsize=(8, 5))
        plt.plot(t[:-1], phi, 'r', lw=2)
        plt.ylabel(r'$\tau_g$ [ps]')
        plt.xlabel('t')
        plt.grid(alpha=0.3)
        plt.show()
    """
    dw = 2*pi*fs/H.size
    return np.diff(phase(H))/dw * 1e12

def dispersion(H: np.ndarray, fs: float, f0: float):
    """
    Calculate the dispersion of a frequency response.

    Parameters
    ----------
    H : :obj:`np.ndarray`
        Frequency response of a system.
    fs : :obj:`float`
        Sampling frequency of the system.
    f0 : :obj:`float`
        Center frequency of the system.

    Returns
    -------
    D : :obj:`np.ndarray`
        Cumulative dispersion of the system, in [ps/nm].
    """
    f = fftshift(fftfreq(H.size, d=1/fs))
    dλ = np.diff(c/(f+f0))[0]*1e9
    D = np.diff(tau_g(H, fs))/dλ
    return D



def bode(H: np.ndarray, 
         fs: float, 
         f0: float=None, 
         xaxis: Literal['f','w','lambda']='f', 
         disp: bool=False,
         yscale : Literal['linear', 'db']='linear',
         ret: bool=False, 
         retAxes: bool=False,
         show_: bool=True, 
         style: Literal['dark', 'light']='dark',
         xlim: tuple=None):
    r"""
    Plot the Bode plot of a given transfer function H (magnitude, phase and group delay).

    Parameters
    ----------
    H : :obj:`np.ndarray`
        The transfer function.
    fs : :obj:`float`
        The sampling frequency.
    f0 : :obj:`float`, default: None
        The center frequency. If not None, dispersion are also plotted.
    xaxis : :obj:`str`, default: 'f'
        The x-axis (frequency, angular velocity, wavelength).
    disp : :obj:`bool`, default: False
        Whether to plot the dispersion.
    ret : :obj:`bool`, default: False
        Whether to return the plotted data.
    show_ : :obj:`bool`, default: True
        Whether to display the plot.
    style : :obj:`str`, default: 'dark'
        The plot style.

    Returns
    -------
    (f, H, phase, tau_g) : :obj:`np.ndarray`
        A tuple containing the frequency, magnitude, phase, and group delay if ``ret=True``.

    Raises
    ------
    ValueError
        If style is not "dark" or "light".

    Example
    -------
    .. code-block:: python

        >>> from opticomlib import bode
        >>> H, phase, tau_g = bode(H, fs, ret=True, show_=False)
    """
    if not isinstance(H, np.ndarray):
        raise ValueError('`H` must be a numpy.ndarray.')
    
    f = fftshift(fftfreq(H.size, d=1/fs))
    w = 2*pi*f
    
    if xaxis == 'f':
        x = f*1e-9
        xlabel = 'Frequency [GHz]'
    elif xaxis == 'w':
        x = w*1e-9
        xlabel = r'$\omega$ [Grad/s]'
    elif xaxis == 'lambda':
        if not f0:
            raise ValueError('`f0` must be specify for determine lambda vector.')
        x = (c/(f+f0) - c/f0)*1e9
        xlabel = r'$\lambda$ [nm]'

    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'light':
        plt.style.use('default')
    else:
        raise ValueError('`style` must be "dark" or "light".')

    nplots = 4 if disp and f0 else 3

    if yscale=='db':
        y = db(np.abs(H)**2)
        ylabel = r'$|H(\omega)|^2$ [dB]'
        ylim = (-60, 1)
        pad = 32
    elif yscale=='linear':
        y = np.abs(H)**2
        ylabel = r'$|H(\omega)|^2$'
        ylim = (-0.1, 1.1)
        pad = 20
    else:
        raise ValueError('`yscale` must be "linear" or "db".')

    _, axs = plt.subplots(nplots, 1, figsize=(8, 6), sharex=True, gridspec_kw={'hspace': 0.02})
    plt.suptitle('Frequency Response')
    
    axs[0].plot(x, y, 'r', lw=2)
    axs[0].set_ylabel(ylabel, rotation=0, labelpad=pad)
    axs[0].grid(alpha=0.3)
    axs[0].yaxis.set_label_position("left")
    axs[0].yaxis.tick_right()
    axs[0].set_ylim(*ylim)
    
    axs[1].plot(x, phase(H), 'b', lw=2)
    axs[1].set_ylabel(r'$\phi$ [rad]', rotation=0, labelpad=pad)
    axs[1].grid(alpha=0.3)
    axs[1].yaxis.set_label_position("left")
    axs[1].yaxis.tick_right()

    axs[2].plot(x[1:], sg.medfilt(tau_g(H, fs), 7), 'g', lw=2)
    axs[2].set_ylabel(r'$\tau_g$ [ps]', rotation=0, labelpad=pad)
    axs[2].grid(alpha=0.3)
    axs[2].yaxis.set_label_position("left")
    axs[2].yaxis.tick_right()

    if disp: 
        if not f0:
            raise ValueError('`f0` must be specify to determine dispersion.')

        axs[3].plot(x[:-2], sg.medfilt(dispersion(H, fs, f0), 7), 'm', lw=2)
        axs[3].set_ylabel(r'D [ps/nm]', rotation=0, labelpad=28)
        axs[3].set_xlabel(xlabel)
        axs[3].grid(alpha=0.3)
        axs[3].yaxis.set_label_position("left")
        axs[3].yaxis.tick_right()
    else:
        axs[2].set_xlabel(xlabel)
    
    if xlim:
        plt.xlim(xlim)

    if retAxes:
        return axs
    
    if show_:
        plt.show()
    
    if ret:
        return f, H, phase, tau_g


def rcos(x, alpha, T):
    r"""
    Raised cosine spectrum function.

    Parameters
    ----------
    x : Number or Array_Like
        Input values.
    alpha : :obj:`float`
        Roll-off factor.
    T : :obj:`float`
        Symbol period.

    Returns
    -------
    :obj:`np.ndarray`
        Raised cosine function.

    Example
    -------
    https://en.wikipedia.org/wiki/Raised-cosine_filter

    .. plot::
        :include-source:
        :alt: Raised cosine function
        :align: center
        
        from opticomlib import rcos
        import matplotlib.pyplot as plt
        import numpy as np

        T = 1
        x = np.linspace(-1.5/T, 1.5/T, 1000)

        plt.figure(figsize=(8, 5))
        
        for alpha in [0, 0.25, 0.5, 1]:
            plt.plot(x, rcos(x, alpha, T), label=r'$\alpha$ = {}'.format(alpha))
        
        plt.ylabel('y')
        plt.xlabel('x')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    """
    
    first_condition = np.abs(x) <= (1-alpha)/(2*T)
    second_condition = (np.abs(x)>(1-alpha)/(2*T)) & (np.abs(x)<=(1+alpha)/(2*T))
    third_condition = np.abs(x) > (1+alpha)/(2*T)

    if isinstance(x, Number):
        return 1 if first_condition else 0 if third_condition else 0.5*(1+np.cos(pi*T/alpha*(np.abs(x)-(1-alpha)/(2*T))))
    
    if not isinstance(x, Array_Like):
        raise ValueError('`x` must be a number or an array_like.')
    
    x = np.array(x)
    H = np.zeros_like(x)

    H[ first_condition ] = 1
    if alpha != 0:
        H[ second_condition ] = 0.5*(1+np.cos(pi*T/alpha*(np.abs(x[second_condition])-(1-alpha)/(2*T))))
    return H


def si(x, unit: Literal['m','s']='s', k: int=1):
    r"""
    Unit of measure classifier.

    Parameters
    ----------
    x : int | float
        Number to classify.
    unit : str, default: 's'
        Unit of measure. Valid options are {'s', 'm', 'Hz', 'rad', 'bit', 'byte', 'W', 'V', 'A', 'F', 'H', 'Ohm'}.
    k : int, default: 1
        Precision of the output.

    Returns
    -------
    str
        String with number and unit.

    Example
    -------
    .. code-block:: python

        >>> si(0.002, 's')
        '2.0 ms'
        >>> si(1e9, 'Hz')
        '1.0 GHz'
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
    if 1e-15<= x <1e-12:
        return f'{x*1e15:.{k}f} f{unit}'
    if x == 0:
        return f'0 {unit}'
    

def norm(x):
    """
    Normalize an array by dividing each element by the maximum value in the array.

    Parameters
    ----------
    x : Array_Like
        Input array to be normalized.

    Returns
    -------
    out : np.ndarray
        Normalized array.

    Raises
    ------
    ValueError
        If ``x`` is not an `array_like`.
    """
    if isinstance(x, Array_Like):
        x = np.array(x)
    else:
        raise ValueError('`x` must be an array_like.')
    
    return x/x.max()


def nearest(x, a):
    """
    Find the nearest value in an array.

    Parameters
    ----------
    x : Array_Like
        Input array.
    a : Number
        Value to find.

    Returns
    -------
    out : Number
        Nearest value in the array.

    Raises
    ------
    ValueError
        If ``x`` is not an `array_like`.
        If ``a`` is not a `number`.
    """
    if isinstance(x, Array_Like):
        x = np.array(x)
    else:
        raise ValueError('`x` must be an array_like.')

    if not isinstance(a, Number):
        raise ValueError('`a` must be a number.')
    
    return x[np.abs(x-a).argmin()]


def theory_BER(
    P_avg, 
    modulation: Literal['ook', 'ppm'], 
    M=None, 
    decision=None, 
    ER=np.inf, 
    amplify=False, 
    f0=193.4145e12, 
    G=None, 
    NF=None, 
    BW_opt=None, 
    r=1.0, 
    BW_el=5e9, 
    R_L=50, 
    T=300, 
    NF_el=0
    ):
    """
    This function calculates the bit error rate (BER) for an OPTICAL RECEIVER, based on a PIN photodetector, from the average input power. 
    It also allows consider the effects of an EDFA amplifier in the results.

    If ``amplify==False``, thermal and shot noise of PIN will be consider in BER calculation. 
    If ``amplify==True``, signal-ase and ase-ase beating noises generated by the EDFA are consider too. 

    In addition, parameter ``NF_el != 0`` (electrical noise figure) can be used to represent another sources of noise after detection, like electrical amplifiers, etc. 

    Parameters
    ----------
    P_avg: float
        Average Received input optical Power (in [dB]).
    modulation: Literal['ook', 'ppm']
        Kind of modulation format {'ook', 'ppm'}, more modulations in future...
    M: int
        Order of M-ary PPM (a power of 2). Only needed if `modulation='ppm'`. 
    decision: Literal['hard', 'soft']
        Kind of PPM decision. 'hard' decision use the optimum threshold to separate ON and OFF slots. 'soft' decision
        use the Maximum a Posteriori (MAP) and it outperform 'hard' decision. Only needed if `modulation='ppm'`.
    ER: float
        Extinction Ratio of the input optical signal, in [dB].
    amplify: bool
        If use an EDFA or not at the receiver (before PIN). Default: `False`.
    f0: float
        Central frequency of communication, in [Hz]. Only used if `amplify=True`. Default: `193.4 THz` corresponding to `1550 nm`.
    G: float
        Gain of EDFA, in [dB]. Only used if `amplify=True`. This parameter is mandatory.
    NF: float
        Noise Figure of EDFA, in [dB]. Only used if `amplify=True`. Mandatory.
    BW_opt: float
        Bandwidth of optical filter that is placed after the EDFA, in [Hz]. Only used if `amplify=True`. Mandatory.
    r: float
        Responsivity of photo-detector. Default: 1.0 [A/W]
    BW_el: float
        Bandwidth of photo-detector or electrical filter, in [Hz]. Default: 5e9 [Hz].
    R_L: float
        Load resistance of photo-detector, in [Ω]. Default: 50 [Ω].
    T: float
        Temperature of photo-detector, in [K]. Default: 300 [K].
    NF_el: float
        Equivalent Noise Figure of electric circuit, in [dB]. Default: 0 [dB] 

    Returns
    -------
    BER : float
        Theoretical Bit Error rate of the system specified.

    Notes
    -----
    The bandwidths are used only for the determination of the noise contribution to the BER value, i.e. the signal 
    distortion due to these bandwidth is not taken into account. Therefore, the bandwidths ``BW_el`` and ``BW_opt`` are 
    not considered to affect the transmitted signal.

    Example
    -------
    .. plot::
        :include-source:
        :alt: Raised cosine function
        :align: center
        
        from opticomlib import theory_BER
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(-40, -20, 1000)  # Average input optical power [dB]

        plt.figure(figsize=(8, 6))
        
        plt.semilogy(x, theory_BER(P_avg=x, modulation='ook'), label='OOK')
        plt.semilogy(x, theory_BER(P_avg=x, modulation='ppm', M=4, decision='soft'), label='4-PPM (soft)')
        plt.semilogy(x, theory_BER(P_avg=x, modulation='ppm', M=4, decision='hard'), label='4-PPM (hard)')
        
        plt.xlabel(r'$P_{avg}$')
        plt.ylabel('BER')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(1e-9,)
        plt.show()
    """
    
    from scipy.constants import h, k as kB, e, pi
    from scipy.integrate import quad

    @np.vectorize(otypes=[np.float64]) # se creó esta función envoltorio para que se mostrara bien luego en la documentación.
    def temp(
        P_avg, 
        modulation: Literal['ook', 'ppm'], 
        M=None, 
        decision=None, 
        ER=np.inf, 
        amplify=False, 
        f0=193.4145e12, 
        G=None, 
        NF=None, 
        BW_opt=None, 
        r=1.0, 
        BW_el=5e9, 
        R_L=50, 
        T=300, 
        NF_el=0
        ):
        if amplify:
            if G is None:
                raise ValueError('Enter the EDFA gain "G" in [dB].')
            if NF is None:
                raise ValueError('Enter the EDFA noise figure "NF" in [dB].')
            if BW_opt is None:
                raise ValueError('Enter the bandwidth of the optical filter "BW_opt" in [Hz].')
                
            g = idb(G)
            nf = idb(NF)
            l = BW_el/BW_opt

            p_ase = nf * h * f0 * (g - 1) * BW_opt # ASE optical noise 
            mu_ASE = r * p_ase * R_L # ASE voltage offset
        else:
            g = 1
            l = 1
            mu_ASE = 0

        M = 2 if modulation.lower() == 'ook' else M

        er = idb(ER)  # extinction ratio
        nf_el = idb(NF_el)  # electrical noise figure
        p_avg = idbm(P_avg)  # average input power, in [W]

        p_ON = p_avg * M / (1 + (M-1)/er) # ON slot average optical power, without amplification
        p_OFF = p_ON/er   # OFF slot average optical power, without amplification

        mu_ON = r * g * p_ON * R_L + mu_ASE # ON slot voltage
        mu_OFF = r * g * p_OFF * R_L + mu_ASE # OFF slot voltage

        S_sig_ase_i = 2 * mu_ASE * np.array([(mu - mu_ASE) for mu in [mu_OFF, mu_ON]]) * l  # signal-ase beating noise variance, in [V^2]
        S_ase_ase = mu_ASE**2 * (1 - l/2) * l                                               # ase-ase beating noise variance, in [V^2]

        S_th = 4 * kB * T * BW_el * R_L * nf_el                  # thermal noise variance, in [V^2]
        S_sh_i = 2 * e * np.array([mu_OFF, mu_ON]) * BW_el       # shot noise variance, in [V^2]
        
        s = (S_th + S_sig_ase_i + S_ase_ase + S_sh_i)**0.5   # santandar desviation of ON and OFF slots

        if modulation.lower() == 'ppm':
            if M is None:
                raise ValueError('Enter a value for "M".')
            if M<2 or (M & (M - 1)):
                raise ValueError('The parameter "M" must be a power of 2 greater than or equal to 2.')
            
            if decision.lower()=='hard':
                SER = lambda x: 1 - Q((x-mu_ON)/s[1]) * (1-Q((x-mu_OFF)/s[0]))**(M-1) # hard decision
                SER = SER(np.linspace(mu_OFF, mu_ON, 5000)).min()
            elif decision.lower()=='soft':
                SER = 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((mu_ON-mu_OFF+s[1]*x)/s[0]))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0]
            else:
                raise ValueError('decision must be "hard" or "soft"')
            BER = SER * M/2/(M-1)
        
        elif modulation.lower() == 'ook':
            BER = lambda x: 0.5*(Q((mu_ON-x)/s[1]) + Q((x-mu_OFF)/s[0]))
            BER = BER(np.linspace(mu_OFF, mu_ON, 5000)).min()
        
        else:
            raise KeyError(f'The modulation type "{modulation}" is invalid.')
        
        return BER
    
    return temp(P_avg, modulation, M, decision, ER, amplify, f0, G, NF, BW_opt, r, BW_el, R_L, T, NF_el)