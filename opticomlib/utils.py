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
    p_ase
    average_voltages
    noise_variances
    optimum_threshold
    shortest_int  
    eyediagram
    rcos
    gauss
    upfirdn
    phase_estimator      
"""

import re
import numpy as np
import timeit, time as tm
import scipy.special as sp # type: ignore
import scipy.signal as sg # type: ignore

from typing import Literal, Union

from scipy.constants import pi, c, h, e, k as kB # type: ignore

import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift

import warnings

Array_Like = (set, list, tuple, np.ndarray)
from numbers import Integral as IntegerNumber
from numbers import Real as RealNumber # inlude integer numbers
from numbers import Complex as ComplexNumber # inlcude real numbers, is not exclusive


def _is_iterable_and_numpy_compatible(obj):
    """
    Check if an object is iterable, can be converted into a NumPy array, 
    and contains only numeric values (complex or real values).

    Parameters
    ----------
    obj : any
        The input object to be checked.

    Returns
    -------
    bool
        True if the object is iterable, can be transformed into a NumPy array, 
        and contains only numeric values. False otherwise.

    Notes
    -----
    - The function first verifies if `obj` is iterable.
    - If it is not iterable, the function returns False immediately.
    - If `obj` can be converted into a NumPy array (`np.array(obj)`), 
      it proceeds to check the numerical validity.
    - `numbers.Complex` is used to ensure all elements are numeric.
    - The function supports various iterable types like lists, tuples, and NumPy arrays.
    - It excludes non-numeric elements such as strings and mixed-type collections.
    """
    # Check if the object is iterable
    from collections.abc import Iterable
    is_iterable = isinstance(obj, Iterable)
    
    if not is_iterable:
        return False
    
    try:
        array = np.array(obj)  # Try to convert it into a NumPy array
    except Exception:
        return False

    # Check if all elements are numeric
    from numbers import Complex
    result = all(isinstance(x, Complex) for x in array.flatten())

    return result

def _is_numeric(obj):
    return isinstance(obj, ComplexNumber)

def _is_real(obj):
    return isinstance(obj, RealNumber)

def _is_integer(obj):
    return isinstance(obj, IntegerNumber)


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
        If ``num`` is not an integer number.
    ValueError
        If ``num`` is too large to be represented with ``digits`` bits.

    Example
    -------
    .. code-block:: python

        >>> dec2bin(5, 4)
        array([0, 1, 0, 1], dtype=uint8)
    """
    if not _is_integer(num):
        raise ValueError('`num` debe ser un número entero.')

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

class _Timer:
    def __init__(self):
        self.tic_stack = []

    def tic(self):
        self.tic_stack.append(tm.time())

    def toc(self):
        if not self.tic_stack:
            raise Exception("toc() called without a matching tic()")
        start_time = self.tic_stack.pop()
        return tm.time() - start_time

# Crear una instancia singleton de Timer
_timer_instance = _Timer()

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
    _timer_instance.tic()

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
    return _timer_instance.toc()


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
    if _is_iterable_and_numpy_compatible(x):
        x = np.array(x)
        if (x<0).any():
            raise ValueError('Some values of input array are negative.')
    elif _is_numeric(x):
        if x<0:
            raise ValueError('`x` must be positive.')
    else:   
        raise TypeError('`x` must be a number or an array_like with positive values.')
    
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
    return db(x) + 30


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
    if _is_iterable_and_numpy_compatible(x):
        x = np.array(x)
    elif not _is_numeric(x):
        raise TypeError('The input value must be a number or an array_like.')
    
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
    if _is_iterable_and_numpy_compatible(x):
        x = np.array(x)
    elif not _is_numeric(x):
        raise TypeError('The input value must be a number or an array_like.')
    
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
    if _is_iterable_and_numpy_compatible(x):
        x = np.array(x)
    elif not _is_numeric(x):
        raise TypeError('The input value must be a number or an array_like.')

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
    if _is_iterable_and_numpy_compatible(x):
        x = np.array(x)
    elif not _is_numeric(x):
        raise TypeError('The input value must be a number or an array_like.')
    
    return 0.5*sp.erfc(x/2**0.5) 


def phase(x: np.ndarray, zero_ref_index: int=None):
    r"""
    Calculate the unwrapped phase the signal.

    Parameters
    ----------
    x : :obj:`np.ndarray`
        Signal to calculate the phase
    zero_ref_index : int, default: ``None``
        Position of signal to take zero phase reference. By default zero phase reference is set to position 0 of x.

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
    if not _is_iterable_and_numpy_compatible(x):
        raise TypeError('The input value must be an array_like.')

    phase_ = np.angle(x)
    
    if zero_ref_index is not None:
        offset = phase_[zero_ref_index]
    else:
        offset = 0

    return np.unwrap(phase_) - offset

def tau_g(x: np.ndarray, fs: float):
    r"""
    Calculate the group delay of a frequency response.

    Parameters
    ----------
    x : :obj:`np.ndarray`
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
    if not _is_iterable_and_numpy_compatible(x):
        raise TypeError('The input value must be an array_like.')
    
    dw = 2*pi*fs/x.size
    phase_ = phase(x)
    return np.diff(phase_, prepend=phase_[0])/dw * 1e12

def dispersion(x: np.ndarray, fs: float, f0: float):
    """
    Calculate the dispersion of a frequency response.

    Parameters
    ----------
    x : :obj:`np.ndarray`
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
    if not _is_iterable_and_numpy_compatible(x):
        raise TypeError('The input value must be an array_like.')
    
    f = fftshift(fftfreq(x.size, d=1/fs))
    dλ = np.diff(c/(f+f0))[0]*1e9
    tau_g_ = tau_g(x, fs)
    D = np.diff(tau_g_, prepend=tau_g_[0])/dλ
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

    axs[2].plot(x, sg.medfilt(tau_g(H, fs), 7), 'g', lw=2)
    axs[2].set_ylabel(r'$\tau_g$ [ps]', rotation=0, labelpad=pad)
    axs[2].grid(alpha=0.3)
    axs[2].yaxis.set_label_position("left")
    axs[2].yaxis.tick_right()

    if disp: 
        if not f0:
            raise ValueError('`f0` must be specify to determine dispersion.')

        axs[3].plot(x, sg.medfilt(dispersion(H, fs, f0), 7), 'm', lw=2)
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

    if _is_numeric(x):
        return 1 if first_condition else 0 if third_condition else 0.5*(1+np.cos(pi*T/alpha*(np.abs(x)-(1-alpha)/(2*T))))
    
    if not _is_iterable_and_numpy_compatible(x):
        raise ValueError('`x` must be a number or an array_like.')
    
    x = np.array(x)
    y = np.zeros_like(x)

    y[ first_condition ] = 1
    if alpha != 0:
        y[ second_condition ] = 0.5*(1+np.cos(pi*T/alpha*(np.abs(x[second_condition])-(1-alpha)/(2*T))))
    return y


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
    if not _is_numeric(x):
        raise TypeError('`x` must be a numeric value.')

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
        return f'{x:.{k}f} {unit}'
    

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
    if _is_iterable_and_numpy_compatible(x):
        x = np.array(x)
    else:
        raise TypeError('`x` must be an array_like.')
    
    return x/x.max()


def nearest(x, a):
    """
    Find the value of X closest to a.

    Parameters
    ----------
    X : Array_Like
        Input array.
    A : Number or Array_Like
        Reference value or values to find in X.

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
    if _is_iterable_and_numpy_compatible(x):
        x = np.array(x)
    else:
        raise ValueError('`x` must be an array_like.')

    if _is_iterable_and_numpy_compatible(a):
        return x[np.argmin(
                np.abs(
                    np.repeat([x], len(a), axis=0) - np.reshape(a, (-1, 1))
                ),
                axis=1,
            )]
    
    if _is_numeric(a):
        return x[np.argmin(np.abs(x - a))]
    
    raise ValueError('`A` must be a number or an array_like')


def nearest_index(X, A):
    """
    Find the indices of X for the values of X closest to the values of A.

    Parameters
    ----------
    X : Array_Like
        Input array.
    A : Number or Array_Like
        Value or values to find in X.

    Returns
    -------
    out : Number or np.ndarray
        Indices of the nearest values in the array.

    Raises
    ------
    ValueError
        If ``X`` is not an `array_like`.
    """
    if _is_iterable_and_numpy_compatible(X):
        X = np.array(X)
    else:
        raise ValueError('`X` must be an array_like.')

    if _is_iterable_and_numpy_compatible(A):
        return np.argmin(
                np.abs(
                    np.repeat([X], len(A), axis=0) - np.reshape(A, (-1, 1))
                ),
                axis=1,
            )
    
    if _is_numeric(A):
        return np.argmin(np.abs(X - A))
    
    raise ValueError('`A` must be a number or an array_like')
    
    

def p_ase(
        amplify=True, 
        wavelength=1550e-9, 
        G=None, 
        NF=None, 
        BW_opt=None,
):
    """
    Calculate the ASE noise power [Watts].
    
    Parameters
    ----------
    amplify : bool, default: True
        If use an EDFA or not at the receiver (before PIN).
    wavelength : float, default: 1550e-9
        Wavelength of the signal.
    G : float
        Gain of EDFA, in [dB]. Only used if `amplify=True`. This parameter is mandatory.
    NF : float
        Noise Figure of EDFA, in [dB]. Only used if `amplify=True`. Mandatory.
    BW_opt : float
        Bandwidth of optical filter that is placed after the EDFA, in [Hz]. Only used if `amplify=True`. Mandatory.

    Returns
    -------
    p_ase : float
        ASE optical noise power, in [W].
    """
    if amplify:
        if not (G is not None and NF is not None and BW_opt is not None):
            raise ValueError('`G`, `NF` and `BW_opt` must be specify.')
        
        nf = idb(NF)
        g = idb(G)
        f0 = c/wavelength

        p_ase = nf * h * f0 * (g - 1) * BW_opt # ASE optical noise 
    else:
        p_ase = 0
    return p_ase
    
def average_voltages(
        P_avg, 
        modulation: Literal['ook', 'ppm'], 
        M=None, 
        ER=np.inf,
        amplify=True, 
        wavelength=1550e-9, 
        G=None, 
        NF=None, 
        BW_opt=None,
        r=1.0,
        R_L=50,
):
    """
    Calculate the average voltages of the ON and OFF slots [Voltages].

    Parameters
    ----------
    P_avg : float
        Average Received input optical Power (in [dBm]).
    modulation : Literal['ook', 'ppm']
        Kind of modulation format {'ook', 'ppm'}, more modulations in future...
    M : int
        Order of M-ary PPM (a power of 2). Only needed if `modulation='ppm'`.
    ER : float, default: np.inf
        Extinction Ratio of the input optical signal, in [dB].
    amplify : bool, default: True
        If use an EDFA or not at the receiver (before PIN).
    wavelength : float, default: 1550e-9
        Wavelength of the signal.
    G : float
        Gain of EDFA, in [dB]. Only used if `amplify=True`. This parameter is mandatory.
    NF : float
        Noise Figure of EDFA, in [dB]. Only used if `amplify=True`. Mandatory.
    BW_opt : float
        Bandwidth of optical filter that is placed after the EDFA, in [Hz]. Only used if `amplify=True`. Mandatory.
    r : float, default: 1.0
        Responsivity of photo-detector.
    R_L : float, default: 50
        Load resistance of photo-detector, in [Ω].

    Returns
    -------
    mu: np.ndarray
        Average voltage of ON and OFF slots. mu[0] is the OFF slot and mu[1] is the ON slot.
    mu_ASE: float
        ASE voltage offset.
    """ 
    M = 2 if modulation.lower() == 'ook' else M

    er = idb(ER)  # extinction ratio
    p_avg = idbm(P_avg)  # average input power, in [W]
    g = idb(G)  # gain of EDFA

    p_ON = p_avg * M / (1 + (M-1)/er) # ON slot average optical power, without amplification
    p_OFF = p_ON/er   # OFF slot average optical power, without amplification

    mu_ASE = r * p_ase(amplify, wavelength, G, NF, BW_opt) * R_L  # ASE voltage offset
    
    mu = r * g * np.array([p_OFF, p_ON]) * R_L + mu_ASE  # average voltage of ON and OFF slots
    return mu, mu_ASE

def noise_variances(
        P_avg,
        modulation: Literal['ook', 'ppm'], 
        M=None,
        ER=np.inf,
        amplify=True,
        wavelength=1550e-9,
        G=None,
        NF=None,
        BW_opt=None,
        r=1.0,
        BW_el=5e9,
        R_L=50,
        T=300,
        NF_el = 0
    ):
    """
    Calculate the theoretical noise variances for OFF and ON slots, include sig-ase, ase-ase, thermal and shot noises [V^2].
    If ``amplify=False`` only thermal and shot are calculated.

    Parameters
    ----------
    P_avg: float
        Average Received input optical Power (in [dBm]).
    modulation: Literal['ook', 'ppm']
        Kind of modulation format {'ook', 'ppm'}, more modulations in future...
    M: int
        Order of M-ary PPM (a power of 2). Only needed if `modulation='ppm'`. 
    ER: float
        Extinction Ratio of the input optical signal, in [dB].
    amplify: bool
        If use an EDFA or not at the receiver (before PIN). Default: `False`.
    wavelength: float
        Central frequency of communication, in [Hz]. Only used if `amplify=True`. Default: `1550 nm`.
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
    """
    mu, mu_ASE = average_voltages(P_avg, modulation, M, ER, amplify, wavelength, G, NF, BW_opt, r, R_L)

    l = BW_el/BW_opt
    nf_el = idb(NF_el)

    S_sig_ase_i = 2 * mu_ASE * (mu-mu_ASE) * l  # signal-ase beating noise variance, in [V^2]
    S_ase_ase = mu_ASE**2 * (1 - l/2) * l       # ase-ase beating noise variance, in [V^2]

    S_th = 4 * kB * T * BW_el * R_L   # thermal noise variance, in [V^2]
    S_sh_i = 2 * e * mu * BW_el * R_L   # shot noise variance, in [V^2]
    
    S = (S_th + S_sig_ase_i + S_ase_ase + S_sh_i) * nf_el   # variance of ON and OFF slots
    return S

def optimum_threshold(mu0,mu1,S0,S1, modulation: Literal['ook', 'ppm'], M=None):
    """
    Calculate the optimum threshold for binary modulation formats.

    Parameters
    ----------
    mu0: float
        Average voltage of OFF slot.
    mu1: float
        Average voltage of ON slot.
    S0: float
        Noise variance of OFF slot.
    S1: float
        Noise variance of ON slot.
    modulation: str
        Modulation format
    M: int
        PPM order

    Returns
    -------
    threshold: float
        Optimum threshold value.
    """

    M = 2 if modulation.lower() == 'ook' else M

    s1=S1**0.5
    s0=S0**0.5

    threshold = 1/(S1-S0)*(mu0*S1 - mu1*S0 + s1*s0*np.sqrt((mu1-mu0)**2 + 2*(S1-S0)*np.log(s1/s0*(M-1))))
    return threshold

def theory_BER(
    P_avg, 
    modulation: Literal['ook', 'ppm'], 
    M=None, 
    decision=None, 
    threshold=None,
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
        Average Received input optical Power (in [dBm]).
    modulation: Literal['ook', 'ppm']
        Kind of modulation format {'ook', 'ppm'}, more modulations in future...
    M: int
        Order of M-ary PPM (a power of 2). Only needed if `modulation='ppm'`. 
    decision: Literal['hard', 'soft']
        Kind of PPM decision. 'hard' decision use the optimum threshold to separate ON and OFF slots. 'soft' decision
        use the Maximum a Posteriori (MAP) and it outperform 'hard' decision. Only needed if `modulation='ppm'`.
    threshold: float 
        Threshold for decision. Only needed if (`modulation='ook'`) or (`modulation='ppm` and `decision='hard'`). Value
        must be in (0, 1), without include edges. By default optimum threshold is used.
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
        threshold=None, 
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
                
                if threshold is not None:
                    if threshold<=0 or threshold>=1:
                        raise ValueError('The threshold value must be in the range (0, 1).')
                    
                    SER = SER(threshold * mu_ON + (1 - threshold) * mu_OFF)
                else:
                    SER = SER(np.linspace(mu_OFF, mu_ON, 5000)).min()
    
            elif decision.lower()=='soft':
                SER = 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((mu_ON-mu_OFF+s[1]*x)/s[0]))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0]
    
            else:
                raise ValueError('decision must be "hard" or "soft"')
    
            BER = SER * M/2/(M-1)
        
        elif modulation.lower() == 'ook':
            BER = lambda x: 0.5*(Q((mu_ON-x)/s[1]) + Q((x-mu_OFF)/s[0]))

            if threshold is not None:
                if threshold<=0 or threshold>=1:
                    raise ValueError('The threshold value must be in the range (0, 1).')
                
                BER = BER(threshold * mu_ON + (1 - threshold) * mu_OFF)
            else:
                BER = BER(np.linspace(mu_OFF, mu_ON, 5000)).min()
    
        else:
            raise KeyError(f'The modulation type "{modulation}" is invalid.')
        
        return BER
    
    return temp(P_avg, modulation, M, decision, threshold, ER, amplify, f0, G, NF, BW_opt, r, BW_el, R_L, T, NF_el)



def shortest_int(x: np.ndarray, percent: float=50) -> tuple[float, float]:
    r"""
    Estimation of the shortest interval of x values, containing {``percent``}% of the samples in 'x'.

    Parameters
    ----------
    x : ndarray
        Data of not complex values. If data is complex, real part will be taken.
    percent : real number
        percent of of data.

    Returns
    -------
    tuple[float, float]
        The shortest interval containing 50% of the samples in 'data'.
    """
    if not _is_iterable_and_numpy_compatible(x):
        raise TypeError('`x` must be an array_like.')
    if not _is_real(percent) or percent <= 0 or percent >100:
        raise ValueError('`percent` must be a real number between (0, 100].')

    diff_lag = (
        lambda data, lag: data[lag:] - data[:-lag]
    )  # Difference between two elements of an array separated by a distance 'lag'

    x = np.sort(x.real)
    lag = int(len(x) * percent/100)
    diff = diff_lag(x, lag)
    i = np.where(np.abs(diff - np.min(diff)) < 1e-10)[0]
    if len(i) > 1:
        i = int(np.mean(i))
    return np.array((x[i], x[i + lag]))


# --- Función para Aplicar el Filtro Gaussiano Optimizado ---
def apply_optimized_gaussian_filter(t, signal, T_bit):
    """
    Applies a Gaussian filter to a NRZ signal. The filter is optimized based on the bit duration, to a sigma of 0.139 * T_bit.

    Args:
        t (np.ndarray): Time vector corresponding to signal_in.
        signal (np.ndarray): NRZ input signal.
        T_bit (float): Bit duration, used as a reference for sigma.

    Returns:
        np.ndarray: The filtered signal with scaled amplitude.
    """
    dt = t[1] - t[0]
    if dt <= 0:
         raise ValueError("El paso de tiempo dt debe ser positivo.")

    # Calcular parámetros del kernel basados en el sigma óptimo
    std_dev_points = T_bit * 0.139 / dt

    # Determinar el tamaño del kernel. Debe ser impar y lo suficientemente grande.
    # Un tamaño basado en 6*sigma_en_puntos cubre >99.7% de la gaussiana.
    kernel_size = int(6 * std_dev_points)
    if kernel_size % 2 == 0:
        kernel_size += 1 # Asegurar que es impar para un centro claro
    if kernel_size < 3: # Asegurar un tamaño mínimo para el kernel
        kernel_size = 3

    # Asegurar que el kernel no sea más grande que la señal de entrada (menos un pequeño margen)
    max_possible_kernel_size = len(signal) - 2
    if kernel_size > max_possible_kernel_size:
         print(f"Advertencia: El tamaño calculado del kernel ({kernel_size}) es mayor que la señal ({len(signal)}). Reduciendo a {max_possible_kernel_size}.")
         kernel_size = max_possible_kernel_size
         if kernel_size < 3:
             print("Error: El tamaño de la señal es demasiado pequeño para aplicar un filtro significativo.")
             return np.zeros_like(signal) # Devolver ceros si no se puede filtrar

    # Crear el kernel gaussiano usando la función corregida
    from scipy.signal.windows import gaussian
    gaussian_kernel = gaussian(kernel_size, std=std_dev_points)

    # Normalizar el kernel para preservar el área (o nivel DC) de la señal
    kernel_sum = np.sum(gaussian_kernel)
    gaussian_kernel /= kernel_sum

    # Realizar la convolución
    # mode='same' asegura que la salida tenga el mismo tamaño y esté centrada
    from scipy.signal import convolve
    convolved_signal = convolve(signal, gaussian_kernel, mode='same')

    return convolved_signal


def eyediagram(y, sps, n_traces=None, cmap='viridis', 
             N_grid_bins=350, grid_sigma=3, ax=None, **plot_kw):
    """Plots a colored eye diagram, internally calculating color density.

    Parameters
    ----------
    y : np.ndarray
        Full amplitude array (1D).
    sps : int
        Samples per symbol. Used to segment the eye traces.
    n_traces : int, optional
        Maximum number of traces to plot. If None, all available traces
        will be plotted. Defaults to None.
    cmap : str, optional
        Name of the matplotlib colormap. Defaults to 'viridis'.
    N_grid_bins : int, optional
        Number of bins for the density histogram. Defaults to 350.
    grid_sigma : float, optional
        Sigma for the Gaussian filter applied to the density. Defaults to 3.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
        Defaults to None.
    **plot_kw : dict, optional
        Additional plotting parameters:
        
        Figure parameters (used only if ax is None):
        - figsize : tuple, default (10, 6)
        - dpi : int, default 100
        
        Line collection parameters:
        - linewidth : float, default 0.75
        - alpha : float, default 0.25
        - capstyle : str, default 'round'
        - joinstyle : str, default 'round'
        
        Axes formatting parameters:
        - xlabel : str, default "Time (2-symbol segment)"
        - ylabel : str, default "Amplitude"
        - title : str, default "Eye Diagram ({num_traces} traces)"
        - grid : bool, default True
        - grid_alpha : float, default 0.3
        - xlim : tuple, optional (xmin, xmax)
        - ylim : tuple, optional (ymin, ymax)
        - tight_layout : bool, default True
        
        Display parameters:
        - show : bool, default False (whether to call plt.show())
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the eye diagram plot.
    """
    # Truncate signals to avoid edge artifacts
    # Remove sps//2 samples from each end
    start_idx = sps // 2
    end_idx = len(y) - sps // 2
    
    if start_idx >= end_idx or end_idx <= start_idx:
        raise ValueError(f"Signal too short for truncation. Need at least {sps} samples, got {len(y)}.")
    
    Y_truncated = y[start_idx:end_idx]
    
    # Eye diagram parameters
    num_points_per_trace = 2 * sps
    if len(Y_truncated) < num_points_per_trace:
        raise ValueError(f"Need at least {num_points_per_trace} points for eye diagram, got {len(Y_truncated)} after truncation.")
    
    available_traces = len(Y_truncated) // num_points_per_trace
    num_traces_to_plot = min(available_traces, n_traces) if n_traces is not None else available_traces
    
    if num_traces_to_plot == 0:
        raise ValueError(f"Not enough points to form even one trace of {num_points_per_trace} points after truncation.")
    
    X_truncated = np.kron(np.ones(num_traces_to_plot), np.linspace(-1, 1 - 1/sps, num_points_per_trace))
    Y_truncated = Y_truncated[:num_traces_to_plot * num_points_per_trace]
    
    # Get colormap
    try:
        cmap_obj = getattr(plt.cm, cmap)
    except AttributeError:
        print(f"Warning: Colormap '{cmap}' not found. Using 'viridis' by default.")
        cmap_obj = plt.cm.viridis

    # Extract plotting parameters with defaults
    # Figure parameters
    figsize = plot_kw.get('figsize', None)
    dpi = plot_kw.get('dpi', 100)
    
    # Line collection parameters
    linewidth = plot_kw.get('linewidth', 1)
    alpha = plot_kw.get('alpha', 0.05)
    capstyle = plot_kw.get('capstyle', 'round')
    joinstyle = plot_kw.get('joinstyle', 'round')
    
    # Axes formatting parameters
    xlabel = plot_kw.get('xlabel', "Time (2-symbol segment)")
    ylabel = plot_kw.get('ylabel', "Amplitude")
    title_template = plot_kw.get('title', "Eye Diagram ({num_traces} traces)")
    grid = plot_kw.get('grid', True)
    grid_alpha = plot_kw.get('grid_alpha', 0.3)
    xlim = plot_kw.get('xlim', None)
    ylim = plot_kw.get('ylim', None)
    tight_layout = plot_kw.get('tight_layout', True)
    
    # Display parameters
    show = plot_kw.get('show', False)

    # Calculate ranges using truncated signals
    min_x, max_x = X_truncated.min(), X_truncated.max()
    min_y, max_y = Y_truncated.min(), Y_truncated.max()
    
    # Normalize coordinates (handle edge cases)
    X_norm = np.zeros_like(X_truncated) if max_x == min_x else (X_truncated - min_x) / (max_x - min_x)
    Y_norm = np.zeros_like(Y_truncated) if max_y == min_y else (Y_truncated - min_y) / (max_y - min_y)

    # Create density grid using truncated signals
    from scipy.ndimage import gaussian_filter
    grid_density, _, _ = np.histogram2d(X_truncated, Y_truncated, bins=N_grid_bins)
    grid_density = gaussian_filter(grid_density, sigma=grid_sigma)

    # Map points to grid indices
    ix_grid = np.clip((X_norm * (N_grid_bins - 1)).astype(int), 0, N_grid_bins - 1)
    iy_grid = np.clip((Y_norm * (N_grid_bins - 1)).astype(int), 0, N_grid_bins - 1)

    # Get and normalize color values
    color_values = grid_density[ix_grid, iy_grid]
    color_range = color_values.max() - color_values.min()
    color_values_norm = np.zeros_like(color_values) if color_range == 0 else (color_values - color_values.min()) / color_range
    

    # Prepare data for plotting using truncated signals
    x_eye_trace = X_truncated[:num_points_per_trace]

    Y_reshaped = Y_truncated.reshape(num_traces_to_plot, num_points_per_trace)
    color_reshaped = color_values_norm.reshape(num_traces_to_plot, num_points_per_trace)

    # Create plot or use existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        created_fig = False

    # Plot eye traces
    from matplotlib.collections import LineCollection
    for i in range(num_traces_to_plot):
        # Create line segments
        points = np.array([x_eye_trace, Y_reshaped[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        if len(segments) > 0:
            colors = cmap_obj(color_reshaped[i][:len(segments)])
            lc = LineCollection(segments, colors=colors, linewidth=linewidth, 
                              alpha=alpha, capstyle=capstyle, joinstyle=joinstyle)
            ax.add_collection(lc)

    # Format plot
    ax.set_xlim(xlim if xlim is not None else (-1,1))
    ax.set_ylim(ylim if ylim is not None else (min_y, max_y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Format title with number of traces
    title_formatted = title_template.format(num_traces=num_traces_to_plot)
    ax.set_title(title_formatted)
    
    if grid:
        ax.grid(True, alpha=grid_alpha)
    
    if created_fig and tight_layout:
        plt.tight_layout()
    
    if created_fig and show:
        plt.show()
    
    return ax



def rcos(beta, span, sps, shape='sqrt'):
    """Generate a raised cosine or root raised cosine filter impulse response.

    This function replicates the MATLAB `rcosdesign()` function, generating the impulse response
    of a raised cosine or root raised cosine filter used in digital communications for
    pulse shaping.

    Parameters
    ----------
    beta : float
        Roll-off factor, must be between 0 and 1.
    span : int
        Number of symbols. The filter length will be span * sps + 1.
    sps : int
        Samples per symbol.
    shape : str, optional
        Filter shape, either 'normal' for raised cosine or 'sqrt' for root raised cosine.
        Default is 'sqrt'.

    Returns
    -------
    numpy.ndarray
        The filter impulse response, normalized to unit energy.

    Raises
    ------
    ValueError
        If beta is not in [0, 1] or shape is not 'normal' or 'sqrt'.

    Notes
    -----
    The filter is normalized such that the sum of squares of the coefficients is 1.
    For beta=0, it reduces to a `sinc()` function.

    Examples
    --------
    >>> import numpy as np
    >>> h = rcos(0.5, 6, 64, 'sqrt')
    >>> h.shape
    (49,)
    """
    if not (0 <= beta <= 1):
        raise ValueError("beta debe estar en [0, 1]")
    if shape.lower() not in ('sqrt', 'normal'):
        raise ValueError("shape debe ser 'sqrt' o 'normal'")
    
    N = span * sps
    t = np.linspace(-span/2, span/2, N+1)
    
    if beta == 0:
        p = np.sinc(t)
    
    elif shape.lower() == 'normal':
        sinc_t = np.sinc(t)
        cos_term = np.cos(np.pi * beta * t)
        den = 1 - (2 * beta * t)**2
        p = np.divide(sinc_t * cos_term, den, out=np.zeros_like(den), where=den != 0)
        
        # Handle the singularity at t = 1/(2*beta)
        special_mask = np.abs(den) < 1e-8
        if np.any(special_mask):
            p[special_mask] = (np.pi / 4) * np.sinc(1 / (2 * beta))
    
    else:  # sqrt
        t_abs = np.abs(t)
        p = np.zeros_like(t)
        
        # Special case at t=0
        mask_zero = t_abs < 1e-8
        p[mask_zero] = (1 - beta) + 4 * beta / np.pi
        
        # Special case at t = 1/(4*beta)
        special_t_sqrt = 1 / (4 * beta)
        mask_special = np.abs(t_abs - special_t_sqrt) < 1e-8
        if np.any(mask_special):
            p[mask_special] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        
        # General case
        mask_general = ~mask_zero & ~mask_special
        if np.any(mask_general):
            ti = t[mask_general]
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti)**2)
            p[mask_general] = num / den
    
    # Normalización de energía
    p = p / np.sqrt(np.sum(p**2))
    return p


def gauss(span, sps, T=1, m=1):
    """Generate a Gaussian or Super-Gaussian filter.

    This function generates the impulse response of a Gaussian filter used in digital
    communications to reduce bandwidth and minimize intersymbol interference.

    Parameters
    ----------
    span : int
        Number of symbols. The filter length will be span * sps + 1.
    sps : int
        Samples per symbol.
    T : float
        Full Width at Half Maximum (FWHM) of the pulse in symbols. Default is 1.
    m : int
        Super-Gaussian order. Default is 1 (standard Gaussian).

    Returns
    -------
    numpy.ndarray
        The impulse response of the Gaussian filter, normalized to unit energy.

    Notes
    -----
    The filter is normalized such that the sum of squares of the coefficients is 1.
    The Gaussian shape helps reduce intersymbol interference in modulation systems.

    Examples
    --------
    >>> import numpy as np
    >>> h = gauss(6, 8, 1.0)
    >>> print(h.shape)
    (49,)
    >>> print('Energy:', np.sum(h**2))
    1.0
    """
    N = span * sps
    t = np.linspace(-span/2, span/2, N+1)
    alpha = 2*np.sqrt(np.log(2)) / T
    p = np.exp(- (alpha * t)**(2*m))
    p = p / np.sqrt(np.sum(p**2))
    return p


def upfirdn(x, h, up=1, dn=1):
    """Replicate MATLAB's upfirdn function for upsampling and FIR filtering.

    This function performs upsampling of the input signal and then applies an FIR filter,
    partially replicating the functionality of MATLAB's `upfirdn`.

    Parameters
    ----------
    x : array_like
        Input signal to process.
    h : array_like
        Impulse response of the FIR filter.
    up : int, optional
        Upsampling factor. Default is 1 (no upsampling).
    dn : int, optional
        Downsampling factor. Default is 1 (no downsampling).

    Returns
    -------
    numpy.ndarray
        The filtered signal after upsampling, convolution and downsampling.

    Notes
    -----
    Upsampling is performed by inserting zeros between the input signal samples.
    Convolution is then applied with the filter using `mode='same'` to maintain length. Finally,
    downsampling is performed by selecting every `dn`-th sample from the filtered signal.
    """
    # Upsample
    xu = np.zeros(len(x) * up)
    xu[::up] = x

    # Filtrado
    y = np.convolve(xu, h, mode='same')
    
    # Downsample
    y = y[::dn]
    return y


def phase_estimator(*signals):
    """
    Estimate the initial phase of a signal using the Hilbert transform.
    This function applies the Hilbert transform to the input signal(s) to obtain their
    analytic representation, computes the instantaneous phase, and performs a linear
    fit to estimate the initial phase offset. It processes multiple signals if provided,
    but returns the initial phase estimate for the last signal.

    Parameters
    ----------
    *signals : array_like
        The input signal(s), each a 1-D array of real or complex values.

    Returns
    -------
    float
        Estimated initial phase in radians for each input signal.

    Notes
    -----
    The initial phase is the intercept of the linear fit of the unwrapped phase versus time.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 1000)
    >>> y = np.sin(2 * np.pi * 10 * t + np.pi/4)  # 10 Hz signal with phase offset
    >>> phase = phase_estimator(y)
    >>> print(f"Estimated phase: {phase:.2f} rad")
    """
    from scipy.signal import hilbert

    phis = []

    for y in signals:
        y_hilb = hilbert(y)
        
        t = np.arange(y.size)
        inst_phase = np.unwrap(np.angle(y_hilb))

        # Perform linear fit of phase vs time to estimate initial phase
        _, phi0 = np.polyfit(t, inst_phase, 1)
        phis.append(phi0)
    
    if len(phis) == 1:
        return phis[0]
    return np.array(phis)