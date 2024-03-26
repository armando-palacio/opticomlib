"""
.. rubric:: Classes
.. autosummary::

    global_variables
    binary_sequence
    electrical_signal
    optical_signal
    eye
"""

from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from pympler.asizeof import asizeof as sizeof

import numpy as np
from scipy.constants import c, pi

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif' 

from matplotlib.widgets import Slider

from typing import Literal, Any, Iterable

import warnings

from .utils import (
    str2array, 
    dbm, 
    si, 
)

Array_Like = (list, tuple, np.ndarray)
Number = (int, float)

class global_variables():
    r"""**Global Variables (gv)**

    This object is used to store global variables that are used in the simulation.
    The global variables are used mainly to define the sampling frequency, the slot rate, 
    the number of samples per slot and the optical wavelength or frequency.

    .. Note:: 
        
        A slot is taken as the smallest time unit representing a binary value of the signal.
        For example, in PPM a bit is not the same as a slot. However, in OOK a bit and a slot are the same.

    This class don't need to be instantiated. It is already instantiated as ``gv``.
    For update or add a variable use the :meth:`__call__` method (i.e gv(\*\*kargs)).
    
    .. rubric:: Attributes
    .. autosummary::

        ~global_variables.sps
        ~global_variables.R
        ~global_variables.fs
        ~global_variables.dt
        ~global_variables.wavelength
        ~global_variables.f0
        ~global_variables.N
        ~global_variables.t
        ~global_variables.dw
        ~global_variables.w

        
    .. rubric:: Methods
    .. autosummary::

        __call__
        __str__
        print

    Examples
    --------
    >>> gv(R=10e9, sps=8, N=100).print()

    ::

        ------------------------
        *** Global Variables ***
        ------------------------
            sps :   8
            R   :   1.00e+10
            fs  :   8.00e+10
            λ0  :   1.55e-06
            f0  :   1.93e+14
            N   :   100
            dt  :   1.25e-11
            dw  :   6.28e+08
            t   :   [0.e+00 1.e-11 3.e-11 ... 1.e-08 1.e-08 1.e-08]
            w   :   [-3.e+11 -3.e+11 -3.e+11 ...  2.e+11  3.e+11  3.e+11]

    Also can be define new variables trough \*\*kwargs. If at least two of this arguments (``sps``, ``fs`` and ``R``) are not provided
    a warning will be raised and the default values will be used.

    >>> gv(alpha=0.5, beta=0.3).print()
    
    ::
    
        UserWarning: `sps`, `R` and `fs` will be set to default values (16 samples per slot, 1.00e+09 Hz, 1.60e+10 Samples/s)
        warnings.warn(msg)

        ------------------------------
        ***    Global Variables    ***
        ------------------------------
                sps :  16
                R   :  1.00e+09
                fs  :  1.60e+10
                λ0  :  1.55e-06
                f0  :  1.93e+14
        
        Custom
        ------
                alpha : 0.5
                beta : 0.3

    """

    def __init__(self):
        self.sps = 16
        """Number of samples per slot, ``16`` by default."""
        self.R = 1e9
        """Slot rate in Hz, ``1e9`` by default."""	
        self.fs = self.R*self.sps
        """Sampling frequency in Samples/s, ``R*sps=16e9`` by default."""	
        self.dt = 1/self.fs
        """Time step in seconds, ``1/fs=62.5e-12`` by default."""
        self.wavelength = 1550e-9
        """Optical communication central wavelength in meters, ``1550e-9`` by default."""
        self.f0 = c/self.wavelength
        """Optical communication central frequency in Hz, ``c/wavelength=193.4e12`` by default."""
        self.N = None
        """Number of slots to simulate (``None`` by default), if provided, it will set the instance's `N` attribute and calculate `t`, `dw`, and `w`."""
        self.t = None
        """Time array in seconds, ``None`` by default."""
        self.dw = None
        """Frequency step in Hz, ``None`` by default."""
        self.w = None
        """Frequency array in Hz, ``None`` by default."""


    def __call__(self, sps: int=None, R: float=None, fs: float=None, wavelength: float=1550e-9, N: int=None, **kargs) -> Any:
        """
        Configures the instance with the provided parameters.

        Parameters
        ----------
        sps : :obj:`int`, optional
            Samples per slot. If provided, it will set the instance's `sps` attribute.
        R : :obj:`float`, optional
            Rate in Hz. If provided, it will set the instance's `R` attribute.
        fs : :obj:`float`, optional
            Sampling frequency in Samples/s. If provided, it will set the instance's `fs` attribute.
        wavelength : :obj:`float`, optional
            Wavelength in meters. Default is 1550e-9.
        N : :obj:`int`, optional
            Number of samples. If provided, it will set the instance's `N` attribute and calculate `t`, `dw`, and `w`.
        **kargs : :obj:`dict`
            Additional attributes to set on the instance.

        Returns
        -------
        self
            The instance itself.

        Notes
        -----
        If `sps` is provided and either `R` or `fs` is provided, it will calculate the missing one.
        If `R` is provided and `fs` is provided, it will calculate `sps`.
        If only `fs` is provided, it will calculate `sps` using the instance's `R` attribute.
        If none of `sps`, `R`, or `fs` is provided, it will use the instance's default values.
        """
        if sps:
            self.sps = sps
            if R:
                self.R = R
                self.fs = R*sps
            elif fs:
                self.fs = fs
                self.R = fs/sps
            else:
                msg = f'`R` will be set to default value ({self.R:.2e} Hz)'
                warnings.warn(msg)
                self.fs = self.R*sps

        elif R: 
            self.R = R
            if fs:
                self.fs = fs
                self.sps = int(fs/R)
            else:
                msg = f'`sps` will be set to default value ({self.sps} samples per slot)'
                warnings.warn(msg)
                self.fs = R*self.sps

        elif fs:
            msg = f'`sps` will be set to default value ({self.sps} samples per slot)'
            warnings.warn(msg)
            self.fs = fs
            self.sps = int(fs/self.R)

        else:
            msg = f'`sps`, `R` and `fs` will be set to default values ({self.sps} samples per slot, {self.R:.2e} Hz, {self.fs:.2e} Samples/s)'
            warnings.warn(msg)

        self.dt = 1/self.fs

        if N is not None:
            self.N = N
            self.t = np.linspace(0, N*self.sps*self.dt, N*self.sps, endpoint=True)
            self.dw = 2*pi*self.fs/(N*self.sps)
            self.w = 2*pi*fftshift(fftfreq(N*self.sps))*self.fs
        else:
            self.N = None
            self.t = None
            self.dw = None
            self.w = None
        
        self.wavelength = wavelength
        self.f0 = c/wavelength

        if kargs:
            for key, value in kargs.items():
                setattr(self, key, value)
        
        return self
        
    def __str__(self):
        """ Returns a formatted string with the global variables of the instance."""
        title = 3*'*' + '    Global Variables    ' + 3*'*'
        sub = len(title)*'-'

        names = list(gv.__dict__.keys())
        others = [name for name in names if name not in ['sps', 'R', 'fs', 'wavelength', 'f0', 'N', 'dt', 'dw', 't', 'w']]

        msg = f'\n{sub}\n{title}\n{sub}\n\t' + \
            f'sps :  {self.sps}\n\t' + \
            f'R   :  {self.R:.2e}\n\t' + \
            f'fs  :  {self.fs:.2e}\n\t' + \
            f'λ0  :  {self.wavelength:.2e}\n\t' + \
            f'f0  :  {self.f0:.2e}\n'
        
        if self.N is not None:
            msg += '\t' + \
                f'N   :  {self.N}\n\t' + \
                f'dt  :  {self.dt:.2e}\n\t' + \
                f'dw  :  {self.dw:.2e}\n\t' + \
                f't   :  {self.t}\n\t' + \
                f'w   :  {self.w}\n'
            
        if others:
            msg += '  Custom\n  ------\n\t' + '\n\t'.join([f'{name} : {getattr(self, name)}' for name in others]) + '\n'

        return msg
    
    def print(self):
        """ Prints the global variables of the instance in a formatted manner.

        Prints the global variables including `sps`, `R`, `fs`, `wavelength`, `f0`, `N`, `dt`, `dw`, `t`, and `w`.
        If there are other attributes defined, they will be printed under the "Custom" section.

        Notes
        -----
        The variables are printed with a precision of 2 in scientific notation, except for `sps` and `N` which are integers.
        """
        np.set_printoptions(precision=0, threshold=10)
        print(self)


gv = global_variables()


class binary_sequence():
    r"""**Binary Sequence**

    This class provides methods and attributes to work with binary sequences. 
    The binary sequence can be provided as a string, list, tuple, or numpy array.

    .. rubric:: Attributes
    .. autosummary::

        ~binary_sequence.data
        ~binary_sequence.execution_time

    .. rubric:: Methods
    .. autosummary::

        __init__
        __str__
        __repr__
        print
        __len__
        __getitem__
        __eq__
        __add__
        __radd__
        __invert__
        len
        ones
        zeros
        type
        sizeof
    """

    def __init__(self, data: str | Iterable): 
        """ Initialize the binary sequence object.

        Parameters
        ----------
        data : :obj:`str`, 1D array_like or scalar
            The binary sequence data.
        """
        if isinstance(data, str):
            data = str2array(data)
        else:
            data = np.array(data)

        if not np.all((data == 0) | (data == 1)): 
            raise ValueError("The array must contain only 0's and 1's!")
        if data.ndim > 1:
            raise ValueError(f"Binary sequence must be 1D array, invalid shape {data.shape}")
        if data.ndim == 0 and data.size == 1:
            data = data[np.newaxis]
        
        self.data = data.astype(np.uint8)
        """The binary sequence data, a 1D numpy array of boolean values."""
        self.execution_time = None
        """The execution time of the last operation performed on the binary sequence."""

    def __str__(self, title: str=None): 
        """Return a formatted string with the binary sequence data, length, size in bytes and time if available."""
        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'

        np.set_printoptions(precision=0, threshold=100)
        data = str(self.data)

        msg = f'\n{sub}\n{title}\n{sub}\n\t' + \
            f'data  :  {data}\n\t' + \
            f'len   :  {self.len()}\n\t' + \
            f'size  :  {self.sizeof()} bytes\n'
        
        if self.execution_time is not None:
            msg += '\t' +\
                f'time  :  {si(self.execution_time, "s", 1)}\n'
        return msg
    
    def __repr__(self):
        np.set_printoptions(threshold=100)
        return f'binary_sequence({str(self.data)})'
    
    def print(self, msg: str=None): 
        """Print object parameters.

        Parameters
        ----------
        msg : str, opcional
            top message to show

        Returns
        -------
        :obj:`binary_sequence`
            The same object.
        """
        print(self.__str__(msg))
        return self
    
    def __len__(self):
        """Get number of slots of the binary sequence. ``len(self)``"""
        return self.len()

    def __getitem__(self, slice: int | slice):
        """Get a slice of the binary sequence (``self[slice]``). 
        
        Parameters
        ----------
        slice : :obj:`int` or :obj:`slice`
            The slice to get. 

        Returns
        -------
        :obj:`int` or :obj:`binary_sequence`
            The value of the slot if `slice` is an integer, or a new binary sequence object with the result of the slice.
        """ 
        return binary_sequence(self.data[slice])
    
    def __eq__(self, other):
        """Compare two binary sequences using ``==`` operator.

        Parameters
        ----------
        other : :obj:`str` or :obj:`binary_sequence` or :obj:`Array_Like`
            The binary sequence to compare.
            
        Returns
        -------
        :obj:`np.ndarray` of :obj:`bool`
            A boolean array with the result of the comparison. ``True`` if the elements are equal, ``False`` otherwise.
        """
        if isinstance(other, binary_sequence):
            other = other.data
        elif isinstance(other, str):
            other = str2array(other, bool)  
        else:
            other = np.array(other, dtype=bool)
            if other.ndim == 0:
                other = other[np.newaxis]

        if other.size != self.data.size and other.size != 1:
            raise ValueError(f"Can't compare binary sequences with shapes {self.data.shape} and {other.shape}")
        
        return np.array_equal(self.data, other)

    def __add__(self, other): 
        """ Concatenate two binary sequences, adding to the end (``+``).

        Parameters
        ----------
        other : :obj:`str` or :obj:`binary_sequence` or Array_Like
            The binary sequence to concatenate.

        Returns
        -------
        binary_sequence
            A new binary sequence object with the result of the concatenation.

        Raises
        ------
        ValueError
            If the sequence to concatenate it's not in an apropiate format.
        TypeError
            If the binary sequence to concatenate is not of type :obj:`str`, :obj:`binary_sequence` or :obj:`Array_Like`.
        
        See Also
        --------
        __radd__ : Concatenates two binary sequence, adding at the beginning (``+``).
        """
        if isinstance(other, binary_sequence):
            other = other.data
        elif isinstance(other, str):
            other = str2array(other)
        elif isinstance(other, Array_Like):
            other = np.array(other)
        else:
            raise TypeError("Can't concatenate binary_sequence with type {}".format(type(other)))
        
        if not np.all((other == 0) | (other == 1)): 
            raise ValueError("Sequence to concatenate must contain only 0's and 1's!")
        if other.ndim != 1:
            raise ValueError(f"Binary sequence must be 1D array, invalid shape {other.shape}")
        
        out = np.concatenate((self.data, other))
        return binary_sequence(out)
    
    def __radd__(self, other): 
        """ Concatenate two binary sequences, adding to the beginning (``+``).

        Parameters
        ----------
        other : :obj:`str` or :obj:`binary_sequence` or Array_Like
            The binary sequence to concatenate.

        Returns
        -------
        binary_sequence
            A new binary sequence object with the result of the concatenation.

        Raises
        ------
        ValueError
            If the sequence to concatenate it's not in an apropiate format.
        TypeError
            If the binary sequence to concatenate is not of type :obj:`str`, :obj:`binary_sequence` or :obj:`Array_Like`.
        
        See Also
        --------
        __add__ : Concatenates two binary sequence, adding at the end.
        """
        if isinstance(other, binary_sequence):
            other = other.data
        elif isinstance(other, str):
            other = str2array(other)
        elif isinstance(other, Array_Like):
            other = np.array(other)
        else:
            raise TypeError("Can't concatenate binary_sequence with type {}".format(type(other)))
        
        if not np.all((other == 0) | (other == 1)): 
            raise ValueError("Sequence to concatenate must contain only 0's and 1's!")
        if other.ndim != 1:
            raise ValueError(f"Binary sequence must be 1D array, invalid shape {other.shape}")

        out = np.concatenate((other, self.data))
        return binary_sequence(out)

    def __invert__(self):
        """Invert the binary sequence using the ``~`` operator. 
        
        Implement a bitwise not ``~`` operation on the binary sequence. Example: ``~binary_sequence([1,0,1,0])`` returns ``binary_sequence([0,1,0,1])``.

        Returns
        -------
        binary_sequence
            A new binary sequence object with the result of the inversion.
        """
        return binary_sequence(~self.data.astype(bool))

    def len(self): 
        """Get number of slots of the binary sequence.
        
        Returns
        -------
        :obj:`int`
            The number of slots of the binary sequence.
        """
        return self.data.size
    
    def ones(self):
        """Return the number of ones in the binary sequence.
        
        Returns
        -------
        :obj:`int`
            The number of ones in the binary sequence.
        """
        return np.sum(self.data)
    
    def zeros(self):
        """Return the number of zeros in the binary sequence.
        
        Returns
        -------
        :obj:`int`
            The number of zeros in the binary sequence.
        """
        return self.len() - self.ones()
    
    def type(self): 
        """Return de object type.
        
        Returns
        -------
        :obj:`type`
            The object type :obj:`binary_sequence`.
        """
        return type(self)
    
    def sizeof(self):
        """Get memory size of object in bytes."""
        return sizeof(self)


class electrical_signal():
    """**Electrical Signal**

    This class provides methods and attributes to work with electrical signals. 
    It has overloaded operators necessary to properly interpret 
    the ``+``, ``-``, ``*`` and ``/``` operations as any numpy array.

    .. rubric:: Attributes
    .. autosummary::

        ~electrical_signal.signal
        ~electrical_signal.noise
        ~electrical_signal.execution_time

    .. rubric:: Methods
    .. autosummary::

        __init__
        __call__
        print
        len
        type
        sizeof
        fs
        sps
        dt
        t
        w
        abs
        power
        phase
        apply
        copy
        plot
        psd
        grid
        legend
        show
    """

    def __init__(self, signal: str | Iterable, noise: str | Iterable = None, dtype: np.dtype=None) -> None:
        """ Initialize the electrical signal object.

        Parameters
        ----------
        signal : :obj:`str` or 1D array_like or scalar
            The signal values.
        noise : :obj:`str` or 1D array_like or scalar, optional
            The noise values. Defaults to `None`.

        Notes
        -----
        The signal and noise can be provided as a string, in which case it will be converted to a 
        ``numpy.array`` using the :func:`str2array` function. For example:
        
        .. code-block:: python

            >>> electrical_signal('1 2 3,4,5')  # separate values by space or comma indistinctly
            electrical_signal(signal=[1.+0.j 2.+0.j 3.+0.j 4.+0.j 5.+0.j],
                              noise=[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j])
            >>> electrical_signal('1+2j, 3+4j, 5+6j') # complex values
        """    
        if isinstance(signal, str):
            signal = str2array(signal)
        else: 
            signal = np.array(signal, dtype=dtype)
        
        if noise is not None:
            if isinstance(noise, str):
                noise = str2array(noise)
            else: 
                noise = np.array(noise, dtype=dtype)
            
            if dtype is None:
                arrays_type = np.result_type(signal, noise) # obtain the most comprehensive type
            else:
                arrays_type = dtype

            signal = signal.astype(arrays_type)
            noise = noise.astype(arrays_type) 

            if signal.shape != noise.shape:
                raise ValueError(f"`signal` and `noise` must have the same shape, missmatch shapes {signal.shape} and {noise.shape}!")
        
        if noise is None and dtype is not None:
            signal = signal.astype(dtype)
            
        if self.__class__ == electrical_signal:
            if signal.ndim > 1 or signal.size < 1:
                raise ValueError(f"Signal must be scalar or 1D array for electrical_signal, invalid shape {signal.shape}")
            
            if signal.ndim == 0:
                signal = signal[np.newaxis]
                if noise is not None:
                    noise = noise[np.newaxis]
        
        self.signal = signal
        """The signal values, a 1D numpy array of complex values."""
        self.noise = noise
        """The noise values, a 1D numpy array of complex values."""
        self.execution_time = None
        """The execution time of the last operation performed on the electrical signal."""

    def __str__(self, title: str=None): 
        """Return a formatted string with the electrical_signal data, length, size in bytes and time if available."""
        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'
        tab = 3*' '

        np.set_printoptions(precision=1, threshold=10)

        if self.signal.ndim == 1:
            signal = str(self.signal)
            noise = str(self.noise)
        else:
            signal = str(self.signal).replace('\n', '\n'+tab + 11*' ')
            noise = str(self.noise).replace('\n', '\n'+tab + 11*' ')
        
        msg = f'\n{sub}\n{title}\n{sub}\n'+ tab + \
            f'signal:    {signal}\n'+ tab + \
            f'noise:     {noise}\n'+ tab + \
            f'len:       {self.len()}\n' + tab + \
            f'elem_type: {self.signal.dtype}\n' + tab + \
            f'mem_size:  {self.sizeof()} bytes\n'
        
        if self.execution_time is not None:
            msg += tab + \
                f'time:      {si(self.execution_time, "s", 1)}\n'
        return msg
    
    def __repr__(self):
        np.set_printoptions(precision=1, threshold=20)
        
        if self.noise is not None:
            return f'electrical_signal({str(self.signal)})'
        return f'electrical_signal(signal={str(self.signal)},\n\t\t   noise={str(self.noise)})'

    def print(self, msg: str=None): 
        """Prints object parameters.
        
        Parameters
        ----------
        msg : :obj:`str`, opcional
            top message to show

        Returns
        -------
        self : electrical_signal
            The same object.
        """
        print(self.__str__(msg))
        return self

    def __len__(self): 
        return self.len()
    
    def __add__(self, other):
        """ Add two electrical signals (``+`` operator). Same that ``__radd__``.
        
        Parameters
        ----------
        other : :obj:`electrical_signal` or :obj:`Array_Like` or :obj:`Number`
            The signal to add.
        
        Returns
        -------
        :obj:`electrical_signal`
            A new electrical signal object with the result of the addition.
        """
        if not isinstance(other, self.type()):
            other = self.__class__(other) # only signal is considered
        
        if self.len() != other.len() and other.len() != 1:
            raise ValueError(f"Can't add {self.__class__.__name__}'s with shapes {self.signal.shape} and {other.signal.shape}")
        
        dtype = np.result_type(self.signal, other.signal)

        if self.noise is None and other.noise is None:
            return self.__class__(self.signal + other.signal, dtype=dtype)
        elif self.noise is None:
            return self.__class__(self.signal + other.signal, other.noise, dtype=dtype)
        elif other.noise is None:
            return self.__class__(self.signal + other.signal, self.noise, dtype=dtype)
        return self.__class__(self.signal + other.signal, self.noise + other.noise, dtype=dtype)
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        """ Substract two electrical signals (``-`` operator).

        Parameters
        ----------
        other : :obj:`electrical_signal` or :obj:`Array_Like` or :obj:`Number`
            The signal to substract.

        Returns
        -------
        :obj:`electrical_signal`
            A new electrical signal object with the result of the substraction.        
        """
        if not isinstance(other, self.__class__):
            other = self.__class__(other) # only signal is considered
        
        if self.len() != other.len() and other.len() != 1:
            raise ValueError(f"Can't substract {self.__class__.__name__}'s with shapes {self.signal.shape} and {other.signal.shape}")
        
        dtype = np.result_type(self.signal, other.signal)

        if self.noise is None and other.noise is None:
            return self.__class__(self.signal - other.signal, dtype=dtype)
        elif self.noise is None:
            return self.__class__(self.signal - other.signal, -other.noise, dtype=dtype)
        elif other.noise is None:
            return self.__class__(self.signal - other.signal, self.noise, dtype=dtype)
        return self.__class__(self.signal - other.signal, self.noise - other.noise, dtype=dtype)
        
    def __rsub__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other) # only signal is considered
        
        if self.len() != other.len() and other.len() != 1:
            raise ValueError(f"Can't substract {self.__class__.__name__}'s with shapes {self.signal.shape} and {other.signal.shape}")
        
        dtype = np.result_type(self.signal, other.signal)

        if self.noise is None and other.noise is None:
            return self.__class__(-self.signal + other.signal, dtype=dtype)
        elif self.noise is None:
            return self.__class__(-self.signal + other.signal, other.noise, dtype=dtype)
        elif other.noise is None:
            return self.__class__(-self.signal + other.signal, -self.noise, dtype=dtype)
        return self.__class__(-self.signal + other.signal, -self.noise + other.noise, dtype=dtype)
        
    def __mul__(self, other):
        """ Multiply two electrical signals (``*`` operator). Same that ``__rmul__``.
        
        Parameters
        ----------
        other : :obj:`electrical_signal` or :obj:`Array_Like` or :obj:`Number`
            The signal to multiply.

        Returns
        -------
        :obj:`electrical_signal`
            A new electrical signal object with the result of the multiplication.
        """
        if not isinstance(other, self.__class__):
            other = self.__class__(other) # only signal is considered
        
        if self.len() != other.len() and other.len() != 1:
            raise ValueError(f"Can't add {self.__class__.__name__}'s with shapes {self.signal.shape} and {other.signal.shape}")
        
        dtype = np.result_type(self.signal, other.signal)

        if self.noise is None and other.noise is None:
            return self.__class__(self.signal * other.signal, dtype=dtype)
        elif self.noise is None:
            return self.__class__(self.signal * other.signal, other.noise, dtype=dtype)
        elif other.noise is None:
            return self.__class__(self.signal * other.signal, self.noise, dtype=dtype)
        return self.__class__(self.signal * other.signal, self.noise * other.noise, dtype=dtype)
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __getitem__(self, slice: int | slice):
        """Slice the signal.

        Parameters
        ----------
        slice : :obj:`int` or :obj:`slice`
            Index or slice to get.

        Returns
        -------
        out : :obj:`optical_signal`
            A new object with the result of the slicing.
        """
        if self.noise is None:
            return electrical_signal( self.signal[slice] ) 
        return electrical_signal( self.signal[slice], self.noise[slice] )

    def __call__(self, domain: Literal['t','w', 'f'], shift: bool=False):
        """ Return a new object with Fast Fourier Transform (FFT) of signal and noise of input object.

        Parameters
        ----------
        domain : {'t', 'w', 'f'}
            Domain to transform. 't' for time domain (ifft is applied), 'w' and 'f' for frequency domain (fft is applied).
        shift : :obj:`bool`, optional
            If True, apply the ``np.fft.fftshift()`` or ``np.fft.ifftshift`` functions as appropriate.

        Returns
        -------
        new_obj : :obj:`electrical_signal` or :obj:`optical_signal`
            A new electrical signal object with the result of the transformation.

        Raises
        ------
        TypeError
            If ``domain`` is not one of the following values ('t', 'w', 'f').
        """
        if domain == 'w' or domain == 'f':
            signal = fft(self.signal, axis=-1)
            if self.noise is not None:
                noise = fft(self.noise, axis=-1)
              
        elif domain == 't':
            signal = ifft(self.signal, axis=-1)
            if self.noise is not None:
                noise = ifft(self.noise, axis=-1)
        
        else:
            raise ValueError("`domain` must be one of the following values ('t', 'w', 'f')")
        
        if shift:
            if domain == 'w' or domain == 'f':
                signal = fftshift(signal, axes=-1)
                if self.noise is not None:
                    noise = fftshift(noise, axes=-1)
            else: 
                signal = ifftshift(signal, axes=-1)
                if self.noise is not None:
                    noise = ifftshift(noise, axes=-1)

        if self.noise is None:
            return self.__class__(signal)
        return self.__class__(signal, noise)
    
    def __gt__(self, other): 
        """ Compare the signal+noise with a threshold (``>`` operator).

        Parameters
        ----------
        other : array_like or :obj:`float`    
            The threshold to compare with. If other is an array, the comparison is element-wise.
        
        Returns
        -------
        out: binary_sequence
            A new binary sequence object with the result of the comparison.

        Raises
        ------
        ValueError
            If the arrays must have the same length.
        TypeError
            If `other` is not of type :obj:`electrical_signal`, :obj:`list`, :obj:`tuple`, :obj:`numpy.array`, :obj:`int` or :obj:`float`.
        """
        if not isinstance(other, electrical_signal):
            other = electrical_signal(other) # only signal is considered
        
        if self.len() != other.len() and other.len() != 1:
            raise ValueError(f"Can't compare electrical_signals with shapes {self.signal.shape} and {other.signal.shape}")

        return binary_sequence(self.abs() > other.abs())
        

    def __lt__(self, other):
        """ Compare the signal+noise with a threshold (``<`` operator).

        Parameters
        ----------
        other : array_like or :obj:`float`    
            The threshold to compare with. If other is an array, the comparison is element-wise.
        
        Returns
        -------
        out: binary_sequence
            A new binary sequence object with the result of the comparison.

        Raises
        ------
        ValueError
            If the arrays must have the same length.
        TypeError
            If `other` is not of type :obj:`electrical_signal`, :obj:`list`, :obj:`tuple`, :obj:`numpy.array`, :obj:`int` or :obj:`float`.
        """
        if not isinstance(other, electrical_signal):
            other = electrical_signal(other) # only signal is considered
        
        if self.len() != other.len() and other.len() != 1:
            raise ValueError(f"Can't compare electrical_signals with shapes {self.signal.shape} and {other.signal.shape}")

        return binary_sequence(self.abs() < other.abs())
             
    def len(self): 
        """Get number of samples of the electrical signal.
        
        Returns
        -------
        :obj:`int`
            The number of samples of the electrical signal.
        """
        if self.signal.ndim > 1:
            return self.signal.shape[1]
        return self.signal.size

    def type(self): 
        """Return de object type (``electrical_signal``).
        
        Returns
        -------
        :obj:`type`
            The object type (``electrical_signal``).
        """
        return type(self)

    def sizeof(self):
        """Get memory size of object in bytes.
        
        Returns
        -------
        :obj:`int`
            The memory size of the object in bytes.
        """
        return sizeof(self)

    def fs(self): 
        """Get sampling frequency of the electrical signal.
        
        Returns
        -------
        :obj:`float`
            The sampling frequency of the electrical signal (``gv.fs``).
        """
        return gv.fs
    
    def sps(self):
        """Get samples per slot of the electrical signal.
        
        Returns
        -------
        :obj:`int`
            The samples per slot of the electrical signal (``gv.sps``).
        """
        return gv.sps
    
    def dt(self): 
        """Get time step of the electrical signal.
        
        Returns
        -------
        :obj:`float`
            The time step of the electrical signal (``gv.dt``).
        """
        return gv.dt
    
    def t(self): 
        """Get time array for the electrical signal.
        
        Returns
        -------
        :obj:`np.ndarray`
            The time array for the electrical signal.
        """
        return np.linspace(0, self.len()*gv.dt, self.len(), endpoint=True)
    
    def w(self, shift: bool=False): 
        """Return angular frequency for spectrum representation.
        
        Parameters
        ----------
        shift : :obj:`bool`, optional
            If True, apply fftshift().

        Returns
        -------
        :obj:`np.ndarray`
            The angular frequency array for signals simulation.
        """
        w = 2*pi*fftfreq(self.len())*self.fs()
        if shift:
            return fftshift(w)
        return w
    
    def power(self, by: Literal['signal','noise','all']='all'): 
        """Get power of the electrical signal.
        
        Parameters
        ----------
        by : :obj:`str`, optional
            Defines from which attribute to obtain the power. If 'all', power of signal+noise is determined.
        
        Returns
        -------
        :obj:`float`
            The power of the electrical signal.
        """
        if by.lower() not in ['signal', 'noise', 'all']:
            raise ValueError('`by` must be one of the following values ("signal", "noise", "all")')
        return np.mean(self.abs(by)**2, axis=-1)
    
    def phase(self):
        """Get phase of the ``signal`` + `noise`.
        
        Returns
        -------
        :obj:`np.ndarray`
            The phase of the electrical signal.
        """
        if self.noise is None:
            return np.unwrap(np.angle(self.signal))
        return np.unwrap(np.angle(self.signal + self.noise))
    
    def apply(self, function, *args, **kargs):
        r"""Apply a function to signal and noise.
        
        Parameters
        ----------
        function : :obj:`callable`
            The function to apply.
        \*args : :obj:`iterable`
            Variable length argument list to pass to the function.
        \*\*kargs : :obj:`dict`
            Arbitrary keyword arguments to pass to the function.

        Returns
        -------
        out : :obj:`electrical_signal`
            A new electrical signal object with the result of the function applied to the signal and noise.
        """
        output = self.copy()
        output.signal = function(self.signal, *args, **kargs)
        if self.noise is not None:
            output.noise = function(self.noise, *args, **kargs)
        output.execution_time = self.execution_time
        return output

    def copy(self, n: int=None):
        """Return a copy of the object.
        
        Parameters
        ----------
        n : :obj:`int`, optional
            Index to truncate original object. If None, the whole object is copied.

        Returns
        -------
        cp : :obj:`electrical_signal`
            A copy of the object.
        """
        if n is None: 
            n = self.len()
        return self[:n]

    def abs(self, by: Literal['signal','noise','all']='all'):
        """Get absolute value of ``signal``, ``noise`` or ``signal+noise``.

        Parameters
        ----------
        by : :obj:`str`, optional
            Defines from which attribute to obtain the absolute value. If 'all', absolute value of ``signal+noise`` is determined.
        
        Returns
        -------
        out : :obj:`np.ndarray`, (1D or 2D, float)
            The absolute value of the object.
        """
        if not isinstance(by, str):
            raise TypeError('`by` must be a string.')
        by = by.lower()
        
        if by == 'signal':
            return np.abs(self.signal)
        elif by == 'noise':
            return np.abs(self.noise) if self.noise is not None else np.zeros(self.signal.shape, dtype=self.signal.dtype)
        elif by == 'all':
            return np.abs(self.signal + self.noise) if self.noise is not None else np.abs(self.signal)
        else:
            raise ValueError('`by` must be one of the following values ("signal", "noise", "all")')
    

    def plot(self, 
             fmt: str='-', 
             n: int=None, 
             xlabel: str=None, 
             ylabel: str=None, 
             style: Literal['dark', 'light'] = 'dark',
             grid: bool=False,
             hold: bool=True,
             **kwargs: dict): 
        r"""Plot real part of electrical signal.

        Parameters
        ----------
        fmt : :obj:`str`
            Format style of line. Example 'b-.', Defaults to '-'.
        n : :obj:`int`, optional
            Number of samples to plot. Defaults to the length of the signal.
        xlabel : :obj:`str`, optional
            X-axis label. Defaults to 'Time [ns]'.
        ylabel : :obj:`str`, optional
            Y-axis label. Defaults to 'Amplitude [V]'.
        style : :obj:`str`, optional
            Style of plot. Defaults to 'dark'.
        grid : :obj:`bool`, optional
            If show grid. Defaults to False.
        hold : :obj:`bool`, optional
            If hold the current plot. Defaults to True.
        \*\*kwargs : :obj:`dict`
            Aditional keyword arguments compatible with matplotlib.pyplot.plot().

        Returns
        -------
        self : :obj:`electrical_signal`
            The same object.
        """
        n = self.len() if not n else n
        t = self.t()[:n]*1e9

        if style == 'dark':
            plt.style.use('dark_background')
            c = 'white'
        elif style == 'light':
            plt.style.use('default')
            c = 'black'
        else:
            raise ValueError('`style` must be "dark" or "light".')
        
        if not hold:
            plt.figure()

        plt.plot(t, (self[:n].signal+self[:n].noise).real, fmt, **kwargs)
        plt.xlabel(xlabel if xlabel else 'Time [ns]')
        plt.ylabel(ylabel if ylabel else 'Amplitude [V]')

        if grid:
            for i in t[:n*gv.sps][::gv.sps]:
                plt.axvline(i, color=c, ls='--', alpha=0.3, lw=1)
            plt.axvline(t[-1] + gv.dt*1e9, color=c, ls='--', alpha=0.3, lw=1)
            plt.grid(alpha=0.3, axis='y')
        
        if 'label' in kwargs.keys():
            plt.legend()
        return self
    

    def psd(self, 
            fmt: str='-', 
            n: int=None, 
            xlabel: str=None,
            ylabel: str=None,
            yscale: Literal['linear','dbm']='dbm', 
            style: Literal['dark', 'light'] = 'dark',
            grid: bool=True,
            hold: bool=True,
            **kwargs: dict):
        """Plot Power Spectral Density (PSD) of the electrical signal.

        Parameters
        ----------
        fmt : :obj:`str`
            Format style of line. Example 'b-.'. Defaults to '-'.
        n : :obj:`int`, optional
            Number of samples to plot. Defaults to the length of the signal.
        xlabel : :obj:`str`, optional
            X-axis label. Defaults to 'Frequency [GHz]'.
        ylabel : :obj:`str`, optional
            Y-axis label. Defaults to 'Power [dBm]' if ``yscale='dbm'`` or 'Power [W]' if ``yscale='linear'``.
        yscale : :obj:`str`, {'linear', 'dbm'}, optional
            Kind of Y-axis plot. Defaults to 'dbm'.
        style : :obj:`str`, {'dark', 'light'}, optional
            Style of plot. Defaults to 'dark'.
        grid : :obj:`bool`, optional
            If show grid. Defaults to True.
        hold : :obj:`bool`, optional
            If hold the current plot. Defaults to True.
        **kwargs : :obj:`dict`
            Aditional matplotlib arguments.

        Returns
        -------
        self : :obj:`electrical_signal`
            The same object.
        """
        n = self.len() if not n else n
        f = self[:n].w(shift=True)/2/pi * 1e-9

        psd = fftshift(self[:n]('w').abs('all')**2/n**2)

        if style == 'dark':
            plt.style.use('dark_background')
            c = 'white'
        elif style == 'light':
            plt.style.use('default')
            c = 'black'
        else:
            raise ValueError('`style` must be "dark" or "light".')
        
        if yscale == 'linear':
            args = (f, psd*1e3, fmt)
            ylabel = ylabel if ylabel else 'Power [mW]'
            ylim = (-0.1,)
        elif yscale == 'dbm':
            args = (f, dbm(psd), fmt)
            ylabel = ylabel if ylabel else 'Power [dBm]'
            ylim = (-100,)
        else:
            raise TypeError('`yscale` must be one of the following values ("linear", "dbm")')
        
        if not hold:
            plt.figure()

        plt.plot( *args, **kwargs )
        plt.ylabel( ylabel )
        plt.xlabel( xlabel if xlabel else 'Frequency [GHz]')
        plt.xlim(-3.5*gv.R*1e-9, 3.5*gv.R*1e-9)
        plt.ylim( *ylim )
        if grid: plt.grid(alpha=0.3, color=c)

        if 'label' in kwargs.keys():
            plt.legend()
        return self
    
    def grid(self, **kwargs):
        r"""Add grid to the plot.

        Parameters
        ----------
        \*\*kwargs : :obj:`dict`
            Arbitrary keyword arguments to pass to the function.

        Returns
        -------
        self : :obj:`electrical_signal`
            The same object.
        """
        kwargs['alpha'] = kwargs.get('alpha', 0.3)
        plt.grid(**kwargs)
        return self
    
    def legend(self, *args, **kwargs):
        r"""Add a legend to the plot.

        Parameters
        ----------
        \*args : :obj:`iterable`
            Variable length argument list to pass to the function.
        \*\*kwargs : :obj:`dict`
            Arbitrary keyword arguments to pass to the function.

        Returns
        -------
        self : :obj:`electrical_signal`
            The same object.
        """
        plt.legend(*args, **kwargs)
        return self
    
    def show(self):
        """Show plots.
        
        Returns
        -------
        self : :obj:`electrical_signal`
            The same object.
        """
        plt.show()
        return self


class optical_signal(electrical_signal):
    """**Optical Signal**
    
    Bases: :obj:`electrical_signal`

    This class provides methods and attributes to work with optical signals.

    .. rubric:: Attributes
    .. autosummary::

        ~optical_signal.signal
        ~optical_signal.noise
        ~optical_signal.execution_time

    .. rubric:: Methods
    .. autosummary::

        __init__
        __call__
        print
        len
        type
        sizeof
        fs
        sps
        dt
        t
        w
        power
        phase
        apply
        copy
        abs
        plot
        psd
        grid
        legend
        show
    """

    def __init__(self, 
                 signal: str | Iterable, 
                 noise: str | Iterable = None, 
                 n_pol: Literal[1, 2] = None,
                 dtype: np.dtype=None):
        """ Initialize the optical signal object.

        Parameters
        ----------
        signal : :obj:`str` or array_like (1D, 2D) or scalar
            The signal values.
        noise : :obj:`str` or array_like (1D, 2D) or scalar, optional
            The noise values, default is `None`.
        n_pol : :obj:`int`, optional
            Number of polarizations. Defaults to 1.
        """
        if isinstance(signal, str):
            signal = str2array(signal)
        else:
            signal = np.array(signal, dtype=dtype)

        if noise is not None:
            if isinstance(noise, str):
                noise = str2array(noise)
            else:
                noise = np.array(noise, dtype=dtype)

            if dtype is None:
                arrays_type = np.result_type(signal, noise) # obtain the most comprehensive type
            else:
                arrays_type = dtype
            
            signal = signal.astype(arrays_type)
            noise = noise.astype(arrays_type) 

            if signal.shape != noise.shape:
                raise ValueError(f"`signal` and `noise` must have the same shape, missmatch shapes {signal.shape} and {noise.shape}!")
            
        if noise is None and dtype is not None:
            signal = signal.astype(dtype)

        if self.__class__ == optical_signal:
            if signal.ndim>2 or (signal.ndim>1 and signal.shape[0]>2) or signal.size<1:
                raise ValueError(f"Signal must be a scalar, 1D or 2D array for optical_signal, invalid shape {signal.shape}")
            
            if signal.ndim == 0:
                if n_pol is None:
                    n_pol = 1
                
                if n_pol == 1:
                    signal = signal[np.newaxis]
                    if noise is not None:
                        noise = noise[np.newaxis]
                else:
                    signal = np.array([[signal], [signal]])
            
            elif signal.ndim == 1:
                if n_pol is None:
                    n_pol = 1
                
                if n_pol == 2:
                    signal = np.array([signal, signal])
                    if noise is not None:
                        noise = np.array([noise, noise])
            
            elif signal.ndim == 2 and signal.shape[0] == 1:
                if n_pol is None:
                    n_pol = 2
                
                if n_pol == 1:
                    signal = signal[0]
                    if noise is not None:
                        noise = noise[0]
                else:
                    signal = np.array([signal[0], signal[0]])
                    if noise is not None:
                        noise = np.array([noise[0], noise[0]])
            
            elif signal.ndim == 2 and signal.shape[0] == 2:
                if n_pol is None:
                    n_pol = 2
                
                if n_pol == 1:
                    signal = signal[0]
                    if noise is not None:
                        noise = noise[0]
        
        self.n_pol = n_pol
        super().__init__( signal, noise, dtype=dtype)  
    
    def __repr__(self):
        np.set_printoptions(precision=1, threshold=20)

        if self.noise is not None:
            signal = str(self.signal).replace('\n', '\n' + 15*' ')
            return f'optical_signal({signal})'
        
        signal = str(self.signal).replace('\n', '\n' + 22*' ')
        noise = str(self.noise).replace('\n', '\n' + 22*' ')
        return f'optical_signal(signal={signal}\n' + 16*' '+ f'noise={noise})'

    
    def __getitem__(self, slice: int | slice): 
        """Slice the optical signal.

        Parameters
        ----------
        slice : :obj:`int` or :obj:`slice`
            Index or slice to get the new optical signal.

        Returns
        -------
        out : :obj:`optical_signal`
            A new optical signal object with the result of the slicing.
        """
        if self.n_pol == 1:
            if self.noise is None:
                return optical_signal( self.signal[slice] )
            return optical_signal( self.signal[slice], self.noise[slice] )
        
        elif isinstance(slice, int):
            if self.noise is None:
                return optical_signal( self.signal[:,slice,np.newaxis] )
            return optical_signal( self.signal[:,slice,np.newaxis], self.noise[:,slice,np.newaxis] )
        
        if self.noise is None:
            return optical_signal( self.signal[:,slice] )
        return optical_signal( self.signal[:,slice], self.noise[:,slice] )
    
    def __gt__(self, other): 
        raise NotImplementedError('The > operator is not implemented for optical_signal objects.')
    
    def __lt__(self, other):
        raise NotImplementedError('The < operator is not implemented for optical_signal objects.')

    def plot(self, 
             fmt: str | list='-', 
             mode: Literal['x','y','both','abs']='abs', 
             n=None, 
             xlabel: str=None,
             ylabel: str=None,
             style: Literal['dark', 'light'] = 'dark',
             grid: bool=False,
             hold: bool=True,
             **kwargs): 
        r"""
        Plot intensity of optical signal for selected polarization mode.

        Parameters
        ----------
        fmt : :obj:`str`, optional
            Format style of line. Example 'b-.'. Default is '-'.
        mode : :obj:`str`
            Polarization mode to show. Default is 'abs'.

            - ``'x'`` plot polarization x.
            - ``'y'`` plot polarization y.
            - ``'both'`` plot both polarizations x and y in the same figure
            - ``'abs'`` plot intensity sum of both polarizations I(x) + I(y).

        n : :obj:`int`, optional
            Number of samples to plot. Default is the length of the signal.  
        xlabel : :obj:`str`, optional
            X-axis label. Default is 'Time [ns]'.
        ylabel : :obj:`str`, optional
            Y-axis label. Default is 'Power [mW]'.
        style : :obj:`str`, optional
            Style of plot. Default is 'dark'.

            - ``'dark'`` use dark background.
            - ``'light'`` use light background.
        
        grid : :obj:`bool`, optional
            If show grid. Default is ``False``.
        hold : :obj:`bool`, optional
            If hold the current figure. Default is ``True``.
        \*\*kwargs: :obj:`dict`
            Aditional matplotlib arguments.

        Returns
        -------
        self : :obj:`optical_signal`
            The same object.
        """
        n = self.len() if not n else n
        t = self.t()[:n]*1e9

        if style == 'dark':
            plt.style.use('dark_background')
            c = 'white'
        elif style == 'light':
            plt.style.use('default')
            c = 'black'
        else:
            raise ValueError('`style` must be "dark" or "light".')
        
        I = self[:n].abs('all')**2 *1e3

        if self.n_pol == 1:
            if not isinstance(fmt, str):
                warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                fmt = '-'
            args = (t, I, fmt)
        else: 
            if mode == 'x':
                if not isinstance(fmt, str):
                    warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                    fmt = '-'
                args = (t, I[0], fmt)
            elif mode == 'y':
                if not isinstance(fmt, str):
                    warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                    fmt = '-'
                args = (t, I[1], fmt)
            elif mode == 'both':
                if isinstance(fmt, (list, tuple)):
                    args = (t, I[0], fmt[0], t, I[1], fmt[1])
                elif isinstance(fmt, str):
                    args = (t, I[0], fmt, t, I[1], fmt)
                else:
                    warnings.warn('`fmt` must be a string or a list of strings for both polarizations signals, using default value.')
                    args = (t, I[0], '-', t, I[1], '-')
                    
            elif mode == 'abs':
                if not isinstance(fmt, str):
                    warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                    fmt = '-'
                args = (t, I[0] + I[1], fmt)
            else:
                raise TypeError('argument `mode` must to be one of the following values ("x","y","both","abs").')
        
        label = kwargs.pop('label', None) if mode == 'both' else None

        if not hold:
            plt.figure()

        plt.plot( *args, **kwargs)
        plt.xlabel(xlabel if xlabel else 'Time [ns]')
        plt.ylabel(ylabel if ylabel else 'Power [mW]')
        
        if grid:
            for i in t[:n*gv.sps][::gv.sps]:
                plt.axvline(i, color=c, ls='--', alpha=0.3, lw=1)
            plt.axvline(t[-1] + gv.dt*1e9, color=c, ls='--', alpha=0.3, lw=1)
            plt.grid(alpha=0.3, axis='y')

        if label is not None:
            if isinstance(label, str):
                label = [label]
            plt.legend(label)
        return self
    

    def psd(self, 
            fmt: str | list='-', 
            mode: Literal['x','y','both']='x', 
            n: int=None,
            xlabel: str=None,
            ylabel: str=None, 
            yscale: Literal['linear', 'dbm']='dbm', 
            style: Literal['dark', 'light'] = 'dark',
            grid: bool=True,
            hold: bool=True,
            **kwargs: dict):
        r"""Plot Power Spectral Density (PSD) of the electrical signal.

        Parameters
        ----------
        fmt : :obj:`str`
            Format style of line. Example 'b-.'. Default is '-'.
        mode : :obj:`str`
            Polarization mode to show. Default is 'x'.

            - ``'x'`` plot polarization x.
            - ``'y'`` plot polarization y.
            - ``'both'`` plot both polarizations x and y in the same figure.

        n : int, optional
            Number of samples to plot. Default is the length of the signal.
        xlabel : :obj:`str`, optional
            X-axis label. Default is 'Frequency [GHz]'.
        ylabel : :obj:`str`, optional
            Y-axis label.
        yscale : :obj:`str`, optional
            Kind of Y-axis plot. Default is 'dbm'.

            - ``'linear'`` plot linear scale.
            - ``'dbm'`` plot dBm scale.

        style : :obj:`str`, optional
            Style of plot. Default is 'dark'.

            - ``'dark'`` use dark background.
            - ``'light'`` use light background.

        grid : bool, optional
            If show grid. Default is ``True``.
        hold : bool, optional
            If hold the current figure. Default is ``True``.
        \*\*kwargs : :obj:`dict`
            Aditional matplotlib arguments.

        Returns
        -------
        self : :obj:`optical_signal`
            The same object.
        """
        n = self.len() if not n else n
        f = self[:n].w(shift=True)/2/pi * 1e-9

        psd = fftshift(self[:n]('w').abs('all')**2/n**2, axes=-1)

        if style == 'dark':
            plt.style.use('dark_background')
            c = 'white'
        elif style == 'light':
            plt.style.use('default')
            c = 'black'
        else:
            raise ValueError('`style` should be ("dark" or "light")')
        
        if yscale == 'linear':
            psd = psd*1e3
            ylabel = ylabel if ylabel else 'Power [mW]'
            ylim = (-0.1,)
        elif yscale == 'dbm':
            psd = dbm(psd)
            ylabel = ylabel if ylabel else 'Power [dBm]'
            ylim = (-100,)
        else:
            raise TypeError('argument `yscale` should be ("linear" or "log")')

        if self.n_pol == 1:
            if not isinstance(fmt, str):
                warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                fmt = '-'
            args = (f, psd, fmt)
        else:
            if mode == 'x':
                if not isinstance(fmt, str):
                    warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                    fmt = '-'
                args = (f, psd[0], fmt)
            elif mode == 'y':
                if not isinstance(fmt, str):
                    warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                    fmt = '-'
                args = (f, psd[1], fmt)
            elif mode == 'both':
                if isinstance(fmt, (list, tuple)):
                    args = (f, psd[0], fmt[0], f, psd[1], fmt[1])
                elif isinstance(fmt, str):
                    args = (f, psd[0], fmt, f, psd[1], fmt)
                else:
                    warnings.warn('`fmt` must be a string or a list of strings for both polarizations signals, using default value.')
                    args = (f, psd[0], '-', f, psd[1], '-')
            else:
                raise TypeError('argument `mode` should be ("x", "y" or "both")')    
        
        label = kwargs.pop('label', None)

        if not hold:
            plt.figure()

        plt.plot( *args, **kwargs)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel if xlabel else 'Frequency [GHz]')
        plt.xlim( -3.5*gv.R*1e-9, 3.5*gv.R*1e-9 )
        plt.ylim( *ylim )
        if grid: plt.grid(alpha=0.3, color=c)
        
        if label is not None:
            if isinstance(label, str):
                label = [label]
            plt.legend(label)
        return self
    

class eye():
    """**Eye Diagram Parameters**.

    This object contains the parameters of an eye diagram and methods to plot it.

    .. rubric:: Methods
    .. autosummary::

        __init__
        __str__
        print
        plot
        show

    Attributes
    ----------
    t : :obj:`np.ndarray`
        The time values resampled. Shape (Nx1).
    y : :obj:`np.ndarray`
        The signal values resampled. Shape (Nx1).
    dt : :obj:`float`
        Time between samples.
    sps : :obj:`int`
        Samples per slot.
    t_left : :obj:`float`
        Cross time of left edge.
    t_right : :obj:`float`
        Cross time of right edge.
    t_opt : :obj:`float`
        Optimal time decision.
    t_dist : :obj:`float`
        Time between slots.
    t_span0 : :obj:`float`
        t_opt - t_dist*5%.
    t_span1 : :obj:`float`
        t_opt + t_dist*5%.
    y_top : :obj:`np.ndarray`
        Samples of signal above threshold and within t_span0 and t_apan1.
    y_bot : :obj:`np.ndarray`
        Samples of signal below threshold and within t_span0 and t_apan1.
    mu0 : :obj:`float`
        Mean of y_bot.
    mu1 : :obj:`float`
        Mean of y_top.
    s0 : :obj:`float`
        Standard deviation of y_bot.
    s1 : :obj:`float`
        Standard deviation of y_top.
    er : :obj:`float`
        Extinction ratio.
    eye_h : :obj:`float`
        Eye height.
    """

    def __init__(self, **kwargs: dict):
        r""" Initialize the eye diagram object.

        Parameters
        ----------
        \*\*kwargs : :obj:`dict`, optional
            Dictionary with the eye diagram parameters.
        """

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.empty = False
        else:
            self.empty = True
        
    def __str__(self, title: str=None): 
        """Return a formatted string with the eye diagram data."""
        if self.empty:
            raise ValueError('Empty eye diagram object.')

        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'

        np.set_printoptions(precision=1, threshold=10)

        msg = f'\n{sub}\n{title}\n{sub}\n ' + '\n '.join([f'{key} : {value}' for key, value in self.__dict__.items() if key != 'execution_time'])
        
        if self.execution_time is not None:
            msg += f'\n time  :  {si(self.execution_time, "s", 1)}\n'
        return msg
    
    def print(self, msg: str=None): 
        """Print object parameters.

        Parameters
        ----------
        msg : :obj:`str`, optional
            Top message to show.

        Returns
        -------
        self: :obj:`eye`
            Same object
        """
        print(self.__str__(msg))
        return self
    
    def plot(self, 
             medias_: bool=True, 
             legend_: bool=True, 
             style: Literal['dark', 'light']='dark', 
             cmap: Literal['viridis', 'plasma', 'inferno', 'cividis', 'magma', 'winter']='winter',
             label: str = '',
             savefig: str=None):
        """ Plot eye diagram.

        Parameters
        ----------
        style : :obj:`str`, optional
            Plot style. 'dark' or 'light'.
        means_ : :obj:`bool`, optional
            If True, plot mean values.
        legend_ : :obj:`bool`, optional
            If True, show legend.
        cmap : :obj:`str`, optional
            Colormap to plot.
        label : :obj:`str`, optional
            Label to show in title.
        savefig : :obj:`str`, optional
            Name of the file to save the plot. If None, the plot is not saved.
            Input just the name of the file without extension (extension is .png by default).

        Returns
        -------
        self: :obj:`eye`
            Same object
        """
        if self.empty:
            raise ValueError('Empty eye diagram object.')

        ## SETTINGS

        if style == 'dark':
            plt.style.use('dark_background')
            t_opt_color = '#60FF86'
            means_color = 'white'
            bgcolor='black'
        elif style == 'light':
            t_opt_color = 'green'#'#229954'
            means_color = '#5A5A5A'
            bgcolor='white'
        else:
            raise TypeError("The `style` argument must be one of the following values ('dark', 'light')")
        
        dt = self.dt

        fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [4,1],  
                                                'wspace': 0.03},
                                                figsize=(8,5))
        if label:
            label = ', '+label
        plt.suptitle(f'Eye diagram{label}')
        
        ax[0].set_xlim(-1-dt,1)
        ax[0].set_ylim(self.mu0-4*self.s0, self.mu1+4*self.s1)
        ax[0].set_ylabel(r'Amplitude [mV]', fontsize=12)
        ax[0].grid(color='grey', ls='--', lw=0.5, alpha=0.5)
        ax[0].set_xticks([-1,-0.5,0,0.5,1])
        ax[0].set_xlabel(r'Time [$t/T_{slot}$]', fontsize=12)
        t_line1 = ax[0].axvline(self.t_opt, color = t_opt_color, ls = '--', alpha = 0.7)
        t_line_span0 = ax[0].axvline(self.t_span0, color = t_opt_color, ls = '-', alpha = 0.4)
        t_line_span1 = ax[0].axvline(self.t_span1, color = t_opt_color, ls = '-', alpha = 0.4)
        
        if legend_: 
            ax[0].legend([r'$t_{opt}$'], fontsize=12, loc='upper right')
        
        if medias_:
            ax[0].axhline(self.mu1, color = means_color, ls = ':', alpha = 0.7)
            ax[0].axhline(self.mu0, color = means_color, ls = '-.', alpha = 0.7)

            ax[1].axhline(self.mu1, color = means_color, ls = ':', alpha = 0.7)
            ax[1].axhline(self.mu0, color = means_color, ls = '-.', alpha = 0.7)
            if legend_:
                ax[1].legend([r'$\mu_1$',r'$\mu_0$'])

        ax[1].sharey(ax[0])
        ax[1].tick_params(axis='x', which='both', length=0, labelbottom=False)
        ax[1].tick_params(axis='y', which='both', length=0, labelleft=False)
        ax[1].grid(color='grey', ls='--', lw=0.5, alpha=0.5)


        ## ADD PLOTS
        y_ = self.y
        t_ = self.t

        ax[0].hexbin( # plot eye
            x = t_, 
            y = y_, 
            gridsize=500, 
            bins='log',
            alpha=0.7, 
            cmap=cmap 
        )
        
        ax[1].hist(  # plot vertical histogram 
            y_[(t_>self.t_opt-0.05*self.t_dist) & (t_<self.t_opt+0.05*self.t_dist)], 
            bins=200, 
            density=True, 
            orientation = 'horizontal', 
            color = t_opt_color, 
            alpha = 0.9,
            histtype='step',
        )

        ## ADD SLIDERS
        p = ax[0].get_position()
        t_slider_ax = fig.add_axes([p.x0,p.y1+0.01,p.x1-p.x0,0.02])

        t_slider = Slider(
            ax=t_slider_ax,
            label='',
            valmin=-1, 
            valmax=1,
            valstep=0.1, 
            valinit=self.t_opt, 
            initcolor=t_opt_color,
            orientation='horizontal',
            color=bgcolor,
            track_color=bgcolor,
            handle_style = dict(facecolor=t_opt_color, edgecolor=bgcolor, size=10)
        )

        ## PLOTS UPDATE
        def update_t_line(val):
            t_line1.set_xdata(val)

            t_line_span0.set_xdata(val-0.05*self.t_dist)
            t_line_span1.set_xdata(val+0.05*self.t_dist)

            ax[1].patches[-1].remove()

            n,_,_ = ax[1].hist(
                y_[(t_>val-0.01) & (t_<val+0.01)], 
                bins=200, 
                density=True, 
                orientation = 'horizontal', 
                color = t_opt_color, 
                alpha = 0.9,
                histtype='step',
            )
            ax[1].set_xlim(0,max(n))

        t_slider.on_changed(update_t_line)

        if savefig: 
            plt.savefig('.'.join((savefig, 'png')), dpi=300)
        return self

    def show(self):
        """Show plot
        
        Returns
        -------
        self : :obj:`eye`
            The same object.
        """
        plt.show()
        return self
