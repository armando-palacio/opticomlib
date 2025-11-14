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
import scipy.signal as sg

import matplotlib.pyplot as plt

from typing import Literal, Any, Iterable

import warnings

from .utils import (
    str2array, 
    dbm, 
    si, 
    upfirdn,
    eyediagram,
    ComplexNumber,
    RealNumber
)

Array_Like = (list, tuple, np.ndarray)

class global_variables():
    r"""**Global Variables (gv)**

    This object is used to store global variables that are used in the simulation.
    The global variables are used mainly to define the sampling frequency, the slot rate, 
    the number of samples per slot, the number of bits of simulation and the optical wavelength or frequency.

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
        ~global_variables.plot_style

        
    .. rubric:: Methods
    .. autosummary::

        __call__
        __str__
        print
        default

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
            dt  :   1.25e-11
            λ0  :   1.55e-06
            f0  :   1.93e+14
            N   :   100
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
                dt  :   1.25e-11
                λ0  :  1.55e-06
                f0  :  1.93e+14
                N   :   128
                t   :   [0.e+00 1.e-11 3.e-11 ... 1.e-08 1.e-08 1.e-08]
                dw  :   6.28e+08
                w   :   [-3.e+11 -3.e+11 -3.e+11 ...  2.e+11  3.e+11  3.e+11]
        
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
        self.N = 128
        """Number of slots to simulate (128 by default)."""
        self.t = np.linspace(0, self.N*self.sps*self.dt, self.N*self.sps, endpoint=True)
        """Time array in seconds"""
        self.dw = 2*pi*self.fs/(self.N*self.sps)
        """Frequency step in Hz"""
        self.w = 2*pi*fftshift(fftfreq(self.N*self.sps))*self.fs
        """Frequency array in Hz"""
        self.plt_style = 'fast'
        plt.style.use(self.plt_style)

    def default(self):
        """ Return all attributes to default values.
        
        """
        self.sps = 16
        self.R = 1e9
        self.fs = self.R*self.sps
        self.dt = 1/self.fs
        self.wavelength = 1550e-9
        self.f0 = c/self.wavelength
        self.N = 128
        self._set_t_dw_w()
        self.plt_style = 'fast'
        plt.style.use(self.plt_style)

        attrs = [attr for attr in dir(gv) if not callable(getattr(gv, attr)) and not attr.startswith("__") and not (attr in ['sps', 'R', 'fs', 'dt', 'wavelength', 'f0', 'N', 't', 'w', 'dw', 'plt_style'])]
        
        for attr in attrs:
            delattr(self, attr)
        return self


    def __call__(
            self, 
            sps: int=None, 
            R: float=None, 
            fs: float=None, 
            wavelength: float=1550e-9, 
            N: int=None, 
            plt_style : Literal['ggplot', 'bmh', 'dark_background', 'fast', 'default']='fast', 
            verbose=True, 
            **kargs
        ) -> Any:
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
            self.sps = int(np.round(sps))
            if R:
                self.R = R
                self.fs = R*self.sps
            elif fs:
                self.fs = fs
                self.R = fs/self.sps
            else:
                if verbose:
                    msg = f'`R` will be set to default value ({self.R:.2e} Hz)'
                    warnings.warn(msg)
                self.fs = self.R*self.sps

        elif R: 
            self.R = R
            if fs:
                self.fs = fs
                self.sps = int(np.round(fs/R))
            else:
                if verbose:
                    msg = f'`sps` will be set to default value ({self.sps} samples per slot)'
                    warnings.warn(msg)
                self.fs = R*self.sps

        elif fs:
            if verbose:
                msg = f'`sps` will be set to default value ({self.sps} samples per slot)'
                warnings.warn(msg)
            self.fs = fs
            self.sps = int(np.round(fs/self.R))

        elif verbose:
            msg = f'`sps`, `R` and `fs` will be set to previous values ({self.sps} samples per slot, {self.R:.2e} Hz, {self.fs:.2e} Samples/s)'
            warnings.warn(msg)

        self.dt = 1/self.fs

        self.N = N if N is not None else self.N
        self._set_t_dw_w()
        
        self.wavelength = wavelength
        self.f0 = c/wavelength

        self.plt_style = plt_style
        plt.style.use(self.plt_style)

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
            f'dt  :  {self.dt:.2e}\n\t' + \
            f'λ0  :  {self.wavelength:.2e}\n\t' + \
            f'f0  :  {self.f0:.2e}\n\t' + \
            f'N   :  {self.N}\n\t' + \
            f'dt  :  {self.dt:.2e}\n\t' + \
            f't   :  {self.t}\n\t' + \
            f'dw  :  {self.dw:.2e}\n\t' + \
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
        The array variables are printed with a precision of 2 in scientific notation, and are presented in a compact from if length exceeds 20 elements.
        """
        np.set_printoptions(precision=2, threshold=20)
        print(self)

    def _set_t_dw_w(self):
        """ Calculate time array, frequency step and frequency array based on current `N`, `sps`, and `fs` values and set them as attributes."""
        self.t = np.linspace(0, self.N*self.sps/self.fs, self.N*self.sps, endpoint=True)
        self.dw = 2*pi*self.fs/(self.N*self.sps)
        self.w = 2*pi*fftshift(fftfreq(self.N*self.sps))*self.fs


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
        self.execution_time = 0.
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
            f'data  :  {data} (shape: {self.data.shape})\n\t' + \
            f'ones  :  {self.ones()}\n\t' + \
            f'zeros :  {self.zeros()}\n\t' + \
            f'size  :  {self.sizeof()} bytes\n\t' + \
            f'time  :  {si(self.execution_time, "s", 2)}\n'
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
        return self.size

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
        return self.size - self.ones()
    
    def dac(self, h: np.ndarray):
        """Apply upsampling and FIR filtering to the binary sequence for digital-to-analog conversion.

        This method upsamples the binary sequence by the global samples per slot (gv.sps) and applies the provided FIR filter to produce an electrical signal.

        Parameters
        ----------
        h : :obj:`np.ndarray`
            The FIR filter coefficients used for shaping the signal.

        Returns
        -------
        :obj:`electrical_signal`
            The resulting electrical signal after upsampling, filtering, and downsampling.
        """
        return electrical_signal(upfirdn(x=self.data, h=h, up=gv.sps, dn=1))
    
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
        size
        shape
        type
        fs
        sps
        dt
        t
        w
        f
        abs
        sum
        power
        phase
        normalize
        apply
        filter
        plot
        psd
        plot_eye
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
        self.execution_time = 0.
        """The execution time of the last operation performed on the electrical signal."""
        self.dtype = signal.dtype
        self.size = signal.size
        self.shape = signal.shape


    def __str__(self, title: str=None): 
        """Return a formatted string with the electrical_signal data, length, size in bytes and time if available."""
        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'
        tab = 3*' '

        np.set_printoptions(precision=3, threshold=20)

        if self.signal.ndim == 1:
            signal = str(self.signal)
            noise = str(self.noise)
        else:
            signal = str(self.signal).replace('\n', '\n'+tab + 11*' ')
            noise = str(self.noise).replace('\n', '\n'+tab + 11*' ')
        
        msg = f'\n{sub}\n{title}\n{sub}\n'+ tab + \
            f'signal:     {signal} (shape: {self.shape})\n'+ tab + \
            f'noise:      {noise} (shape: {self.shape if self.noise is not None else None})\n'+ tab + \
            f'pow_signal: {si(self.power('W', 'signal'), 'W', 1)} ({self.power('dbm', 'signal'):.1f} dBm)\n'+ tab + \
            f'pow_noise:  {si(self.power('W', 'noise'), 'W', 1)} ({self.power('dbm', 'noise'):.1f} dBm)\n'+ tab + \
            f'pow_total:  {si(self.power('W', 'all'), 'W', 1)} ({self.power('dbm', 'all'):.1f} dBm)\n'+ tab + \
            f'len:        {self.size}\n' + tab + \
            f'elem_type:  {self.dtype}\n' + tab + \
            f'mem_size:   {self.sizeof()} bytes\n' + tab + \
            f'time:       {si(self.execution_time, "s", 2)}\n'
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
        return self.size
    
    def _parse(self, other):
        if not isinstance(other, self.type()):
            other = self.__class__(other)
        else:
            other = other[:]

        if other.noise is None:
            other.noise = np.zeros_like(other.signal)
        
        if self.size != other.size:
            l_min = min(self.size, other.size)
            l_max = max(self.size, other.size)
            
            if l_min != 1 and l_min != l_max: 
                raise ValueError(f"Can't add {self.__class__.__name__}'s with shapes {self.signal.shape} and {other.signal.shape}")
        
        dtype = np.result_type(self.signal, other.signal)
        return other, dtype
    
    def __add__(self, other):
        other, dtype = self._parse(other)
        self_, _ = self._parse(self)

        sig = self_.signal + other.signal
        noi = self_.noise + other.noise

        return self.__class__(sig, noi if all(noi) != 0 else None, dtype=dtype)
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        sig = -self.signal
        noi = -self.noise if self.noise is not None else None
        return self.__class__(sig, noi)

    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return (-self) + other
    
    def __mul__(self, other):
        other, dtype = self._parse(other)
        self_, _ = self._parse(self)

        sig = self_.signal*other.signal
        noi = self_.signal*other.noise + self_.noise*other.signal + self_.noise*other.noise

        return self.__class__(sig, noi if all(noi) !=0 else None, dtype=dtype)
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, number: int):
        if not isinstance(number, ComplexNumber):
            raise TypeError(f"Can't divide electrical_signal by type {type(number)}")
        if number == 0:
            raise ZeroDivisionError("Can't divide electrical_signal by zero")

        if self.noise is None:
            return self.__class__(self.signal / number)
        return self.__class__(self.signal / number, self.noise / number)
    
    def __floordiv__(self, other):
        x = (self/other)
        if x.noise is None:
            return self.__class__( np.floor(x.signal) )
        return self.__class__( np.floor(x.signal), np.floor(x.noise) )
        
    def __getitem__(self, slice: int | slice):
        if self.noise is None:
            return self.__class__( self.signal[slice] ) 
        return self.__class__( self.signal[slice], self.noise[slice] )
    
    def __gt__(self, other): 
        other, _ = self._parse(other)
        self_, _ = self._parse(self)
        
        x_r = self_.signal + self_.noise
        x_l = other.signal + other.noise

        return binary_sequence(x_r > x_l)
        
    def __lt__(self, other):
        return other - self > 0 
    
    def __pow__(self, other):
        if not isinstance(other, RealNumber):
            raise TypeError(f"Can't exponentiate electrical_signal by type {type(other)}")
        
        self_, _ = self._parse(self)

        sig = (self_.signal + self_.noise) ** other 
        
        return self.__class__( sig , dtype=self_.dtype)
    
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
        self_, _ = self._parse(self)

        if domain == 'w' or domain == 'f':
            signal = fft(self_.signal, axis=-1)
            noise = fft(self_.noise, axis=-1)
              
        elif domain == 't':
            signal = ifft(self_.signal, axis=-1)
            noise = ifft(self_.noise, axis=-1)
        
        else:
            raise ValueError("`domain` must be one of the following values ('t', 'w', 'f')")
        
        if shift:
            if domain == 'w' or domain == 'f':
                signal = fftshift(signal, axes=-1)
                noise = fftshift(noise, axes=-1)
            else: 
                signal = ifftshift(signal, axes=-1)
                noise = ifftshift(noise, axes=-1)

        return self.__class__(signal, noise if all(noise) !=0 else None)


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
        return gv.t
    
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
        w = gv.w
        if shift:
            return fftshift(w)
        return w
    
    def f(self, shift: bool=False):
        return self.w(shift)/(2*pi)

    def sum(self, of: Literal['signal','noise','all']='all'):
        """Get sum of ``signal``, ``noise`` or ``signal+noise``.

        Parameters
        ----------
        of : :obj:`str`, optional
            Defines from which attribute to obtain the sum. If 'all', sum of ``signal+noise`` is determined.
        
        Returns
        -------
        out : :obj:`np.ndarray`, (1D or 2D, float)
            The sum of the object.
        """
        self_,_ = self._parse(self)

        if not isinstance(of, str):
            raise TypeError('`of` must be a string.')
        of = of.lower()
        
        if of == 'signal':
            return np.sum(self_.signal, axis=-1)
        elif of == 'noise':
            return np.sum(self_.noise, axis=-1) 
        elif of == 'all':
            return np.sum(self_.signal + self_.noise, axis=-1)
        else:
            raise ValueError('`of` must be one of the following values ("signal", "noise", "all")')

    def abs(self, of: Literal['signal','noise','all']='all'):
        """Get absolute value of ``signal``, ``noise`` or ``signal+noise``.

        Parameters
        ----------
        of : :obj:`str`, optional
            Defines from which attribute to obtain the absolute value. If 'all', absolute value of ``signal+noise`` is determined.
        
        Returns
        -------
        out : :obj:`np.ndarray`, (1D or 2D, float)
            The absolute value of the object.
        """
        self_,_ = self._parse(self)

        if not isinstance(of, str):
            raise TypeError('`of` must be a string.')
        of = of.lower()
        
        if of == 'signal':
            return np.abs(self_.signal)
        elif of == 'noise':
            return np.abs(self_.noise)
        elif of == 'all':
            return np.abs(self_.signal + self_.noise)
        else:
            raise ValueError('`of` must be one of the following values ("signal", "noise", "all")')
    
    def power(self, unit : Literal['W', 'dBm']='W', of: Literal['signal','noise','all']='all'): 
        """Get power of the electrical signal.
        
        Parameters
        ----------
        of : :obj:`str`, optional
            Defines from which attribute to obtain the power. If 'all', power of signal+noise is determined.
        
        Returns
        -------
        :obj:`float`
            The power of the electrical signal.
        """
        if of.lower() not in ['signal', 'noise', 'all']:
            raise ValueError('`of` must be one of the following values ("signal", "noise", "all")')
        p = np.mean(self.abs(of)**2, axis=-1)
        unit = unit.lower()
        if unit == 'w':
            return p
        elif unit == 'dbm':
            return dbm(p)
        else:
            raise ValueError('`unit` must be one of the following values ("W", "dBm")')
    
    def normalize(self, by: Literal['power', 'amplitude']='power'):
        """Return the power-normalized signal
        """
        x = self[:]
        if by == 'power':
            pw = x.power('W', 'signal')
            return x / pw**0.5
        elif by == 'amplitude':
            amp = x.abs('signal').max()
            return x / amp
        else:
            raise ValueError('`by` must be one of the following values ("power", "amplitude")')
    
    def phase(self):
        """Get phase of the ``signal`` + `noise`.
        
        Returns
        -------
        :obj:`np.ndarray`
            The phase of the electrical signal.
        """
        self_, _ = self._parse(self)
        return np.unwrap(np.angle(self_.signal + self_.noise))
    
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
        self_, _ = self._parse(self)
        
        return self.__class__( function(self_.signal + self_.noise, *args, **kargs) )
    
    def filter(self, h: np.ndarray):
        """Apply FIR filter to the electrical signal.

        Parameters
        ----------
        h : :obj:`np.ndarray`
            The FIR filter coefficients used for filtering the signal.

        Returns
        -------
        self : :obj:`electrical_signal`
            The same object with the filtered signal and noise.
        """
        self_, _ = self._parse(self)
        
        sig = np.convolve(self_.signal, h, mode='same')
        noi = np.convolve(self_.noise, h, mode='same')

        return self.__class__(sig, noi if all(noi) !=0 else None, dtype=self_.dtype) 

    def plot(self, 
             fmt: str='-', 
             n: int=None, 
             xlabel: str=None, 
             ylabel: str=None, 
             grid: bool=False,
             hold: bool=True,
             show: bool=False,
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
        self_, _ = self._parse(self)

        n = self_.size if not n else n
        t = self_.t()[:n]*1e9
        
        if not hold:
            plt.figure()

        y = (self_[:n].signal + self_[:n].noise)
        
        plt.plot(t, y, fmt, **kwargs)
        plt.xlabel(xlabel if xlabel else 'Time [ns]')
        plt.ylabel(ylabel if ylabel else 'Amplitude [V]')

        if grid:
            for i in t[:n*gv.sps][::gv.sps]:
                plt.axvline(i, color=c, ls='--', alpha=0.3, lw=1)
            plt.axvline(t[-1] + gv.dt*1e9, color=c, ls='--', alpha=0.3, lw=1)
            plt.grid(alpha=0.3, axis='y')
        
        if 'label' in kwargs.keys():
            plt.legend()

        if show:
            plt.show()
        return self

    def psd(self, 
            fmt: str='-', 
            n: int=None, 
            xlabel: str=None,
            ylabel: str=None,
            yscale: Literal['linear','dbm']='dbm', 
            grid: bool=True,
            hold: bool=True,
            show: bool=False,
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
        n = self.size if not n else n

        f, psd = sg.welch(self[:n].signal, fs=gv.fs*1e-9, nperseg=2048, scaling='spectrum', return_onesided=False, detrend=False)
        f, psd = np.fft.fftshift(f), np.fft.fftshift(psd)
        
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

        if show:
            plt.show()
        return self
    
    def plot_eye(self, n_traces=None, cmap='viridis', 
             N_grid_bins=350, grid_sigma=3, ax=None, **plot_kw):
        r"""Plots a colored eye diagram, internally calculating color density.

        Parameters
        ----------
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
        \*\*plot_kw : dict, optional
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
            - show : bool, default True (whether to call plt.show())
        
        Returns
        -------
        self
            The same object with the plotted eye diagram.
        """

        eyediagram(self.signal, self.sps(), n_traces, cmap, N_grid_bins, grid_sigma, ax, **plot_kw)
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
        size
        shape
        type
        fs
        sps
        dt
        t
        w
        f
        abs
        sum
        power
        phase
        normalize
        apply
        filter
        plot
        psd
        plot_eye
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
             grid: bool=False,
             M: int=None,
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
        n = self.size if not n else n
        t = self.t()[:n]*1e9
        
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

        ls = plt.plot( *args, **kwargs)
        plt.xlabel(xlabel if xlabel else 'Time [ns]')
        plt.ylabel(ylabel if ylabel else 'Power [mW]')
        
        if grid:
            for i,t_ in enumerate(t[:n*gv.sps][::gv.sps]):
                plt.axvline(t_, color=c, ls='--', alpha=0.3, lw=1)
                if M is not None and i%M == 0:
                    plt.axvline(t_, color=c, ls='--', alpha=0.7, lw=1)
            plt.axvline(t[-1] + gv.dt*1e9, color=c, ls='--', alpha=0.3, lw=1)
            plt.grid(alpha=0.3, axis='y')

        if label is not None:
            if isinstance(label, str):
                ls[0].set_label(label)
                ls[1].set_label(label)
            elif isinstance(label, (list, tuple)):
                ls[0].set_label(label[0])
                ls[1].set_label(label[1])
            else:
                raise ValueError('`label` must be a string or a list of strings.')
            plt.legend()
        if 'label' in kwargs.keys():
            plt.legend()

        return self
    

    def psd(self, 
            fmt: str | list='-', 
            mode: Literal['x','y','both']='x', 
            n: int=None,
            xlabel: str=None,
            ylabel: str=None, 
            yscale: Literal['linear', 'dbm']='dbm', 
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
        n = self.size if not n else n
        
        f, psd = sg.welch(self[:n].signal, fs=gv.fs*1e-9, nperseg=2048, scaling='spectrum', return_onesided=False, detrend=False)
        f, psd = np.fft.fftshift(f), np.fft.fftshift(psd)

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
        
        label = kwargs.pop('label', None) if mode == 'both' else None

        if not hold:
            plt.figure()

        ls = plt.plot( *args, **kwargs)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel if xlabel else 'Frequency [GHz]')
        plt.xlim( -3.5*gv.R*1e-9, 3.5*gv.R*1e-9 )
        plt.ylim( *ylim )
        if grid: plt.grid(alpha=0.3, color=c)
        
        if label is not None:
            if isinstance(label, str):
                ls[0].set_label(label)
                ls[1].set_label(label)
            elif isinstance(label, (list, tuple)):
                ls[0].set_label(label[0])
                ls[1].set_label(label[1])
            else:
                raise ValueError('`label` must be a string or a list of strings.')
            plt.legend()
        if 'label' in kwargs.keys():
            plt.legend()

        return self
    
class EyeShowOptions():
    def __init__(self, 
            averages : bool = None, 
            threshold : bool = None, 
            cross_points : bool = None, 
            legends : bool = None, 
            t_opt : bool = None,
            histogram : bool = None,
            all_none : bool = False
        ):

        self.averages = averages if averages is not None else all_none
        self.threshold = threshold if threshold is not None else all_none
        self.cross_points = cross_points if cross_points is not None else all_none
        self.legends = legends if legends is not None else all_none
        self.t_opt = t_opt if t_opt is not None else all_none
        self.histogram = histogram if histogram is not None else all_none

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
             show_options: EyeShowOptions=EyeShowOptions(),
             hlines: list=[],
             vlines: list=[], 
             style: Literal['dark', 'light']='dark', 
             cmap: Literal['viridis', 'plasma', 'inferno', 'cividis', 'magma', 'winter']='winter',
             smooth: bool=True,
             title: str = '',
             savefig: str=None,
             ax = None
        ):
        """ Plot eye diagram.

        Parameters
        ----------
        show_options : :obj:`typing.EyeShowOptions`, optional
            Options to show in the plot. Default show all.
        hlines : :obj:`list`, optional
            A list of time values in which hlines will be set.
        vlines : :obj:`list`, optional
            A list of voltage values in which vlines will be set.
        style : :obj:`str`, optional
            Plot style. 'dark' or 'light'.
        cmap : :obj:`str`, optional
            Colormap to plot.
        title : :obj:`str`, optional
            Title of plot.
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
            if ax is None:
                plt.style.use('dark_background')
            t_opt_color = '#60FF86'
            means_color = 'white'
            bgcolor='black'
        elif style == 'light':
            if ax is None:
                plt.style.use('default')
            t_opt_color = 'green'#'#229954'
            means_color = '#5A5A5A'
            bgcolor='white'
        else:
            raise TypeError("The `style` argument must be one of the following values ('dark', 'light')")
        
        dt = self.dt

        if show_options.histogram:
            fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [4,1],  
                                                    'wspace': 0.03},
                                                    figsize=(8,5))
        elif ax is None:
            fig, ax = plt.subplots(1,1)
            ax = [ax, ax]
        else:
            ax = [ax, ax]
            
        if title:
            plt.suptitle(f'Eye diagram {title}')
        
        ax[0].set_xlim(-1-dt,1)
        ax[0].set_ylim(self.mu0-4*self.s0, self.mu1+4*self.s1)
        ax[0].set_ylabel(r'Amplitude [V]', fontsize=12)
        ax[0].grid(color='grey', ls='--', lw=0.5, alpha=0.5)
        ax[0].set_xticks([-1,-0.5,0,0.5,1])
        ax[0].set_xlabel(r'Time [$t/T_{slot}$]', fontsize=12)
        
        if show_options.t_opt:
            ax[0].axvline(self.t_opt, color = t_opt_color, ls = '--', alpha = 0.7)
            ax[0].axvline(self.t_span0, color = t_opt_color, ls = '-', alpha = 0.4)
            ax[0].axvline(self.t_span1, color = t_opt_color, ls = '-', alpha = 0.4)

        # crossing points
        if show_options.cross_points:
            if self.y_right and self.y_left:
                ax[0].plot([self.t_left, self.t_right], [self.y_left, self.y_right], 'xr')

        # threshold
        if show_options.threshold:
            ax[0].axhline(self.threshold, c='r', ls='--')
            
            if show_options.histogram:
                ax[1].axhline(self.threshold, c='r', ls='--', label='th')
                if show_options.legends:
                    ax[1].legend()
        
        # horizontal lines
        for hl in hlines:
            ax[0].axhline(hl, c='y')
            
            if show_options.histogram:
                ax[1].axhline(hl, c='y')
        
        # vertical lines
        for vl in vlines:
            ax[0].axvline(vl, c='y')
            
            if show_options.histogram:
                ax[1].axvline (vl, c='y')
        
        # legend
        if show_options.legends: 
            ax[0].legend([r'$t_{opt}$'], fontsize=12, loc='upper right')
        
        # means
        if show_options.averages:
            ax[0].axhline(self.mu1, color = means_color, ls = ':', alpha = 0.7)
            ax[0].axhline(self.mu0, color = means_color, ls = '-.', alpha = 0.7)

            if show_options.histogram:
                ax[1].axhline(self.mu1, color = means_color, ls = ':', alpha = 0.7, label=r'$\mu_1$')
                ax[1].axhline(self.mu0, color = means_color, ls = '-.', alpha = 0.7, label=r'$\mu_0$')
                if show_options.legends:
                    ax[1].legend()

        if show_options.histogram:
            ax[1].sharey(ax[0])
            ax[1].tick_params(axis='x', which='both', length=0, labelbottom=False)
            ax[1].tick_params(axis='y', which='both', length=0, labelleft=False)
            ax[1].grid(color='grey', ls='--', lw=0.5, alpha=0.5)


        ## ADD PLOTS
        y_ = np.roll(self.y, -self.sps//2)[self.sps//2 : -self.sps//2]  # shift signal to center
        t_ = self.t[:-self.sps]

        from scipy.ndimage import gaussian_filter
        from scipy.special import expit
        from matplotlib.collections import LineCollection

        N = 350  # number of bins
        heatmap, xedges, yedges = np.histogram2d(t_, y_, bins=N)
        heatmap_smooth = gaussian_filter(heatmap, sigma=3)

        if smooth:
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            vmin, vmax = heatmap.min(), heatmap.max()
            alpha_values = expit((heatmap_smooth - (vmin + 0.05 * (vmax - vmin))) * 100 / (vmax - vmin)).T*0.8

            img = ax[0].imshow(
                heatmap_smooth.T, 
                extent=extent, 
                origin='lower', 
                aspect='auto', 
                alpha=alpha_values,
                cmap=cmap,
                interpolation='bicubic',
                resample=True,
            )
            # fig.colorbar(img, ax=ax[0])
        else:
            # ax[0].hexbin( # plot eye
            #     x = t_, 
            #     y = y_, 
            #     gridsize=500, 
            #     bins='log',
            #     alpha=0.7, 
            #     cmap=cmap 
            # )
            t_norm = (t_ - t_.min()) / (t_.max() - t_.min())
            y_norm = (y_ - y_.min()) / (y_.max() - y_.min())

            it = np.clip((t_norm * (N - 1)).astype(int), 0, N - 1)
            iy = np.clip((y_norm * (N - 1)).astype(int), 0, N - 1)

            color_values = heatmap_smooth[it, iy]
            color_values = (color_values - color_values.min()) / (color_values.max() - color_values.min())

            t = t_[:2*self.sps]

            n_traces = len(y_) // (2*self.sps)

            Y_reshaped = y_[:n_traces * 2*self.sps].reshape(-1, 2*self.sps)
            color_values_reshaped = color_values[:n_traces * 2*self.sps].reshape(-1, 2*self.sps)

            for c, y in zip(color_values_reshaped, Y_reshaped):
                points = np.array([t, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Asignar color a cada segmento según la grilla
                colors = getattr(plt.cm, cmap)(c[:-1])

                lc = LineCollection(segments, colors=colors, linewidth=1, alpha=0.05)
                ax[0].add_collection(lc)

        if show_options.histogram:
            if smooth:
                ax[1].plot(heatmap_smooth[170:180].sum(axis=0), np.linspace(y_.min(), y_.max(), 350), color=t_opt_color)
            else:
                ax[1].hist(  # plot vertical histogram 
                    y_[(t_>self.t_opt-0.05*self.t_dist) & (t_<self.t_opt+0.05*self.t_dist)], 
                    bins=200, 
                    density=True, 
                    orientation = 'horizontal', 
                    color = t_opt_color, 
                    alpha = 0.9,
                    histtype='step',
                )

        if savefig: 
            if savefig.endswith('.png'):
                plt.savefig(savefig, dpi=300)
            else:
                plt.savefig(savefig)
        
        if ax is None:
            plt.style.use('default')
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

