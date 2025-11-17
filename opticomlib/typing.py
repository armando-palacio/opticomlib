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
from scipy.ndimage import gaussian_filter
from scipy.special import expit
import scipy.signal as sg

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from typing import Literal, Any, Iterable

import warnings

from .utils import (
    str2array, 
    dbm, 
    si, 
    upfirdn,
    eyediagram,
    ComplexNumber,
    RealNumber,
    tic, toc,
)

from .logger import logging, HierLogger
logger = HierLogger(__name__)
INFO, DEBUG, WARNING = logging.INFO, logging.DEBUG, logging.WARNING

Array_Like = (list, tuple, np.ndarray)











class NULLType: 
    def __add__(self, other):  # n + null -> n
        return other
    __radd__ = __add__
    def __mul__(self, other):  # n * null -> 0
        return self
    __rmul__ = __mul__
    def __repr__(self):
        return "NULL"
    def __str__(self):
        return "NULL"
    def __sub__(self, other):
        return -other
    __rsub__ = __add__
    def __neg__(self):
        return self
    def __truediv__(self, other):
        return self
    __floordiv__ = __truediv__
    def __pow__(self, other):
        return self
    def __array_function__(self, func, types, args, kwargs):
        return self
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__' and not kwargs.get('out'):
            if ufunc == np.add or ufunc == np.subtract:
                lhs, _ = inputs
                return lhs
        return self
    def __getattr__(self, name):
        # Return self so that any attribute access returns NULL
        # This also allows chaining: NULL.real.imag -> NULL
        return self
    def __call__(self, *args, **kwargs):
        # Return self so that any method call returns NULL
        # This handles cases like NULL.conj() -> NULL
        return self
NULL = NULLType()











@logger.auto_indent_methods
class global_variables():
    r"""**Global Variables (gv)**

    This object stores the simulation-wide parameters required across the pipeline.
    It keeps track of the sampling frequency, slot rate, samples per slot, number of
    simulated slots, optical wavelength/frequency, preferred plotting style, and logging verbosity.

    .. Note:: 
        
        A slot is taken as the smallest time unit representing a binary value of the signal.
        For example, in PPM a bit is not the same as a slot. However, in OOK a bit and a slot are the same.

    This class doesn't need to be instantiated; it is already exposed as ``gv``.
    Use the :meth:`__call__` method (e.g. ``gv(**kwargs)``) to refresh parameters,
    update Matplotlib's style via ``plt_style`` or adjust the logger through ``verbose``.
    
    .. rubric:: Attributes
    .. autosummary::

        ~global_variables.sps
        ~global_variables.R
        ~global_variables.fs
        ~global_variables.wavelength
        ~global_variables.f0
        ~global_variables.N
        ~global_variables.dt
        ~global_variables.t
        ~global_variables.dw
        ~global_variables.w
        ~global_variables.plt_style
        ~global_variables.verbose

        
    .. rubric:: Methods
    .. autosummary::

        __call__
        print
        default

    Examples
    --------
    >>> gv(R=10e9, sps=8, N=100).print()

    ::

        ------------------------------
        ***    Global Variables    ***
        ------------------------------
                sps :  8
                R   :  1.00e+10
                fs  :  8.00e+10
                λ0  :  1.55e-06
                f0  :  1.93e+14
                N   :  100
                dt  :  1.25e-11
                t   :  [0.00e+00 1.25e-11 2.50e-11 ... 9.97e-09 9.99e-09 1.00e-08]
                dw  :  6.28e+08
        Config
        ------
                plt_style :  "fast"
                verbose   :  None

    Also can be define new variables trough \*\*kwargs. If at least two of this arguments (``sps``, ``fs`` and ``R``) are not provided
    a warning will be raised and the default values will be used.

    >>> gv(alpha=0.5, beta=0.3).print()
    
    ::

        ------------------------------
        ***    Global Variables    ***
        ------------------------------
                sps :  16
                R   :  1.00e+09
                fs  :  1.60e+10
                λ0  :  1.55e-06
                f0  :  1.93e+14
                N   :  128
                dt  :  6.25e-11
                t   :  [0.00e+00 6.25e-11 1.25e-10 ... 1.28e-07 1.28e-07 1.28e-07]
                dw  :  4.91e+07
        Config
        ------
                plt_style :  "fast"
                verbose   :  None
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
        """Frequency step in rad/s"""
        self.w = 2*pi*fftshift(fftfreq(self.N*self.sps))*self.fs
        """Frequency array in rad/s"""
        self.plt_style = 'fast'
        """Matplotlib plot style, ``"fast"`` by default."""
        plt.style.use(self.plt_style)
        self.verbose = None
        """Logging verbosity level, ``None`` by default. Can be set to ``DEBUG`` or 10, ``INFO`` or 20, ``WARNING`` or 30."""

            
    def __str__(self):
        title = 3*'*' + '    Global Variables    ' + 3*'*'
        sub = len(title)*'-'

        names = list(gv.__dict__.keys())
        others = [name for name in names if name not in ['sps', 'R', 'fs', 'wavelength', 'f0', 'N', 'dt', 'dw', 't', 'w', 'plt_style', 'verbose']]

        msg = f'\n{sub}\n{title}\n{sub}\n\t' + \
            f'sps :  {self.sps}\n\t' + \
            f'R   :  {self.R:.2e}\n\t' + \
            f'fs  :  {self.fs:.2e}\n\t' + \
            f'λ0  :  {self.wavelength:.2e}\n\t' + \
            f'f0  :  {self.f0:.2e}\n\t' + \
            f'N   :  {self.N}\n\t' + \
            f'dt  :  {self.dt:.2e}\n\t' + \
            f't   :  {self.t}\n\t' + \
            f'dw  :  {self.dw:.2e}\n'

        msg += f'  Config\n  ------\n\t' + \
            f'plt_style :  "{self.plt_style}"\n\t' + \
            f'verbose   :  {self.verbose}\n'
            
        if others:
            msg += '  Custom\n  ------\n\t' + '\n\t'.join([f'{name} : {getattr(self, name)}' for name in others]) + '\n'

        return msg
    
    def print(self):
        """ Prints the global variables in a formatted manner"""
        np.set_printoptions(precision=2, threshold=20)
        print(self)

    def __call__(
            self, 
            sps: int=None, 
            R: float=None, 
            fs: float=None, 
            wavelength: float=1550e-9, 
            N: int=None, 
            plt_style : Literal['ggplot', 'bmh', 'dark_background', 'fast', 'default']='fast', 
            verbose=None,
            **kargs
        ) -> Any:
        """
        Configures the instance with the provided parameters.

        Parameters
        ----------
        sps : int, optional
            Samples per slot.
        R : float, optional
            Rate in Hz.
        fs : float, optional
            Sampling frequency in Samples/s.
        wavelength : float, optional
            Wavelength in meters. Default is 1550e-9.
        N : int, optional
            Number of samples.
        plt_style : str, optional
            Matplotlib plot style. Default is "fast".
        verbose : int | None, optional
            Verbosity level for logging.
        **kargs : dict
            Additional custom parameters.

        Returns
        -------
        gv
            The instance itself.

        Notes
        -----
        In the absence of parameters, default values are used. Missing parameters are calculated from the provided ones, prioritizing the default value of **gv.R** when more than one of **sps**, **fs**, and **R** is not provided.
        """
        logger.debug('setting gv()')

        if verbose is not None :
            self.verbose = verbose
            logger.logger.setLevel(self.verbose)

        if sps:
            self.sps = int(np.round(sps))
            if R:
                self.R = R
                self.fs = R*self.sps
            elif fs:
                self.fs = fs
                self.R = fs/self.sps
            else:
                logger.warning("'R' set to default value (%.2e bits/s)", self.R)
                self.fs = self.R*self.sps

        elif R: 
            self.R = R
            if fs:
                self.fs = fs
                self.sps = int(np.round(fs/R))
            else:
                logger.warning("'sps' set to default value (%d S/bit)", self.sps)
                self.fs = R*self.sps

        elif fs:
            logger.warning("'R' set to default value (%.2e bits/s)", self.R)
            self.fs = fs
            self.sps = int(np.round(fs/self.R))

        else:
            logger.warning("'sps', 'R' and 'fs' will be set to default values (%d S/bit, %.2e bits/s, %.2e Hz)", self.sps, self.R, self.fs)
        
        self.dt = 1/self.fs

        self.N = N if N is not None else self.N
        self._set_t_dw_w()
        
        self.wavelength = wavelength
        self.f0 = c/wavelength

        if plt_style != self.plt_style:
            self.plt_style = plt_style
            plt.rcdefaults()
            plt.style.use(self.plt_style)

        logger.info('Global variables set to, sps: %d, R: %.2e, fs: %.2e, N: %d, wavelength: %.2e', self.sps, self.R, self.fs, self.N, self.wavelength)

        if kargs:
            for key, value in kargs.items():
                setattr(self, key, value)
        
        return self

    def _set_t_dw_w(self):
        self.t = np.linspace(0, self.N*self.sps/self.fs, self.N*self.sps, endpoint=True)
        self.dw = 2*pi*self.fs/(self.N*self.sps)
        self.w = 2*pi*fftshift(fftfreq(self.N*self.sps))*self.fs

    def default(self):
        """ Return all parameters to default values."""
        logger.debug('resetting gv to default()')

        self.sps = 16
        self.R = 1e9
        self.fs = self.R*self.sps
        self.dt = 1/self.fs
        self.wavelength = 1550e-9
        self.f0 = c/self.wavelength
        self.N = 128
        self._set_t_dw_w()
        self.plt_style = 'fast'
        plt.rcdefaults()
        plt.style.use(self.plt_style)
        self.verbose = None
        # Reset logger level to default (NOTSET allows propagation to parent)
        logger.logger.setLevel(logging.NOTSET)

        attrs = [attr for attr in dir(gv) if not callable(getattr(gv, attr)) and not attr.startswith("__") and not (attr in ['sps', 'R', 'fs', 'dt', 'wavelength', 'f0', 'N', 't', 'w', 'dw', 'plt_style'])]

        logger.info('Global variables set to default, sps: %d, R: %.2e, fs: %.2e, N: %d, wavelength: %.2e', self.sps, self.R, self.fs, self.N, self.wavelength)
        
        for attr in attrs:
            delattr(self, attr)
        return self

gv = global_variables()












@logger.auto_indent_methods
class binary_sequence():
    r"""**Binary Sequence**

    This class provides methods and attributes to work with binary sequences. 
    The binary sequence can be provided as a string, list, tuple, or numpy array.

    .. rubric:: Attributes
    .. autosummary::

        ~binary_sequence.data
        ~binary_sequence.execution_time
        ~binary_sequence.ones
        ~binary_sequence.zeros
        ~binary_sequence.size
        ~binary_sequence.type
        ~binary_sequence.sizeof

    .. rubric:: Methods
    .. autosummary::

        prbs
        print
        to_numpy
        flip
        hamming_distance
        dac
        plot

    .. table:: **Implemented Operators**
        :widths: 10 90
        :align: center

        +--------------------------------+--------------------------------------------------------------+
        | Operator                       | Description                                                  |
        +================================+==============================================================+
        | ``~``                          | ``~a`` NOT operation, bit by bit.                            |
        +--------------------------------+--------------------------------------------------------------+
        | ``&``                          | ``a & b`` AND operation, bit by bit                          |
        +--------------------------------+--------------------------------------------------------------+
        | ``|``                          | ``a | b`` OR operation, bit by bit                           |
        +--------------------------------+--------------------------------------------------------------+
        | ``^``                          | ``a ^ b`` XOR operation, bit by bit                          |
        +--------------------------------+--------------------------------------------------------------+
        | ``+``                          | ``a + b`` concatenate ``a ∪ b``; ``b + a`` concatenate       |
        |                                | ``b ∪ a``.                                                   |
        +--------------------------------+--------------------------------------------------------------+
        | ``*``                          | ``a * n`` (n > 1 integer) repeats ``a`` n times; ``a * b``   |
        |                                | equivalent to ``&`` operator.                                |
        +--------------------------------+--------------------------------------------------------------+
        | ``==``                         | ``a == b`` compares elements, returning a ``binary_sequence``|
        |                                | mask of matches.                                             |
        +--------------------------------+--------------------------------------------------------------+
        | ``!=``                         | ``a != b`` compares elements, returning a ``binary_sequence``|
        |                                | mask of differences.                                         |
        +--------------------------------+--------------------------------------------------------------+
        | ``[:]``                        | ``a[i]`` returns the integer value at index ``i``;           |
        |                                | ``a[i:j]`` returns a sliced ``binary_sequence``.             |
        +--------------------------------+--------------------------------------------------------------+
        | ``<``, ``<=``, ``>``, ``>=``   | Not implemented                                              |
        | ``-``, ``/``, ``//``, ``<<``,  |                                                              |
        | ``>>``,                        |                                                              |
        +--------------------------------+--------------------------------------------------------------+
    """

    def __init__(self, data: str | Iterable): 
        logger.debug('%s.__init__(%s)', self.__class__.__name__, type(data).__name__)

        if isinstance(data, binary_sequence):
            data = data.data
        elif isinstance(data, str):
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
        logger.debug('__str__()')

        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'

        np.set_printoptions(precision=0, threshold=100)
        data = str(self.data)

        msg = f'\n{sub}\n{title}\n{sub}\n\t' + \
            f'data  :  {data} (shape: {self.data.shape})\n\t' + \
            f'ones  :  {self.ones}\n\t' + \
            f'zeros :  {self.zeros}\n\t' + \
            f'size  :  {self.sizeof} bytes\n\t' + \
            f'time  :  {si(self.execution_time, "s", 2)}\n'
        return msg
    
    def __repr__(self):
        logger.debug('__repr__()')

        np.set_printoptions(threshold=100)
        return f'binary_sequence({str(self.data)})'

    def __len__(self):
        logger.debug('__len__()')
        return self.size
    
    def __array__(self, dtype=None):
        """Return the array representation of the binary sequence.
        
        This method provides the basic array conversion for NumPy compatibility.
        It returns the data.
        This is the fundamental protocol that allows the object to be converted to a NumPy array
        when needed, enabling direct use in NumPy functions that expect array-like objects.
        
        Unlike ``__array_ufunc__`` and ``__array_function__``, this method is called for basic array
        conversion and does not handle specific NumPy operations - it simply provides the
        underlying data as an array.
        
        Parameters
        ----------
        dtype : np.dtype, optional
            Desired data type of the array.
                
        Returns
        -------
        np.ndarray
            The array representation of the binary data.
        """
        logger.debug('__array__()')
        arr = self.data
        return arr
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying NumPy array for array-like methods.
        
        This method allows instances of ``binary_sequence`` to access NumPy array methods
        (like max, min, sum, etc.) directly as if they were arrays. If the requested attribute
        is a method or property of np.ndarray, it will be called on the array representation
        of this object.
        
        Parameters
        ----------
        name : str
            The name of the attribute being accessed.
            
        Returns
        -------
        result
            The result of calling the attribute on the array representation.
            
        Raises
        ------
        AttributeError
            If the attribute is not found in np.ndarray.
        """
        logger.debug('__getattr__(%s)', name)
        # Check if the attribute exists in np.ndarray and is not a private/internal attribute
        if hasattr(np.ndarray, name) and not name.startswith('__'):
            logger.debug("Delegating attribute '%s' to ndarray for %s", name, self.__class__.__name__)
            return getattr(self.__array__(), name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle NumPy universal functions (ufuncs) by converting the object to an array.
        
        This method is specifically designed for NumPy's universal functions, which are
        element-wise operations like np.add, np.sin, np.multiply, etc. When a ufunc is
        called on an instance of this class, this method intercepts the call, converts
        any instances of this class in the inputs to arrays using ``__array__()``, and then
        applies the ufunc to the converted arrays.
        
        Unlike ``__array__``, which provides basic array conversion, this method handles
        the execution of specific ufunc operations. Unlike ``__array_function__``, which
        handles higher-level array functions, this focuses on element-wise operations.
        
        Parameters
        ----------
        ufunc : numpy.ufunc
            The NumPy universal function being called.
        method : str
            The method of the ufunc (e.g., '__call__', 'reduce').
        *inputs : tuple
            The input arguments to the ufunc.
        **kwargs : dict
            Keyword arguments passed to the ufunc.
            
        Returns
        -------
        result
            The result of applying the ufunc to the converted inputs, wrapped in the
            appropriate class if the result is an array with compatible shape.
        """
        logger.debug('__array_ufunc__(%s, %s)', ufunc.__name__, method)
        
        if method == '__call__' and not kwargs.get('out'):
            if ufunc == np.add:
                lhs, rhs = inputs
                if isinstance(rhs, binary_sequence):
                    return rhs.__radd__(lhs)
            if ufunc == np.multiply:
                lhs, rhs = inputs
                if isinstance(rhs, binary_sequence):
                    return rhs.__mul__(lhs)
        
        # Convert inputs that are instances of this class to arrays
        new_inputs = []
        for inp in inputs:
            if isinstance(inp, self.__class__):
                new_inputs.append(inp.__array__())
            else:
                new_inputs.append(inp)
        
        result = getattr(ufunc, method)(*new_inputs, **kwargs)

        try: 
            if isinstance(result, np.ndarray):
                return binary_sequence(result)
        except (ValueError, TypeError) as e:
            logger.debug('Failed to convert ufunc result to binary_sequence: %s', e)
        return result
    
    def __array_function__(self, func, types, args, kwargs):
        """Handle NumPy array functions by converting the object to an array.
        
        This method is called for NumPy array functions that are not universal functions (ufuncs).
        It handles higher-level array operations like np.sum, np.mean, np.concatenate, np.fft.fft,
        and other array manipulation functions. When such a function is called on an instance of
        this class, this method converts any instances in the arguments to arrays and then calls
        the function with the converted arguments.
        
        Key differences from other protocols:

        - ``__array__``: Provides basic array conversion without handling specific operations.
        - ``__array_ufunc__``: Handles element-wise universal functions (ufuncs) like np.add, np.sin, etc.
        - ``__array_function__``: Handles higher-level array functions and transformations like np.abs, etc.
        
        This enables seamless integration with NumPy's array function ecosystem, allowing
        instances to be used directly in functions like np.sum(x), np.mean(x), np.concatenate([x, y]),
        etc., without manual conversion.
        
        Parameters
        ----------
        func : callable
            The NumPy array function being called (e.g., np.sum, np.mean).
        types : tuple
            The types of all arguments passed to the function.
        args : tuple
            The positional arguments passed to the function.
        kwargs : dict
            The keyword arguments passed to the function.
        
        Returns
        -------
        result
            The result of applying the NumPy array function to the converted arguments, wrapped in the
            appropriate class if the result is an array with compatible shape.
        """
        logger.debug('__array_function__(%s)', func.__name__)

        def _convert(obj):
            if isinstance(obj, self.__class__):
                return obj.__array__()
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_convert(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            else:
                return obj
        
        # Convert args and kwargs that contain instances to arrays
        new_args = _convert(args)
        new_kwargs = _convert(kwargs)
        
        result = func(*new_args, **new_kwargs)

        try:
            if isinstance(result, np.ndarray):
                return binary_sequence(result)
        except (ValueError, TypeError) as e:
            logger.debug('Failed to convert array_function result to binary_sequence: %s', e)
        return result

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
        logger.debug('__getitem__(%s)', slice)
        if isinstance(slice, int):
            return self.data[slice]
        return binary_sequence(self.data[slice])
    
    def __eq__(self, other):
        logger.debug('__eq__(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)

        return binary_sequence(self.data == other.data)

    def __ne__(self, other):
        logger.debug('__ne__(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)

        return binary_sequence(self.data != other.data)


    def __add__(self, other): 
        logger.debug('__add__(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)
        
        out = np.concatenate((self.data, other.data))
        return binary_sequence(out)
    
    def __radd__(self, other): 
        logger.debug('__radd__(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)

        out = np.concatenate((other.data, self.data))
        return binary_sequence(out)

    def __mul__(self, other):
        logger.debug('__mul__(%s)', type(other).__name__)

        if isinstance(other, int) and other > 1:
            # Repeat the sequence other times
            repeated = np.tile(self.data, other)
            return binary_sequence(repeated)
        else:
            # Convert other to binary_sequence if necessary
            if not isinstance(other, binary_sequence):
                other = binary_sequence(other)
            
            result = self.data * other.data
            return binary_sequence(result)
    __rmul__ = __mul__

    def __invert__(self):
        logger.debug('__invert__()')

        return binary_sequence(~self.data.astype(bool))
    
    def __or__(self, other):
        logger.debug('__or__(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)

        return binary_sequence(self.data | other.data)
    __ror__ = __or__

    def __and__(self, other):
        logger.debug('__and__(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)

        return binary_sequence(self.data & other.data)
    __rand__ = __and__

    def __xor__(self, other):
        logger.debug('__xor__(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)

        return binary_sequence(self.data ^ other.data)
    __rxor__ = __xor__
    
    # properties
    @property
    def ones(self):
        """Number of ones in the binary sequence."""
        x = np.sum(self.data==1)
        logger.debug('ones: %d', x)
        return x
    
    @property
    def zeros(self):
        """Number of zeros in the binary sequence."""
        x = np.sum(self.data == 0)
        logger.debug('zeros: %d', x)
        return x
    
    @property
    def size(self):
        """Number of slots of the binary sequence."""
        x = self.data.size
        logger.debug('size: %d', x)
        return x

    @property
    def type(self): 
        """Object type."""
        x = type(self)
        logger.debug('type: %s', x.__name__)
        return x
    
    @property
    def sizeof(self):
        """Memory size of object in bytes."""
        logger.debug('sizeof')
        x = sizeof(self)
        logger.debug('sizeof: %d bytes', x)
        return x
    
    # static methods
    @staticmethod
    def prbs(order: int, seed: int=None, len: int=None, return_seed: bool=False):
        r"""Pseudorandom binary sequence generator (PRBS) (*static method*).

        Parameters
        ----------
        order : :obj:`int`, {7, 9, 11, 15, 20, 23, 31}
            degree of the generating pseudorandom polynomial
        len : :obj:`int`, optional
            lenght of output binary sequence
        seed : :obj:`int`, optional
            seed of the generator (initial state of the LFSR).
            It must be provided if you want to continue the sequence.
            Default is 2**order-1.
        return_seed : :obj:`bool`, optional
            If True, the last state of LFSR is returned. Default is False.

        Returns
        -------
        out : :obj:`binary_sequence`
            generated pseudorandom binary sequence if `return_seed` is False
        out, last_seed : :obj:`tuple` of (:obj:`binary_sequence`, : obj:`int`)
            generated pseudorandom binary sequence and last state of LFSR if `return_seed`
        """
        tic()
        taps = {
            7: [7, 6],
            9: [9, 5],
            11: [11, 9],
            15: [15, 14],
            20: [20, 3],
            23: [23, 18],
            31: [31, 28],
        }
        seed = seed % (2**order) if seed is not None else (1 << order) - 1
        if seed == 0:
            seed = 1
            warnings.warn(
                "The seed can't be 0 or a multiple of 2**order. It has been changed to 1.",
                UserWarning,
            )

        if len is not None:
            if not isinstance(len, int):
                raise TypeError("The parameter `len` must be an integer.")
            elif len <= 0:
                raise ValueError(
                    "The parameter `len` must be an integer greater than cero."
                )
        else:
            len = 2**order - 1

        if order not in taps.keys():
            raise ValueError(
                "The parameter `order` must be one of the following values (7, 9, 11, 15, 20, 23, 31)."
            )

        prbs = np.empty((len,), dtype=np.uint8)  # Preallocate memory for the PRBS
        lfsr = seed  # initial state of the LFSR
        tap1, tap2 = np.array(taps[order]) - 1

        index = 0
        while index < len:
            prbs[index] = lfsr & 1
            new = ((lfsr >> tap1) ^ (lfsr >> tap2)) & 1
            lfsr = ((lfsr << 1) | new) & (1 << order) - 1
            index += 1

        output = binary_sequence(prbs)
        output.execution_time = toc()

        if not return_seed:
            return output
        return output, lfsr

    # methods
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
        logger.debug('print()')
        print(self.__str__(msg))
        return self
    
    def to_numpy(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Return a NumPy representation of the binary sequence. This method is similar to ``__array__``, the diference is that some libraries, as matplotlib, used to call this method to get the numpy array.
        """
        logger.debug("to_numpy(dtype=%s)", dtype)
        return np.array(self.data, dtype=dtype)
    
    def flip(self):
        """Invert the binary sequence. Equivalent to the ``~`` operator.

        Returns
        -------
        binary_sequence
            A new binary sequence object with the result of the inversion.
        """
        logger.debug('flip()')

        return ~self

    def hamming_distance(self, other):
        """Calculate the Hamming distance to another binary sequence of the same length.

        Parameters
        ----------
        other : :obj:`str` or :obj:`binary_sequence` or :obj:`Array_Like`
            The binary sequence to compare.
        Returns
        -------
        :obj:`int`
            The Hamming distance between the two binary sequences.
        """
        logger.debug('hamming_distance(%s)', type(other).__name__)

        if not isinstance(other, binary_sequence):
            other = binary_sequence(other)
        return np.sum(self != other)
    
    def dac(self, h: np.ndarray):
        """Apply upsampling and FIR filtering to the binary sequence for digital-to-analog conversion.

        This method upsamples the binary sequence by the global samples per slot (gv.sps) and applies the provided FIR filter to produce an electrical signal.

        Parameters
        ----------
        h : :obj:`np.ndarray`
            The FIR filter impulse response to use for shaping the signal.

        Returns
        -------
        :obj:`electrical_signal`
            The resulting electrical signal after upsampling, filtering, and downsampling.
        """
        logger.debug('dac()')
        return electrical_signal(upfirdn(x=self.data, h=h, up=gv.sps, dn=1))
    
    def plot(self, **kwargs):
        """Plot the binary sequence using matplotlib.

        Parameters
        ----------
        **kwargs : :obj:`dict`
            Additional keyword arguments to customize the plot.

        Returns
        -------
        :obj:`matplotlib.axes.Axes`
            The axes object of the plot.
        """
        logger.debug('plot()')

        _, ax = plt.subplots()
        ax.step(np.arange(self.size), self.data, where='post', **kwargs)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Binary Sequence')
        ax.set_yticks([0, 1])
        ax.grid(True)

        return self











@logger.auto_indent_methods
class electrical_signal():
    """**Electrical Signal**

    This class provides methods and attributes to work with electrical signals. 
    It has overloaded operators necessary to properly interpret 
    the ``+``, ``-``, ``*``, ``/``, ``**``, and comparison operations as any numpy array.

    .. rubric:: Attributes
    .. autosummary::

        ~electrical_signal.signal
        ~electrical_signal.noise
        ~electrical_signal.execution_time
        ~electrical_signal.size
        ~electrical_signal.real
        ~electrical_signal.imag
        ~electrical_signal.type
        ~electrical_signal.fs
        ~electrical_signal.sps
        ~electrical_signal.dt
        ~electrical_signal.t
        ~electrical_signal.sizeof

    .. rubric:: Methods
    .. autosummary::

        __init__
        __call__
        print
        to_numpy
        conj
        sum
        w
        f
        abs
        power
        normalize
        phase
        filter
        plot
        psd
        plot_eye
        grid
        legend
        show

    .. table:: **Implemented Operators**
        :widths: 10 90
        :align: center

        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | Operator                       | Description                                                                                      |
        +================================+==================================================================================================+
        | ``+``                          | ``a + b`` adds signals and noises element-wise;                                                  |
        |                                | ``sig = (a.signal + b.signal), noi = (a.noise + b.noise)``                                       |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``-``                          | ``a - b`` subtracts signals and noises element-wise;                                             |
        |                                | ``sig = (a.signal - b.signal), noi = (a.noise - b.noise)``                                       |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``*``                          | ``a * b`` multiplies two ``electrical_signal``;                                                  |
        |                                | ``sig = (a.signal*b.signal)``                                                                    |
        |                                | ``noi = (a.signal*b.noise + a.noise*b.signal + a.noise*b.noise)``                                |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``/``                          | ``a / n`` divides signal and noise by a scalar;                                                  |
        |                                | ``sig = (a.signal/n), noi = (a.signal/n)``                                                       |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``//``                         | ``a // n`` floor divides signal and noise by a scalar;                                           |
        |                                | ``sig = (a.signal//n), noi = (a.signal//n)``                                                     |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``**``                         | ``a ** n`` raises ``electrical_signal`` to a power;                                              |
        |                                | ``- n=1  -->  sig = (a.signal), noi = (a.noise)``;                                               |
        |                                | ``- n=2  -->  sig = (a.signal**2), noi = (2*a.signal*a.noise + a.noise**2)``                     |
        |                                | ``- n=other  -->  sig = (a.signal + a.noise)**n, noi=NULL``                                      |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``>``                          | ``a > b`` compares signals element-wise, returns                                                 |
        |                                | ``binary_sequence`` mask.                                                                        |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``<``                          | ``a < b`` compares signals element-wise, returns                                                 |
        |                                | ``binary_sequence`` mask.                                                                        |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``==``                         | ``a == b`` compares signals element-wise, returns                                                |
        |                                | ``np.ndarray`` mask.                                                                             |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``[:]``                        | ``a[i]`` returns the value at index ``i``;                                                       |
        |                                | ``a[i:j]`` returns a sliced ``electrical_signal``.                                               |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
        | ``-`` (unary)                  | ``-a`` negates signal and noise.                                                                 |
        +--------------------------------+--------------------------------------------------------------------------------------------------+
    """
    def __init__(self, signal: str | Iterable, noise: str | Iterable = NULL, dtype: np.dtype=None) -> None:
        """ Initialize the electrical signal object.

        Parameters
        ----------
        signal : :obj:`str` or 1D array_like or scalar
            The signal values.
        noise : :obj:`str` or 1D array_like or scalar, optional
            The noise values. Defaults to ``NULL``.
        dtype : :obj:`np.dtype`, optional
            The desired data type for the signal and noise arrays. If not provided, the data type
            will be inferred from the input data. Defaults to ``None``.

        Notes
        -----
        The signal and noise can be provided as a string, in which case it will be converted to a 
        ``numpy.array`` using the :func:`str2array` function. For example:
        
        .. code-block:: python

            >>> electrical_signal('1 2 3,4,5')  # separate values by space or comma indistinctly
            electrical_signal(signal=[1 2 3 4 5],
                               noise=NULL)
            >>> electrical_signal('1+2j, 3+4j, 5+6j') # complex values
            electrical_signal(signal=[1.+2.j 3.+4.j 5.+6.j],
                               noise=NULL)
        """    
        if self.__class__ == electrical_signal:
            logger.debug("%s.__init__()", self.__class__.__name__)

            if isinstance(signal, electrical_signal):
                signal, noise = signal.signal, signal.noise
            else:
                signal, noise = self._prepare_arrays(signal, noise, dtype)

            if signal.ndim > 1 or signal.size < 1:
                raise ValueError(f"Signal must be scalar or 1D array for electrical_signal, invalid shape {signal.shape}")
            
            if signal.ndim == 0:
                signal = signal[np.newaxis]
                if noise is not NULL:
                    noise = noise[np.newaxis]
        
        self.signal = signal
        """The signal values, a 1D array-like values."""
        self.noise = noise
        """The noise values, a 1D array-like values."""
        self.execution_time = 0.
        """The execution time of the last operation performed."""

    def __str__(self, title: str=None): 
        logger.debug("__str__()")
        
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

        pw_sig_w = self.power('W', 'signal')
        pw_sig_dbm = dbm(pw_sig_w)

        pw_noi_w = self.power('W', 'noise')
        pw_noi_dbm = dbm(pw_noi_w)

        pw_all_w = self.power('W', 'all')
        pw_all_dbm = dbm(pw_all_w) 
        
        msg = f'\n{sub}\n{title}\n{sub}\n'+ tab + \
            f'signal:     {signal} (shape: {self.shape})\n'+ tab + \
            f'noise:      {noise} (shape: {self.shape if self.noise is not NULL else None})\n'+ tab + \
            f'pow_signal: {si(pw_sig_w, 'W', 1)} ({pw_sig_dbm:.1f} dBm)\n'+ tab + \
            f'pow_noise:  {si(pw_noi_w, 'W', 1)} ({pw_noi_dbm:.1f} dBm)\n'+ tab + \
            f'pow_total:  {si(pw_all_w, 'W', 1)} ({pw_all_dbm:.1f} dBm)\n'+ tab + \
            f'len:        {self.size}\n' + tab + \
            f'elem_type:  {self.dtype}\n' + tab + \
            f'mem_size:   {self.sizeof} bytes\n' + tab + \
            f'time:       {si(self.execution_time, "s", 2)}\n'
        return msg
    
    def __repr__(self):
        logger.debug("__repr__()")

        np.set_printoptions(precision=3, threshold=20)
        
        if self.noise is not NULL:
            return f'electrical_signal({str(self.signal)})'
        return f'electrical_signal(signal={str(self.signal)},\n\t\t   noise={str(self.noise)})'

    def __len__(self): 
        logger.debug("__len__()")
        return self.size
    
    def __iter__(self):
        logger.debug("__iter__()")
        return iter(self.__array__())
    
    def __array__(self, dtype=None):
        logger.debug("__array__()")
        arr = self.signal + self.noise
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    
    def __getattr__(self, name):
        logger.debug("__getattr__('%s')", name)
        # Check if the attribute exists in np.ndarray and is not a private/internal attribute
        if hasattr(np.ndarray, name) and not name.startswith('__'):
            logger.debug("Delegating attribute '%s' to ndarray for %s", name, self.__class__.__name__)
            return getattr(self.__array__(), name)
        logger.debug("Attribute '%s' not found in %s", name, self.__class__.__name__)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        logger.debug("__array_ufunc__(%s, %s)", getattr(ufunc, '__name__', ufunc), method)

        if method == '__call__' and not kwargs.get('out'):
            if ufunc == np.add:
                lhs, rhs = inputs
                if isinstance(rhs, electrical_signal):
                    return rhs.__add__(lhs)
            if ufunc == np.subtract:
                lhs, rhs = inputs
                if isinstance(rhs, electrical_signal):
                    return (-rhs).__add__(lhs)
            if ufunc == np.multiply:
                lhs, rhs = inputs
                if isinstance(rhs, electrical_signal):
                    return rhs.__mul__(lhs)

        # Convert inputs that are instances of this class to arrays
        new_inputs = []
        for inp in inputs:
            if isinstance(inp, self.__class__):
                new_inputs.append(inp.__array__())
            else:
                new_inputs.append(inp)
        # Call the ufunc
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        
        # If the result is an array with compatible shape, wrap it in the class
        if isinstance(result, np.ndarray):
            if self.__class__ == electrical_signal and result.ndim == 1:
                return self.__class__(result)
            elif self.__class__ == optical_signal and result.ndim in [1, 2]:
                return self.__class__(result)
            else:
                return result
        return result
    
    def __array_function__(self, func, types, args, kwargs):
        logger.debug("__array_function__(%s, %s, %s)", getattr(func, '__name__', func), args, kwargs)

        @logger.auto_indent
        def _convert(obj):
            logger.debug("_convert(%s)", type(obj).__name__)
            if isinstance(obj, self.__class__):
                return obj.__array__()
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_convert(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            else:
                return obj
        
        # Convert args and kwargs that contain instances to arrays
        new_args = _convert(args)
        new_kwargs = _convert(kwargs)
    
        result = func(*new_args, **new_kwargs)
        
        # If the result is an array with compatible shape, wrap it in the class
        if isinstance(result, np.ndarray):
            if self.__class__ == electrical_signal and result.ndim == 1:
                return self.__class__(result)
            elif self.__class__ == optical_signal and result.ndim in [1, 2]:
                return self.__class__(result)
            else:
                return result
        return result
    
    def __add__(self, other):
        logger.debug("__add__()")
        other, _ = self._parse(other)

        sig = self.signal + other.signal
        noi = self.noise + other.noise

        return self.__class__(sig, noi)
    
    def __radd__(self, other):
        logger.debug("__radd__()")
        return self.__add__(other)
    
    def __neg__(self):
        logger.debug("__neg__()")
        sig = -self.signal
        noi = -self.noise 
        return self.__class__(sig, noi)

    def __sub__(self, other):
        logger.debug("__sub__()")
        other, _ = self._parse(other)
        return self + (-other)
        
    def __rsub__(self, other):
        logger.debug("__rsub__()")
        other, _ = self._parse(other)
        return (-self) + other
    
    def __mul__(self, other):
        logger.debug("__mul__()")
        other, _ = self._parse(other)

        sig = self.signal*other.signal
        noi = self.signal*other.noise + self.noise*other.signal + self.noise*other.noise

        return self.__class__(sig, noi)
        
    def __rmul__(self, other):
        logger.debug("__rmul__()")
        return self.__mul__(other)
        
    def __truediv__(self, number: int):
        logger.debug("__truediv__()")
        if not isinstance(number, ComplexNumber):
            logger.error("Division by unsupported type %s", type(number))
            raise TypeError(f"Can't divide electrical_signal by type {type(number)}")
        if number == 0:
            logger.error("Attempted division by zero in %s", self.__class__.__name__)
            raise ZeroDivisionError("Can't divide electrical_signal by zero")

        return self.__class__(self.signal / number, self.noise / number)
    
    def __floordiv__(self, other):
        logger.debug("__floordiv__()")
        x = (self/other)
        return self.__class__( np.floor(x.signal), np.floor(x.noise) )
        
    def __getitem__(self, key):
        logger.debug("__getitem__(%s)", key)
        if isinstance(key, slice):
            if self.noise is NULL:
                return self.__class__( self.signal[key] ) 
            return self.__class__( self.signal[key], self.noise[key] )
        elif isinstance(key, int):
            if self.noise is NULL:
                return self.signal[key]
            return self.__class__( self.signal[key], self.noise[key] )
        raise TypeError(f"Invalid argument type. {key} of type {type(key)}")
    
    def __gt__(self, other):
        logger.debug("__gt__()") 
        other, _ = self._parse(other)
        
        x_r = self.signal + self.noise
        x_l = other.signal + other.noise

        return binary_sequence(x_r > x_l)
        
    def __lt__(self, other):
        logger.debug("__lt__()")
        return other - self > 0 
    
    def __eq__(self, other):
        logger.debug("__eq__()")
        other, _ = self._parse(other)

        x_r = self.signal + self.noise
        x_l = other.signal + other.noise

        return x_r == x_l
    
    def __pow__(self, other):
        logger.debug("__pow__()")

        if not isinstance(other, RealNumber):
            raise TypeError(f"Can't exponentiate electrical_signal by type {type(other)}")
        
        if other == 0:
            sig = np.ones_like(self.signal)
            noi = NULL
        elif other == 1:
            sig = self.signal
            noi = self.noise 
        elif other == 2:
            sig = self.signal**2
            noi = 2*self.signal*self.noise + self.noise**2
        else:
            sig = (self.signal + self.noise) ** other 
            noi = NULL

        return self.__class__(sig, noi)
    
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
        logger.debug("__call__(domain='%s', shift=%s)", domain, shift)

        if domain == 'w' or domain == 'f':
            signal = fft(self.signal, axis=-1)
            noise = fft(self.noise, axis=-1)
              
        elif domain == 't':
            signal = ifft(self.signal, axis=-1)
            noise = ifft(self.noise, axis=-1)
        
        else:
            raise ValueError("`domain` must be one of the following values ('t', 'w', 'f')")
        
        if shift:
            if domain == 'w' or domain == 'f':
                signal = fftshift(signal, axes=-1)
                noise = fftshift(noise, axes=-1)
            else: 
                signal = ifftshift(signal, axes=-1)
                noise = ifftshift(noise, axes=-1)

        return self.__class__(signal, noise)

    # properties
    @property
    def index(self) -> np.ndarray:
        logger.debug("index")
        return np.arange(self.signal.size)
    
    @property
    def size(self) -> np.ndarray:
        """Number of samples of the electrical signal."""
        logger.debug("size")
        return self.signal.size
    
    @property
    def real(self) -> np.ndarray:
        """Real part of the electrical signal (signal + noise)."""
        logger.debug("real")
        print(self.noise.real)
        return self.__class__(self.signal.real, self.noise.real)
    
    @property
    def imag(self) -> np.ndarray:
        """Imaginary part of the electrical signal (signal + noise)."""
        logger.debug("imag")
        return self.__class__(self.signal.imag, self.noise.imag)
    
    @property
    def type(self): 
        """Object type."""
        logger.debug("type")
        return type(self)

    @property
    def sizeof(self):
        """Memory size of object in bytes."""
        logger.debug("sizeof")
        return sizeof(self)

    @property
    def fs(self): 
        """Sampling frequency of the electrical signal."""
        logger.debug("fs()")
        return gv.fs
    
    @property
    def sps(self):
        """Samples per slot of the electrical signal."""
        logger.debug("sps()")
        return gv.sps
    
    @property
    def dt(self): 
        """Time step of the electrical signal."""
        logger.debug("dt")
        return gv.dt
    
    @property
    def t(self): 
        """Time array for the electrical signal."""
        logger.debug("t")
        return gv.t[:self.size]
    
    # static and private methods
    @staticmethod  # can be used without instantiating the class, eg: electrical_signal._prepare_arrays()
    def _prepare_arrays(signal, noise, dtype):
        logger.debug("_prepare_arrays()")

        @logger.auto_indent
        def _convert_to_array(value, dtype, text=''):
            logger.debug("_convert_to_array(%s(%s))", text, type(value).__name__)
            if isinstance(value, str):
                return str2array(value)
            return np.array(value)
        
        signal = _convert_to_array(signal, dtype, 'signal')
        
        if noise is not NULL:
            noise = _convert_to_array(noise, dtype, 'noise')
            
            if dtype is None:
                arrays_type = np.result_type(signal, noise)  # obtain the most comprehensive type
            else:
                arrays_type = dtype

            signal = signal.astype(arrays_type)
            noise = noise.astype(arrays_type) 

            if signal.shape != noise.shape:
                raise ValueError(f"`signal` and `noise` must have the same shape, mismatch shapes {signal.shape} and {noise.shape}!")
        else:
            if dtype is not None:
                signal = signal.astype(dtype)
        
        return signal, noise
    
    def _parse(self, other):
        logger.debug("_parse()")

        if not isinstance(other, self.type):
            other = self.__class__(other)
        else:
            other = other[:]
        
        if self.size != other.size:
            l_min = min(self.size, other.size)
            l_max = max(self.size, other.size)
            
            if l_min != 1 and l_min != l_max:
                raise ValueError(f"Can't operate '{self.__class__.__name__}'s with shapes {self.shape} and {other.shape}")
        
        dtype = np.result_type(self.signal, other.signal)
        return other, dtype
    
    # public methods
    def print(self, msg: str=None): 
        """Print object parameters.
        
        Parameters
        ----------
        msg : :obj:`str`, opcional
            top message to show

        Returns
        -------
        self : electrical_signal
            The same object.
        """
        logger.debug("print()")
        print(self.__str__(msg))
        return self

    def to_numpy(self, dtype: np.dtype | None = None, copy: bool = False) -> np.ndarray:
        """Return a NumPy representation of the electrical signal (signal + noise)."""
        logger.debug("to_numpy(dtype=%s, copy=%s)", dtype, copy)
        data = self.signal + self.noise
        return np.array(data, dtype=dtype, copy=copy)
    
    def conj(self):
        """Return the complex conjugate of the electrical signal.

        Returns
        -------
        :obj:`electrical_signal`
            The complex conjugate of the electrical signal.
        """
        logger.debug("conj()")
        return self.__class__(self.signal.conj(), self.noise.conj())

    def sum(self, axis: int=None):
        """Return the sum of the elements over a given axis.

        Parameters
        ----------
        axis : :obj:`int`, optional
            Axis along which the sum is computed. By default, the sum is computed over the entire array.

        Returns
        -------
        :obj:`electrical_signal`
            New object with signal and noise summed over the specified axis.
        """
        logger.debug("sum(axis=%s)", axis)
        sig = self.signal.sum(axis=axis)
        noi = self.noise.sum(axis=axis) if self.noise is not NULL else NULL
        return self.__class__(sig, noi)
    
    def w(self, shift: bool=False): 
        """Return angular frequency (rad/s) for spectrum representation.
        
        Parameters
        ----------
        shift : :obj:`bool`, optional
            If True, apply fftshift().

        Returns
        -------
        :obj:`np.ndarray`
            The angular frequency array for signals simulation.
        """
        w = fftfreq(self.T.shape[0], gv.dt)*2*pi
        if shift:
            return fftshift(w, axes=-1)
        return w
    
    def f(self, shift: bool=False):
        """Return frequency (Hz) for spectrum representation.
        
        Parameters
        ----------
        shift : :obj:`bool`, optional
            If True, apply fftshift().
        
        Returns
        -------
        :obj:`np.ndarray`

            The frequency array for signals simulation.
        """
        return self.w(shift)/(2*pi)


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
        logger.debug("abs(of='%s')", of)

        if not isinstance(of, str):
            raise TypeError('`of` must be a string.')
        of = of.lower()
        
        if of == 'signal':
            return self.__class__(np.abs(self.signal))
        elif of == 'noise':
            if self.noise is NULL:
                return self.__class__(np.zeros_like(self.signal.real))
            return self.__class__(np.abs(self.noise))
        elif of == 'all':
            return np.abs(self)
        else:
            raise ValueError('`of` must be one of the following values ("signal", "noise", "all")')
    
    def power(self, unit : Literal['W', 'dBm']='W', of: Literal['signal','noise','all']='all'): 
        """Get power of the electrical signal.
        
        Parameters
        ----------
        unit : :obj:`str`, optional
            Defines the unit of power. 'W' for Watts, 'dBm' for decibels-milliwatts.
        of : :obj:`str`, optional
            Defines from which attribute to obtain the power. If 'all', power of ``signal+noise`` is determined.
        
        Returns
        -------
        :obj:`float`
            The power of the electrical signal.
        """
        logger.debug("power(unit='%s', of='%s')", unit, of)

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

        Parameters
        ----------
        by : :obj:`str`, optional
            Defines the normalization method. ``'power'`` for power normalization, ``'amplitude'`` for amplitude normalization.
        
        Returns
        -------
        :obj:`electrical_signal`
            The normalized electrical signal.
        """
        logger.debug("normalize(by='%s')", by)

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
        """Get phase of the electrical signal: ``unwrap(angle(signal+noise))``.
        
        Returns
        -------
        :obj:`np.ndarray`
            the unwrapped phase of the electrical signal.
        """
        logger.debug("phase()")
        return np.unwrap(np.angle(self))

    def filter(self, h: np.ndarray):
        """Apply FIR filter of impulse response **h** to the electrical signal: ``np.convolve(signal + noise, h, mode='same')``.

        Parameters
        ----------
        h : :obj:`np.ndarray`
            The FIR filter impulse response.

        Returns
        -------
        :obj:`electrical_signal`
            An electrical signal object with the result of the filtering.
        """
        logger.debug("filter()")

        sig = np.convolve(self.signal, h, mode='same')
        noi = np.convolve(self.noise, h, mode='same')

        return self.__class__(sig, noi) 

    def plot(self, 
             fmt: str | list='-', 
             n: int=None, 
             xlabel: str=None, 
             ylabel: str=None, 
             grid: bool=False,
             hold: bool=True,
             show: bool=False,
             **kwargs: dict): 
        r"""Plot signal in time domain.

        For electrical_signal: plots the real part of the signal.
        For optical_signal: plots the intensity/power.

        Parameters
        ----------
        fmt : :obj:`str` or :obj:`list`, optional
            Format style of line. Example ``'b-.'``, Defaults to ``'-'``.
        n : :obj:`int`, optional
            Number of samples to plot. Defaults to the length of the signal.
        xlabel : :obj:`str`, optional
            X-axis label. Defaults to ``'Time [ns]'``.
        ylabel : :obj:`str`, optional
            Y-axis label. Defaults to ``'Amplitude [V]'``
        grid : :obj:`bool`, optional
            If show grid. Defaults to ``False``.
        hold : :obj:`bool`, optional
            If hold the current plot. Defaults to ``True``.
        **kwargs : :obj:`dict`
            Additional keyword arguments compatible with ``matplotlib.pyplot.plot()``.

        Returns
        -------
        :obj:`electrical_signal`
            The same object.
        """
        logger.debug("plot()")

        n = min(self.size, gv.t.size) if n is None else n
        t = gv.t[:n]*1e9
        y = self[:n]
        
        args = (t, y, fmt)
        ylabel = ylabel if ylabel else 'Amplitude [V]'
        xlabel = xlabel if xlabel else 'Time [ns]'

        label = kwargs.pop('label', None)

        if not hold:
            plt.figure()

        plt.plot(*args, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            for i, t_ in enumerate(t[:n*gv.sps][::gv.sps]):
                plt.axvline(t_, color='gray', ls='--', alpha=0.3, lw=1)
            plt.axvline(t[-1] + gv.dt*1e9, color='gray', ls='--', alpha=0.3, lw=1)
            plt.grid(alpha=0.3, axis='y')

        if label is not None:
            plt.legend()
        if 'label' in kwargs.keys():
            plt.legend()

        if show:
            plt.show()
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
            show: bool=False,
            **kwargs: dict):
        """Plot Power Spectral Density (PSD) of the electrical/optical signal.

        Parameters
        ----------
        fmt : :obj:`str` or :obj:`list`
            Format style of line. Example ``'b-.'``. Defaults to ``'-'``.
        mode : :obj:`str`, optional
            Polarization mode to show (for optical signals). Defaults to ``'x'``.
            
            - ``'x'`` plot polarization x.
            - ``'y'`` plot polarization y.
            - ``'both'`` plot both polarizations x and y in the same figure.

        n : :obj:`int`, optional
            Number of samples to plot. Defaults to the length of the signal.
        xlabel : :obj:`str`, optional
            X-axis label. Defaults to ``'Frequency [GHz]'``.
        ylabel : :obj:`str`, optional
            Y-axis label. Defaults to ``'Power [dBm]'`` if ``yscale='dbm'`` or ``'Power [mW]'`` if ``yscale='linear'``.
        yscale : :obj:`str`, {'linear', 'dbm'}, optional
            Kind of Y-axis plot. Defaults to ``'dbm'``.
        grid : :obj:`bool`, optional
            If show grid. Defaults to ``True``.
        hold : :obj:`bool`, optional
            If hold the current plot. Defaults to ``True``.
        **kwargs : :obj:`dict`
            Additional matplotlib arguments.

        Returns
        -------
        obj:`electrical_signal`
            The same object.
        """
        logger.debug("psd()")

        n = self.size if not n else n
        
        f, psd = sg.welch(self[:n].signal, fs=gv.fs*1e-9, nperseg=2048, scaling='spectrum', return_onesided=False, detrend=False)
        f, psd = fftshift(f), fftshift(psd, axes=-1)

        if yscale == 'linear':
            psd = psd*1e3
            ylabel = ylabel if ylabel else 'Power [mW]'
            ylim = (-0.1,)
        elif yscale == 'dbm':
            psd = dbm(psd)
            ylabel = ylabel if ylabel else 'Power [dBm]'
            ylim = (-100,)
        else:
            raise TypeError('`yscale` must be one of the following values ("linear", "dbm")')
        
        n_pol = getattr(self, 'n_pol', 1)
        
        if n_pol == 1:
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
        
        label = kwargs.pop('label', None) if mode == 'both' and n_pol > 1 else None

        if not hold:
            plt.figure()

        ls = plt.plot( *args, **kwargs)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel if xlabel else 'Frequency [GHz]')
        plt.xlim( -3.5*gv.R*1e-9, 3.5*gv.R*1e-9 )
        plt.ylim( *ylim )
        if grid:
            plt.grid(alpha=0.3)
        
        if label is not None:
            if isinstance(label, str):
                ls[0].set_label(label + ' X')
                ls[1].set_label(label + ' Y')
            elif isinstance(label, (list, tuple)):
                ls[0].set_label(label[0])
                ls[1].set_label(label[1])
            else:
                raise ValueError('`label` must be a string or a list of strings.')
            plt.legend()
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
        n_traces : :obj:`int`, optional
            Maximum number of traces to plot. If ``None``, all available traces
            will be plotted. Defaults to ``None``.
        cmap : :obj:`str`, optional
            Name of the matplotlib colormap. Defaults to ``'viridis'``.
        N_grid_bins : :obj:`int`, optional
            Number of bins for the density histogram. Defaults to ``350``.
        grid_sigma : :obj:`float`, optional
            Sigma for the Gaussian filter applied to the density. Defaults to ``3``.
        ax : :obj:`matplotlib.axes.Axes`, optional
            Axes object to plot on. If ``None``, creates new figure and axes.
            Defaults to ``None``.
        \*\*plot_kw : :obj:`dict`, optional
            Additional plotting parameters:
            
            *Figure parameters (used only if ax is ``None``):*
            
            - **figsize** : :obj:`tuple`, default ``(10, 6)``
            - **dpi** : :obj:`int`, default ``100``
            
            *Line collection parameters:*
            
            - **linewidth** : :obj:`float`, default ``0.75``
            - **alpha** : :obj:`float`, default ``0.25``
            - **capstyle** : :obj:`str`, default ``'round'``
            - **joinstyle** : :obj:`str`, default ``'round'``
            
            *Axes formatting parameters:*
            
            - **xlabel** : :obj:`str`, default ``"Time (2-symbol segment)"``
            - **ylabel** : :obj:`str`, default ``"Amplitude"``
            - **title** : :obj:`str`, default ``"Eye Diagram ({num_traces} traces)"``
            - **grid** : bool, default ``True``
            - **grid_alpha** : :obj:`float`, default ``0.3``
            - **xlim** : :obj:`tuple`, optional (xmin, xmax)
            - **ylim** : :obj:`tuple`, optional (ymin, ymax)
            - **tight_layout** : :obj:`bool`, default ``True``
            
            *Display parameters:*

            - **show** : :obj:`bool`, default ``True`` (whether to call ``plt.show()``)
        
        Returns
        -------
        :obj:`electrical_signal`
            The same object with the plotted eye diagram.
        """
        logger.debug("plot_eye()")
        eyediagram(self, gv.sps, n_traces, cmap, N_grid_bins, grid_sigma, ax, **plot_kw)
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
        logger.debug("grid()")
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
        logger.debug("legend()")
        plt.legend(*args, **kwargs)
        return self
    
    def show(self):
        """Show plots.
        
        Returns
        -------
        self : :obj:`electrical_signal`
            The same object.
        """
        logger.debug("show()")
        plt.show()
        return self











@logger.auto_indent_methods
class optical_signal(electrical_signal):
    """**Optical Signal**
    
    Bases: :obj:`electrical_signal`

    This class provides methods and attributes to work with optical signals. Attributes and some methods are inherited from the :obj:`electrical_signal` class.

    .. rubric:: Attributes
    .. autosummary::

        ~optical_signal.signal
        ~optical_signal.noise
        ~optical_signal.execution_time

    .. rubric:: Methods
    .. autosummary::

        __init__
        plot
    """

    def __init__(self, 
                 signal: str | Iterable, 
                 noise: str | Iterable = NULL, 
                 n_pol: Literal[1, 2] = None,
                 dtype: np.dtype=None):
        """ Initialize the optical signal object.

        Parameters
        ----------
        signal : :obj:`str` or array_like (1D, 2D) or scalar
            The signal values.
        noise : :obj:`str` or array_like (1D, 2D) or scalar, optional
            The noise values, default is ``NULL``.
        n_pol : :obj:`int`, optional
            Number of polarizations. Defaults to ``1``.
        """

        if self.__class__ == optical_signal:
            logger.debug("%s.__init__(n_pol=%s, dtype=%s)", self.__class__.__name__, n_pol, dtype)

            if isinstance(signal, (electrical_signal, optical_signal)):
                signal, noise = signal.signal, signal.noise
            else:
                signal, noise = self._prepare_arrays(signal, noise, dtype)
            
            if signal.ndim>2 or (signal.ndim>1 and signal.shape[0]>2) or signal.size<1:
                raise ValueError(f"Signal must be a scalar, 1D or 2D array for optical_signal, invalid shape {signal.shape}")
            if n_pol is not None and n_pol not in [1, 2]:
                raise ValueError("n_pol must be either 1 or 2")
            
            if signal.ndim == 0:
                if n_pol is None or n_pol == 1: 
                    signal = signal[np.newaxis]
                    if noise is not NULL:
                        noise = noise[np.newaxis]
                    n_pol=1
                else:
                    signal = np.array([[signal], [signal]])
                    if noise is not NULL:
                        noise = np.array([[noise], [noise]])
            
            elif signal.ndim == 1:
                if n_pol is None or n_pol == 1: 
                    n_pol = 1
                else:
                    signal = np.array([signal, signal])
                    if noise is not NULL:
                        noise = np.array([noise, noise])
            
            elif signal.ndim == 2 and signal.shape[0] == 1:
                if n_pol is None or n_pol == 2:
                    signal = np.tile(signal, (2, 1))
                    if noise is not NULL:
                        noise = np.tile(noise, (2, 1))
                    n_pol = 2
                else:
                    signal = signal[0]
                    if noise is not NULL:
                        noise = noise[0]
                    
            elif signal.ndim == 2 and signal.shape[0] == 2:
                if n_pol is None or n_pol == 2:
                    n_pol = 2
                else:
                    signal = signal[0]
                    if noise is not NULL:
                        noise = noise[0]
        
        self.n_pol = n_pol
        super().__init__( signal, noise, dtype=dtype)  
    
    def __repr__(self):
        logger.debug("__repr__()")

        np.set_printoptions(precision=1, threshold=20)

        if self.noise is not NULL:
            signal = str(self.signal).replace('\n', '\n' + 15*' ')
            return f'optical_signal({signal})'
        
        signal = str(self.signal).replace('\n', '\n' + 22*' ')
        noise = str(self.noise).replace('\n', '\n' + 22*' ')
        return f'optical_signal(signal={signal}\n' + 16*' '+ f'noise={noise})'

    def __str__(self, title: str=None): 
        logger.debug("__str__()")
        
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
        
        # Get power values
        pw_sig_w = self.power('W', 'signal')
        pw_sig_dbm = self.power('dbm', 'signal')
        pw_noise_w = self.power('W', 'noise')
        pw_noise_dbm = self.power('dbm', 'noise')
        pw_all_w = self.power('W', 'all')
        pw_all_dbm = self.power('dbm', 'all')

        # Format power strings
        if np.isscalar(pw_sig_w):
            pow_sig_str = f"{si(pw_sig_w, 'W', 1)} ({pw_sig_dbm:.1f} dBm)"
            pow_noise_str = f"{si(pw_noise_w, 'W', 1)} ({pw_noise_dbm:.1f} dBm)"
            pow_all_str = f"{si(pw_all_w, 'W', 1)} ({pw_all_dbm:.1f} dBm)"
        else:
            # For multi-polarization
            pow_sig_str = ', '.join([f"Pol{i}: {si(pw_sig_w[i], 'W', 1)} ({pw_sig_dbm[i]:.1f} dBm)" for i in range(len(pw_sig_w))])
            pow_noise_str = ', '.join([f"Pol{i}: {si(pw_noise_w[i], 'W', 1)} ({pw_noise_dbm[i]:.1f} dBm)" for i in range(len(pw_noise_w))])
            pow_all_str = ', '.join([f"Pol{i}: {si(pw_all_w[i], 'W', 1)} ({pw_all_dbm[i]:.1f} dBm)" for i in range(len(pw_all_w))])

        msg = f'\n{sub}\n{title}\n{sub}\n'+ tab + \
            f'signal:     {signal} (shape: {self.shape})\n'+ tab + \
            f'noise:      {noise} (shape: {self.shape if self.noise is not NULL else None})\n'+ tab + \
            f'pow_signal: {pow_sig_str}\n'+ tab + \
            f'pow_noise:  {pow_noise_str}\n'+ tab + \
            f'pow_total:  {pow_all_str}\n'+ tab + \
            f'elem_type:  {self.dtype}\n' + tab + \
            f'mem_size:   {self.sizeof} bytes\n' + tab + \
            f'time:       {si(self.execution_time, "s", 2)}\n'
        return msg

    
    def __getitem__(self, key): 
        logger.debug("__getitem__(%s)", key)

        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError('Too many indices for optical_signal object.')
            pol_idx, time_idx = key
            if self.n_pol == 1 and pol_idx not in [0, -1, slice(None)]:
                raise IndexError('Optical signal has only one polarization (index 0).')
            sig = self.signal[pol_idx, time_idx] if self.n_pol == 2 else self.signal[time_idx]
            if self.noise is not NULL:
                noi = self.noise[pol_idx, time_idx] if self.n_pol == 2 else self.noise[time_idx]
            elif isinstance(time_idx, int):
                return sig[time_idx]
            else:
                noi = NULL
            return self.__class__(sig, noi, n_pol=1 if sig.ndim!=2 else self.n_pol)
        elif isinstance(key, slice):
            if self.n_pol == 1:
                sig = self.signal[key]
                if self.noise is not NULL:
                    noi = self.noise[key]
                else:
                    noi = NULL
            else:
                sig = self.signal[:, key]
                if self.noise is not NULL:
                    noi = self.noise[:, key]
                else:
                    noi = NULL
            return self.__class__(sig, noi, n_pol=self.n_pol)
        else:
            if self.n_pol == 1:
                sig = self.signal[key]
                if self.noise is not NULL:
                    noi = self.noise[key]
                else:
                    return sig
            else:
                sig = self.signal[key, :]
                if self.noise is not NULL:
                    noi = self.noise[key, :]
                else:   
                    noi = NULL
            return self.__class__(sig, noi, n_pol=1 if sig.ndim!=2 else self.n_pol)            
            
    def __gt__(self, other): 
        raise NotImplementedError('The > operator is not implemented for optical_signal objects.')
    
    def __lt__(self, other):
        raise NotImplementedError('The < operator is not implemented for optical_signal objects.')

    @property
    def size(self) -> np.ndarray:
        """Number of samples of one polarization of the optical signal."""
        logger.debug("size")
        if self.n_pol == 1:
            return self.signal.size
        else:
            return self.signal[0].size

    def plot(self, 
             fmt: str | list='-', 
             mode: Literal['field', 'power'] = 'power', 
             n: int=None, 
             xlabel: str=None, 
             ylabel: str=None, 
             grid: bool=False,
             hold: bool=True,
             show: bool=False,
             **kwargs: dict): 
        r"""Plot signal in time domain.

        For optical_signal: plots the intensity/power or field.

        Parameters
        ----------
        fmt : :obj:`str` or :obj:`list`, optional
            Format style of line. Example ``'b-.'``, Defaults to ``'-'``.
        mode : :obj:`str`, optional
            Plot mode. ``'field'``, ``'power'`` (default).

            - ``'field'`` plot real and imaginary parts of the field one-polarization.
            - ``'power'`` plot power/intensity (one or two polarizations).

        n : :obj:`int`, optional
            Number of samples to plot. Defaults to the length of the signal.
        xlabel : :obj:`str`, optional
            X-axis label. Defaults to ``'Time [ns]'``.
        ylabel : :obj:`str`, optional
            Y-axis label. Defaults to ``'Power [mW]'`` for power, ``'Field [W**0.5]'`` for field.
        grid : :obj:`bool`, optional
            If show grid. Defaults to ``False``.
        hold : :obj:`bool`, optional
            If hold the current plot. Defaults to ``True``.
        **kwargs : :obj:`dict`
            Additional keyword arguments compatible with ``matplotlib.pyplot.plot()``.

        Returns
        -------
        self : optical_signal
            The same object.
        """
        logger.debug("plot()")

        n = min(self.size, gv.t.size) if n is None else n
        t = gv.t[:n]*1e9
        y = self[:n]

        if not hold:
            plt.figure()
        xlabel = xlabel if xlabel else 'Time [ns]'
        
        # Optical signal: plot intensity or field
        if mode == 'power':
            I = y.abs('all')**2 * 1e3  # mW
            if y.n_pol == 1:
                if not isinstance(fmt, str):
                    warnings.warn('`fmt` must be a string for single polarization signals, using default value.')
                    fmt = '-'
                args = (t, I, fmt)
                label = kwargs.pop('label', None)
                plt.plot(*args, **kwargs, label=label)
                legend = True if label is not None else False
            else:
                if not isinstance(fmt, str):
                    warnings.warn('`fmt` must be a string for multi-polarization signals, using default value.')
                    fmt = '-'
                args0 = (t, I[0], fmt)  # total power
                args1 = (t, I[1], fmt)  # total power
                label = kwargs.pop('label', None) 
                plt.plot(*args0, **kwargs, label='Pol X' if label is None else label + ' Pol X')
                plt.plot(*args1, **kwargs, label='Pol Y' if label is None else label + ' Pol Y')
                legend = True
                
            ylabel = ylabel if ylabel else 'Power [mW]'
    
        elif mode == 'field':
            if y.n_pol > 1:
                raise ValueError('`field` mode is only supported for single polarization signals.')
            if not isinstance(fmt, str):
                warnings.warn('`fmt` must be a string for field mode, using default value.')
                fmt = '-'
            label = kwargs.pop('label', None)
            plt.plot(t, y.real, fmt, label='Real' if label is None else label + 'Real')
            plt.plot(t, y.imag, fmt, label='Imag' if label is None else label + 'Imag')
            legend = True
            ylabel = ylabel if ylabel else r'Field [$\sqrt{W}$]'

        else:
            raise ValueError('`mode` must be one of ("power", "field") for optical signals.')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            for i, t_ in enumerate(t[:n*gv.sps][::gv.sps]):
                plt.axvline(t_, color='gray', ls='--', alpha=0.3, lw=1)
            plt.axvline(t[-1] + gv.dt*1e9, color='gray', ls='--', alpha=0.3, lw=1)
            plt.grid(alpha=0.3, axis='y')

        if legend:
            plt.legend()

        if show:
            plt.show()
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











@logger.auto_indent_methods
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
        logger.debug("%s.__init__()", self.__class__.__name__)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.empty = False
        else:
            self.empty = True
        
    def __str__(self, title: str=None): 
        """Return a formatted string with the eye diagram data."""
        logger.debug("__str__()")

        if self.empty:
            raise ValueError('Empty eye diagram object.')

        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'

        np.set_printoptions(precision=1, threshold=10)

        msg = f'\n{sub}\n{title}\n{sub}\n ' + '\n '.join([f'{key} : {value}' for key, value in self.__dict__.items() if key != 'execution_time'])
        
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
        logger.debug("print()")
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
        logger.debug("plot()")

        if self.empty:
            raise ValueError('Empty eye diagram object.')

        ## SETTINGS
        
        # Determine style context for temporary style changes
        if style == 'dark':
            style_context = 'dark_background'
            t_opt_color = '#60FF86'
            means_color = 'white'
            bgcolor='black'
        elif style == 'light':
            style_context = 'default'
            t_opt_color = 'green'#'#229954'
            means_color = '#5A5A5A'
            bgcolor='white'
        else:
            raise TypeError("The `style` argument must be one of the following values ('dark', 'light')")
        
        dt = self.dt

        # Use style context manager only when creating new axes to avoid global state pollution
        # When ax is provided by caller, respect their style settings
        from contextlib import nullcontext
        style_mgr = plt.style.context(style_context) if ax is None else nullcontext()
        
        with style_mgr:
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
        
        return self

    def show(self):
        """Show plot
        
        Returns
        -------
        self : :obj:`eye`
            The same object.
        """
        logger.debug("show()")
        plt.show()
        return self

