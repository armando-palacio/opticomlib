"""
.. autosummary::
   :toctree: generated/

   gv                    -- Global variables instance
   binary_sequence       -- Binary sequence class
   electrical_signal     -- Electrical signal class
   optical_signal        -- Optical signal class
   eye                   -- Eye diagram class
"""

from numpy.fft import fft, ifft, fftfreq, fftshift
from pympler.asizeof import asizeof as sizeof

import numpy as np
from scipy.constants import c, pi

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif' 

from matplotlib.widgets import Slider

from typing import Literal, Union, Any

import warnings

from .utils import (
    str2array, 
    dbm, 
    si, 
)

Array_Like = (list, tuple, np.ndarray)
Number = (int, float)

class global_variables():
    r"""
    **Global Variables Object**

    This object is used to store global variables that are used in the simulation.
    The global variables are used mainly to define the sampling frequency, the slot rate, 
    the number of samples per slot and the optical wavelength or frequency.

    .. Note:: 
        
        A slot is taken as the smallest time unit representing a binary value of the signal.
        For example, in PPM a bit is not the same as a slot. However, in OOK a bit and a slot are the same.

    This class don't need to be instantiated. It is already instantiated as ``gv``.
    For update or add a variable use the :meth:`__call__` method (i.e gv(\*\*kargs)).
    
    Attributes
    ----------
    sps : :obj:`int` 
        Samples per slot. Defaults to 16
    R : :obj:`float` 
        Slot rate in Hz. Defaults to 1e9
    fs : :obj:`float`
        Sampling frequency in Hz. Defaults to sps*R = 16e9
    dt : :obj:`float`
        Time step in seconds. Defaults 1/fs = 62.5e-12
    wavelength : :obj:`float`
        Optical wavelength in meters. Defaults to 1550e-9
    f0 : :obj:`float` 
        Central frequency in Hz. Defaults to c/wavelength = 193.4e12
    N : :obj:`int`
        Number of slots. Defaults to :obj:`None`
    t : :obj:`np.ndarray`
        Time array for signal simulation. If ``N`` is defined. Defaults to :obj:`None`
    dw : :obj:`float`
        Frequency step in Hz. If ``N`` is defined. Defaults to :obj:`None`
    w : :obj:`np.ndarray`
        Angular frequency array for signals simulation. If ``N`` is defined. Defaults to :obj:`None`

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

    Also can be define new variables trough \*\*kwargs. If at least two of the this arguments (``sps``, ``fs`` and ``R``) are not provided
    a warning will be raised and the default values will be used.

    >>> gv(alpha=0.5, beta=0.3).print()
    
    ::
    
        UserWarning: `sps`, `R` and `fs` will be set to default values (16 samples per slot, 1e+09 Hz, 2e+10 Samples/s)
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
        self.R = 1e9
        self.fs = self.R*self.sps
        self.dt = 1/self.fs
        self.dw = None
        self.wavelength = 1550e-9
        self.f0 = c/self.wavelength
        self.N = None
        self.t = None
        self.w = None


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
                msg = f'`R` will be set to default value ({self.R:.0e} Hz)'
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
            msg = f'`sps`, `R` and `fs` will be set to default values ({self.sps} samples per slot, {self.R:.0e} Hz, {self.fs:.0e} Samples/s)'
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
    r"""Binary sequence class.

    This class provides methods and attributes to work with binary sequences. 
    The binary sequence can be provided as a string, list, tuple, or numpy array.

    Parameters
    ----------
    data : :obj:`str` or Array_Like
        The binary sequence data.
    
    Attributes
    ----------
    data : :obj:`np.ndarray`, (1D, bool)
        The binary sequence data.
    execution_time : :obj:`float`
        The execution time of the last operation performed on the binary sequence.
    """

    def __init__(self, data: Union[str, list, tuple, np.ndarray]): 
        if not isinstance(data, ((str,) + Array_Like)):
            raise TypeError("The argument must be an str or array_like!")
        
        if isinstance(data, str):
            data = str2array(data, dtype=bool)
        else:
            data = np.array(data)
            if not np.all((data == 0) | (data == 1)): 
                raise ValueError("The array must contain only 0's and 1's!")

        self.data = np.array(data, dtype=bool)
        self.ejecution_time = None

    def __str__(self, title: str=None): 
        """Return a formatted string with the binary sequence data, length, size in bytes and time if available."""
        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'

        np.set_printoptions(precision=0, threshold=10)
        data = str(self.data.astype(np.uint8))

        msg = f'\n{sub}\n{title}\n{sub}\n\t' + \
            f'data  :  {data}\n\t' + \
            f'len   :  {self.len()}\n\t' + \
            f'size  :  {self.sizeof()} bytes\n'
        
        if self.ejecution_time is not None:
            msg += '\t' +\
                f'time  :  {si(self.ejecution_time, "s", 1)}\n'
        return msg
    
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
    
    def __len__(self): return self.len()
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("The index must be an integer!")
        return binary_sequence(self.data[key])
    
    def __add__(self, other): 
        """ Concatenate two binary sequences, adding to the end.

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
        __radd__ : Concatenates two binary sequence, adding at the beginning.
        """
        if isinstance(other, str):
            other = str2array(other, bool)
        elif isinstance(other, binary_sequence):
            other = other.data
        elif isinstance(other, Array_Like):
            other = np.array(other)
            if not np.all((other == 0) | (other == 1)): 
                raise ValueError("Sequence to concatenate must contain only 0's and 1's!")
        else:
            raise TypeError("Can't concatenate binary_sequence with type {}".format(type(other)))
        out = np.concatenate((self.data, other))
        return binary_sequence(out)
    
    def __radd__(self, other): 
        """ Concatenate two binary sequences, adding to the beginning.

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
        if isinstance(other, str):
            other = str2array(other, bool)
        elif isinstance(other, binary_sequence):
            other = other.data
        elif isinstance(other, Array_Like):
            other = np.array(other)
            if not np.all((other == 0) | (other == 1)): 
                raise ValueError("Sequence to concatenate must contain only 0's and 1's!")
        else:
            raise TypeError("Can't concatenate binary_sequence with type {}".format(type(other)))
        out = np.concatenate((other, self.data))
        return binary_sequence(out)

    def len(self): 
        """Get number of slots of the binary sequence.
        
        Returns
        -------
        :obj:`int`
            The number of slots of the binary sequence.
        """
        return self.data.size
    
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
    """Electrical signal class.

    Parameters
    ----------
    signal : array_like, optional
        The signal values. Defaults to `None`.
    noise : array_like, optional
        The noise values. Defaults to `None`.

    Attributes
    ----------
    signal : array_like, (1D, complex)
        a complex-valued signal
    noise : array_like, (1D, complex)
        a complex-valued noise associated with the signal
    execution_time : float
        the time taken for the execution of the signal
    """

    def __init__(self, signal: Union[list, tuple, np.ndarray]=None, noise: Union[list, tuple, np.ndarray]=None) -> None:
        if signal is None and noise is None:
            raise KeyError("`signal` or `noise` must be provided!")
        if (signal is not None) and (noise is not None) and (len(signal)!=len(noise)):
            raise ValueError(f"The arrays `signal`{signal.shape} and `noise`{noise.shape} must have the same length!")

        if signal is None:
            self.signal = np.zeros_like(noise, dtype=complex)
        elif not isinstance(signal, (list, tuple, np.ndarray)):
            raise TypeError("`signal` must be of type list, tuple or numpy array!")
        else:
            self.signal = np.array(signal, dtype=complex) # shape (1xN)
        
        if noise is None:
            self.noise = np.zeros_like(signal, dtype=complex)
        elif not isinstance(noise, (list, tuple, np.ndarray)):
            raise TypeError("`noise` must be of type list, tuple or numpy array!")
        else:
            self.noise = np.array(noise, dtype=complex)
        
        self.ejecution_time = None

    def __str__(self, title: str=None): 
        """Return a formatted string with the electrical_signal data, length, size in bytes and time if available."""
        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'

        np.set_printoptions(precision=0, threshold=10)

        if self.signal.ndim == 1:
            signal = str(self.signal)
            noise = str(self.noise)
        else:
            signal = str(self.signal).replace('\n', '\n\t' + 11*' ')
            noise = str(self.noise).replace('\n', '\n\t' + 11*' ')
        
        msg = f'\n{sub}\n{title}\n{sub}\n\t' + \
            f'signal  :  {signal}\n\t' + \
            f'noise   :  {noise}\n\t' + \
            f'len     :  {self.len()}\n\t' + \
            f'size    :  {self.sizeof()} bytes\n'
        
        if self.ejecution_time is not None:
            msg += '\t' + \
                f'time    :  {si(self.ejecution_time, "s", 1)}\n'
        return msg
    
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
        if isinstance(other, electrical_signal): 
            return electrical_signal(self.signal + other.signal, self.noise + other.noise)
        if isinstance(other, (int, float, complex, np.ndarray)):
            return electrical_signal(self.signal + other, self.noise + other)
        
    def __mul__(self, other):
        if isinstance(other, electrical_signal):
            return electrical_signal(self.signal * other.signal, self.noise * other.noise)
        if isinstance(other, (int, float, complex, np.ndarray)):
            return electrical_signal(self.signal * other, self.noise * other)
    
    def __getitem__(self, key):
        return electrical_signal( self.signal[key], self.noise[key] )

    def __call__(self, domain: Literal['t','w', 'f'], shift: bool=False):
        """ Return a new object with Fast Fourier Transform (FFT) of signal and noise of input object.

        Parameters
        ----------
        domain : {'t', 'w', 'f'}
            Domain to transform. 't' for time domain (ifft is applied), 'w' and 'f' for frequency domain (fft is applied).
        shift : :obj:`bool`, optional
            If True, apply ``np.fft.fftshift()`` function.

        Returns
        -------
        new_obj : electrical_signal
            A new electrical signal object with the result of the transformation.

        Raises
        ------
        TypeError
            If ``domain`` is not one of the following values ('t', 'w', 'f').
        """
        if domain == 'w' or domain == 'f':
            signal = fft(self.signal)
            noise = fft(self.noise)
              
        elif domain == 't':
            signal = ifft(self.signal)
            noise = ifft(self.noise)
        
        else:
            raise TypeError("`domain` must be one of the following values ('t', 'w', 'f')")
        
        if shift:
            signal = fftshift(signal)
            noise = fftshift(noise)

        return electrical_signal(signal, noise)
    
    def __gt__(self, other): 
        """ Compare the signal+noise with a threshold.

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
        if isinstance(other, electrical_signal):
            if self.len() != other.len():
                raise ValueError("The arrays must have the same length!")
            threshold = other.signal 
        elif isinstance(other, (list, tuple, np.ndarray)):
            if self.len() != len(other):
                raise ValueError("The arrays must have the same length!")   
            threshold = np.array(other)     
        elif isinstance(other, (int, float)):
            threshold = other
        else:
            raise TypeError("`other` must be of type electrical_signal, list, tuple, numpy array, int or float!")       
        return binary_sequence( self.signal+self.noise > threshold )
             
    def len(self): 
        """Get number of samples of the electrical signal.
        
        Returns
        -------
        :obj:`int`
            The number of samples of the electrical signal.
        """
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
        if by not in ['signal', 'noise', 'all']:
            raise TypeError('`by` must be one of the following values ("signal", "noise", "all")')
        return np.mean(self.abs(by)**2, axis=-1)
    
    def phase(self):
        """Get phase of the electrical signal.
        
        Returns
        -------
        :obj:`np.ndarray`
            The phase of the electrical signal.
        """
        return np.unwrap(np.angle(self.signal))
    
    def apply(self, function, *args, **kargs):
        """Apply a function to signal and noise.
        
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
        if np.sum(self.noise):
            output.noise = function(self.noise, *args, **kargs)
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
        """Get absolute value of the electrical signal.

        Parameters
        ----------
        by : :obj:`str`, optional
            Defines from which attribute to obtain the absolute value. If 'all', absolute value of signal+noise is determined.
        
        Returns
        -------
        out : :obj:`np.ndarray`, (1D, float)
            The absolute value of the electrical signal.
        """
        if by == 'signal':
            return np.abs(self.signal)
        elif by == 'noise':
            return np.abs(self.noise)
        elif by == 'all':
            return np.abs(self.signal + self.noise)
        else:
            raise TypeError('`by` must be one of the following values ("signal", "noise", "all")')
    

    def plot(self, 
             fmt: str='-', 
             n: int=None, 
             xlabel: str=None, 
             ylabel: str=None, 
             style: Literal['dark', 'light'] = 'dark',
             grid: bool=True,
             **kwargs: dict): 
        """Plot real part of electrical signal.

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
            If show grid. Defaults to True.
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
        
        plt.plot( *args, **kwargs )
        plt.ylabel( ylabel )
        plt.xlabel( xlabel if xlabel else 'Frequency [GHz]')
        plt.xlim(-3.5*gv.R*1e-9, 3.5*gv.R*1e-9)
        plt.ylim( *ylim )
        if grid: plt.grid(alpha=0.3, color=c)

        if 'label' in kwargs.keys():
            plt.legend()
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
    """Optical signal class.
    
    Bases: :obj:`electrical_signal`

    Parameters
    ----------
    signal : array_like, (1D, 2D)
        The signal values, default is `None`.
    noise : array_like, (1D, 2D)
        The noise values, default is `None`.  

    Attributes
    ----------
    signal : :obj:`np.ndarray`, (2D, complex)
        a complex-valued signal
    noise : :obj:`np.ndarray`, (2D, complex)
        a complex-valued noise associated with the signal
    execution_time : :obj:`float`
        the time taken for the execution of the signal
    """

    def __init__(self, signal: Union[list, tuple, np.ndarray]=None, noise: Union[list, tuple, np.ndarray]=None) -> None:
        if signal is not None:
            ndim = np.array(signal).ndim
            if ndim > 2 or ndim < 1:
                raise ValueError("`signal` must be a 1D or 2D array!")
            if ndim == 1:
                signal = np.array([signal, np.zeros_like(signal)])
        elif noise is not None:
            ndim = np.array(noise).ndim
            if ndim > 2 or ndim < 1:
                raise ValueError("`noise` must be a 1D or 2D array!")
            if ndim == 1:
                noise = np.array([noise, np.zeros_like(noise)])
        else:
            raise KeyError("`signal` or `noise` must be provided!")
        
        super().__init__( signal, noise )  
    
    def len(self): 
        """Get number of samples of the optical signal.

        Returns
        -------
        :obj:`int`
            The number of samples of the optical signal.
        """
        return self.signal.shape[1]

    def __add__(self, other): 
        """ Sum two optical signals.
        
        Parameters
        ----------
        other : :obj:`optical_signal` or :obj:`array_like` or :obj:`Number`
            Optical_signal or value to sum. If `other` is an array_like or a Number it will be added to the signal only. 
            If `other` is an optical_signal, it will be added the corresponding signal and noise.

        Returns
        -------
        out : :obj:`optical_signal`
            A new optical signal object with the result of the sum.
        """
        if isinstance(other, optical_signal):
            if self.len() != other.len():
                raise ValueError("Can't sum signals with different lengths! ({} and {})".format(self.len(), other.len()))
            return optical_signal(self.signal + other.signal, self.noise + other.noise)
        if isinstance(other, (int, float, complex)):
            return optical_signal(self.signal + other, self.noise)
        if isinstance(other, (list, tuple, np.ndarray)):
            other = np.array(other)
            if other.ndim == 1:
                l = other.size
            elif other.ndim == 2:
                l = other.shape[1]
            else:
                raise ValueError("`other` must be a 1D or 2D array!")

            if self.len() != l:
                raise ValueError("Can't sum signals with different lengths! ({} and {})".format(self.len(), l))
            return optical_signal(self.signal + other, self.noise)
        
    def __radd__(self, other):
        """ Sum two optical signals. __radd__() = __add__()."""
        return self.__add__(other)
    
    def __mul__(self, other):
        """ Multiply two optical signals.
        
        Parameters
        ----------
        other : :obj:`optical_signal` or :obj:`array_like` or :obj:`Number`
            Optical_signal or value to multiply. If `other` is an array_like or a Number it will be multiply to the signal only. 
            If `other` is an optical_signal, it will be multiply the corresponding signal and noise.

        Returns
        -------
        out : :obj:`optical_signal`
            A new optical signal object with the result of the multiply.
        """
        if isinstance(other, optical_signal):
            if self.len() != other.len():
                raise ValueError("Can't multiply signals with different lengths! ({} and {})".format(self.len(), other.len()))
            return optical_signal(self.signal * other.signal, self.noise * other.noise)
        if isinstance(other, (int, float, complex)):
            return optical_signal(self.signal * other, self.noise)
        if isinstance(other, (list, tuple, np.ndarray)):
            other = np.array(other)
            if other.ndim == 1:
                l = other.size
            elif other.ndim == 2:
                l = other.shape[1]
            else:
                raise ValueError("`other` must be a 1D or 2D array!")

            if self.len() != l:
                raise ValueError("Can't multiply signals with different lengths! ({} and {})".format(self.len(), l))
            return optical_signal(self.signal * other, self.noise)
        
    def __rmul__(self, other):
        """ Multiply two optical signals. __rmul__() = __mul__()."""
        return self.__mul__(other)
    
    def __getitem__(self, key): 
        """Slice the optical signal.

        Parameters
        ----------
        key : :obj:`int` or :obj:`slice`
            Index or slice to get the new optical signal.

        Returns
        -------
        out : :obj:`optical_signal`
            A new optical signal object with the result of the slicing.
        """
        return optical_signal( self.signal[:,key], self.noise[:,key] )
    
    def __call__(self, domain: Literal['t','w','f'], shift: bool=False):
        """ Return a new object with Fast Fourier Transform (FFT) of signal and noise of input object.

        Parameters
        ----------
        domain : {'t', 'w', 'f'}
            Domain to transform. 't' for time domain (ifft is applied), 'w' or 'f' for frequency domain (fft is applied).
        shift : :obj:`bool`, optional
            If True, apply ``np.fft.fftshift()`` function.

        Returns
        -------
        new_obj : optical_signal
            A new optical signal object with the result of the transformation.

        Raises
        ------
        TypeError
            If ``domain`` is not one of the following values ('t', 'w', 'f').
        """
        if domain == 'w' or domain == 'f':
            signal = fft(self.signal, axis=-1)
            noise = fft(self.noise, axis=-1)
              
        elif domain == 't':
            signal = ifft(self.signal, axis=-1)
            noise = ifft(self.noise, axis=-1)
        
        else:
            raise TypeError("`domain` must be one of the following values ('t', 'w', 'f')")
        
        if shift:
            signal = fftshift(signal, axes=-1)
            noise = fftshift(noise, axes=-1)

        return optical_signal(signal, noise)

    def plot(self, 
             fmt=None, 
             n=None, 
             mode: Literal['x','y','both','abs']='abs', 
             xlabel: str=None,
             ylabel: str=None,
             style: Literal['dark', 'light'] = 'dark',
             grid: bool=True,
             **kwargs): 
        r"""
        Plot intensity of optical signal for selected polarization mode.

        Parameters
        ----------
        fmt : :obj:`str`, optional
            Format style of line. Example 'b-.'. Default is '-'.
        n : :obj:`int`, optional
            Number of samples to plot. Default is the length of the signal.
        mode : :obj:`str`
            Polarization mode to show. Default is 'abs'.

            - ``'x'`` plot polarization x.
            - ``'y'`` plot polarization y.
            - ``'both'`` plot both polarizations x and y in the same figure
            - ``'abs'`` plot intensity sum of both polarizations I(x) + I(y).
            
        xlabel : :obj:`str`, optional
            X-axis label. Default is 'Time [ns]'.
        ylabel : :obj:`str`, optional
            Y-axis label. Default is 'Power [mW]'.
        style : :obj:`str`, optional
            Style of plot. Default is 'dark'.

            - ``'dark'`` use dark background.
            - ``'light'`` use light background.
        
        grid : :obj:`bool`, optional
            If show grid. Default is ``True``.
        \*\*kwargs: :obj:`dict`
            Aditional matplotlib arguments.

        Returns
        -------
        self : :obj:`optical_signal`
            The same object.
        """
        fmt = '-' if not fmt and mode != 'both' else ['-','-'] if not fmt and mode == 'both' else fmt 
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

        if mode == 'x':
            args = (t, I[0], fmt)
        elif mode == 'y':
            args = (t, I[1], fmt)
        elif mode == 'both':
            args = (t, I[0], fmt[0], t, I[1], fmt[1])
        elif mode == 'abs':
            args = (t, I[0] + I[1], fmt)
        else:
            raise TypeError('argument `mode` must to be one of the following values ("x","y","both","abs").')
        
        label = kwargs.pop('label', None)

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
            fmt: Union[str, list]='-', 
            mode: Literal['x','y','both']='x', 
            n: int=None,
            xlabel: str=None,
            ylabel: str=None, 
            yscale: Literal['linear', 'dbm']='dbm', 
            style: Literal['dark', 'light'] = 'dark',
            grid: bool=True,
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
        \*\*kwargs : :obj:`dict`
            Aditional matplotlib arguments.

        Returns
        -------
        self : :obj:`optical_signal`
            The same object.
        """
        if mode == 'both' and isinstance(fmt, str):
            fmt = [fmt, fmt]
        elif mode != 'both' and isinstance(fmt, (list, tuple)):
            fmt = fmt[0]

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

        if mode == 'x':
            args = (f, psd[0], fmt)
        elif mode == 'y':
            args = (f, psd[1], fmt)
        elif mode == 'both':
            args = (f, psd[0], fmt[0], f, psd[1], fmt[1])
        else:
            raise TypeError('argument `mode` should be ("x", "y" or "both")')    
        
        label = kwargs.pop('label', None)

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
    """A class to represent the parameters of an eye diagram.

    This object contains the parameters of an eye diagram and methods to plot it.

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

    def __init__(self, eye_dict={}):
        if eye_dict:
            for key, value in eye_dict.items():
                setattr(self, key, value)
        
    def __str__(self, title: str=None): 
        """Return a formatted string with the eye diagram data."""
        if title is None:
            title = self.__class__.__name__
        
        title = 3*'*' + f'    {title}    ' + 3*'*'
        sub = len(title)*'-'

        np.set_printoptions(precision=1, threshold=10)

        msg = f'\n{sub}\n{title}\n{sub}\n ' + '\n '.join([f'{key} : {value}' for key, value in self.__dict__.items() if key != 'ejecution_time'])
        
        if self.ejecution_time is not None:
            msg += f'\n time  :  {si(self.ejecution_time, "s", 1)}\n'
        return msg
    
    def print(self, msg: str=None): 
        """Print object parameters.

        Parameters
        ----------
        msg : :obj:`str`, optional
            Top message to show.

        Returns
        -------
        :obj:`eye`
            same object
        """
        print(self.__str__(msg))
        return self
    
    def plot(self, 
             medias_=True, 
             legend_=True, 
             show_=True, 
             save_=False, 
             filename=None, 
             style: Literal['dark', 'light']='dark', 
             cmap:Literal['viridis', 'plasma', 'inferno', 'cividis', 'magma', 'winter']='winter',
             label: str = ''):
        """ Plot eye diagram.

        Parameters
        ----------
        style : :obj:`str`, optional
            Plot style. 'dark' or 'light'.
        means_ : :obj:`bool`, optional
            If True, plot mean values.
        legend_ : :obj:`bool`, optional
            If True, show legend.
        show_ : :obj:`bool`, optional
            If True, show plot.
        save_ : :obj:`bool`, optional
            If True, save plot.
        filename : :obj:`str`, optional
            Filename to save plot.
        cmap : :obj:`str`, optional
            Colormap to plot.
        label : :obj:`str`, optional
            Label to show in title.

        Returns
        -------
        eye : same object
        """

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

        if save_: 
            if filename is None:
                filename = 'eyediagram.png'
            plt.savefig(filename, dpi=300)
        if show_: 
            plt.show()
        plt.style.use('default')
        return self


if __name__ == '__main__':
    print('done')