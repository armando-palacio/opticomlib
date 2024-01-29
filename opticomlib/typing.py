"""
=====================================
Data types (:mod:`opticomlib.typing`)
=====================================

.. autosummary::
   :toctree: generated/

   global_vars           -- Global variables instance
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

from typing import Literal, Union, Any

from .utils import (
    str2array, 
    dbm, 
)

class global_variables():
    """
    global variables object
    
    Args:
        sps (int): muestras por slot (default: 16) 
        R (float): tasa de slots en [Hz] (default: 1e9)
        fs (float): frecuencia de muestreo en [Hz] (default: R*sps)
        lambda_opt (float): longitud de onda del canal óptico en [m] (default: 1550e-9)
    
    Attributes:
        sps (int): muestras por slot
        R (float): tasa de slots en [Hz]
        fs (float): frecuencia de muestreo en [Hz]
        dt (float): paso temporal en [s]
        lambda_opt (float): longitud de onda del canal óptico en [m]
        f0 (float): frecuencia central en [Hz]

    Methods:
        __call__(sps, R=None, fs=None, lambda_opt=1550e-9): Updates the global variables.
        print(): Prints the global variables.
    """
    def __init__(self):
        self.sps = 16
        self.R = 1e9
        self.fs = self.R*self.sps
        self.dt = 1/self.fs
        self.lambda_opt = 1550e-9
        self.f0 = c/self.lambda_opt


    def __call__(self, sps: int, R: float=None, fs: float=None, lambda_opt: float=1550e-9) -> Any:
        self.sps = sps
        
        if R: 
            self.R = R
            self.fs = fs = R*sps
        elif fs:
            self.fs = fs
            self.R = fs/sps
            self.dt = 1/fs
        
        self.lambda_opt = lambda_opt
        self.f0 = c/lambda_opt
        
        return self
        
    def __str__(self):
        return str(self.__dict__)
    
    def print(self):
        for key, value in self.__dict__.items():
            print(f'{key} : {value}')

global_vars = global_variables()


class binary_sequence():
    """
    binary sequence object

    Args:
        data: The data to initialize the binary sequence. It can be a string, list, tuple, or numpy array.

    Attributes:
        data (ndarray): The binary sequence data stored as a numpy array. Shape (Nx1).
        ejecution_time (float): The execution time of previous device.

    Raises:
        TypeError: If the data argument is not a string, list, tuple, or numpy array.
        ValueError: If the string data contains non-binary numbers.

    """

    def __init__(self, data: Union[str, list, tuple, np.ndarray]):
        if not isinstance(data, (str, list, tuple, np.ndarray)):
            raise TypeError("El argumento debe ser una cadena de texto, una lista, una tupla o un array de numpy!")
        
        if isinstance(data, str):
            data = str2array(data)
            if set(data)-set([0,1]):
                raise ValueError("La cadena de texto debe contener únicamente números binarios!")

        self.data = np.array(data, dtype=np.uint8)

        self.ejecution_time = None

    def __str__(self): return f'data : {self.data}\nlen : {self.len()}\nsize : {self.sizeof()} bytes\ntime : {self.ejecution_time} s'
    def __len__(self): return self.len()
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("El índice debe ser un número entero!")
        return binary_sequence(self.data[key])
    
    def __add__(self, other): 
        if isinstance(other, str):
            other = str2array(other)
        elif isinstance(other, binary_sequence):
            other = other.data.tolist()
        return binary_sequence(self.data.tolist() + other)
    
    def __radd__(self, other): 
        other = binary_sequence(other)
        return other.__add__(self)

    def len(self): 
        """
        Get the length of the binary sequence data.
        """
        return self.data.size
    
    def type(self): 
        """
        Get the type of the object
        """
        return type(self)
    def print(self, msg: str=None): 
        """
        Print object parameters.

        Args:
            msg (str) [Opcional]: top message to show
        """
        if msg: print(3*'*',msg,3*'*')
        print(self, end='\n\n')
        return self
    
    def sizeof(self):
        """
        Get memory size of object in bytes.
        """
        return sizeof(self)


class electrical_signal():
    """
    electrical signal object

    Args:
        signal (Union[list, tuple, np.ndarray]) [Optional]: The signal values. Defaults to None.
        noise (Union[list, tuple, np.ndarray]) [Optional]: The noise values. Defaults to None.

    If signal or noise are not passed as arguments it will be initialized to zeros((N,))    
    
    Attributes:
        signal (ndarray): The signal values. Shape (Nx1).
        noise (ndarray): The noise values. Shape (Nx1).
        ejecution_time (None): The execution time of previous device.
    """
    def __init__(self, signal: Union[list, tuple, np.ndarray]=None, noise: Union[list, tuple, np.ndarray]=None) -> None:
        if signal is None and noise is None:
            raise KeyError("Se debe pasar como argumento 'signal' o 'noise'")
        if (signal is not None) and (noise is not None) and (len(signal)!=len(noise)):
            raise ValueError(f"Los vectores 'signal'({len(signal)}) y 'noise'({len(noise)}) deben coincidir en longitud")

        if signal is None:
            self.signal = np.zeros_like(noise, dtype=complex)
        elif not isinstance(signal, (list, tuple, np.ndarray)):
            raise TypeError("'signal' debe ser de tipo lista, tupla o array de numpy!")
        else:
            self.signal = np.array(signal, dtype=complex) # shape (1xN)
        
        if noise is None:
            self.noise = np.zeros_like(signal, dtype=complex)
        elif not isinstance(noise, (list, tuple, np.ndarray)):
            raise TypeError("'noise' debe ser de tipo lista, tupla o array de numpy!")
        else:
            self.noise = np.array(noise, dtype=complex)
        
        self.ejecution_time = None

    def __str__(self): 
        return f'signal : {self.signal}\nnoise : {self.noise}\nlen : {self.len()}\npower : {self.power():.1e} W ({dbm(self.power()):.2f} dBm)\nsize : {self.sizeof()} bytes\ntime : {self.ejecution_time}'
    
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

    def __call__(self, dominio):
        if dominio == 'w':
            return electrical_signal( fft(self.signal), fft(self.noise) )
        if dominio == 't':
            return electrical_signal( ifft(self.signal), ifft(self.noise) )
        else:
            raise TypeError("solo se aceptan los argumentos 'w' o 't'")
    
    def __gt__(self, other): 
        if isinstance(other, electrical_signal): 
            return binary_sequence( (self.signal+self.noise > other.signal).astype(int) )
        if isinstance(other, (int, float, complex, np.ndarray)):
            return binary_sequence( (self.signal+self.noise > other).astype(int) )
             
    def len(self): 
        """
        Get the length of the electrical signal data.
        """
        return self.signal.size

    def type(self): 
        """
        Get the type of the object
        """
        return type(self)
    
    def print(self, msg: str=None): 
        """
        Print object parameters.

        Args:
            msg (str) [Optional]: top message to show

        Returns:
            electrical_signal: Same object.
        """
        if msg: print(3*'*',msg,3*'*')
        print(self, end='\n\n')
        return self

    def sizeof(self):
        """
        Get memory size of object in bytes.
        """
        return sizeof(self)

    def fs(self): 
        """
        Get sampling rate of signal.
        """
        return global_vars.fs
    
    def sps(self):
        """
        Get samples por slot of signal.
        """
        return global_vars.sps
    
    def dt(self): 
        """
        Get time between samples.
        """
        return global_vars.dt
    
    def t(self): 
        """
        Return time array for electrical signal.
        """
        return np.linspace(0, self.len()*global_vars.dt, self.len(), endpoint=True)
    
    def w(self, shift: bool=False): 
        """
        Return angular frequency for spectrum representation.

        Args:
            shift (bool) [Optional]: If ``True`` apply fftshift(), default ``False``
        """
        w = 2*pi*fftfreq(self.len())*self.fs()
        if shift:
            return fftshift(w)
        return w
    
    def power(self, by: Literal['signal','noise','all']='all'): 
        """
        Get power of the electrical signal.

        Args:
            by (str) [Optional]: defines from which attribute to obtain the power. If ``'all'``, power of signal+noise is determinated.
        """
        if by not in ['signal', 'noise', 'all']:
            raise TypeError('`by` debe tomar los valores ("signal", "noise", "all")')
        return np.mean(self.abs(by)**2, axis=-1)

    ## Métodos propios de esta clase
    def copy(self, n: int=None):
        """
        Return a copy of the object

        Args:
            n (int) [Optional]: index to truncate original object.
        """
        if n is None: 
            n = self.len()
        return self[:n]

    def abs(self, by: Literal['signal','noise','all']='all'):
        """
        Get abs of electrical signal.

        Args:
            by (str) [Optional] : defines from which attribute to obtain the abs. If ``'all'``, abs of signal+noise is determinated.
        """
        if by == 'signal':
            return np.abs(self.signal)
        elif by == 'noise':
            return np.abs(self.noise)
        elif by == 'all':
            return np.abs(self.signal + self.noise)
        else:
            raise TypeError('`by` debe tomar los valores ("signal", "noise", "all")')
    

    def plot(self, fmt: str=None, n: int=None, xlabel: str=None, ylabel: str=None, **kargs): 
        """
        Plot real part of electrical signal.

        Args:
            fmt (str): Format style of line. Example ``'b-.'``
            n (int) : number of samples to plot (default: ``self.len()``).
            xlabel (str): X-axis label (default - ``'Tiempo [ns]'``).
            ylabel (str): Y-axis label (default - ``'Amplitud [u.a]'``).
            **kargs : all arguments compatible with ``matplotlib.pyplot.plot()``.

        Returns:
            electrical_signal: Same object.
        """
        if fmt is None:
            fmt = '-'  
        if n is None: 
            n = self.len()
        plt.plot(self.t()[:n]*1e9, (self[:n].signal+self[:n].noise).real, fmt, **kargs)
        plt.xlabel(xlabel if xlabel else 'Tiempo [ns]')
        plt.ylabel(ylabel if ylabel else 'Amplitud [u.a.]')
        
        if 'label'  in kargs.keys():
            plt.legend()
        return self
    
    def psd(self, fmt=None, kind: Literal['linear','log']='log', n=None, **kargs):
        """
        Plot Power Spectral Density (PSD) of the electrical signal.

        Args:
            fmt (str): Format style of line. Example ``'b-.'``
            kind (str): kind of Y-axis plot.
            n (int) : number of samples to plot (default: ``self.len()``).
            **kargs : all arguments compatible with ``matplotlib.pyplot.plot()``.
        if ``'label'`` is in ``kargs.keys()``, legend will be show be default. 

        Returns:
            electrical_signal: Same object.
        """
        if fmt is None:
            fmt = '-'
        if n is None:
            n = self.len()

        # f = fftshift( fftfreq(n, d=self.dt())*1e-9)  # GHz
        f = self[:n].w(shift=True)/2/pi * 1e-9
        psd = fftshift(self[:n]('w').abs('signal')**2/n**2)
        
        if kind == 'linear':
            plt.plot(f, psd*1e3, fmt, **kargs)
        elif kind == 'log':
            plt.semilogy(f, psd*1e3, fmt, **kargs)
        else:
            raise TypeError('El argumento `kind` debe ser uno de los siguientes valores ("linear", "log")')
        plt.xlabel('Frecuencia [GHz]')
        plt.ylabel('Potencia [mW]')
        if 'label'  in kargs.keys():
            plt.legend()
        return self
    
    def grid(self, n=None):
        """
        Plot vertical grid each every time slot. 

        Args:
            n (int) : number of samples to plot (default: ``self.len()``).

        Returns:
            electrical_signal: Same object.
        """
        sps,t = self.sps(), self.t()
        if n is None: 
            n = self.len()
        for i in t[:n*sps+1][::sps]:
            plt.axvline(i, color='k', ls='--', alpha=0.3, lw=1)
        return self


class optical_signal(electrical_signal):
    """
    optical signal object

    Args:
        signal (Union[list, tuple, np.ndarray]) [Optional]: The signal values. Defaults to None.
        noise (Union[list, tuple, np.ndarray]) [Optional]: The noise values. Defaults to None.

    If signal or noise are not passed as arguments it will be initialized to zeros((2,N))    
    
    Attributes:
        signal (ndarray): The signal values. Shape (2, N).
        noise (ndarray): The noise values. Shape (2, N).
        ejecution_time (None): The execution time of previous device.
    """
    def __init__(self, signal: Union[list, tuple, np.ndarray], noise: Union[list, tuple, np.ndarray]=None) -> None:
        if np.array(signal).ndim==1:
            signal = np.array([signal, np.zeros_like(signal)])
        super().__init__( signal, noise )

    def __str__(self): 
        return f'signal : {self.signal}\nnoise : {self.noise}\nlen : {self.len()}\ntotal power : {sum(self.power()):.1e} W ({dbm(sum(self.power())):.2f} dBm)\nsize : {self.sizeof()} bytes\ntime : {self.ejecution_time}'
    
    def len(self): 
        return self.signal.shape[1]

    def __add__(self, other): 
        if isinstance(other, optical_signal):
            return optical_signal(self.signal + other.signal, self.noise + other.noise)
        if isinstance(other, (int, float, complex, np.ndarray)):
            return optical_signal(self.signal + other, self.noise + other)
    
    def __mul__(self, other):
        if isinstance(other, optical_signal):
            return optical_signal(self.signal * other.signal, self.noise * other.noise)
        if isinstance(other, (int, float, complex, np.ndarray)):
            return optical_signal(self.signal * other, self.noise * other)
    
    def __getitem__(self, key): 
        return optical_signal( self.signal[:,key], self.noise[:,key] )
    
    def __call__(self, dominio: Literal['t','w']):
        if dominio == 'w':
            return optical_signal( fft(self.signal, axis=-1), fft(self.noise, axis=-1) )
        if dominio == 't':
            return optical_signal( ifft(self.signal, axis=-1), ifft(self.noise, axis=-1) )
        else:
            raise TypeError("solo se aceptan los argumentos 'w' o 't'")

    def plot(self, fmt=None, mode: Literal['x','y','both','abs']='abs', n=None, **kargs): 
        """
        Plot intensity of optical signal for selected mode.

        Args:
            fmt (str): Format style of line. Example ``'b-.'``
            mode (str): Polarization mode to show. ``'abs'`` plot intensity of signal x+y.
            n (int) : number of samples to plot (default: ``self.len()``).
            **kargs : all arguments compatible with ``matplotlib.pyplot.plot()``.

        Returns:
            optical_signal: Same object.
        """
        t = self.t()[:n]*1e9
        if fmt is None:
            fmt = ['-', '-']
        if isinstance(fmt, str):
            fmt = [fmt, fmt]
        if n is None: 
            n = self.len()

        if 'label' in kargs.keys():
            label = kargs['label']
            label_flag = True
            kargs.pop('label')
        else:
            label_flag = False

        if mode == 'x':
            label = label if label_flag else 'Polarización X'
            plt.plot(t, np.abs(self.signal[0,:n] + self.noise[0,:n])**2, fmt[0], label=label, **kargs)
        elif mode == 'y':
            label = label if label_flag else 'Polarización Y'
            plt.plot(t, np.abs(self.signal[1,:n] + self.noise[1,:n])**2, fmt[0], label=label, **kargs)
        elif mode == 'both':
            plt.plot(t, np.abs(self.signal[0,:n]+self.noise[0,:n])**2, fmt[0], t, np.abs(self.signal[1,:n]+self.noise[1,:n])**2, fmt[1], label=['Polarización X', 'Polarización Y'], **kargs)
        elif mode == 'abs':
            label = label if label_flag else 'Abs'
            s = self[:n].abs()
            plt.plot(t, (s[0]**2 + s[1]**2), fmt[0], label=label, **kargs)
        else:
            raise TypeError('El argumento `mode` debe se uno de los siguientes valores ("x","y","xy","abs").')

        plt.legend()
        plt.xlabel('Tiempo [ns]')
        plt.ylabel('Potencia [W]')
        return self
    
    def psd(self, fmt=None, kind: Literal['linear', 'log']='log', mode: Literal['x','y']='x', n=None, **kargs):
        """
        Plot Power Spectral Density (PSD) of the optical signal.

        Args:
            fmt (str): Format style of line. Example ``'b-.'``
            kind (str): kind of Y-axis plot.
            mode (str): polarization mode to show.
            n (int) : number of samples to plot (default: ``self.len()``).
            **kargs : all arguments compatible with ``matplotlib.pyplot.plot()``.
        if ``'label'`` is in ``kargs.keys()``, legend will be show be default. 

        Returns:
            optical_signal: Same object.
        """
        if fmt is None:
            fmt = '-'
        if mode is None:
            mode = 'x'
        if n is None:
            n = self.len()
        
        # f = fftshift( fftfreq(n, d=self.dt())*1e-9)  # GHz
        f = self[:n].w(shift=True)/2/pi * 1e-9

        if mode =='x':
            psd = fftshift(self[:n]('w').abs('signal')[0]**2/n**2)
        elif mode == 'y':
            psd = fftshift(self[:n]('w').abs('signal')[1]**2/n**2)
        else:
            raise TypeError('El argumento `mode` debe ser uno de los siguientes valores ("x" o "y")')    
        
        if kind == 'linear':
            plt.plot(f, psd*1e3, fmt, **kargs)
        elif kind == 'log':
            plt.semilogy(f, psd*1e3, fmt, **kargs)
        else:
            raise TypeError('El argumento `kind` debe ser uno de los siguientes valores ("linear", "log")')
        plt.xlabel('Frecuencia [GHz]')
        plt.ylabel('Potencia [mW]')
        if 'label'  in kargs.keys():
            plt.legend()
        return self
    
    def grid(self, n=None):
        """
        Plot vertical grid each every time slot. 

        Args:
            n (int) : number of samples to plot (default: ``self.len()``).

        Returns:
            optical_signal: Same object.
        """
        sps,t = global_vars.sps, self.t()
        if n is None: 
            n = self.len()
        for i in t[:n*sps+1][::sps]:
            plt.axvline(i,color='k', ls='--', alpha=0.3, lw=1)
        return self
    

class eye():
    """
    Eye diagram parameters object.

    Attributes:
        t (ndarray): the time values resampled. Shape (Nx1).
        y (ndarray): the signal values resampled. Shape (Nx1).
        dt (float): time between samples.
        sps (int): samples per slot.
        t_left (float): cross time of left edge.
        t_right (float): cross time of right edge.
        t_opt (float): optimal time decision.
        t_dist (float): time between slots.
        t_span0 (float): t_opt - t_dist*5%.
        t_span1 (float): t_opt + t_dist*5%.
        y_top (ndarray): samples of signal above threshold and within t_span0 and t_apan1.
        y_bot (ndarray): samples of signal below threshold and within t_span0 and t_apan1.
        mu0 (float): mean of y_bot.
        mu1 (float): mean of y_top.
        s0 (float): standard deviation of y_bot.
        s1 (float): standard deviation of y_top.
        er (float): extinsion ratio.
        eye_h (float): eye height.
    """
    def __init__(self, eye_dict={}):
        if eye_dict:
            for key, value in eye_dict.items():
                setattr(self, key, value)
        
    def __str__(self): return str(self.__dict__)
    
    def print(self, msg: str=None): 
        """
        Print object parameters.

        Args:
            msg (str) [Optional]: top message to show
        
        Returns:
            eye: Same object.
        """
        if msg: print(3*'*', msg, 3*'*')
        print(self, end='\n\n')
        return self
    
    def print_(self, msg: str=None):
        """
        Print object parameters as (key: value).

        Args:
            msg (str) [Optional]: top message to show
        
        Returns:
            eye: Same object.
        """
        if msg: print(3*'*', msg, 3*'*')
        for key, value in self.__dict__.items():
            print(f'{key} : {value}')
        return self
    
    def plot(self, umbral, medias_=True, legend_=True, show_=True, save_=False, filename=None, style: Literal['dark', 'light']='dark', cmap:Literal['viridis', 'plasma', 'inferno', 'cividis', 'magma', 'winter']='winter'):
        """
        Plot eye diagram.

        Args:
            umbral (float): threshold value.
            medias_ (bool) [Optional]: if ``True`` plot mean values.
            legend_ (bool) [Optional]: if ``True`` show legend.
            show_ (bool) [Optional]: if ``True`` show plot.
            save_ (bool) [Optional]: if ``True`` save plot.
            filename (str) [Optional]: filename to save plot.
            style (str) [Optional]: plot style. ``'dark'`` or ``'light'``.
            cmap (str) [Optional]: colormap to plot.

        Returns:
            eye: Same object.
        """
        if not show_:
            return self

        from matplotlib.widgets import Slider
        if style == 'dark':
            plt.style.use('dark_background')
            t_opt_color = '#60FF86'
            r_th_color = '#FF345F'
            means_color = 'white'
            bgcolor='black'
        elif style == 'light':
            t_opt_color = 'green'#'#229954'
            r_th_color = 'red'#'#B03A2E'
            means_color = '#5A5A5A'
            bgcolor='white'
        else:
            raise TypeError("El argumento `style` debe ser uno de los siguientes valores ('dark', 'light')")
        
        nslots = min(nslots, self.y.size//self.sps)
        
        fig,ax = plt.subplots(2,2, gridspec_kw={'width_ratios': [4,1], 
                                            'height_ratios': [2,6], 
                                            'wspace': 0.03,
                                            'hspace': 0.05},
                                            figsize=(8,6))
        
        # suptitle('Diagrama de ojo')
        dt = self.dt
        ax[1,0].set_xlim(-1-dt,1)
        ax[1,0].set_ylim(self.mu0-4*self.s0, self.mu1+4*self.s1)
        ax[1,0].set_ylabel(r'Amplitude [mV]', fontsize=12)
        ax[1,0].grid(color='grey', ls='--', lw=0.5, alpha=0.5)
        ax[1,0].set_xticks([-1,-0.5,0,0.5,1])
        ax[1,0].set_xlabel(r'Time [$t/T_{slot}$]', fontsize=12)
        t_line1 = ax[1,0].axvline(self.t_opt, color = t_opt_color, ls = '--', alpha = 0.7)
        y_line1 = ax[1,0].axhline(umbral, color = r_th_color, ls = '--', alpha = 0.7)

        t_line_span0 = ax[1,0].axvline(self.t_span0, color = t_opt_color, ls = '-', alpha = 0.4)
        t_line_span1 = ax[1,0].axvline(self.t_span1, color = t_opt_color, ls = '-', alpha = 0.4)

        if legend_: 
            ax[1,0].legend([r'$t_{opt}$', r'$r_{th}$'], fontsize=12, loc='upper right')
        
        if medias_:
            ax[1,0].axhline(self.mu1, color = means_color, ls = ':', alpha = 0.7)
            ax[1,0].axhline(self.mu0, color = means_color, ls = '-.', alpha = 0.7)

            ax[1,1].axhline(self.mu1, color = means_color, ls = ':', alpha = 0.7)
            ax[1,1].axhline(self.mu0, color = means_color, ls = '-.', alpha = 0.7)
            if legend_:
                ax[1,1].legend([r'$\mu_1$',r'$\mu_0$'])

        ax[1,1].sharey(ax[1,0])
        ax[1,1].tick_params(axis='x', which='both', length=0, labelbottom=False)
        ax[1,1].tick_params(axis='y', which='both', length=0, labelleft=False)
        ax[1,1].grid(color='grey', ls='--', lw=0.5, alpha=0.5)
        y_line2 = ax[1,1].axhline(umbral, color = r_th_color, ls = '--', alpha = 0.7)

        ax[0,0].sharex(ax[1,0])
        ax[0,0].tick_params(axis='y', which='both', length=0, labelleft=False)
        ax[0,0].tick_params(axis='x', which='both', length=0, labelbottom=False)
        ax[0,0].grid(color='grey', ls='--', lw=0.5, alpha=0.5)
        t_line2 = ax[0,0].axvline(self.t_opt, color = t_opt_color, ls = '--', alpha = 0.7)

        ax[0,1].set_visible(False)

        y_ = self.y
        t_ = self.t

        ax[1,0].hexbin(
            x = t_, 
            y = y_, 
            gridsize=500, 
            bins='log',
            alpha=0.7, 
            cmap=cmap 
        )
        
        ax[1,1].hist(
            y_[(t_>self.t_opt-0.05*self.t_dist) & (t_<self.t_opt+0.05*self.t_dist)], 
            bins=200, 
            density=True, 
            orientation = 'horizontal', 
            color = r_th_color, 
            alpha = 0.9,
            histtype='step',
        )

        ax[0,0].hist(
            t_[(y_>umbral*0.95) & (y_<umbral*1.05)], 
            bins=200, 
            density=True, 
            orientation = 'vertical', 
            color = t_opt_color, 
            alpha = 0.9,
            histtype='step',
        )

        y_min, y_max = ax[1,0].get_ylim()

        p = ax[1,1].get_position() 
        y_slider_ax = fig.add_axes([p.x1+0.006,p.y0,0.02,p.y1-p.y0])
        p = ax[0,0].get_position() 
        t_slider_ax = fig.add_axes([p.x0,p.y1+0.01,p.x1-p.x0,0.02])

        y_slider = Slider(
            ax=y_slider_ax,
            label='',
            valmin=y_min, 
            valmax=y_max,
            valstep=(y_max-y_min)/20,
            valinit=umbral, 
            initcolor=r_th_color,
            orientation='vertical',
            valfmt='',
            color=bgcolor,
            track_color=bgcolor,
            handle_style = dict(facecolor=r_th_color, edgecolor=bgcolor, size=10)
        )

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

        def update_y_line(val):
            y_line1.set_ydata(val)
            y_line2.set_ydata(val)

            ax[0,0].patches[-1].remove()

            n,_,_ = ax[0,0].hist(
                t_[(y_>val*0.95) & (y_<val*1.05)], 
                bins=200, 
                density=True, 
                orientation = 'vertical', 
                color = r_th_color, 
                alpha = 0.9,
                histtype='step',
            )
            ax[0,0].set_ylim(0,max(n))

        def update_t_line(val):
            t_line1.set_xdata(val)
            t_line2.set_xdata(val)

            t_line_span0.set_xdata(val-0.05*self.t_dist)
            t_line_span1.set_xdata(val+0.05*self.t_dist)

            ax[1,1].patches[-1].remove()

            n,_,_ = ax[1,1].hist(
                y_[(t_>val-0.01) & (t_<val+0.01)], 
                bins=200, 
                density=True, 
                orientation = 'horizontal', 
                color = t_opt_color, 
                alpha = 0.9,
                histtype='step',
            )
            ax[1,1].set_xlim(0,max(n))

        y_slider.on_changed(update_y_line)
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