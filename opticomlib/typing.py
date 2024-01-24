"""
======================================
Data types (:mod:`opticomlib._types_`)
======================================

.. autosummary::
   :toctree: generated/

   global_variables      -- Global variables class
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
    def __init__(self):
        """
            ### Descripción:
            Variables globales para la simulación de sistemas ópticos con modulación PPM.
            
            ---

            ### Parámetros de la clase:
            - `sps` (int) - muestras por slot (default: 16) 
            - `slot_rate` (float) - tasa de slots en [Hz] (default: 1e9)
            - `fs` (float) - frecuencia de muestreo en [Hz] (default: slot_rate*sps)
            - `dt` (float) - periodo de muestreo en [s] (default: 1/fs)
            - `lambda_opt` (float) - longitud de onda del canal óptico en [m] (default: 1550e-9)
            - `f0` (float) - frecuencia central del canal óptico en [Hz] (default: c/lambda_opt)
        """

        self.sps = 16
        self.R = 1e9
        self.fs = self.R*self.sps
        self.dt = 1/self.fs
        self.lambda_opt = 1550e-9
        self.f0 = c/self.lambda_opt
    
    def update(self):
        self.R = self.bit_rate*self.M/np.log2(self.M)
        self.fs = self.R*self.sps
        self.dt = 1/self.fs
        self.f0 = c/self.lambda_opt

    def __call__(self, sps, R=None, fs=None, lambda_opt=1550e-9) -> Any:
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
        
    def print(self):
        for key, value in self.__dict__.items():
            print(f'{key} : {value}')

    def __str__(self):
        return str(self.__dict__)

global_vars = global_variables()


class binary_sequence():
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
    def __getitem__(self, key): return binary_sequence(self.data[key])
    def __add__(self, other): 
        if isinstance(other, str):
            other = str2array(other)
        elif isinstance(other, binary_sequence):
            other = other.data.tolist()
        return binary_sequence(self.data.tolist() + other)
    def __radd__(self, other): 
        other = binary_sequence(other)
        return other.__add__(self)

    def len(self): return self.data.size
    def type(self): return type(self)
    def print(self, msg: str=None): 
        if msg: print(3*'*',msg,3*'*')
        print(self, end='\n\n')
        return self
    def sizeof(self) -> int:
        return sizeof(self)


class electrical_signal():
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
    
    def type(self): 
        return type(self)
    
    def print(self, msg: str=None): 
        if msg: print(3*'*',msg,3*'*')
        print(self, end='\n\n')
        return self

    def sizeof(self) -> int:
        return sizeof(self)

    def fs(self): 
        return global_vars.fs
    
    def sps(self):
        return global_vars.sps
    
    def dt(self): 
        return global_vars.dt
    
    def t(self): 
        return np.linspace(0, self.len()*global_vars.dt, self.len(), endpoint=True)
    
    def w(self): 
        return 2*pi*fftfreq(self.len())*self.fs()
    
    def power(self, by: Literal['signal','noise','all']='all'): 
        if by not in ['signal', 'noise', 'all']:
            raise TypeError('`by` debe tomar los valores ("signal", "noise", "all")')
        return np.mean(self.abs(by)**2, axis=-1)

    ## Métodos propios de esta clase
    def copy(self, n=None):
        if n is None: 
            n = self.len()
        return self[:n]

    def abs(self, by: Literal['signal','noise','all']='all'):
        if by == 'signal':
            return np.abs(self.signal)
        elif by == 'noise':
            return np.abs(self.noise)
        elif by == 'all':
            return np.abs(self.signal + self.noise)
        else:
            raise TypeError('`by` debe tomar los valores ("signal", "noise", "all")')
    
    def len(self): 
        return self.signal.size
    
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

    def plot(self, fmt=None, n=None, xlabel=None, ylabel=None, **kargs): 
        if fmt is None:
            fmt = '-'  
        if n is None: 
            n = self.len()
        plt.plot(self.t()[:n], np.abs(self[:n].signal+self[:n].noise), fmt, **kargs)
        plt.xlabel(xlabel if xlabel else 'Tiempo [s]')
        plt.ylabel(ylabel if ylabel else 'Amplitud [u.a.]')
        
        if 'label'  in kargs.keys():
            plt.legend()
        return self
    
    def psd(self, fmt=None, kind: Literal['linear','log']='log', n=None, **kargs):
        if fmt is None:
            fmt = '-'
        if n is None:
            n = self.len()

        f = fftshift( fftfreq(n, d=self.dt())*1e-9)  # GHz
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
        sps,t = global_vars.sps, self.t()
        if n is None: 
            n = self.len()
        for i in t[:n*sps][::sps]:
            plt.axvline(i, color='k', ls='--', alpha=0.3,lw=1)
        return self


class optical_signal(electrical_signal):
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

    def plot(self, fmt=None, mode: Literal['x','y','xy','abs']='abs', n=None, **kargs): 
        t = self.t()[:n]
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
            plt.plot(t, (self.signal[0,:n] + self.noise[0,:n]).real, fmt[0], label=label, **kargs)
        elif mode == 'y':
            label = label if label_flag else 'Polarización Y'
            plt.plot(t, (self.signal[1,:n] + self.noise[1,:n]).real, fmt[0], label=label, **kargs)
        elif mode == 'xy':
            plt.plot(t, (self.signal[0,:n]+self.noise[0,:n]).real, fmt[0], t, (self.signal[1,:n]+self.noise[1,:n]).real, fmt[1], label=['Polarización X', 'Polarización Y'], **kargs)
        elif mode == 'abs':
            label = label if label_flag else 'Abs'
            s = self[:n].abs()
            plt.plot(t, (s[0]**2 + s[1]**2)**0.5, fmt[0], label=label, **kargs)
        else:
            raise TypeError('El argumento `mode` debe se uno de los siguientes valores ("x","y","xy","abs").')

        plt.legend()
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Potencia [W]')
        return self
    
    def psd(self, fmt=None, kind: Literal['linear', 'log']='log', mode: Literal['x','y']=None, n=None, **kargs):
        if fmt is None:
            fmt = '-'
        if mode is None:
            mode = 'x'
        if n is None:
            n = self.len()
        
        f = fftshift( fftfreq(n, d=self.dt())*1e-9)  # GHz

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
        sps,t = global_vars.sps, self.t()
        if n is None: 
            n = self.len()
        for i in t[:n*sps][::sps]:
            plt.axvline(i,color='k', ls='--', alpha=0.3, lw=1)
        return self
    

class eye():
    def __init__(self, eye_dict={}):
        if eye_dict:
            for key, value in eye_dict.items():
                setattr(self, key, value)
        
    def __str__(self): return str(self.__dict__)
    
    def print(self, msg: str=None): 
        if msg: print(3*'*', msg, 3*'*')
        print(self, end='\n\n')
        return self
    
    def print_(self, msg: str=None):
        if msg: print(3*'*', msg, 3*'*')
        for key, value in self.__dict__.items():
            print(f'{key} : {value}')
        return self
    
    def plot(self, umbral, nbits=1000, medias_=True, legend_=True, show_=True, save_=False, filename=None, style: Literal['dark', 'light']='dark', cmap:Literal['viridis', 'plasma', 'inferno', 'cividis', 'magma', 'winter']='winter'):
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
        
        nbits = min(nbits, self.y.size//self.sps)
        
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