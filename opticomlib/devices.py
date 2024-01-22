"""
===================================================================
Models for optoelectronic components (:mod:`opticomlib.components`)
===================================================================

.. autosummary::
   :toctree: generated/

   PRBS                  -- Pseudorandom binary sequence generator
   DAC                   -- Digital-to-analog converter (DAC) model
   BPF                   -- Optical band-pass filter (BPF) bessel model
   EDFA                  -- Erbium-doped fiber amplifier (EDFA) simple model
   DM                    -- Dispersion medium model
   FIBER                 -- Optical fiber model (dispersion, attenuation and non-linearities, Split-Step Fourier Method)
   LPF                   -- Electrical low-pass filter (LPF) bessel model
   PD                    -- Photodetector (PD) model
   ADC                   -- Analog-to-digital converter (ADC) model
   GET_EYE               -- Eye diagram parameters and metrics estimator
   SAMPLER               -- Sampler device
"""


"""Basic physical models for optical/electronic components."""
import numpy as np
import scipy.signal as sg
from typing import Literal, Union
from numpy import ndarray
from scipy.constants import pi, k as kB, e, h
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
import sklearn.cluster as sk
from tqdm.auto import tqdm # barra de progreso

from ._types_ import (
    electrical_signal,
    binary_sequence,
    optical_signal,
    global_vars,
    eye,
)

from ._utils_ import (
    generate_prbs,
    idbm,
    idb,
    tic,
    toc,
)



def PRBS(n: int=2**8, user: list=[], defined: int=None) -> binary_sequence:
    """
    ### Descripción:
    Generador Pseudoaleatorio de secuencias binarias.

    ---
    
    ### Args:
    - `n` [Opcional] - cantidad de dígitos a generar (default: `n=2**8`)
    - `user` [Opcional] - secuencia binaria de entrada (default: `user=[]`)
    - `defined` [Opcional] - grado del polinómio generador (default: `defined=None`)

    ---
    
    ### Returns:
    - `binary_sequence`
    """
    tic()

    if user:
        output = binary_sequence( user )
    elif defined:
        output = binary_sequence( generate_prbs(defined) )
    else:
        output = binary_sequence( np.random.randint(0, 2, n) )
    output.ejecution_time = toc()
    return output



def DAC(input: Union[str, list, tuple, ndarray, binary_sequence], 
        Vout: float=None,
        pulse_shape: Literal['rect','gaussian']='rect', 
        **kargs) -> electrical_signal:  
    """
    ### Descripción:
    Conversor digital a analógico. Convierte una secuencia binaria en una señal eléctrica, muestreada a una frecuencia `fs`.

    ---

    ### Args:
    - `input` - secuencia binaria de entrada
    - `Vout` [Opcional] - amplitud de la señal de salida [-15 a 15 Voltios] (default: `amplitud=5.0`)
    - `pulse_shape` [Opcional] - forma de pulso a la salida, puede ser "rect" o "gaussian" (default: `type="rect"`)

    ---
    
    ### Kargs:
    - `c` [Opcional] - chirp del pulso gaussiano. Solo si `type=gaussian` (default: `c=0.0`)
    - `m` [Opcional] - orden del pulso supergausiano. Solo si `type=gaussian` (default: `m=1`)
    - `T` [Opcional] - ancho a mitad de altura del pulso gaussiano en cantidad de muestras. Solo si `type=gaussian` (default: `T=sps`)
    ---

    ### Returns:
    - `electrical_signal`
    """
    tic()
    if not isinstance(input, binary_sequence):
        input = binary_sequence(input)
    
    sps = kargs['sps'] if 'sps' in kargs.keys() else global_vars.sps

    if pulse_shape == 'rect':
        x = np.kron(input.data, np.ones(sps))
    
    elif pulse_shape == 'gaussian':
        c = kargs['c'] if 'c' in kargs.keys() else 0.0
        m = kargs['m'] if 'm' in kargs.keys() else 1
        T = kargs['T'] if 'T' in kargs.keys() else sps

        p = lambda t, T: np.exp(-(1+1j*c)/2 * (t/T)**(2*m))

        t = np.linspace(-2*sps, 2*sps, 4*sps) # vector de tiempo del pulso gaussiano
        k = 2*(2*np.log(2))**(1/(2*m)) # factor de escala entre el ancho de un slot y la desviación estándar del pulso gaussiano
        pulse = p(t, T/k) # pulso gaussiano

        s = np.zeros(input.len()*sps)
        s[int(sps//2)::sps]=input.data
        s[int(sps//2-1)::sps]=input.data

        x = sg.fftconvolve(s, pulse, mode='same')/2
    else:
        raise NameError('El parámetro `type` debe ser uno de los siguientes valores ("rect","gaussian").')

    if Vout:
        x = x * Vout / x.max()

    output = electrical_signal( x )

    output.ejecution_time = toc()
    return output


def MODULATOR(input: electrical_signal, p_laser: float=None, pol: str='x') -> optical_signal:
    """
    ### Descripción:
    Modula la señal óptica de un láser de potencia dada,79 a partir de una señal eléctrica de entrada.
    
    ---

    ### Args:
    - `input` - señal eléctrica moduladora
    - `p_laser` [Opcional] - potencia del laser en [dBm] (default: `p_laser=global_vars.p_laser`)
    - `pol` [Opcional] - eje de polarización de la señal óptica de salida (default: `x`)
    
    ---

    ### Returns:
    - `optical_signal`
    """
    tic()
    
    i = {'x':0, 'y':1}

    if not isinstance(input, electrical_signal):
        raise TypeError("`input` debe ser del tipo (electrical_signal).")
    if not p_laser:
        p_laser = idbm(global_vars.p_laser)
    if pol not in ['x','y']:
        raise TypeError("`pol` debe ser ('x' o 'y').")

    output = optical_signal( np.zeros((2,input.len())) )

    opt_signal = input.signal/(input.power('signal')**0.5) * p_laser**0.5
    output.signal[i[pol]] = opt_signal
    
    if input.power('noise'):
        opt_noise = input.noise/(input.power('noise')**0.5) * p_laser**0.5
        output.noise[i[pol]] = opt_noise
    
    output.ejecution_time = toc()
    return output


def BPF(input: optical_signal, BW: float, n: int=4, fs: float=None) -> optical_signal:
    """
    ### Descripción:
    Filtro Pasa Banda (BPF) Óptico. Filtra la señal óptica de entrada, dejando pasar la banda de frecuencias deseada. 
    
    ---

    ### Args:
    - `input` - señal óptica a filtrar
    - `BW` - ancho de banda del filtro en [Hz]
    - `n` [Opcional] - orden del filtro (default: `n=4`)
    - `fs` [Opcional] - frecuencia de muestreo de la señal de entrada (default: `fs=globals_vars.fs`)
    
    ---
    
    ### Returns:
    - `optical_signal`
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("`input` debe ser del tipo (optical_signal).")
    if not fs:
        fs = global_vars.fs

    sos_band = sg.bessel(N = n, Wn = BW/2, btype = 'low', fs=fs, output='sos', norm='mag')

    output = optical_signal( np.zeros((2,input.len())) )

    output.signal = sg.sosfiltfilt(sos_band, input.signal, axis=-1)

    if np.sum(input.noise):
        output.noise = sg.sosfiltfilt(sos_band, input.noise, axis=-1)

    output.ejecution_time = toc()
    return output


def EDFA(input: optical_signal, G: float, NF: float, BW: float=None) -> tuple: # modelo simplificado del EDFA (no satura)
    """
    ### Descripción:
    Amplificador de fibra dopada con Erbium. Amplifica la señal óptica a la entrada, agregando ruido de emisión espontánea amplificada (ASE). 
    
    ---

    ### Args: 
    - `input` - señal óptica a amplificar
    - `G` - ganancia del amplificador en [dB]
    - `NF` - figura de ruido del amplificador en [dB]
    - `BW` [Opcional] - ancho de banda del amplificador en [Hz] (default: `BW=global_vars.BW_opt`)
    
    ---

    ### Returns:
    -
    - `output` (optical_signal) - señal óptica de salida
    """

    if not isinstance(input, optical_signal):
        raise TypeError("`input` debe ser del tipo (optical_signal).")
    if BW is None:
        BW = global_vars.BW_opt
     
    output = BPF( input * idb(G)**0.5, BW )
    ase = BPF( optical_signal( np.zeros_like(input.signal), np.exp(-1j*np.random.uniform(0, 2*pi, input.noise.shape)) ), BW )
    ejc_time = output.ejecution_time + ase.ejecution_time
    
    tic()
    P_ase = idb(NF) * h * global_vars.f0 * (idb(G)-1) * BW

    norm_x, norm_y = ase.power('noise') # potencia de ruido de ASE en [W] para cada polarización

    ase.noise[0] /= norm_x**0.5 / (P_ase/2)**0.5
    ase.noise[1] /= norm_y**0.5 / (P_ase/2)**0.5

    output += ase

    output.ejecution_time = ejc_time + toc()
    return output


def DM(input: optical_signal, beta_2: float, length: float) -> optical_signal:
    """
    ### Descripción:
    Medio Dispersivo. Emula un medio con solo la propiedad de dispersión, es decir solo `beta_2` diferente de cero. 
    
    ---
    
    ### Args:
    - `signal` - señal óptica de entrada
    - `beta_2` - coeficiente de dispersión de la fibra en [ps^2/km]
    - `length` - longitud del medio dispersivo en [km]
    
    ---
    
    ### Returns:
    - `optical_signal`
    """
    tic()

    if not isinstance(input, optical_signal):
        raise TypeError("El argumento debe ser una señal óptica!") 

    # cambio las unidades de beta_2 y length:
    beta_2 = beta_2 * 1e-12**2/1e3
    length = length * 1e3 
    
    output = (input('w') * np.exp(-beta_2/2 * input.w()**2 * 1j * length))('t')
    
    output.ejecution_time = toc()
    return output


def FIBER(input: optical_signal, length: float, alpha: float=0.0, beta_2: float=0.0, beta_3: float=0.0, gamma: float=0.0, phi_max:float=0.05, show_progress=False) -> optical_signal:
    """
    ### Descripción:
    Fibra Óptica. Simula la transmisión por fibra de una señal óptica de entrada teniendo en cuenta los efectos de la atenuación, dispersión y no linealidades. 
    
    ---

    ### Args:
    - `input` - señal óptica de entrada
    - `length` - longitud de la fibra en [km]
    - `alpha` [Opcional] - coeficiente de atenuación de la fibra en [dB/km] (default: `alpha=0.0`)
    - `beta_2` [Opcional] - coeficiente de dispersión de segundo orden de la fibra en [ps^2/km] (default: `beta_2=0.0`)
    - `beta_3` [Opcional] - coeficiente de dispersión de tercer orden de la fibra en [ps^3/km] (default: `beta_3=0.0`)
    - `gamma` [Opcional] - coeficiente de no linealidad de la fibra en [(W·km)^-1] (default: `gamma=0.0`)
    - `phi_max` [Opcional] - cota superior de la fase no lineal en [rad] (default: `phi_max=0.05`)
    - `show_progress` [Opcional] - mostrar barra de progreso (default: `show_progress=False`)
    
    ---
    
    ### Returns:
    - `optical_signal`
    """

    tic()
    alpha  = alpha/4.343 # [1/km]

    w = input.w()*1e-12 # [rad/ps]
    D_op = -alpha/2 - 1j/2 * beta_2 * w**2 - 1j/6 * beta_3 * w**3

    A = input.signal

    h = length if (beta_2==0 and beta_3==0) or gamma==0 else phi_max / (gamma * (np.abs(A[0])**2 + np.abs(A[1])**2)).max()

    x_length = h

    if show_progress:
        barra_progreso = tqdm(total=100)

    while True:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )

        if show_progress:
            barra_progreso.update( 100 * h / length )

        h = phi_max / (gamma * (np.abs(A[0])**2 + np.abs(A[1])**2)).max() if gamma != 0 else length 

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length
    
    if h != 0:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        
        if show_progress:
            barra_progreso.update( 100 * h / length )

    output = optical_signal( A, input.noise )
    output.ejecution_time = toc()
    return output



def LPF(input: Union[ndarray, electrical_signal], BW: float, n: int=4, fs: float=None) -> electrical_signal:
        """
        ### Descripción:
        Filtro Pasa Bajo (LPF) Eléctrico. Filtra la señal eléctrica de entrada, dejando pasar la banda de frecuencias deseada. 
        
        ---

        ### Args:
        - `input` - señal eléctrica a filtrar
        - `BW` - ancho de banda del filtro en [Hz]
        - `n` [Opcional] - orden del filtro (default: `n=4`)
        - `fs` [Opcional] - frecuencia de muestreo de la señal de entrada (default: `fs=globals_vars.fs`)
        
        ---
        
        ### Returns:
        - `electrical_signal`
        """
        tic()

        if not isinstance(input, electrical_signal):
            if not isinstance(input, ndarray):
                raise TypeError("`input` debe ser del tipo (ndarray ó electrical_signal).")
            else:
                input = electrical_signal(input)
        if not fs:
            fs = global_vars.fs

        sos_band = sg.bessel(N = n, Wn = BW, btype = 'low', fs=fs, output='sos', norm='mag')

        output = electrical_signal( np.zeros(input.len()) )

        output.signal = sg.sosfiltfilt(sos_band, input.signal)

        if np.sum(input.noise):
            output.noise = sg.sosfiltfilt(sos_band, input.noise)
        
        output.ejecution_time = toc()
        return output



def PD(input: optical_signal, BW: float=None, R: float=1.0, T: float=300.0, R_load: float=50.0, noise: Literal['ase-only','thermal-only','shot-only','ase-thermal','ase-shot','thermal-shot','all']='all') -> electrical_signal:
    """
    ### Descripción:
    Photodetector. Simula la detección de una señal óptica por un fotodetector.
    
    ---

    ### Args:
    - `input` - señal óptica a detectar
    - `BW` - ancho de banda del detector en [Hz]
    - `R` - Responsividad del detector en [A/W] (default: `R=1.0`)
    - `T` - Temperatura del detector en [K] (default: `T=300.0`)
    - `R_load` - Resistencia de carga del detector en [Ohm] (default: `R_load=50.0`)
    
    ---
    
    ### Returns:
    - `electrical_signal`
    """
    tic()
    if BW is None:
        BW = global_vars.BW_elec

    i_sig = R * np.sum(input.abs('signal')**2, axis=0) # se suman las dos polarizaciones

    if 'thermal' in noise or 'all' in noise:
        S_T = 4 * kB * T * BW / R_load # Density of thermal noise in [A^2]
        i_T = np.random.normal(0, S_T**0.5, input.len())
    
    if 'shot' in noise or 'all' in noise:
        S_N = 2 * e * i_sig * BW # Density of shot noise in [A^2]
        i_N = np.vectorize(lambda s: np.random.normal(0,s))(S_N**0.5)

    if 'ase' in noise or 'all' in noise:
        i_sig_sp = R * np.abs(input.signal[0]*input.noise[0].conjugate() + \
                            input.signal[0].conjugate()*input.noise[0] + \
                            input.signal[1]*input.noise[1].conjugate() + \
                            input.signal[1].conjugate()*input.noise[1])
        i_sp_sp = R * np.sum(input.abs('noise')**2, axis=0) # se suman las dos polarizaciones

    if noise == 'ase-only':
        noise = i_sig_sp  + i_sp_sp
    elif noise == 'thermal-only':
        noise = i_T 
    elif noise == 'shot-only':
        noise = i_N
    elif noise == 'ase-shot':
        noise = i_sig_sp  + i_sp_sp + i_N 
    elif noise == 'ase-thermal':
        noise = i_sig_sp  + i_sp_sp + i_T 
    elif noise == 'thermal-shot':
        noise = i_T + i_N 
    elif noise == 'all':
        noise = i_sig_sp  + i_sp_sp + i_N + i_T 
    else:
        raise ValueError(f"El argumento `noise` debe ser uno de los siguientes: 'ase-only','thermal-only','shot-only','ase-thermal','ase-shot','thermal-shot','all'.")
    
    time = toc()
    filt = LPF(i_sig, BW, n=4)
    output = electrical_signal(filt.signal, noise)

    output.ejecution_time = filt.ejecution_time + time
    return output



def ADC(input: electrical_signal, fs: float=None, BW: float=None, nbits: int=8) -> binary_sequence:
    """
    ### Descripción:
    Conversor analógico a digital. Convierte una señal eléctrica analógica en una señal digital de amplitud cuantizada, muestreada a una frecuencia `fs`
    y filtrada a un ancho de banda BW.

    ---

    ### Args:
    - `input` - señal eléctrica a cuantizar
    - `fs` [Opcional] - frecuencia de muestreo de la señal de salida (default: `global_vars.fs`)
    - `BW` [Opcional] - ancho de banda del ADC en [Hz] (default: `inf` no se filtra la señal)
    - `nbits` [Opcional] - cantidad de bits del ADC (default: `nbits=8`)

    ---
    
    ### Returns:
    - `electrical_signal`
    """
    tic()

    if not isinstance(input, electrical_signal):
        raise TypeError("`input` debe ser del tipo (electrical_signal).")
    
    if not fs:
        fs = global_vars.fs

    if BW:
        filt = sg.bessel(N = 4, Wn = BW, btype = 'low', fs=input.fs(), output='sos', norm='mag')
        signal = sg.sosfiltfilt(filt, input.signal)
    else: 
        signal = input.signal

    signal = sg.resample(signal, int(input.len()*fs/input.fs()))

    V_min = signal.min()
    V_max = signal.max()
    dig_signal = np.round( (signal - V_min) / (V_max - V_min) * (2**nbits - 1) ) # normalizo la señal entre 0 y 2**nbits-1
    dig_signal = dig_signal / (2**nbits - 1) * (V_max - V_min) + V_min # vuelvo a la amplitud original

    if np.sum(input.noise):
        noise = sg.sosfiltfilt(filt, input.noise)
        noise = sg.resample(noise, int(input.len()*fs//input.fs()))
        V_min = noise.min()
        V_max = noise.max()
        dig_noise = np.round( (noise - V_min) / (V_max - V_min) * (2**nbits - 1) ) # normalizo la señal entre 0 y 2**nbits-1
        dig_noise = dig_noise / (2**nbits - 1) * (V_max - V_min) + V_min # vuelvo a la amplitud original

        output = electrical_signal( dig_signal, dig_noise )
    else:
        output = electrical_signal( dig_signal )

    output.ejecution_time = toc()
    return output


def GET_EYE(input: Union[electrical_signal, optical_signal], nslots: int=4096, sps_resamplig: int = None):
    """
    ### Descripción:
    Estima todos los parámetros fundamentales y métricas del diagrama de ojo de la señal eléctrica de entrada.
    
    ---

    ### Args:
    - `input` - señal eléctrica a partir de la cual se estimará el diagrama de ojos
    - 'nslots' [Opcional] - cantidad de slots a considerar para la estimación de los parámetros (default: `nslots=4096`)
    - 'sps_resamplig' [Opcional] - cantidad de muestras por slot a las que se desea resamplear la señal a analizar (default: `sps_resamplig=256`)
    
    ---
    
    ### Returns:
    - `eye` - objeto de la clase `Eye` con todos los parámetros y métricas del diagrama de ojo
    """
    tic()


    def shorth_int(data: np.ndarray) -> tuple[float, float]:
        """
        ### Descripción:
        Estimación del intervalo más corto que contiene el 50% de las muestras de 'data'.
        
        ---
        
        ### Args:
        - `data` - array de datos

        ---

        ### Returns:
        - `tuple[float, float]` - intervalo más corto que contiene el 50% de las muestras de 'data'
        """
        diff_lag = lambda data,lag: data[lag:]-data[:-lag]  # Diferencia entre dos elemento de un array separados una distancia 'lag'
        
        data = np.sort(data)
        lag = len(data)//2
        diff = diff_lag(data,lag)
        i = np.where(np.abs(diff - np.min(diff))<1e-10)[0]
        if len(i)>1:
            i = int(np.mean(i))
        return (data[i], data[i+lag])
    
    def find_nearest(levels: np.ndarray, data: Union[np.ndarray, float]) -> Union[np.ndarray, float]: 
        """
        ### Descripción:
        Encuentra el elemento de 'levels' más cercano a cada valor de 'data'.
        
        ---

        ### Args:
        - `levels` - niveles de referencia.
        - `data` - valores a comparar.

        ### Returns:
        - `Union[ndarray, float]` - vector o float con los valores de 'levels' correspondientes a cada valor de 'data'
        """

        if type(data) == float or type(data) == np.float64:
            return levels[np.argmin(np.abs(levels - data))]
        else:
            return levels[np.argmin( np.abs( np.repeat([levels],len(data),axis=0) - np.reshape(data,(-1,1)) ),axis=1 )]

    eye_dict = {}

    sps = input.sps(); eye_dict['sps'] = sps
    dt = input.dt(); eye_dict['dt'] = dt

    
    n = input[sps:].len()%(2*input.sps())
    if n: input = input[sps:-n]
    nslots = min(input.len()//sps, nslots)
    input = input[:nslots*sps]

    if isinstance(input, optical_signal):
        s = input.abs()
        input = (s[0]**2 + s[1]**2)**0.5
    elif isinstance(input, electrical_signal):
        input = (input.signal+input.noise).real
    else:
        raise TypeError("El argumento 'input' debe ser de la clase 'optical_signal' o 'electrical_signal'.")

    input = np.roll(input, -sps//2+1) # Para centrar el ojo en el gráfico
    y_set = np.unique(input)

    # realizamos un resampling de la señal para obtener una mayor resolución en ambos ejes
    if sps_resamplig is not None:
        input = sg.resample(input, nslots*sps_resamplig); eye_dict['y'] = input
        t = np.kron(np.ones(nslots//2), np.linspace(-1, 1-dt, 2*sps_resamplig)); eye_dict['t'] = t
    else:
        eye_dict['y'] = input
        t = np.kron(np.ones((len(input)//sps)//2), np.linspace(-1,1,2*sps, endpoint=False)); eye_dict['t'] = t

    # Obtenemos el centroide de las muestras en el eje Y
    vm = np.mean(sk.KMeans(n_clusters=2, n_init=10).fit(input.reshape(-1,1)).cluster_centers_)

    # obtenemos el intervalo más corto de la mitad superior que contiene al 50% de las muestras
    top_int = shorth_int(input[input>vm]) 
    # Obtenemos el LMS del nivel 1
    state_1 = np.mean(top_int)
    # obtenemos el intervalo más corto de la mitad inferior que contiene al 50% de las muestras
    bot_int = shorth_int(input[input<vm])
    # Obtenemos el LMS del nivel 0
    state_0 = np.mean(bot_int)

    # Obtenemos la amplitud entre los dos niveles 0 y 1
    d01 = state_1 - state_0

    # Tomamos el 75% de nivel de umbral
    v75 = state_1 - 0.25*d01

    # Tomamos el 25% de nivel de umbral
    v25 = state_0 + 0.25*d01

    t_set = np.array(list(set(t)))

    # El siguiente vector se utilizará solo para determinar los tiempos de cruce
    tt = t[(input>v25)&(input<v75)]

    # Obtenemos el centroide de los datos de tiempo
    tm = np.mean(sk.KMeans(n_clusters=2, n_init=10).fit(tt.reshape(-1,1)).cluster_centers_)

    # Obtenemos el tiempo de cruce por la izquierda
    t_left = find_nearest(t_set, np.mean(tt[tt<tm])); eye_dict['t_left'] = t_left

    # Obtenemos el tiempo de cruce por la derecha
    t_right = find_nearest(t_set, np.mean(tt[tt>tm])); eye_dict['t_right'] = t_right

    # Determinamos el centro del ojo
    t_center = find_nearest(t_set, (t_left + t_right)/2); eye_dict['t_opt'] = t_center

    # Para el 20% del centro del diagrama de ojo
    t_dist = t_right - t_left; eye_dict['t_dist'] = t_dist
    t_span0 = t_center - 0.05*t_dist; eye_dict['t_span0'] = t_span0
    t_span1 = t_center + 0.05*t_dist; eye_dict['t_span1'] = t_span1

    # Dentro del 20% de los datos del centro del diagrama de ojo, separamos en dos clusters superior e inferior
    y_center = find_nearest(y_set, (state_0 + state_1)/2)

    # Obtenemos el instante óptimo para realizar el down sampling
    instant = np.abs(t-t_center).argmin() - sps//2 + 1; eye_dict['i'] = instant

    # Obtenemos el cluster superior
    y_top = input[(input > y_center) & ((t_span0 < t) & (t < t_span1))]; eye_dict['y_top'] = y_top

    # Obtenemos el cluster inferior
    y_bot = input[(input < y_center) & ((t_span0 < t) & (t < t_span1))]; eye_dict['y_bot'] = y_bot

    # Para cada cluster calculamos las medias y desviaciones estándar
    mu1 = np.mean(y_top); eye_dict['mu1'] = mu1
    s1 = np.std(y_top); eye_dict['s1'] = s1
    mu0 = np.mean(y_bot); eye_dict['mu0'] = mu0
    s0 = np.std(y_bot); eye_dict['s0'] = s0

    # Obtenemos la relación de extinción
    er = 10*np.log10(mu1/mu0) if mu0>0 else np.nan; eye_dict['er'] = er

    # Obtenemos la apertura del ojo
    eye_h = mu1 - 3 * s1 - mu0 - 3 * s0; eye_dict['eye_h'] = eye_h

    eye_dict['ejecution_time'] = toc()
    return eye(eye_dict)


def SAMPLER(input: electrical_signal, _eye_: eye) -> electrical_signal:
    """
    ### Descripción:
    Recibe una señal de tipo `electrical_signal` y un objeto de tipo `eye` y realiza el muestreo de la señal 
    en el instante óptimo determinado por el objeto `eye`.
    
    ---

    ### Args:
    - `input` - señal eléctrica a muestrear
    - `eye`  - objeto de tipo `eye` que contiene la información del diagrama de ojo

    ### Returns:
    - `electrical_signal` - señal electrica muestreada a una muestra por slot
    """
    tic()
    output = input[_eye_.i::_eye_.sps]

    output.ejecution_time = toc()
    return output






### algunas funciones de prueba
def animated_fiber_propagation(input: optical_signal, M :int, length_: float, alpha_: float=0.0, beta_2_: float=0.0, beta_3_: float=0.0, gamma_: float=0.0, phi_max:float=0.05):
    from matplotlib.animation import FuncAnimation

    # cambio las unidades
    length = length_ * 1e3 
    alpha  = alpha_ * 1/(4.343*1e3)
    beta_2 = beta_2_ * 1e-12**2/1e3
    beta_3 = beta_3_ * 1e-12**3/1e3
    gamma  = gamma_ * 1/1e3

    w = input.w()
    D_op = -alpha/2 - 1j/2 * beta_2 * w**2 - 1j/6 * beta_3 * w**3

    A = input.signal[0]

    h = length if (beta_2==0 and beta_3==0) or gamma==0 else phi_max / (gamma * np.abs(A)**2).max()

    x_length = h
    A_z = [A]
    hs = [0]

    while True:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A)
        hs.append(h)

        h = phi_max / (gamma * np.abs(A)**2).max() if gamma != 0 else length 

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length
    
    if h > 0:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A)
        hs.append(h)

    t = input.t()*global_vars.slot_rate

    fig, ax = plt.subplots()

    line, = ax.plot(t, np.abs(A_z[0]), lw=2, color = 'red', ls='--')
    line, = ax.plot([], [], lw=2, color = 'k')

    plt.suptitle(r'Fiber: $\alpha = {:.2f}$ dB/km, $\beta_2 = {}$ ps^2/km, $\gamma = {}$ (W·km)^-1'.format(alpha_, beta_2_, gamma_))
    ax.set_xlabel(r'$t/T_{slot}$')
    ax.set_ylabel('|A(z,t)|')
    ax.set_xlim((0, t.max()))
    ax.set_ylim((abs(A_z[0]).min()*0.95, np.abs(A_z).max()*1.05))

    time_text = ax.text(0.05,0.9, '', transform = ax.transAxes)


    def init():
        line.set_data([],[])
        time_text.set_text('z = 0.0 Km')
        for i in t[::M*global_vars.sps]:
           plt.axvline(i, color='k', ls='--')
        for i in t[::global_vars.sps]:
           plt.axvline(i, color='k', ls='--', alpha=0.3,lw=1)
        return [line, time_text] 

    def animate(i):
        y = np.abs(A_z[i])
        line.set_data(t, y)
        time_text.set_text('z = {:.2f} Km'.format(np.cumsum(hs)[i]/1e3))
        return [line, time_text] 

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(A_z), interval=100, blit=True, repeat=False)
    plt.show()


def animated_fiber_propagation_with_psd(input: optical_signal, M :int, length_: float, alpha_: float=0.0, beta_2_: float=0.0, beta_3_: float=0.0, gamma_: float=0.0, phi_max:float=0.05, n:int=None):
    from matplotlib.animation import FuncAnimation

    n = input.len() if n is None else n*M*input.sps() 
    
    length = length_
    alpha  = alpha_/4.343
    beta_2 = beta_2_
    beta_3 = beta_3_
    gamma  = gamma_

    w = input.w() * 1e-12 # rad/ps
    D_op = -alpha/2 - 1j/2 * beta_2 * w**2 - 1j/6 * beta_3 * w**3

    A = input.signal[0]

    h = length if (beta_2==0 and beta_3==0) or gamma==0 else phi_max / (gamma * np.abs(A)**2).max()

    x_length = h
    A_z = [A]
    A_z_w = [fft(A)]
    hs = [0]

    while True:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A*np.exp(alpha*x_length/2))
        A_z_w.append(fft(A*np.exp(alpha*x_length/2)))
        hs.append(h)

        h = phi_max / (gamma * np.abs(A)**2).max() if gamma != 0 else length 

        if x_length + h > length:
            break

        x_length += h

    h = length - x_length
    
    if h > 0:
        exp_NL = np.exp(1j * gamma * (h/2) * np.abs(A)**2)
        exp_L = np.exp(D_op * h)
        A = exp_NL * ifft( exp_L * fft( exp_NL * A ) )
        A_z.append(A*np.exp(alpha*length/2))
        A_z_w.append(fft(A*np.exp(alpha*length/2)))
        hs.append(h)

    t = input.t()*global_vars.slot_rate

    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(6,6))

    line1, = ax1.plot(t[:n], np.abs(A_z[0])[:n], lw=2, color = 'red', ls='--')
    line1, = ax1.plot([], [], lw=2, color = 'k')

    plt.suptitle(r'Fiber: $\alpha = {:.2f}$ dB/km, $\beta_2 = {}$ ps^2/km, $\gamma = {}$ (W·km)^-1'.format(alpha_, beta_2_, gamma_))
    ax1.set_xlabel('t/T')
    ax1.set_ylabel('|A(z,t)|')
    ax1.set_xlim((0, t[:n].max()))
    ax1.set_ylim((0, np.abs(A_z[:n]).max()))

    z_text = ax2.text(0.05,0.9, '', transform = ax2.transAxes)

    f = fftshift(w/2/np.pi)*1e3 # GHz

    line2, = ax2.plot(f, fftshift(np.abs(A_z_w[0])**2)/input.len(), '--g', lw=2)
    line2, = ax2.plot([], [], 'k', lw=2)

    ax2.set_xlabel('f [GHz]')
    ax2.set_ylabel(r'$|A(z,w)|^2$')
    sigma = -f[np.cumsum(np.abs(fftshift(A_z_w[0]))**2)<0.001*np.sum(np.abs(A_z_w[0])**2)][-1]
    ax2.set_xlim((-2*sigma, 2*sigma))
    ax2.set_ylim((0, np.abs(A_z_w).max()**2*1.05/input.len()))
    ax2.grid()

    plt.tight_layout()


    def init():
        line1.set_data([],[])
        z_text.set_text('z = 0.0 Km')
        for i in t[:n:M*global_vars.sps]:
            ax1.axvline(i, color='k', ls='--')
        for i in t[:n:global_vars.sps]:
            ax1.axvline(i, color='k', ls='--', alpha=0.3,lw=1)
        return [line1, z_text] 

    def animate(i):
        y = np.abs(A_z[i])
        line1.set_data(t[:n], y[:n])
        z_text.set_text('z = {:.2f} Km'.format(np.cumsum(hs)[i]))

        y = fftshift(np.abs(A_z_w[i])**2)/input.len()
        line2.set_data(f, y)
        return [line1, line2, z_text] 

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(A_z), interval=100, blit=True, repeat=False)
    plt.show()



if __name__ == '__main__':
    # def Spectrogram(signal, T_L):
    #     N = signal.len()//T_L
    #     y = np.zeros(N*T_L)
    #     for i in range(N):
    #         y[i*T_L:(i+1)*T_L] = fftshift(np.abs( fft(signal.signal[i*T_L:(i+1)*T_L]) )**2)
    #     y = y.reshape((N, T_L)).T
    #     return y


    # sps = 2048
    # global_vars(M=8, sps=sps, R=10e9)
    # y = DAC('0 1 0 1 0 0 1', pulse_shape='gaussian', T=sps, m=1, c=-50)
    # N_TL = 128
    # # g = Spectrogram(y, N_TL)
    # # f = y.w()/(2*np.pi)
    # # plt.imshow(g, cmap='hot', aspect='auto', interpolation='blackman', extent=[y.t().min()*1e9, y.t().max()*1e9, f.min()*1e-9, f.max()*1e-9])
    # # plt.ylim(-1000,1000)
    # # y.plot(ylabel='Voltaje (V)',c='red').grid()
    # plt.figure()
    # plt.plot(y.t(), y.abs('signal'))
    # # for i in range(y.len()//N_TL):
    # #     plt.axvline(y.t()[i*N_TL], ls='--')
    # plt.show()
    pass



