"""  
Este módulo está orientado al trabajo en el laboratorio.  
"""


import os, sys; sys.path.append(os.path.dirname(__file__)+'\..') # Agrego el directorio anterior al path 

from opticomlib._types_ import *
from opticomlib.electrics import PPM_ENCODER, PRBS, DSP, GET_EYE, BER_COUNTER, BER_FROM_EYE
import pyvisa as visa
from time import sleep


def SEARCH_INSTRUMENTS() -> list[str]:
    """
    ### Descripción:
    Busca los instrumentos conectados al PC y devuelve una lista con sus nombres.   

    ---

    ### Returns:
    - `resources` - lista con los nombres de los instrumentos conectados al PC.
    """

    return visa.ResourceManager().list_resources()



def CONNECT_INSTRUMENT(resource_addr: str) -> visa.resources.Resource:
    """
    ### Descripción:
    Conecta un instrumento al PC y devuelve un objeto 'pyvisa.resources.Resource' para la comunicación con el instrumento.   

    ---

    ### Args:
    - `resource_addr` - dirección del instrumento a conectar.

    ### Returns:
    - `inst` - objeto 'pyvisa.resources.Resource' para la comunicación con el instrumento.
    """

    inst = visa.ResourceManager().open_resource(resource_addr)
    print(inst.query('*IDN?'))

    return inst



def GET_WAVEFORM_FROM_MDO3054(inst: visa.resources.Resource, ch: int, points: int=10**5, save: bool=False, filename: str=None) -> np.ndarray:
    """
    ### Descripción:
    Obtiene la forma de onda del instrumento Tektronik MDO3054 y la guarda en un archivo de texto si `save=True`.
    
    ---

    ### Args:
    - `inst` - Objeto 'pyvisa.resources.Resource' para la comunicación con el instrumento
    - `ch` - canal a leer
    - `points` [Opcional] - número de puntos de la forma de onda [1k, 10k, 100k, 1M, 10M] (por defecto 100k)
    - `save` [Opcional] - si es `True` guarda la forma de onda en un archivo de texto (por defecto `False`)
    - `filename` [Opcional] - nombre del archivo de texto donde se guarda la forma de onda (si `save=True`, por defecto 'waveform.txt')

    ### Returns:
    - `data_rx` - forma de onda en [mV]
    """

    inst.write('HEADer 0')
    inst.write(f'HOR:RECO {int(points)}')
    inst.write(f'DAT:SOURce CH{ch}')
    inst.write('DAT:Start 1')
    inst.write(f'DAT:STOP {int(points)}')
    inst.write('WFMOutpre:ENCdg ASCIi')
    inst.write('WFMOutpre:BYT_Nr 1') # 1 byte por muestra

    y_offset = float(inst.query('WFMP:YOF?').replace('\n',''))
    y_scale = float(inst.query('WFMP:YMU?').replace('\n',''))

    inst.write('ACQuire:MODe SAMple')
    inst.write('ACQUIRE:STOPAFTER SEQUENCE'); sleep(1)
    inst.write('ACQUIRE:STATE RUN')

    y = inst.query_ascii_values('CURV?', container=np.array)
    data_rx = y_scale*(y - y_offset)*1000 # mV

    # obtener el tiempo de la forma de onda
    x_incr = float(inst.query('WFMP:XIN?').replace('\n',''))
    t = np.arange(0, x_incr*points, x_incr)*1e9 # ns

    inst.write('ACQUIRE:STOPAFTER RUNSTOP')
    inst.write('ACQUIRE:STATE RUN')

    if save:
        if filename is None:
            filename = f'waveform-CH{ch}.txt'
        filename = '.'.join(filename.split('.')[:-1]) + f'-{y_scale*25*1000}mVdiv' + '.txt'
        np.savetxt(filename, np.column_stack((t, data_rx)), fmt='%.2e', header=f't [ns]\tdata_rx [mV] (escala{y_scale*25*1000} mV/div)')
    return t, data_rx



def DSP_tx(filename_bits_tx_in: str=None, nbits: int=2**8, save: bool=True, filename_bits_tx_out: str=None, filename_slots_tx_out: str=None, marker1: str=False, marker2: str=False) -> binary_sequence:
    """
    ### Descripción:
    Acondiciona la secuencia binaria a transmitir. Si `filename` es `None` genera una secuencia aleatoria de `nbits` bits, sino lee el archivo de texto `filename`.
    Dichos bits son codificados en PPM.
    
    ---

    ### Args:
    - `filename_bits_tx_in` [Opcional] - archivo (.txt) que contiene la secuencia binaria a transmitir (por defecto `None`)
    - `nbits` [Opcional] - si `filename==None` se genera una secuencia binaria aleatoria de longitud nbits (por defecto `2**8`)
    - `filename_bits_tx_out` [Opcional] - nombre del archivo (.txt) donde se almacenará la secuencia de bits (por defecto `None`). 
    - `filename_slots_tx_out` [Opcional] - nombre del archivo (.txt) donde se almacenará la secuencia de slots (por defecto `None`). Este archivo es utilizado
    solo para cargar la secuencia al generador de señales.
    - `marker1` [Opcional] - agrega un marker al inicio de la trama (por defecto `False`)
    - `marker2` [Opcional] - agrega un marker al inicio de cada símbolo de la trama (por defecto `False`)


    ### Returns:
    - `seq_ppm` - retorna la secuencia binaria codificada en PPM lista para cargar al generador
    """

    if filename_bits_tx_in:
        bits_tx = binary_sequence(np.loadtxt(filename_bits_tx_in, dtype=np.uint8))
    else:
        bits_tx = PRBS(nbits)

        if save:
            if filename_bits_tx_out is None:
                filename_bits_tx_out = 'bits_tx.txt'
            np.savetxt(filename_bits_tx_out, bits_tx.data, fmt='%d')

    seq_ppm = PPM_ENCODER( bits_tx ).data

    if save:
        out = seq_ppm
        if filename_slots_tx_out is None:
            filename_slots_tx_out = 'slots_tx.txt'
        if marker1:
            marker_1 = np.zeros_like(seq_ppm); marker_1[0]=1
            out = np.column_stack((out, marker_1))
        if marker2:
            marker_2 = np.zeros_like(seq_ppm); marker_2[::global_vars.M]=1
            out = np.column_stack((out, marker_2))

        np.savetxt(filename_slots_tx_out, out, fmt='%.2f', delimiter=',')

    return seq_ppm


def Digital_filter(signal: np.ndarray, BW: float, fs: float, orden: int=4)-> electrical_signal:
    sos_band = sg.bessel(N = orden, Wn = BW, btype = 'low', fs=fs, output='sos', norm='mag')
    return electrical_signal(sg.sosfiltfilt(sos_band, signal))


def DSP_rx(filename_in: str=None, inst: visa.resources.Resource=None, ch: int=1, points: int=10**4, save: bool=True, filename_out: str=None, ber_type: Literal['count, estimate']='count', filename_bits_tx: str=None) -> tuple[float, float]:
    """
    ### Descripción:
    Obtiene la forma de onda del instrumento Tektronik MDO3054 y la guarda en un archivo de texto si `save=True`. Luego realiza un resampling
    de la señal y la sincroniza con los datos transmitidos. Luego, obtiene los parámetros del ojo en el caso de la decisión dura, realiza la decisión y decodificación. 
    Finalmente, calcula la BER. 
    
    ---

    ### Args:
    - `filename_in` [Opcional] - nombre del archivo que contiene la señal digital a analizar (por defecto `None`)
    - `inst` [Opcional] - Objeto 'pyvisa.resources.Resource' para la comunicación con el instrumento (por defecto `None`). Solo se utiliza si `filename_in=None`
    - `ch` [Opcional] - canal a leer (por defecto `1`). Solo se utiliza si `filename_in=None`
    - `points` [Opcional] - número de puntos de la forma de onda [1k, 10k, 100k, 1M, 10M] (por defecto `10k`). Solo se utiliza si `filename_in=None`
    - `save` [Opcional] - si es `True` guarda la forma de onda en un archivo de texto (por defecto `True`). Solo se utiliza si `filename_in=None`
    - `filename_out` [Opcional] - nombre del archivo (.txt) donde se guarda la forma de onda (si `save=True`, por defecto 'waveform.txt'). Solo se utiliza si `filename_in=None`
    - `ber_type` [Opcional] - tipo de BER a calcular (por defecto 'count')
    - `filename_bits_tx` [Opcional] - nombre del archivo (.txt) que contiene la secuencia binaria transmitida (por defecto `None`). Solo se utiliza si `ber_type='count'`

    ### Returns:
    - `(BER_S, BER_H)` - retorna las probabilidades de error de bit para la decisión suave y dura respectivamente
    """

    if filename_in is None:
        assert inst is not None, 'Especifique el instrumento en el argumento "inst : visa.resources.Resource"'
        _, data_rx = GET_WAVEFORM_FROM_MDO3054(inst, ch, points, save, filename_out)
    else:
        _, data_rx = np.loadtxt(filename_in, dtype=np.float64, unpack=True)

    BW_elec, fs, sps_osc, k, M = global_vars.BW_elec, global_vars.fs, global_vars.sps, np.log2(global_vars.M), global_vars.M
    
    #--------------------------------------------------------------------------
    # filtramos la señal adquirida
    rx = Digital_filter(data_rx, BW=BW_elec, fs=fs, orden=4)

    #--------------------------------------------------------------------------
    # realizamos un resampling
    sps_osc = global_vars.sps
    sps_dsp = 64

    rx_ = sg.resample(rx.signal, int(rx.len()*sps_dsp//sps_osc))

    global_vars.sps = sps_dsp
    global_vars.update()

    rx_ = electrical_signal(rx_)
    # (rx_*1000).plot('.g',label='Re', n=n).grid(n)

    #--------------------------------------------------------------------------
    # sincronizamos los datos con los transmitidos
    assert filename_bits_tx is not None, 'Introduce el nombre del archivo que contiene los bits transmitidos en el argumento "filename_bits_tx : str"'
    bits_tx = binary_sequence(np.loadtxt(filename_bits_tx, dtype=np.uint8)); bits_tx=bits_tx[:int(bits_tx.len()//k*k)]

    sy,_ = SYNC(rx_, bits_tx)

    l = len(bits_tx) # cantidad de bits transmitidos
    m = int((l//k)*M) # cantidad de slots transmitidos
    n = int((sy.len()//sps_dsp)//m) # cantidad de veces que se repite la secuencia de slots transmitida en la señal recibida

    re = sy[:n*m*sps_dsp]

    #--------------------------------------------------------------------------
    # obtenemos los parámetros del ojo y graficamos
    GET_EYE(re[:4096*sps_dsp]).plot()
    _eye_ = GET_EYE_PARAMS_v2(re, bits_tx, nslots=m)

    if ber_type == 'count':
        bits_rx_H = DSP(re, _eye_, decision='hard')
        bits_rx_S = DSP(re, _eye_, decision='soft')

        bits_tx = binary_sequence(np.kron(np.ones(n), bits_tx.data))

        ber_H = BER_COUNTER(bits_tx, bits_rx_H)
        ber_S = BER_COUNTER(bits_tx, bits_rx_S)
    
    elif ber_type == 'estimate':
        ber_H = BER_FROM_EYE(_eye_, decision='hard')
        ber_S = BER_FROM_EYE(_eye_, decision='soft')

    
    global_vars.sps = sps_osc
    global_vars.update()
    print()
    print(ber_H, ber_S)

    return ber_S, ber_H, _eye_


def SYNC(signal_rx: electrical_signal, bits_tx: binary_sequence, sps_osc:int=None) -> tuple[electrical_signal, int]:
    """
    ### Descripción:
    Se realiza una sincronización de la señal recibida con la señal transmitida para saber a 
    partir de que posición de la señal recibida se debe comenzar a procesar. Para ello se realiza una correlación 
    entre la señal recibida y la señal transmitida y se busca el máximo de la correlación.
    
    ---

    ### Args:
    - `signal_rx` - señal digital recibida (del osciloscopio).
    - `bits_tx` - secuencia de bits transmitida.

    ### Returns:
    - `signal_sync` - señal digital sincronizada.
    - `i` - posición del vector 'signal' a partir de la cual se realiza la sincronización.  
    """

    M, sps = global_vars.M, signal_rx.sps()
    if not isinstance(sps, int):
        raise TypeError('Los sps deben ser un número entero para realizar la sincronización.')

    slots = PPM_ENCODER(bits_tx, M)
    signal_tx = np.kron(slots.data, np.ones(sps))
    signal_rx = signal_rx.signal

    if len(signal_rx)<len(signal_tx): raise BufferError('La longitud del vector recibido debe ser mayor al vector transmitido!!')

    l = len(signal_tx)
    corr = sg.fftconvolve(signal_rx[:2*l], signal_tx[l::-1], mode='valid') # Correlación de la señal transmitida con la señal recibida en una ventana de 2*l (suficiente para encontrar un máximo)

    if np.max(corr) < 3*np.std(corr): raise ValueError('No se encontró un máximo de correlación!!') # falso positivo
    
    i = np.argmax(corr)

    signal_sync = electrical_signal(signal_rx[i:-(l-i)])
    signal_sync.ejecution_time = toc()
    return signal_sync, i



def GET_EYE_PARAMS_v2(sync_signal: electrical_signal, bits_tx: binary_sequence, nslots:int=8192) -> tuple[float, float, float, float]:
    """
    ### Descripción:
    Esta función obtiene las medias y desviaciones estándar de los niveles 0 y 1 de la señal recibida. Utiliza los
    slots de la señal transmitida para determinar los instantes de tiempo en los que se encuentran los niveles 0 y 1.
    
    ---

    ### Args:
    - `sync_signal` - señal digital sincronizada en tiempo con la señal transmitida.
    - `bits_tx` - secuencia de bits transmitida.
    - `nslots` [Opcional] - cantidad de slots a utilizar para la estimación (default: 8192).

    ### Returns:
    - `eye` - clase eye con los parámetros de la estimación.  
    """
    sps = sync_signal.sps()
    M = global_vars.M

    eye_dict = {}

    eye_dict['M'] = M
    eye_dict['sps'] = sps


    rx = sync_signal[:nslots*sps].signal + sync_signal[:nslots*sps].noise; eye_dict['y'] = rx
    tx = np.kron(PPM_ENCODER(bits_tx, M).data[:nslots], np.ones(sps))

    unos = rx[tx==1]; eye_dict['unos']=unos
    zeros = rx[tx==0]; eye_dict['zeros']=zeros

    t0 = np.kron(np.ones(zeros.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False)); eye_dict['t0']=t0
    t1 = np.kron(np.ones(unos.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False)); eye_dict['t1']=t1

    eye_dict['i']=sps//2

    unos_ = unos[(t1<0.05) & (t1>-0.05)]
    zeros_ = zeros[(t0<0.05) & (t0>-0.05)]

    mu0 = np.mean(zeros_).real; eye_dict['mu0'] = mu0
    mu1 = np.mean(unos_).real; eye_dict['mu1'] = mu1

    s0 = np.std(zeros_).real; eye_dict['s0'] = s0
    s1 = np.std(unos_).real; eye_dict['s1'] = s1

    r = np.linspace(mu0,mu1,1000)
    umbral = r[np.argmin( 1 - Q((r-mu1)/s1) * (1-Q((r-mu0)/s0))**(M-1) )]; eye_dict['umbral'] = umbral

    return eye(eye_dict)