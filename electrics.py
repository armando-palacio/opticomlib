import os, sys; sys.path.append(os.path.dirname(__file__)+'\..') # Agrego el directorio anterior al path 

from opticomlib._types_ import *



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



def PPM_ENCODER(input: Union[str, list, tuple, np.ndarray, binary_sequence], M: int=None) -> binary_sequence:
    """
    ### Descripción:
    Codificador digital PPM. Convierte una secuencia binaria de entrada en una secuencia binaria codificada en PPM. 

    ---
    
    ### Args:
    - `input` - secuencia binaria de entrada
    - `M` [Opcional] - cantidad de slots que contiene un símbolo (default: `M=global_vars.M`)

    ---
    
    ### Returns:
    - `binary_sequence`
    """
    tic()

    if isinstance(input, binary_sequence):
        input = input.data
    elif isinstance(input, str):
        input = str2ndarray(input)
    elif isinstance(input, (list, tuple)):
        input = np.array(input)
    else:
        raise TypeError("El argumento `input` debe ser del tipo (str, list, tuple, ndarray, binary_sequence).")
    
    if not M:
        M = global_vars.M
    
    k = int(np.log2(M))

    input = input[:input.len()//k*k] # truncamos la secuencia de bits a un múltiplo de k

    decimal = np.sum(input.reshape(-1,k)*2**np.arange(k)[::-1], axis=-1) # convertimos los símbolos a decimal
    ppm_s = np.zeros(decimal.size*M, dtype=np.uint8)

    ppm_s[np.arange(decimal.size)*M + decimal] = 1 # codificamos los símbolos en PPM
   
    output = binary_sequence(ppm_s) 
    output.ejecution_time = toc()
    return output



def PPM_DECODER(input: Union[str, list, tuple, np.ndarray, binary_sequence], M: int=None) -> binary_sequence:
    """
    ### Descripción:
    Recibe una secuencia de bits codificada en PPM y la decodifica.
    
    ---

    ### Args:
    - `input` - secuencia binaria codificada en PPM
    - `M` [Opcional] - orden de modulación PPM

    ### Returns:
    - `binary_sequence` - secuencia binaria decodificada
    """
    tic()

    if isinstance(input, binary_sequence):
        input = input.data
    elif isinstance(input, str):
        input = str2ndarray(input)
    elif isinstance(input, (list, tuple)):
        input = np.array(input)
    else:
        raise TypeError("El argumento `input` debe ser del tipo (str, list, tuple, ndarray, binary_sequence).")

    if not M:
        M = global_vars.M
    
    k = int(np.log2(M))

    decimal = np.where(input==1)[0]%M # obtenemos el decimal de cada símbolo

    output = np.array(list(map(lambda x: dec2bin(x,k), decimal))).ravel() # convertimos a binario cada decimal
    output= binary_sequence(output)

    output.ejecution_time = toc()
    return output



def DAC(seq: Union[str, list, tuple, np.ndarray, binary_sequence], sps: int=None, type: Literal['rect','gaussian']='rect', c: float=0.0, m: int=1) -> electrical_signal:  
    """
    ### Descripción:
    Conversor digital a analógico. Convierte una secuencia binaria en una señal eléctrica, muestreada a una frecuencia `fs`.

    ---

    ### Args:
    - `seq` - secuencia binaria de entrada
    - `sps` [Opcional] - cantidad de muestras por slot (default: `sps=global_vars.sps`)
    - `type` [Opcional] - forma de pulso a la salida, puede ser "rect" o "gaussian" (default: `type="rect"`)
    - `c` [Opcional] - chirp del pulso. Solo para el caso de pulsos gausianos (default: `c=0.0`)
    - `m` [Opcional] - orden del pulso supergausiano (default: `m=1`)
    
    ---

    ### Returns:
    - `electrical_signal`
    """
    tic()
    if not isinstance(seq, binary_sequence):
        seq = binary_sequence(seq)
    if not sps:
        sps = global_vars.sps

    if type == 'rect':
        x = np.kron(seq.data, np.ones(sps))
        output = electrical_signal( x )
    
    elif type == 'gaussian':
        p = lambda t, T0: np.exp(-(1+1j*c)/2 * (t/T0)**(2*m))

        M = global_vars.M
        T = 1/global_vars.slot_rate # tiempo de slot
        T0 = T/2.35482 # desviación estándar del pulso gaussiano (considerando T como el ancho a la mitad de la potencia)

        tp = np.linspace(-M/2*T, M/2*T, M*sps) # vector de tiempo del pulso gaussiano
        pulse = p(tp, T0) # pulso gaussiano

        s = np.zeros(seq.len()*sps)
        s[int(sps//2)::sps]=seq.data
        s[int(sps//2-1)::sps]=seq.data

        output = electrical_signal( sg.fftconvolve(s, pulse, mode='same')/2 )
    
    else:
        raise NameError('El parámetro `type` debe ser uno de los siguientes valores ("rect","gaussian").')

    output.ejecution_time = toc()
    return output



def LPF(input: Union[np.ndarray, electrical_signal], BW: float, n: int=4, fs: float=None) -> electrical_signal:
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
            if not isinstance(input, np.ndarray):
                raise TypeError("`input` debe ser del tipo (ndarray ó electrical_signal).")
            else:
                input = electrical_signal(input)
        if not fs:
            fs = global_vars.fs

        sos_band = sg.bessel(N = n, Wn = BW, btype = 'low', fs=fs, output='sos', norm='mag')

        output = electrical_signal( np.zeros((2,input.len())) )

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



def GET_EYE(input: Union[electrical_signal, optical_signal], M: int=None, nslots: int=4096, sps_resamplig: int = None):
    """
    ### Descripción:
    Estima todos los parámetros fundamentales y métricas del diagrama de ojo de la señal eléctrica de entrada.
    
    ---

    ### Args:
    - `input` - señal eléctrica a partir de la cual se estimará el diagrama de ojos
    - `M` [Opcional] - orden de modulación PPM (default: `M=global_vars.M`)
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
    
    import sklearn.cluster as sk

    eye_dict = {}

    M = M if M is not None else global_vars.M; eye_dict['M'] = M
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

    # obtenemos el umbral de decisión para PPM
    r = np.linspace(mu0, mu1, 1000)
    umbral = r[np.argmin(1 - Q((r-mu1)/s1) * (1-Q((r-mu0)/s0))**(M-1))]; eye_dict['umbral'] = umbral

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



def THRESHOLD(input: electrical_signal, _eye_: eye) -> binary_sequence:
    """
    ### Descripción:
    Realiza la comparación con un umbral de decisión de la señal eléctrica, muestreando a una mustra por slot.
    
    ---

    ### Args:
    - `input` - señal eléctrica a muestrear
    - `eye`  - objeto de tipo `eye` que contiene la información del diagrama de ojo

    ### Returns:
    - `binary_sequence` - secuencia binaria obtenida
    """
    tic()
    output = SAMPLER(input, _eye_) > _eye_.umbral
    output.ejecution_time = toc()
    return output



def HARD_DECISION(input: binary_sequence, M: int=None) -> binary_sequence:
    """
    ### Descripción:
    Estima los símbolos PPM más probables a partir de la secuencia binaria dada como entrada.
    
    ---

    ### Args:
    - `input` - secuencia binaria a estimar

    ### Returns:
    - `binary_sequence` - secuencia de símbolos estimados
    """
    tic()

    if not M:
        M = global_vars.M
    assert M < 256, 'El máximo orden de modulación PPM es 256!!'

    n_simb = int(input.len()/M)

    s = np.sum(input.data.reshape(n_simb, M), axis=-1)

    output = np.array(input.data, dtype=np.uint8)

    for i in np.where(s==0)[0]: # si existe algún símbolo sin ningún slot encendidos, se prende uno al azar
        output[i*M + np.random.randint(M)] = 1

    for i in np.where(s>1)[0]: # si existe algún símbolo con más de 1 slot encendido, se elige uno de ellos al azar)
        j = np.where(output[i*M:(i+1)*M]==1)[0]
        output[i*M:(i+1)*M] = 0
        output[i*M + np.random.choice(j)]=1

    output = binary_sequence(output)
    output.ejecution_time = toc()
    return output



def SOFT_DECISION(input: electrical_signal, M: int=None) -> binary_sequence:
    """
    ### Descripción:
    Estima los símbolos PPM más probables a partir de la señal eléctrica dada como entrada.
    
    ---

    ### Args:
    - `input` - señal eléctrica sin muestrear

    ### Returns:
    - `binary_sequence` - secuencia de símbolos estimados
    """
    tic()

    if not M:
        M = global_vars.M
    assert M < 256, 'El máximo orden de modulación PPM es 256!!'

 
    signal = np.sum( (input.signal + input.noise).reshape(-1, input.sps()), axis=-1)

    i = np.argmax( signal.reshape(-1, M), axis=-1)

    output = np.zeros_like(signal, dtype=np.uint8)
    output[np.arange(i.shape[0])*M+i] = 1

    output = binary_sequence(output)
    output.ejecution_time = toc()
    return output



def DSP(input: electrical_signal, _eye_: eye=eye(), decision: Literal['hard','soft']='hard') -> binary_sequence:
    """
    ### Descripción:
    Este componente realiza todas las tareas de decisión y decodificación de la señal digital. 
    
    Si se selecciona la decisión dura, se realiza el submuestreo de la señal digital a 1 muestra por slot en el instante óptimo 
    determinado por el objeto `eye`, luego se realiza la comparación con un umbral de decisión estimado por `eye`, se decide que 
    símbolo PPM se transmitió y se decodifica la secuencia binaria.

    Si se selecciona la decisión blanda, se realiza la integración de cada slot de la señal digital, se decide que símbolo PPM tiene
    mayor verosimilitud y se decodifica la secuencia binaria.
    
    ---

    ### Args:
    - `input` - secuencia binaria codificada en PPM
    - `_eye_` [Opcional] - objeto eye con los parámetros del diagrama de ojos (default: `_eye_=eye()`)
    - `decision` [Opcional] - tipo de decisión a realizar (default: `decision='hard'`)

    ### Returns:
    - `binary_sequence` - secuencia binaria decodificada
    """
    
    if decision == 'hard':
        output = THRESHOLD(input, _eye_)
        simbols = HARD_DECISION(output); simbols.ejecution_time += output.ejecution_time
    elif decision == 'soft':
        simbols = SOFT_DECISION(input)
    else:
        raise TypeError('No existe el tipo de decisión seleccionada!!')

    output = PPM_DECODER(simbols) 

    output.ejecution_time += simbols.ejecution_time 
    return output



def BER_COUNTER(Tx: binary_sequence, Rx: binary_sequence) -> float:
    """
    ### Descripción:
    Calcula la tasa de error de bits (BER) entre dos secuencias binarias.
    
    ---

    ### Args:
    - `Tx` - secuencia binaria transmitida
    - `Rx` - secuencia binaria recibida

    ### Returns:
    - `float` - BER entre las secuencias dadas
    """
    Tx = Tx[:Rx.len()]
    assert Tx.len() == Rx.len(), 'Las secuencias deben tener la misma longitud!!'
    return np.sum(Tx.data != Rx.data)/Tx.len()



def BER_FROM_EYE(_eye_: eye, decision: Literal['hard','soft']='hard') -> float:
    """
    ### Descripción:
    Calcula la tasa de error de bits (BER) a partir de los parámetros estimados del diagrama de ojos
    
    ---

    ### Args:
    - `Tx` - secuencia binaria transmitida
    - `Rx` - secuencia binaria recibida

    ### Returns:
    - `float` - BER entre las secuencias dadas
    """
    M = _eye_.M
    I1 = _eye_.mu1
    I0 = _eye_.mu0
    s1 = _eye_.s1
    s0 = _eye_.s0
    um = _eye_.umbral

    if decision == 'hard':
        Pe_sym = 1 - Q((um-I1)/s1) * (1-Q((um-I0)/s0))**(M-1)
    elif decision == 'soft':
        Pe_sym = 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((I1-I0+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0]
    else:
        raise TypeError('decision debe ser "hard" o "soft"!!') 
    return M/2/(M-1)*Pe_sym