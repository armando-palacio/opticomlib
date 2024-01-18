import os, sys; sys.path.append(os.path.dirname(__file__)+'\..') # Agrego el directorio anterior al path 

from opticomlib._types_ import *



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



def DISPERSIVE_MEDIUM(input: optical_signal, beta_2: float, length: float) -> optical_signal:
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
    from tqdm.auto import tqdm

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




### algunas funciones de prueba
def animated_fiber_propagation(input: optical_signal, length_: float, alpha_: float=0.0, beta_2_: float=0.0, beta_3_: float=0.0, gamma_: float=0.0, phi_max:float=0.05, n:int=None):
    from matplotlib.animation import FuncAnimation

    n = input.len() if n is None else n*global_vars.M*input.sps() 
    
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

    fig, (ax1,ax2) = subplots(2,1, figsize=(6,6))

    line1, = ax1.plot(t[:n], np.abs(A_z[0])[:n], lw=2, color = 'red', ls='--')
    line1, = ax1.plot([], [], lw=2, color = 'k')

    suptitle(r'Fiber: $\alpha = {:.2f}$ dB/km, $\beta_2 = {}$ ps^2/km, $\gamma = {}$ (W·km)^-1'.format(alpha_, beta_2_, gamma_))
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
        for i in t[:n:global_vars.M*global_vars.sps]:
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
    show()