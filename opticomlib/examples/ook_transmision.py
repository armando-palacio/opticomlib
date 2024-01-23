from opticomlib.ook import (
    DSP,
    BER_analizer,
    theory_BER
)

from opticomlib.devices import (
    PRBS,
    DAC,
    MODULATOR,
    FIBER,
    PD,
    GET_EYE,
    global_vars
)

""" 
    En este ejemplo se simulará la transmisión sobre 50 km de fibra óptica SMF de una señal OOK de 10 Gbps con un PRBS de 2^7-1 bits.
    Se utilizará un DAC con una frecuencia de muestreo de 20 GSPS y una modulación NRZ.
"""

# Parámetros de la simulación
global_vars(sps=2, R=20e9)

# prbs
tx_seq = PRBS(order=7)

# DAC y pulse shaping
dig_signal = DAC(tx_seq, Vout=1, pulse_shape='rect')

# modulador
mod_signal = MODULATOR(dig_signal, p_laser=0)

# fibra óptica
fiber_signal = FIBER(mod_signal, length=50, alpha=0.2, beta_2=-20, gamma=1.5, show_progress=True)

# fotodetector
pd_signal = PD(fiber_signal, BW=global_vars.R*0.75, responsivity=1, noise='all')

# parámetros del diagrama de ojos
eye_ = GET_EYE(pd_signal)

# DSP
rx_seq = DSP(pd_signal, eye_)

# BER
ber = BER_analizer('counter', tx_seq, rx_seq)

# BER teórico
mu0 = 0
mu1 = fiber_signal.power('signal')
ber_theory = theory_BER()

print(f'BER obtenida: {ber:.2e}')
print(f'BER teórico:  {ber_theory:.2e}')
