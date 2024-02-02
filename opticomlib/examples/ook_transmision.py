from opticomlib.ook import (
    DSP,
    THRESHOLD_EST,
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
    gv
)

import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
from scipy.constants import k as kB, e

""" 
    En este ejemplo se simulará la transmisión sobre 50 km de fibra óptica SMF de una señal OOK de 10 Gbps con un PRBS de 2^7-1 bits.
    Se utilizará un DAC con una frecuencia de muestreo de 20 GSPS y una modulación NRZ.
"""

# Parámetros de la simulación
gv(sps=16, R=1e9)

# prbs
tx_seq = PRBS(order=15)

# DAC y pulse shaping
dig_signal = DAC(tx_seq, Vout=1, pulse_shape='rect')

# modulador
mod_signal = MODULATOR(dig_signal, p_laser=-21)

# fibra óptica
fiber_signal = FIBER(mod_signal, length=50, alpha=0.1, beta_2=20, gamma=2, show_progress=True)

# fotodetector
pd_signal = PD(fiber_signal, BW=gv.R*0.75, responsivity=1, noise='all')

# parámetros del diagrama de ojos
eye_ = GET_EYE(pd_signal, sps_resamplig=128)

# DSP
rx_seq = DSP(pd_signal, eye_)

# BER
ber = BER_analizer('counter', Tx=tx_seq, Rx=rx_seq)


# Cálculo teórico de la BER


responsivity = 1 # Responsividad del fotodetector
T = 300 # Temperatura equivalente del receptor
BW = gv.R*0.75 # Ancho de banda del receptor
R_load = 50 # Resistencia de carga del fotodetector

mu0 = 0
mu1 = fiber_signal.power('signal')[0] * responsivity * 2

sT = (4*kB*T*BW/R_load)**0.5
sS = 2*e*mu1*BW

s0 = sT
s1 = (sT**2 + sS**2)**0.5

ber_theory = theory_BER(mu0, mu1, s0, s1)

print(f'BER medida: {ber:.2e} ({ber*tx_seq.len():.0f} errores de {tx_seq.len()} bits transmitidos)')
print(f'BER teórica:  {ber_theory:.2e}')


## PLOTS

mod_signal.psd(label='Señal modulada')
fiber_signal.psd(label='Señal de salida de la fibra')
plt.ylim(1e-11, 1e-1)
plt.grid()

plt.figure()
mod_signal.plot('r', n=50*gv.sps, label = 'Señal modulada')
fiber_signal.plot('b', n=50*gv.sps, label='Señal de salida de la fibra')
pd_signal.plot('g', n=50*gv.sps, label='Señal photodetectada').grid(n=50)
plt.axhline(THRESHOLD_EST(eye_), color='r', linestyle='--', label='Umbral de decisión')
plt.legend(loc='upper right')

eye_.plot(THRESHOLD_EST(eye_))
plt.show()
