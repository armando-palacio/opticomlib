""" 
    This example will simulate the transmission over 50 km of SMF optical fiber of a 10 Gbps OOK 
    signal with a PRBS of 2^7-1 bits. NRZ modulation with gaussian pulse shaping will be used.

    required opticomlib-v1.5
"""

import opticomlib.ook as ook

import opticomlib.devices as dvs

from opticomlib import (
    dbm,
    idbm, 
    optical_signal, 
    gv,
    theory_BER
)

nm = 1e-9

import numpy as np
import matplotlib.pyplot as plt


# Simulation parameters
gv(sps=64, R=10e9, wavelength=1550*nm, Vpi=5, N=2**10).print()

# prbs
tx_seq = dvs.PRBS(order=9, len=gv.N)

# DAC and pulse shaping
v = dvs.DAC(tx_seq, Vpp=gv.Vpi, offset=-gv.Vpi/2, pulse_shape='gaussian')

# optical source
cw_laser = dvs.LASER(P0=5) # 5 dBm CW optical source, 1-polarization

# Mach-Zehnder modulator
mod_signal = dvs.MZM(cw_laser, v, bias=-gv.Vpi/2, Vpi=gv.Vpi, loss_dB=3, ER_dB=26)

# optical fiber
fiber_signal = dvs.FIBER(mod_signal, length=50, alpha=0.2, beta_2=-20, gamma=2, show_progress=True)
P_avg = fiber_signal.power('dBm')

# photo-detector
pd_signal = dvs.PD(fiber_signal, BW=gv.R*0.75, r=1, include_noise='all')

# DSP
rx_seq, eye_, rth = ook.DSP(pd_signal) # return received bits, eye diagram and decision threshold

# BER
ber = ook.BER_analizer('counter', Tx=tx_seq, Rx=rx_seq)


# Theoretical BER
ber_theory = theory_BER(
    P_avg = P_avg,
    modulation = 'ook',
    ER = 26,
    amplify=False,
    BW_el=0.75*gv.R,
    r=1.0,
    T=300,
    R_L=50,
)

print(f'BER counts: {ber:.2e} ({ber*tx_seq.size:.0f} errors of {tx_seq.size} transmitted bits)')
print(f'BER theoretical:  {ber_theory:.2e}')


## PLOTS
mod_signal.psd(label='Fiber input PSD')
fiber_signal.psd(label='Fiber output PSD', grid=True)


mod_signal.plot('r', n=50*gv.sps, label = 'Fiber input', hold=False)
fiber_signal.plot('b', n=50*gv.sps, label='Fiber output', grid=True)


pd_signal.plot('g', n=50*gv.sps, label='Photodetected signal', grid=True, hold=False)
plt.axhline(rth, color='r', linestyle='--', label='Decision threshold')
plt.legend()

pd_signal.plot_eye()
plt.show()
# eye_.plot().show()
