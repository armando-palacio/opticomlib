from opticomlib import idbm, dbm, optical_signal, gv
from opticomlib.devices import MZM, LASER

import numpy as np
import matplotlib.pyplot as plt

gv(sps=128, R=10e9, Vpi=5, N=10)

tx_seq = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0], bool)

V = DAC(tx_seq, Vpp=gv.Vpi, offset=gv.Vpi/2, pulse_shape='nrz')

input = LASER(P0=10) + np.random.normal(0, 0.01, gv.t.size)

mod_sig = MZM(input, el_input=V, bias=-gv.Vpi/2, Vpi=gv.Vpi, loss_dB=2, ER_dB=40, BW=40e9)

fig, axs = plt.subplots(3,1, sharex=True, tight_layout=True)

# Plot input and output power
axs[0].plot(gv.t, dbm(input.abs()**2), 'r-', label='input', lw=3)
axs[0].plot(gv.t, dbm(mod_sig.abs()**2), 'C1-', label='output', lw=3)
axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
axs[0].set_ylabel('Potencia [dBm]')
for i in gv.t[::gv.sps]:
    axs[0].axvline(i, color='k', linestyle='--', alpha=0.5)

# # Plot fase
phi_in = input.phase()
phi_out = mod_sig.phase()

axs[1].plot(gv.t, phi_in, 'b-', label='Fase in', lw=3)
axs[1].plot(gv.t, phi_out, 'C0-', label='Fase out', lw=3)
axs[1].set_ylabel('Fase [rad]')
axs[1].legend(bbox_to_anchor=(1, 1), loc='upper left')
for i in gv.t[::gv.sps]:
    axs[1].axvline(i, color='k', linestyle='--', alpha=0.5)

# Frecuency chirp
freq_in = 1/2/np.pi*np.diff(phi_in)/gv.dt
freq_out = 1/2/np.pi*np.diff(phi_out)/gv.dt

axs[2].plot(gv.t[:-1], freq_in, 'k', label='Frequency in', lw=3)
axs[2].plot(gv.t[:-1], freq_out, 'C7', label='Frequency out', lw=3)
axs[2].set_xlabel('Tiempo [ns]')
axs[2].set_ylabel('Frequency Chirp [Hz]')
axs[2].legend(bbox_to_anchor=(1, 1), loc='upper left')
for i in gv.t[::gv.sps]:
    axs[2].axvline(i, color='k', linestyle='--', alpha=0.5)
plt.show()