from opticomlib.typing import optical_signal, gv
from opticomlib.devices import MZM
from opticomlib.utils import idbm, dbm

import numpy as np
import matplotlib.pyplot as plt

# Global variables
gv(sps=100, R=10e9)

Vpi = 5
tx_seq = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0], bool)
not_tx_seq = ~tx_seq
V = 2*(np.kron(not_tx_seq, np.ones(100)) - 0.5 )*Vpi/2

# Test MZM
input = optical_signal( np.ones(1000)*idbm(10)**0.5 )

mod_sig = MZM(input, V, bias=Vpi/2, Vpi=Vpi, loss_dB=3, eta=0.01, BW=20e9)

# PLOTS
fig, axs = plt.subplots(3,1, sharex=True, tight_layout=True)

t = input.t()*1e9

# Plot input and output power
axs[0].plot(t, dbm(input.signal[0].real**2), 'r-', label='input', lw=3)
axs[0].plot(t, dbm(mod_sig.abs('signal')[0]**2), 'C1-', label='output', lw=3)
axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
axs[0].set_ylabel('Potencia [dBm]')
for i in t[::100]:
    axs[0].axvline(i, color='k', linestyle='--', alpha=0.5)

# Plot fase
phi_in = np.arctan2(input.signal[0].imag, input.signal[0].real)
phi_out = np.arctan2(mod_sig.signal[0].imag, mod_sig.signal[0].real)

axs[1].plot(t, phi_in, 'b-', label='Fase in', lw=3)
axs[1].plot(t, phi_out, 'C0-', label='Fase output', lw=3)
# axs[1].set_xlabel('Tiempo [ns]')
axs[1].set_ylabel('Fase [rad]')
axs[1].legend(bbox_to_anchor=(1, 1), loc='upper left')
for i in t[::100]:
    axs[1].axvline(i, color='k', linestyle='--', alpha=0.5)

# Frecuency chirp
freq_in = 1/2/np.pi*np.diff(phi_in)/np.diff(t)
freq_out = 1/2/np.pi*np.diff(phi_out)/np.diff(t)

axs[2].plot(t[:-1], freq_in, 'k', label='Frequency in', lw=3)
axs[2].plot(t[:-1], freq_out, 'C7', label='Frequency out', lw=3)
axs[2].set_xlabel('Tiempo [ns]')
axs[2].set_ylabel('Frequency Chirp [Hz]')
axs[2].legend(bbox_to_anchor=(1, 1), loc='upper left')
for i in t[::100]:
    axs[2].axvline(i, color='k', linestyle='--', alpha=0.5)

plt.show()