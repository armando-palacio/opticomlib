from opticomlib.devices import DM, DAC
from opticomlib import optical_signal, gv, idbm, bode

import matplotlib.pyplot as plt
import numpy as np

gv(N=7, sps=32, R=10e9)

signal = DAC('0,0,0,1,0,0,0', pulse_shape='gaussian')
input = optical_signal( signal.signal/signal.power()**0.5*idbm(20)**0.5 )

output, H = DM(input, D=4000, retH=True)

t = gv.t*1e9

plt.style.use('dark_background')
fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.05})

ax[0].plot(t, input.abs()[0], 'r-', lw=3, label='input')
ax[0].plot(t, output.abs()[0], 'b-', lw=3, label='output')

ax[0].set_ylabel(r'$|E(t)|$')

ax[1].plot(t[:-1], np.diff(input.phase()[0])/gv.dt*1e-9, 'r-', lw=3)
ax[1].plot(t[:-1], np.diff(output.phase()[0])/gv.dt*1e-9, 'b-', lw=3)

plt.xlabel('Time (ns)')
plt.ylabel(r'$f_i(t)$ (GHz)')
plt.ylim(-150, 150)
plt.show()