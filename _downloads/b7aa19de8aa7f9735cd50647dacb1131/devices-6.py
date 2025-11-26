from opticomlib.devices import LPF
from opticomlib import gv, electrical_signal
import matplotlib.pyplot as plt
import numpy as np

gv(N = 10, sps=128, R=1e9)

t = gv.t
c = 20e9/t[-1]   # frequency chirp from 0 to 20 GHz

input = electrical_signal( np.sin( np.pi*c*t**2) )
output = LPF(input, 10e9)

input.psd('r', label='input', lw=2)
output.psd('b', label='output', lw=2)

plt.xlim(-30,30)
plt.ylim(-20, 5)
plt.annotate('-6 dB', xy=(10, -5), xytext=(10, 2), c='r', arrowprops=dict(arrowstyle='<->'), fontsize=12, ha='center', va='center')
plt.show()