from opticomlib.devices import DAC, gv
from opticomlib.ook import DSP

import numpy as np
import matplotlib.pyplot as plt

gv(sps=64, R=1e9)

x = DAC('01000100100000', pulse_shape='gaussian')
x.noise = np.random.normal(0, 0.1, x.size)

y, eye_, xth = DSP(x)

x.plot('y', label='Photodetected signal')
DAC(y).plot(c='r', lw=2, label='Received sequence')
plt.axhline(xth, color='b', linestyle='--', label='Threshold')
plt.legend(loc='upper right')
plt.show()