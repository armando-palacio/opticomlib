from opticomlib.devices import DAC, gv
from opticomlib.ppm import DSP

import numpy as np
import matplotlib.pyplot as plt

gv(sps=64, R=1e9)

x = DAC('0100 1010 0000', pulse_shape='gaussian')
x.noise = np.random.normal(0, 0.1, x.size)

y = DSP(x, M=4, decision='soft')

DAC(y).plot(c='r', lw=3, label='Received sequence').show()