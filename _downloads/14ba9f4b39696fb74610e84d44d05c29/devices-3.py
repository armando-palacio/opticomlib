from opticomlib.devices import FIBER, DAC
from opticomlib import optical_signal, gv, idbm

gv(sps=32, R=10e9)

signal = DAC('0,0,0,1,0,0,0', pulse_shape='gaussian')
input = optical_signal( signal.signal/signal.power()**0.5*idbm(20)**0.5, n_pol=2)

output = FIBER(input, length=50, alpha=0.01, beta_2=-20, gamma=0.1, show_progress=True)

input.plot('r-', label='input', lw=3)
output.plot('b-', label='output', lw=3).show()