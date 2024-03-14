from opticomlib.devices import DAC
from opticomlib import gv

gv(sps=32) # set samples per bit

DAC('0 0 1 0 0', Vout=5, pulse_shape='gaussian', m=2).plot('r', lw=3).show()