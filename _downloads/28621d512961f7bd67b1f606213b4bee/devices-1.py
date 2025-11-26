from opticomlib.devices import DAC
from opticomlib import gv

gv(sps=32) # set samples per bit

DAC('0 0 1 0 0', Vpp=5, pulse_shape='gaussian', m=2).plot('r', lw=3, grid=True).show()