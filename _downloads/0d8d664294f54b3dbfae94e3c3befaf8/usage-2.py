from opticomlib.devices import DAC, FIBER
from opticomlib import optical_signal, gv

gv(sps=32, R=10e9) # set again samples per slot and slot rate

P = 1e-3 # 1 mW of peak power

pulse_i = optical_signal(DAC('0001000', pulse_shape='gaussian', Vpp=P**0.5)) # create a gaussian pulse as before with (1 mW) of peak power and convert it as an optical_signal,
                                                                             # because of the FIBER device only accepts optical_signal as input

pulse_o = FIBER(pulse_i, length=100, alpha=0, beta_2=-20, beta_3=0, gamma=1.5) # propagate the pulse through a fiber of 100km length,
                                                                               # with alpha=0.2 dB/km, beta_2=-20 ps^2/km, beta_3=0 ps^3/km and gamma=1.5 1/W/km
                                                                               # see "Devices/FIBER" documentation page for more details of the FIBER device
pulse_i.plot('r.-', label='Input')
pulse_o.plot('b.-', label='Output').grid().legend().show()  # plot, grid, legend and show the input and output pulses