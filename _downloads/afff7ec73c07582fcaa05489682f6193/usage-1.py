from opticomlib.devices import DAC, gv

gv(sps=32, R=1e9) # set samples per slot and slot rate, it automatically will set de sampling frequency (fs),
                  # see "Data Types/global_variables" documentation page for more details

pulse = DAC('0001000', pulse_shape='gaussian', Vpp=1) # create a gaussian pulse with 1V of amplitude peak-to-peak
                                                      # see "Devices/DAC" documentation page for more details

pulse.plot('r.-').grid().show() # plot, grid and show the pulse,
                                # see "Data Types/electrical_signal.plot" documentation page for more details