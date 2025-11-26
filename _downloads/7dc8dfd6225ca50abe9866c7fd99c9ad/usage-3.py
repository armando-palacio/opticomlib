from opticomlib.devices import PRBS, DAC, gv, np

gv(sps=128, R=10e9) # set again samples per slot and slot rate

x = DAC(PRBS(order=15), pulse_shape='gaussian', Vpp=1) # create a PRBS signal and pass it through a gaussian pulse shaping filter with 1V output
x.noise = np.random.normal(0, 0.05, len(x)) # add gaussian noise to the signal

x.plot_eye(n_traces=1024, cmap='inferno', alpha=0.2)
x.show()