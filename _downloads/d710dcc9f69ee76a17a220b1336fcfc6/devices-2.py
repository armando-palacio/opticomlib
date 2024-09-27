from opticomlib.devices import EDFA
from opticomlib import optical_signal, gv, np, plt

gv(sps=256, R=1e9, N=5, G=20, NF=5, BW=50e9)

x = optical_signal(
    signal=[
        (1e-3)*np.sin(2*np.pi*gv.R*gv.t),
        np.zeros_like(gv.t)
    ],
    n_pol=2
)

y = EDFA(x, G=gv.G, NF=gv.NF, BW=gv.BW)

fig, axs = plt.subplots(2,1, sharex=True, figsize=(8,6))
plt.suptitle(f"EDFA input-output (G={gv.G} dB, NF={gv.NF} dB, BW={gv.BW*1e-9} GHz)")

axs[0].set_title('Input')
axs[0].plot(gv.t*1e9, x.signal.T)
axs[0].set_ylim(-0.015, 0.015)

axs[1].set_title('Output')
axs[1].plot(gv.t*1e9, y.signal.T + y.noise.T.real)
axs[1].set_ylim(-0.015, 0.015)

plt.legend(['x-pol', 'y-pol'])
plt.xlabel('t [ns]')
plt.show()