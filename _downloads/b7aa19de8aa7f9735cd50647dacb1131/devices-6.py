from opticomlib.devices import ADC
from opticomlib import gv, electrical_signal
import numpy as np

gv(sps=64, R=1e9, N=2)

y = electrical_signal( np.sin(2*np.pi*gv.R*gv.t) )

yn = ADC(y, n=2)

y.plot(
    style='light',
    grid=True,
    lw=5
)
yn.plot('.-', lw=2).show()