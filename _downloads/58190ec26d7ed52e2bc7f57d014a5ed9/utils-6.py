from opticomlib import theory_BER
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-40, -20, 1000)  # Average input optical power [dB]

plt.figure(figsize=(8, 6))

plt.semilogy(x, theory_BER(P_avg=x, modulation='ook'), label='OOK')
plt.semilogy(x, theory_BER(P_avg=x, modulation='ppm', M=4, decision='soft'), label='4-PPM (soft)')
plt.semilogy(x, theory_BER(P_avg=x, modulation='ppm', M=4, decision='hard'), label='4-PPM (hard)')

plt.xlabel(r'$P_{avg}$')
plt.ylabel('BER')
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(1e-9,)
plt.show()