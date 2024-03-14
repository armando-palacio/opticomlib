from opticomlib import tau_g
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-5, 5, 1000)
y = np.exp(1j*t**2)
phi = tau_g(y, 1e2)

plt.figure(figsize=(8, 5))
plt.plot(t[:-1], phi, 'r', lw=2)
plt.ylabel(r'$\tau_g$ [ps]')
plt.xlabel('t')
plt.grid(alpha=0.3)
plt.show()