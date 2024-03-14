from opticomlib import phase
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-5, 5, 1000)
y = np.exp(1j*t**2)
phi = phase(y)

plt.figure(figsize=(8, 5))
plt.plot(t, phi, 'r', lw=2)
plt.ylabel('phase [rad]')
plt.xlabel('t')
plt.grid(alpha=0.3)
plt.show()