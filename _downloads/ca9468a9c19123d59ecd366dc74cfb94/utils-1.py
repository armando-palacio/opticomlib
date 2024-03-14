from opticomlib import gaus
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 1000)
y = gaus(x, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'r', lw=2)
plt.ylabel('y')
plt.xlabel('x')
plt.grid(alpha=0.3)
plt.show()