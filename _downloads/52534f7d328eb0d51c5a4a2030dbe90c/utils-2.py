from opticomlib import Q
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(8, 5))
plt.plot(x, Q(x), 'r', lw=3, label='Q(x)')
plt.plot(x, Q(-x), 'b', lw=3, label='Q(-x)')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.grid()
plt.show()