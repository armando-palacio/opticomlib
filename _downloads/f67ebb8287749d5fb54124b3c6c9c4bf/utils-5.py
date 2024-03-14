from opticomlib import rcos
import matplotlib.pyplot as plt
import numpy as np

T = 1
x = np.linspace(-1.5/T, 1.5/T, 1000)

plt.figure(figsize=(8, 5))

for alpha in [0, 0.25, 0.5, 1]:
    plt.plot(x, rcos(x, alpha, T), label=r'$\alpha$ = {}'.format(alpha))

plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.grid(alpha=0.3)
plt.show()