from opticomlib.devices import PM

from opticomlib.typing import optical_signal
import numpy as np
import matplotlib.pyplot as plt

# Test PM
input = optical_signal(np.exp(1j*np.linspace(0,4*np.pi, 1000)))

# Fase constante
output = PM(input, v=2.5, Vpi=5)

fig, axs = plt.subplots(3,1, sharex=True, tight_layout=True)
axs[0].set_title(r'Variación de fase constante ($\Delta f=0$)')
axs[0].plot(input.t()*1e9, input.signal[0].real, 'r-', label='input', lw=3)
axs[0].plot(output.t()*1e9, output.signal[0].real, 'b-', label='output', lw=3)
axs[0].grid()

# Fase variable (lineal)
output = PM(input, v=np.linspace(0,30,input.len()), Vpi=5)


axs[1].set_title(r'Variación de fase lineal ($\Delta f \rightarrow cte.$)')
axs[1].plot(input.t()*1e9, input.signal[0].real, 'r-', label='input', lw=3)
axs[1].plot(output.t()*1e9, output.signal[0].real, 'b-', label='output', lw=3)
axs[1].grid()


# Fase variable (cuadrática)
output = PM(input, v=np.linspace(0,30**0.5,input.len())**2, Vpi=5)

plt.title(r'Variación de fase cuadrática ($\Delta f \rightarrow lineal$)')
axs[2].plot(input.t()*1e9, input.signal[0].real, 'r-', label='input', lw=3)
axs[2].plot(output.t()*1e9, output.signal[0].real, 'b-', label='output', lw=3)
axs[2].grid()
plt.xlabel('Tiempo [ns]')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.show()