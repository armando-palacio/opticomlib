# Simulación de Comunicaciones Ópticas No Coherentes

La librería `opticomlib` fue desarrollada para facilitar la simulación de comunicaciones ópticas no coherentes en $\texttt{Python}$. Está desarrollada especialmente para la modulación $M$-PPM. Este código proporciona la implementación de cada componente óptico de forma modular, lo que permite al usuario armar su propio sistema de comunicación personalizado.

## Características clave

- Código en Python para simular comunicaciones ópticas no coherentes.
- Soporte para los formatos de modulación OOK y PPM
- Implementación modular de cada componente óptico, en forma de librería.
- Los componentes incluyen fuentes de luz, moduladores, demoduladores, filtros, etc.
- Código fácil de leer y modificar.

## Uso

Para utilizar el código de simulación, el usuario puede importar las librerías del componente óptico requerido y configurar los parámetros de entrada para generar la señal de salida.

Por ejemplo:

```
from opticomlib.components import *

global_vars.M = M = 8            # Orden de modulación PPM
global_vars.bit_rate = Rb = 3e9  # Tasa de transmisión de bits en [Hz] 
global_vars.sps = sps = 64       # Muestras por slot 
global_vars.update()

bits = PRBS(n=2**10)              # generamos algunos bits de forma aleatoria
simbs_ppm = PPM_ENCODER(bits, M)  # mapeamos los bits a simbolos ppm

elec_sig = DAC(simb_ppm, sps=sps, fs=fs, type='gaussian', m=3)  # Convertimos la señal digital en 'analógica'

laser = LASER(...)  # Definimos una instancia del laser

opt_sig = MODULATOR(elec_sig, laser)  # Realizamos la modulación electro-óptica
```

## Contribución

Las contribuciones a este repositorio son bienvenidas. Si desea corregir un error, mejorar la documentación o agregar una nueva función, envíe una solicitud de extracción.

## Licencia

Este proyecto tiene licencia MIT - vea el archivo [LICENSE.md](LICENSE.md) para detalles.
