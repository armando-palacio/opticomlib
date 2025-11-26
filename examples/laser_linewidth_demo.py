from opticomlib.devices import LASER
from opticomlib import *

gv(sps=32, R=1e9, N=100000, plt_style='ggplot')
plt.rcParams['font.family'] = 'serif'

l1 = LASER(P0 = 0, lw=20e6)
l2 = LASER(P0 = 0, lw=100e6)

## Plot Spectrum
plt.figure(figsize=(7,5))

nperseg = 4*2048

f, psd = get_psd(l1, fs=gv.fs, nperseg=nperseg)
plt.plot(f*1e-6, dbm(psd), label='20 MHz', lw=3)

f, psd = get_psd(l2, fs=gv.fs, nperseg=nperseg)
plt.plot(f*1e-6, dbm(psd), label='100 MHz', lw=3)

plt.text(-280, -9, f'Freq. Res.: {gv.fs/nperseg*1e-6:.2f} MHz', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [dBm]')
plt.xlim(-300, 300)
plt.ylim(-30, -6)
plt.legend()

plt.show()