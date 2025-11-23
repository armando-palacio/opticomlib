from opticomlib.devices import LASER
from opticomlib import *

gv(sps=32, R=1e9, N=100000)

l1 = LASER(gv.t, p = 0, lw=20e6)
l2 = LASER(gv.t, p = 0, lw=100e6)


## Plot Spectrum
plt.figure(figsize=(7,5))

nperseg = 4*2048
nfft = 4*4096

f, psd = sg.welch(l1.signal, fs=gv.fs*1e-6, nperseg=nperseg, nfft=nfft, scaling='spectrum', return_onesided=False, detrend=False)
f, psd = np.fft.fftshift(f), np.fft.fftshift(psd)

plt.plot(f, dbm(psd), label='20 MHz', lw=3)

f, psd = sg.welch(l2.signal, fs=gv.fs*1e-6, nperseg=nperseg, nfft=nfft, scaling='spectrum', return_onesided=False, detrend=False)
f, psd = np.fft.fftshift(f), np.fft.fftshift(psd)

plt.plot(f, dbm(psd), label='100 MHz', lw=3)

plt.text(-280, -9, f'Sinc Bandwidth: {4/(nperseg*gv.dt)*1e-6:.2f} MHz\n         Freq. Res.: {gv.fs/nfft*1e-6:.2f} MHz', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Spectrum (dBm)')
plt.xlim(-300, 300)
plt.ylim(-30, -6)
plt.legend()
plt.grid()

plt.show()