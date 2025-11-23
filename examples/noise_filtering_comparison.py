from opticomlib import *
from opticomlib.devices import LPF

gv(sps=16, R=10e9, N=100000)

M=gv.N * gv.sps

noise_power = -150 # dBm/Hz

noise = electrical_signal( np.random.normal(0, np.sqrt(idbm(noise_power)*gv.fs), gv.N * gv.sps))

noise_1 = LPF(noise, BW=gv.R)

#filtro cuadrado ideal
def ideal_square_filter(signal, BW):
    fft_signal = fft(signal.signal)
    df = gv.fs / (gv.N * gv.sps)
    cutoff_bin = int(BW / (df))
    fft_signal_filtered = np.zeros_like(fft_signal)
    fft_signal_filtered[:cutoff_bin] = fft_signal[:cutoff_bin]
    fft_signal_filtered[-cutoff_bin:] = fft_signal[-cutoff_bin:]
    filtered_signal = electrical_signal(ifft(fft_signal_filtered))
    return filtered_signal
noise_2 = ideal_square_filter(noise, BW=gv.R)

nperseg = 512

f,psd = sg.welch(noise.signal, fs=gv.fs, nperseg=nperseg, return_onesided=False, scaling='density', noverlap=int(0.75*nperseg), detrend=False, window='hann')

f,psd_1 = sg.welch(noise_1.signal, fs=gv.fs, nperseg=nperseg, return_onesided=False, scaling='density', noverlap=int(0.75*nperseg), detrend=False, window='hann')

f,psd_2 = sg.welch(noise_2.signal, fs=gv.fs, nperseg=nperseg, return_onesided=False, scaling='density', noverlap=int(0.75*nperseg), detrend=False, window='hann')

plt.plot(fftshift(f)/1e9, dbm(fftshift(psd)), lw=3, label='AWG Noise')
plt.plot(fftshift(f)/1e9, dbm(fftshift(psd_1)), lw=3, label='Bessel Filtered Noise')
plt.plot(fftshift(f)/1e9, dbm(fftshift(psd_2)), lw=3, label='Ideal Square Filtered Noise')
plt.xlabel('Frequency (GHz)')
plt.ylabel('PSD (dBm/Hz)')

print(noise_1.power(), noise_2.power())

plt.legend()
plt.grid()
plt.ylim(noise_power-10, noise_power+2)
plt.xlim(-20, 20)
plt.show()


