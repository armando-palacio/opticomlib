from opticomlib.ppm import PPM_ENCODER
from opticomlib.devices import PRBS, DAC
from scipy.signal import welch
from matplotlib.pyplot import *
from opticomlib import gv, dbm

rcParams['text.usetex'] = True
rcParams['font.size'] = 16
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']

gv(sps=16)
ak = PRBS(15)

vs_Rb = 1

for M in [4, 8, 16, 32]:
    x = DAC(PPM_ENCODER(ak, M))
    x = x.signal/x.power()**0.5

    f, P = welch(x, fs=gv.fs, nperseg=1024, return_onesided=True, scaling='spectrum', nfft=4096,)
    
    R = gv.R*np.log2(M)/M if vs_Rb else gv.R
    plot(f/R, P, lw=3, label=f'M={M}')
    
    if vs_Rb:
        xlim(-0.5, 11)
        xlabel(r'$f/R_b$', fontsize=18)
    else:
        xlim(0,2)
        xlabel(r'$f/R$', fontsize=18)

# ook
x = DAC(ak)
x = x.signal/x.power()**0.5

f, P = welch(x, fs=gv.fs, nperseg=1024, return_onesided=True, scaling='spectrum', nfft=4096,)
R = gv.R
plot(f/R, P, 'k', lw=3, label='OOK')

grid()
ylabel('PSD', fontsize=16)
# legend()
tight_layout()

# if vs_Rb:
#     savefig('ppm_psd_0.pdf')
# else:
#     savefig('ppm_psd_1.pdf')
show()


