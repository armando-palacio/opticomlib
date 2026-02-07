from opticomlib import *
bs = binary_sequence

gv(sps=128, plt_style='dark_background')
plt.rcParams['font.family']='serif'
plt.rcParams['font.size']=14

beta = 0.5
hn = rcos_pulse(beta=beta, span=22, sps=gv.sps, shape='normal')

bs.prbs(9).dac(hn).plot_eye(cmap='cool', alpha=0.3, title=r'Raised Cosine Pulse ($\beta$={})'.format(beta))
plt.ylim(-0.3,1.3)
plt.show()