import numpy as np
import timeit, time as tm
import scipy.special as sp
import scipy.signal as sg
from scipy.constants import pi, c, h, k as kB, e
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.pyplot import plot, subplots, psd, axvline, legend, xlabel, ylabel, savefig, show, suptitle

plt.rcParams['font.family'] = 'serif'


def generate_prbs(k):
    taps = {7: [7,6], 9: [9,5], 11: [11,9], 15: [15,14], 20: [20,3], 23: [23,18], 31: [31,28]}
    if k not in taps.keys():
        raise ValueError(f'k must be in {list(taps.keys())}')

    prbs = []
    lfsr = (1<<k)-1
    tap1, tap2 = np.array(taps[k])-1

    while True:
        prbs.append(lfsr&1)
        new = ((lfsr>>tap1)^(lfsr>>tap2))&1
        lfsr = ((lfsr<<1) | new) & (1<<k)-1
        if lfsr == (1<<k)-1:
            break
    return np.array(prbs, np.uint8)


def dec2bin(n, k=8):
    binary = np.zeros(k, np.uint8)
    if n > 2**k-1: raise ValueError('El número es demasiado grande para ser representado con k bits.')
    i = k - 1
    while n > 0 and i >= 0:
        binary[i] = n % 2
        n //= 2
        i -= 1
    return binary


def str2ndarray(string: str): 
    return np.array(list(map(int, string.replace(',','').replace(' ',''))))


def get_time(line_of_code: str, n:int): return timeit.timeit(line_of_code, number=n)/n


def tic(): global __; __ = tm.time()
def toc(): global __; return tm.time()-__ 


db = lambda x: 10*np.log10(x) # función de conversión a dB


dbm = lambda x: 10*np.log10(x*1e3) # función de conversión a dBm (x en Watts)


idb = lambda x: 10**(x/10) # función de conversión a veces


idbm = lambda x: 10**(x/10-3) # función de conversión a Watts


gaus = lambda x,mu,std: 1/std/(2*pi)**0.5*np.exp(-0.5*(x-mu)**2/std**2) # función gaussiana


Q = lambda x: 0.5*sp.erfc(x/2**0.5) # función Q


def Pe_OOK(mu1,s0,s1):
    def fun(mu1,s0,s1):
        r = np.linspace(0,mu1,1000)
        return 0.5*np.min(Q((mu1-r)/s1) + Q(r/s0))
    return np.vectorize(fun)(mu1,s0,s1)


def Pe_H(mu1,s0,s1,M):
    def fun(mu1,s0,s1,M):
        r = np.linspace(0,mu1,1000)
        return np.min(1 - Q((r-mu1)/s1) * (1-Q((r)/s0))**(M-1))
    return np.vectorize(fun)(mu1,s0,s1,M)*0.5*M/(M-1)


def Pe_S(mu1,s0,s1,M):
    fun = np.vectorize(lambda mu1,s0,s1,M: 1-1/(2*pi)**0.5*quad(lambda x: (1-Q((mu1+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0])
    return fun(mu1,s0,s1,M)*0.5*M/(M-1)

        

