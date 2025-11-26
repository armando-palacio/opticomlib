from opticomlib import *


def gmzm(Ein, phi, N=2):
    pos = np.array([np.exp(1j * n *phi) for n in range(1, N//2+1)]) if N > 1 else np.zeros_like(phi)

    if N>1:
        return Ein/N * np.sum( (N%2) + pos + np.conj(pos) , axis=0)
    else:
        return Ein/N * (N%2 + pos + np.conj(pos))


# plt.plot(abs(gmzm(np.cos(2*np.pi*1*np.linspace(0,10,1000)), np.pi/2)**2))

phi = np.linspace(-2*np.pi, 2*np.pi, 1000)
plt.plot(phi/(np.pi), abs(gmzm(1, phi, N=2)**2), 'r--')
plt.plot(phi/(np.pi), abs(gmzm(1, phi, N=4)**2), 'b--')
plt.plot(phi/(np.pi), abs(gmzm(1, phi, N=10)**2), 'y--')
# plt.plot(phi/np.pi, np.sin(1.5*phi + 0.5*np.pi)/2+0.5)
plt.grid()
plt.title('MZM3 Transfer Function')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.tight_layout()

plt.figure()
plt.plot(phi/(np.pi), gmzm(1, phi, N=2).real, 'r--')
plt.plot(phi/(np.pi), gmzm(1, phi, N=4).real, 'b--')
plt.plot(phi/(np.pi), gmzm(1, phi, N=10).real, 'y--')
plt.grid()
plt.title('MZM3 Transfer Function (Real Part)')

plt.show()