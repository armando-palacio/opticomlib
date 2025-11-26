from opticomlib.devices import LASER, MZM, PD
from matplotlib.widgets import Slider, CheckButtons
from opticomlib import (
    electrical_signal as el_sig, 
    np, plt, 
    gv,
    idb, dbm, pi, 
    get_psd
)

plt.rcParams['font.family'] = 'serif'

gv(sps=32, R=1e9, Vpi=5, N=100)

# optical carrier 
E_in = LASER(P0=30) # 30 dBm (1 W)

# electrical message signal
m = np.sin( el_sig(2*np.pi*gv.R*gv.t) )
m_pre = np.arcsin(m)

def mzm_transfer_func(v, bias=0, ER=np.inf):
    v = pi / 2 / gv.Vpi * (v + bias)
    return np.cos(v)**2 + 1/idb(ER) * np.sin(v)**2


# Create figure with subplots
fig = plt.figure(figsize=(10, 6))
fig.text(0.5, 0.92, r'$\frac{P_{out}}{P_{in}} = \cos^2[\frac{\pi}{2V_\pi}(\mathbf{g_m} v(t) + \mathbf{V_{bias}})] + \frac{1}{\mathbf{ER}}\sin^2[\frac{\pi}{2V_\pi}(\mathbf{g_m} v(t) + \mathbf{V_{bias}})]$', ha='center', fontsize=14)

ax1 = fig.add_subplot(222)  # Time domain
ax2 = fig.add_subplot(224)  # Frequency domain
ax3 = fig.add_subplot(221)  # MZM response

# Sliders
# Position sliders in the bottom-left quadrant (subplot 3 area)
ax_ER = plt.axes([0.12, 0.35, 0.25, 0.03])
ax_gm = plt.axes([0.12, 0.25, 0.25, 0.03])
ax_bias = plt.axes([0.12, 0.15, 0.25, 0.03])
ax_check = plt.axes([0.12, 0.04, 0.25, 0.06])

# Style
slider_kwargs = {'color': 'teal', 'alpha': 0.8}

slider_ER = Slider(ax_ER, r'$ER$ [dB]', 0, 30, valinit=29, valstep=1, valfmt='%1.0f', **slider_kwargs)
slider_gm = Slider(ax_gm, r'$g_m$', 0.0, 2.0, valinit=1, valstep=0.1, valfmt='%1.1f', **slider_kwargs)
slider_bias = Slider(ax_bias, r'$V_{bias}$', -1, 0, valinit=-1/2, valstep=0.05, valfmt=r'%1.2f $V_\pi$', **slider_kwargs)

# Checkbox
check = CheckButtons(ax_check, ['Pre-compensation (Arcsin)'], [False])
ax_check.set_frame_on(False)

check.set_frame_props({'facecolor': 'teal', 'alpha': 0.5, 'sizes': [100]})
plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.83, wspace=0.3, hspace=0.4)

# Initial Plotting - Time Domain
line_m_t, = ax1.plot(gv.t[:len(m)]*1e9, m.signal, label='m(t)', lw=2, c='r')
line_m_est_t, = ax1.plot([], [], label=r'$\hat{m}(t)$', lw=2, c='b') # Initialize empty
ax1.legend(loc='upper right')
ax1.set_xlim(0, 4)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(alpha=0.3)
ax1.set_xlabel('Time [ns]')
ax1.set_ylabel('Amplitude [V]')

# Initial Plotting - Frequency Domain
f, psd_m = get_psd(m, gv.fs)
line_m_f, = ax2.plot(f*1e-9, dbm(psd_m), label='m(t)', lw=2, c='r')
line_m_est_f, = ax2.plot([], [], label=r'$\hat{m}(t)$', lw=2, c='b') # Initialize empty
ax2.legend(loc='upper right')
ax2.set_xlim(-8, 8)
ax2.set_ylim(-100, 30) # Fixed ylim for stability
ax2.grid(alpha=0.3)
ax2.set_xlabel('Frequency [GHz]')
ax2.set_ylabel('Power [dBm]')

# Initial Plotting - Transfer Function
V_range = np.linspace(-2, 1, 1000) * gv.Vpi
line_tf_red, = ax3.plot(V_range, mzm_transfer_func(V_range, ER=29), lw=2, c='r') # Initial ER=29

line_tf_blue, = ax3.plot([], [], 'b', alpha=1, lw=3)
scatter_blue = ax3.scatter([], [], c='b', marker='o', s=50)
scatter_black, = ax3.plot([], [], 'ko', markersize=10)

ax3.set_title('MZM Transfer Function')
ax3.set_xlabel(r'Voltage')
ax3.set_ylabel(r'$T = P_{out}/P_{in}$')
ticks = np.arange(-2, 1.1, 0.5) * gv.Vpi
labels = [r'$-2V_\pi$', r'$-3V_\pi/2$', r'$-V_\pi$', r'$-V_\pi/2$', r'$0$', r'$V_\pi/2$', r'$V_\pi$']
ax3.set_xticks(ticks)
ax3.set_xticklabels(labels)
ax3.set_ylim(-0.05, 1.05)
ax3.grid(alpha=0.3)

# Update function
def update(val):
    gm = slider_gm.val
    bias = slider_bias.val * gv.Vpi
    ER = slider_ER.val
    status = check.get_status()[0]

    if status:
        v = gv.Vpi/np.pi * np.arcsin(m)
    else:
        v = gv.Vpi/2 * m

    x = MZM(E_in, gm*v, bias=bias, Vpi=gv.Vpi, ER_dB=ER)
    m_est = 2 * PD(x, BW=gv.R * 8) / 50 - E_in.power()
    
    # Update Time Domain
    line_m_est_t.set_data(gv.t[:len(m_est)]*1e9, m_est.signal)
    
    # Update Frequency Domain
    f, psd_est = get_psd(m_est, gv.fs)
    line_m_est_f.set_data(f*1e-9, dbm(psd_est))
    
    # Update Transfer Function
    # Red curve (depends on ER)
    line_tf_red.set_ydata(mzm_transfer_func(V_range, ER=ER))
    
    # Blue curve (depends on bias, gm, ER)
    V_applied = np.linspace(bias - gm * gv.Vpi / 2, bias + gm * gv.Vpi / 2, 100)
    I_applied = mzm_transfer_func(V_applied, ER=ER)
    line_tf_blue.set_data(V_applied, I_applied)
    
    # Points
    scatter_blue.set_offsets(np.c_[[V_applied[0], V_applied[-1]], [I_applied[0], I_applied[-1]]])
    scatter_black.set_data([bias], [mzm_transfer_func(0, bias, ER)])
    
    fig.canvas.draw_idle()

# Connect sliders and checkbox to update function
slider_gm.on_changed(update)
slider_bias.on_changed(update)
slider_ER.on_changed(update)
check.on_clicked(update)

update(None)

plt.show()
