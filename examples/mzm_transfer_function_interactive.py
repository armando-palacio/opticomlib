from opticomlib.devices import MZM, PD, DAC, PRBS
from opticomlib import *
from matplotlib.widgets import Slider, CheckButtons

gv(sps=64, R=1e9, Vpi=5, N=100)

f0 = gv.R

# optical carrier (complex envelope)
E_in = optical_signal(np.ones(gv.N * gv.sps))

# message signal
m = electrical_signal(np.sin(2 * np.pi * f0 * gv.t))
m_pre = 2 / np.pi * np.arcsin(m.signal)

# Initial values
gm = 1.0
bias = -gv.Vpi / 2
ER = np.inf  # Extinction ratio in dB

# Compute transfer function data once
V_range = np.linspace(-2 * gv.Vpi, gv.Vpi, 1000)
I_range = []
for V in V_range:
    x_temp = MZM(E_in, V, bias=0, Vpi=gv.Vpi, loss_dB=0, ER_dB=ER)
    y_temp = PD(x_temp, BW=gv.R * 10) * (1 / 50) 
    I_range.append(y_temp.signal.mean())
I_range = np.array(I_range)


# Helper functions for plotting
def plot_time_domain(ax, ideal, y):
    ax.clear()
    plt.sca(ax)
    ideal.plot(label='m(t)')
    y.plot(label=r'$\hat{m}(t)$')
    plt.legend(loc='upper left')
    plt.xlim(0, 4)
    plt.ylim(-0.6, 0.6)
    plt.grid()

def plot_freq_domain(ax, m, y):
    ax.clear()
    plt.sca(ax)
    m.psd(label='m(t)', yscale='dbm')
    y.psd(label=r'$\hat{m}(t)$', yscale='dbm')
    plt.legend(loc='upper left')
    plt.xlim(-8, 8)
    plt.grid()

def plot_transfer(ax, V_range, I_range, bias, gm):
    ax.clear()
    plt.sca(ax)
    V_norm = V_range / gv.Vpi
    plt.grid()
    plt.plot(V_norm, I_range, lw=2, c='r')
    V_applied = np.linspace(bias - gm * gv.Vpi / 2, bias + gm * gv.Vpi / 2, 100)
    I_applied = np.interp(V_applied, V_range, I_range)
    V_applied_norm = V_applied / gv.Vpi
    plt.plot(V_applied_norm, I_applied, 'b', alpha=1, lw=3)
    plt.scatter([V_applied_norm[0], V_applied_norm[-1]], [I_applied[0], I_applied[-1]], c='b', marker='o', s=50)
    bias_norm = bias / gv.Vpi
    plt.plot(bias_norm, np.interp(bias, V_range, I_range), 'ko', markersize=10)
    plt.xlabel(r'Voltage')
    plt.ylabel('Intensity')
    plt.title('MZM Transfer Function')
    ticks = np.arange(-2, 1.1, 0.5)
    labels = [r'$-2V_\pi$', r'$-3V_\pi/2$', r'$-V_\pi$', r'$-V_\pi/2$', r'$0$', r'$V_\pi/2$', r'$V_\pi$']
    plt.xticks(ticks, labels)
    plt.ylim(-0.05, 1.05)

# Create figure with subplots
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(221)  # Time domain
ax2 = fig.add_subplot(223)  # Frequency domain
ax3 = fig.add_subplot(222)  # MZM response

# Dragging variables
dragging = False

# Event handlers for dragging bias point
def on_press(event):
    global dragging
    if event.inaxes == ax3:
        bias_x = bias / gv.Vpi
        bias_y = np.interp(bias, V_range, I_range)
        if abs(event.xdata - bias_x) < 0.1 and abs(event.ydata - bias_y) < 0.1:  # threshold
            dragging = True

def on_motion(event):
    global dragging, bias
    if dragging and event.inaxes == ax3:
        new_bias = event.xdata * gv.Vpi
        new_bias = np.clip(new_bias, -gv.Vpi, 0)  # clamp to slider range
        bias = new_bias
        slider_bias.set_val(bias)  # update slider
        update(None)

def on_release(event):
    global dragging
    dragging = False

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# Initial computation
x = MZM(E_in, gm * gv.Vpi / 2 * m_pre, bias=bias, Vpi=gv.Vpi, loss_dB=0, ER_dB=ER)
y = PD(x, BW=gv.R * 8) * (1 / 50) - 1/2

# Initial plots
plot_time_domain(ax1, m*0.5, y)
plot_freq_domain(ax2, m, y)
plot_transfer(ax3, V_range, I_range, bias, gm)

# Sliders
ax_gm = plt.axes([0.55, 0.25, 0.35, 0.1])
ax_bias = plt.axes([0.55, 0.1, 0.35, 0.1])

slider_gm = Slider(ax_gm, 'gm', 0.0, 2.0, valinit=gm, valstep=0.1, valfmt='%1.2f')
slider_bias = Slider(ax_bias, 'Vbias', - gv.Vpi, 0, valinit=bias, valstep=0.05, valfmt='%1.2f')

# Checkbox
ax_check = plt.axes([0.55, 0.02, 0.35, 0.05])
check = CheckButtons(ax_check, ['Precompensación'], [True])


# Update function
def update(val):
    gm_val = slider_gm.val
    bias_val = slider_bias.val
    status = check.get_status()[0]
    # signal = m_pre if status else m.signal
    signal = gm_val * gv.Vpi / np.pi * np.arcsin(m.signal) if status else gm_val * m.signal * gv.Vpi / 2
    x = MZM(E_in, signal, bias=bias_val, Vpi=gv.Vpi, loss_dB=0, ER_dB=ER)
    y_new = PD(x, BW=gv.R * 8) * (1 / 50) - E_in.power()/2
    plot_time_domain(ax1, m*0.5, y_new)
    plot_freq_domain(ax2, m, y_new)
    plot_transfer(ax3, V_range, I_range, bias_val, gm_val)
    fig.canvas.draw_idle()

# Connect sliders and checkbox to update function
slider_gm.on_changed(update)
slider_bias.on_changed(update)
check.on_clicked(update)

plt.show()
