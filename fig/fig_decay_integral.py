# 多尺度指数衰减积分特征示意：上图为扰动信号，下图为三种衰减常数与累积积分对比。
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.sans-serif': ['SimHei', 'Songti SC', 'Arial'],
    'font.family': 'sans-serif',
    'font.size': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 300,
    'axes.unicode_minus': False,
    'savefig.bbox': 'tight',
})

# ---------- parameters ----------
dt = 0.005
T = 0.5
t = np.arange(0, T + dt, dt)  # 101 steps
lam_list = [0.1, 1.0, 10.0]
colors = {0.1: '#E63946', 1.0: '#457B9D', 10.0: '#2A9D8F', 'cum': '#F4A261'}

# ---------- perturbation signal ----------
p_t = np.zeros_like(t)
cut_time = 0.2
ramp_mask = t <= cut_time
p_t[ramp_mask] = 1.0 * (t[ramp_mask] / cut_time)   # ramp 0 -> 1
p_t[~ramp_mask] = 1.0                                # hold at 1

# ---------- decay integrals ----------
def compute_decay_integrals(p, dt, lam):
    """Recursive exponential decay integral: I[t] = I[t-1]*exp(-lam*dt) + p[t]*dt"""
    I = np.zeros_like(p)
    for k in range(1, len(p)):
        I[k] = I[k - 1] * np.exp(-lam * dt) + p[k] * dt
    return I

integrals = {}
for lam in lam_list:
    integrals[lam] = compute_decay_integrals(p_t, dt, lam)

# cumulative integral (lambda -> 0 limit)
integrals_cum = np.cumsum(p_t) * dt

# ---------- plot ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# --- upper: perturbation ---
ax1.fill_between(t, 0, p_t, color='#1D3557', alpha=0.25)
ax1.plot(t, p_t, color='#1D3557', linewidth=2.2)
ax1.axvline(x=cut_time, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax1.text(cut_time + 0.008, 0.05, f'$t_{{cut}}={cut_time}$s', color='gray', fontsize=20,
         va='bottom')
ax1.set_ylabel('归一化扰动 $p(t)$', fontsize=20)
ax1.set_ylim(-0.05, 1.15)
ax1.set_yticks([0, 0.5, 1.0])
ax1.grid(True, linestyle=':', alpha=0.4)
ax1.text(0.01, 0.92, '(a) 瞬态扰动信号',
         transform=ax1.transAxes, fontsize=16, fontweight='bold')

# --- lower: decay integrals ---
for lam in lam_list:
    ax2.plot(t, integrals[lam], color=colors[lam], linewidth=1.8,
             label=f'$\\lambda={lam:.1f}$  $(T_{{1/2}}={np.log(2)/lam:.2f}\\,\\mathrm{{s}})$')
ax2.plot(t, integrals_cum, color=colors['cum'], linewidth=2.0, linestyle='--',
         label='累积积分 $(\\lambda\\to 0)$')

# mark t=cut in lower panel
ax2.axvline(x=cut_time, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax2.set_xlabel('时间 $t$ (s)', fontsize=20)
ax2.set_ylabel('衰减积分值', fontsize=20)
ax2.legend(loc='upper left', framealpha=0.9, ncol=2)
ax2.grid(True, linestyle=':', alpha=0.4)
ax2.text(0.01, 0.92, '(b) 指数衰减积分特征',
         transform=ax2.transAxes, fontsize=16, fontweight='bold')

# annotations: explain the physics
ax2.annotate('$\\lambda=10$ 在 $t_{cut}$\n附近趋于饱和',
             xy=(0.25, integrals[10.0][50]), xytext=(0.32, integrals[10.0][50] * 0.5),
             fontsize=16, color=colors[10.0],
             arrowprops=dict(arrowstyle='->', color=colors[10.0], lw=1.2))

ax2.annotate('$\\lambda=0.1$ 持续\n累积增长',
             xy=(0.42, integrals[0.1][84]), xytext=(0.28, integrals[0.1][84] * 0.65),
             fontsize=16, color=colors[0.1],
             arrowprops=dict(arrowstyle='->', color=colors[0.1], lw=1.2))

ax2.annotate('$\\int p\\,dt$ = 注入\n总反应性',
             xy=(0.47, integrals_cum[94]), xytext=(0.34, integrals_cum[94] * 0.72),
             fontsize=16, color=colors['cum'],
             arrowprops=dict(arrowstyle='->', color=colors['cum'], lw=1.2))

plt.tight_layout(pad=0.5)
plt.savefig('fig/fig_decay_integral.svg')
print('Saved: fig/fig_decay_integral.svg')
plt.close()
