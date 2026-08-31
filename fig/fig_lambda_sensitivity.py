# 多尺度衰减积分验证：为什么单一物理λ不够用
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.sans-serif': ['SimHei', 'Songti SC', 'Arial'],
    'font.family': 'sans-serif',
    'font.size': 20, 'axes.labelsize': 20,
    'legend.fontsize': 16, 'figure.dpi': 300,
    'axes.unicode_minus': False,
})

dt = 0.005
T_win = 0.5
t = np.arange(0, T_win + dt, dt)

p_t = np.minimum(t / 0.2, 1.0)

def compute_I(lam):
    I = np.zeros_like(t)
    for k in range(1, len(t)):
        I[k] = I[k - 1] * np.exp(-lam * dt) + p_t[k] * dt
    return I

I_008 = compute_I(0.08)
I_01  = compute_I(0.1)
I_10  = compute_I(1.0)
I_100 = compute_I(10.0)
I_cum = np.cumsum(p_t) * dt

r_008_01 = np.corrcoef(I_008, I_01)[0, 1]

A = np.column_stack([I_008, np.ones_like(I_008)])
coeff_10, *_  = np.linalg.lstsq(A, I_10, rcond=None)
coeff_100, *_ = np.linalg.lstsq(A, I_100, rcond=None)
resid_10  = np.linalg.norm(I_10  - A @ coeff_10)  / np.linalg.norm(I_10)
resid_100 = np.linalg.norm(I_100 - A @ coeff_100) / np.linalg.norm(I_100)

# ──────────────────────────────────
fig = plt.figure(figsize=(22, 7))

# --- (a): physical ≈ engineering ---
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(t, I_008, '#1D3557', linewidth=2.2, label=r'$\lambda=0.08$ (物理值)')
ax1.plot(t, I_01,  '#E63946', linewidth=2.0, linestyle='--',
         label=r'$\lambda=0.1$ (工程取值)')
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('积分值')
ax1.legend(framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.4)
ax1.tick_params(labelsize=17)

# --- (b): single λ cannot reconstruct ---
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(t, I_10,  '#457B9D', linewidth=2.2, label=r'$I_{1.0}$ (中期记忆)')
ax2.plot(t, A @ coeff_10, '#457B9D', linewidth=1.4, linestyle='--',
         label=r'由 $I_{{0.08}}$ 重建 (残差 {:.1%})'.format(resid_10))
ax2.plot(t, I_100, '#E63946', linewidth=2.2, label=r'$I_{10.0}$ (短期记忆)')
ax2.plot(t, A @ coeff_100, '#E63946', linewidth=1.4, linestyle='--',
         label=r'由 $I_{{0.08}}$ 重建 (残差 {:.1%})'.format(resid_100))
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('积分值')
ax2.legend(framealpha=0.9, fontsize=13)
ax2.grid(True, linestyle=':', alpha=0.4)
ax2.tick_params(labelsize=17)

# --- (c): four memory features ---
ax3 = fig.add_subplot(1, 3, 3)
colors = ['#1D3557', '#457B9D', '#E63946', '#F4A261']
labels = [r'$I_{0.1}$ (长期)', r'$I_{1.0}$ (中期)', r'$I_{10.0}$ (短期)', r'$I_{cum}$ (累积)']
for I_val, c, lb in zip([I_01, I_10, I_100, I_cum], colors, labels):
    ax3.plot(t, I_val, color=c, linewidth=2.2, label=lb)
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('积分值')
ax3.legend(framealpha=0.9)
ax3.grid(True, linestyle=':', alpha=0.4)
ax3.tick_params(labelsize=17)

# ── subplot labels at consistent height ──
plt.tight_layout(pad=2.5, rect=[0, 0.10, 1, 1])
fig.canvas.draw()

y_label = 0.035
labels = [
    r'(a) $\lambda=0.08 \approx \lambda=0.1$ ($r={:.6f}$)'.format(r_008_01),
    r'(b) 单一物理 $\lambda$ 无法表达不同时间尺度',
    '(c) 四组记忆特征：覆盖长期惯性至瞬时响应',
]
for ax, label in zip([ax1, ax2, ax3], labels):
    bbox = ax.get_position()
    fig.text(bbox.x0 + bbox.width / 2, y_label, label,
             ha='center', va='bottom', fontsize=19)

plt.savefig('fig/fig_lambda_sensitivity.svg', bbox_inches='tight', facecolor='white')
plt.close()

print(f'Saved: fig/fig_lambda_sensitivity.svg')
print(f'  r(0.08, 0.1) = {r_008_01:.6f}')
print(f'  Residual I_1.0  <- I_0.08: {resid_10:.4f}')
print(f'  Residual I_10.0 <- I_0.08: {resid_100:.4f}')
