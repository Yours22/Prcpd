# 图 5-9: 总功率守恒曲线对比 — 基线 / no_amp_shape / no_symlog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Songti SC', 'Arial'],
    'font.family': 'sans-serif',
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 16,
    'figure.dpi': 300,
    'axes.unicode_minus': False,
    'savefig.bbox': 'tight',
})

DATA = '2D-PINN/data-processed'
BASE_V = '2D-PINN/ablation/no_cumsum/results/val'
BASE_T = '2D-PINN/ablation/no_cumsum/results/test_extrap'
AMP_V  = '2D-PINN/ablation/no_amp_shape/results/val'
AMP_T  = '2D-PINN/ablation/no_amp_shape/results/test_extrap'
SYM_V  = '2D-PINN/ablation/no_symlog/results/val'
SYM_T  = '2D-PINN/ablation/no_symlog/results/test_extrap'

Y_val_true  = np.load(os.path.join(DATA, 'Y_val_raw.npy'))
Y_test_true = np.load(os.path.join(DATA, 'Y_test_raw.npy'))

def load_pred(base_dir, test_dir):
    """Load pred and compute total power for both val and test_extrap."""
    yv = np.load(os.path.join(base_dir, 'Y_pred.npy'))
    yt = np.load(os.path.join(test_dir, 'Y_pred.npy'))
    return yv, yt

yv_base, yt_base = load_pred(BASE_V, BASE_T)
yv_amp, yt_amp     = load_pred(AMP_V, AMP_T)
yv_sym, yt_sym     = load_pred(SYM_V, SYM_T)

t = np.arange(0, 0.505, 0.005)

# Compute total power: sum over all 800 spatial nodes
def total_power(Y):
    return Y.sum(axis=2)  # (N, T)

case_v = 0
case_t = 0

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: validation
ax = axes[0]
ax.plot(t, total_power(Y_val_true)[case_v], color='#212121', linewidth=2.2, label='真实值')
ax.plot(t, total_power(yv_base)[case_v], color='#2E7D32', linewidth=1.5,
        linestyle='--', label='基线 (no_cumsum)')
ax.plot(t, total_power(yv_amp)[case_v], color='#E53935', linewidth=1.5,
        linestyle='--', label='no_amp_shape')
ax.plot(t, total_power(yv_sym)[case_v], color='#7B1FA2', linewidth=1.5,
        linestyle='--', label='no_symlog')
ax.set_title('验证集 Case 0', fontsize=20, fontweight='bold')
ax.set_ylabel('总功率 P(t)', fontsize=20)
ax.set_xlabel('时间 t (s)', fontsize=20)
ax.set_yscale('log')
ax.legend(loc='upper left', framealpha=0.9, fontsize=16)
ax.grid(True, linestyle=':', alpha=0.4)

# Right: extrapolation
ax = axes[1]
ax.plot(t, total_power(Y_test_true)[case_t], color='#212121', linewidth=2.2, label='真实值')
ax.plot(t, total_power(yt_base)[case_t], color='#2E7D32', linewidth=1.5,
        linestyle='--', label='基线 (no_cumsum)')
ax.plot(t, total_power(yt_amp)[case_t], color='#E53935', linewidth=1.5,
        linestyle='--', label='no_amp_shape')
ax.plot(t, total_power(yt_sym)[case_t], color='#7B1FA2', linewidth=1.5,
        linestyle='--', label='no_symlog')
ax.set_title('外推集 Case 0', fontsize=20, fontweight='bold')
ax.set_xlabel('时间 t (s)', fontsize=20)
ax.set_yscale('log')
ax.legend(loc='upper left', framealpha=0.9, fontsize=16)
ax.grid(True, linestyle=':', alpha=0.4)

# Annotations — 上下错开避免文字重叠
axes[1].annotate('no_symlog:\n系统性低估 >50%',
                 xy=(0.45, total_power(yt_sym)[case_t, 90]),
                 xytext=(0.2, total_power(Y_test_true)[case_t, 70]),
                 fontsize=16, color='#7B1FA2',
                 arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.2))

axes[1].annotate('no_amp_shape:\n末期偏离',
                 xy=(0.42, total_power(yt_amp)[case_t, 84]),
                 xytext=(0.35, total_power(Y_test_true)[case_t, 50]),
                 fontsize=16, color='#E53935',
                 arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.2))

fig.suptitle('总功率守恒验证 — 全堆通量求和 P(t)', fontsize=22, fontweight='bold', y=1.01)
plt.tight_layout(pad=0.8)
out = 'fig/fig_ch5_power.svg'
plt.savefig(out, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
