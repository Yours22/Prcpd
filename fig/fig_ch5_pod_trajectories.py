# 图 5-2: 基线模型 POD 系数轨迹对比 — 验证集和外推集各一个典型算例
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
ABL  = '2D-PINN/ablation/no_cumsum/results'

A_val_true  = np.load(os.path.join(DATA, 'A_val.npy'))    # (150, 101, 8)
A_test_true = np.load(os.path.join(DATA, 'A_test.npy'))   # (148, 101, 8)
A_val_pred  = np.load(os.path.join(ABL, 'val', 'A_pred.npy'))
A_test_pred = np.load(os.path.join(ABL, 'test_extrap', 'A_pred.npy'))

t = np.arange(0, 0.505, 0.005)

# POD 系数拼接顺序: 快群 4 阶在前, 热群 4 阶在后
# 此处只展示快群 4 阶 (Mode 1–4), 热群 4 阶 (Mode 5–8) 模式类似
mode_names = ['快群 Mode 1\n(全场振幅)', '快群 Mode 2',
              '快群 Mode 3',           '快群 Mode 4']
n_modes = 4

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

def plot_case(ax_row, true, pred, case_idx, title_prefix):
    for m in range(n_modes):
        ax = ax_row[m]
        ax.plot(t, true[case_idx, :, m], color='#1565C0', linewidth=1.2, label='真实值')
        ax.plot(t, pred[case_idx, :, m], color='#E53935', linewidth=1.0,
                linestyle='--', dashes=(4, 2), label='预测')
        ax.set_title(mode_names[m], fontsize=18, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=14)
        if m == 0:
            ax.set_ylabel(title_prefix, fontsize=20, fontweight='bold')
        if m == 0:
            ax.legend(fontsize=14, loc='upper left')

# Pick representative cases — case 0 for val, case 0 for test (most typical)
val_case = 0
test_case = 0

fig, axes = plt.subplots(2, n_modes, figsize=(14, 7))
plot_case(axes[0], A_val_true, A_val_pred, val_case, '验证集\nCase 0')
plot_case(axes[1], A_test_true, A_test_pred, test_case, '外推集\nCase 0')

fig.suptitle('基线模型 (no_cumsum) POD 系数轨迹 — 快群 4 阶',
             fontsize=22, fontweight='bold', y=1.01)
plt.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.6)
out = 'fig/fig_ch5_pod_trajectories.svg'
plt.savefig(out, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
