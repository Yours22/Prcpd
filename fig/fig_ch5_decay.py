# 图 5-4: 衰减记忆特征消融对比 — 同一外推算例
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
BASE = '2D-PINN/ablation/no_cumsum/results'
ABL  = '2D-PINN/ablation/no_decay/results'

A_true = np.load(os.path.join(DATA, 'A_test.npy'))
A_base = np.load(os.path.join(BASE, 'test_extrap', 'A_pred.npy'))
A_abl  = np.load(os.path.join(ABL, 'test_extrap', 'A_pred.npy'))

t = np.arange(0, 0.505, 0.005)
case_idx = 0

# POD 系数拼接顺序: 快群 4 阶在前, 热群 4 阶在后
# 此处只展示快群 4 阶 (Mode 1–4), 热群 4 阶 (Mode 5–8) 模式类似
mode_names = ['快群 Mode 1\n(全场振幅)', '快群 Mode 2',
              '快群 Mode 3',           '快群 Mode 4']
n_modes = 4

fig, axes = plt.subplots(2, n_modes, figsize=(14, 7))

for m in range(n_modes):
    # Row 0: baseline (with decay features)
    ax0 = axes[0, m]
    ax0.plot(t, A_true[case_idx, :, m], color='#1565C0', linewidth=1.2, label='真实值')
    ax0.plot(t, A_base[case_idx, :, m], color='#2E7D32', linewidth=1.0,
             linestyle='--', dashes=(4, 2), label='含记忆特征')
    ax0.set_title(mode_names[m], fontsize=18, fontweight='bold')
    ax0.grid(True, linestyle=':', alpha=0.4)
    ax0.tick_params(labelsize=14)
    if m == 0:
        ax0.set_ylabel('含记忆特征\n(基线)', fontsize=20, fontweight='bold')
        ax0.legend(fontsize=14, loc='upper left')

    # Row 1: ablation (no decay)
    ax1 = axes[1, m]
    ax1.plot(t, A_true[case_idx, :, m], color='#1565C0', linewidth=1.2, label='真实值')
    ax1.plot(t, A_abl[case_idx, :, m], color='#F4A261', linewidth=1.0,
             linestyle='--', dashes=(4, 2), label='无记忆特征')
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.tick_params(labelsize=14)
    if m == 0:
        ax1.set_ylabel('无记忆特征\n(no_decay)', fontsize=20, fontweight='bold')
        ax1.legend(fontsize=14, loc='upper left')

fig.suptitle('衰减记忆特征消融 — 外推算例 POD 系数轨迹（快群 4 阶）',
             fontsize=22, fontweight='bold', y=1.01)
plt.tight_layout(pad=0.8, h_pad=1.5, w_pad=0.6)
out = 'fig/fig_ch5_decay.svg'
plt.savefig(out, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
