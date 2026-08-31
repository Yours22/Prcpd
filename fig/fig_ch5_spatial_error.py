# 图 5-8: 空间误差热力图 — 基线模型，验证/外推 × 快群/热群
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Songti SC', 'Arial'],
    'font.family': 'sans-serif',
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'figure.dpi': 300,
    'axes.unicode_minus': False,
    'savefig.bbox': 'tight',
})

DATA = '2D-PINN/data-processed'
BASE = '2D-PINN/ablation/no_cumsum/results'

N_NODES = 400
NX = NY = 20

Y_val_true  = np.load(os.path.join(DATA, 'Y_val_raw.npy'))
Y_test_true = np.load(os.path.join(DATA, 'Y_test_raw.npy'))
Y_val_pred  = np.load(os.path.join(BASE, 'val', 'Y_pred.npy'))
Y_test_pred = np.load(os.path.join(BASE, 'test_extrap', 'Y_pred.npy'))

t_step = 100  # final time step
val_case = 0
test_case = 0

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

plot_specs = [
    (axes[0, 0], Y_val_true[val_case, t_step, :N_NODES],
     Y_val_pred[val_case, t_step, :N_NODES], '验证集 快群'),
    (axes[0, 1], Y_val_true[val_case, t_step, N_NODES:],
     Y_val_pred[val_case, t_step, N_NODES:], '验证集 热群'),
    (axes[1, 0], Y_test_true[test_case, t_step, :N_NODES],
     Y_test_pred[test_case, t_step, :N_NODES], '外推集 快群'),
    (axes[1, 1], Y_test_true[test_case, t_step, N_NODES:],
     Y_test_pred[test_case, t_step, N_NODES:], '外推集 热群'),
]

for ax, true, pred, title in plot_specs:
    rel_error = np.abs(pred - true) / (np.abs(true) + 1e-10) * 100  # 转为百分比
    field = rel_error.reshape(NY, NX)
    im = ax.imshow(field, cmap='YlOrRd', origin='lower', aspect='equal')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=ticker.FormatStrFormatter('%.1f%%'))
    cbar.ax.tick_params(labelsize=14)

fig.suptitle(f'基线模型空间相对误差 — t=0.50s（步 {t_step}），Case 0',
             fontsize=22, fontweight='bold', y=1.01)

plt.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.4)
out = 'fig/fig_ch5_spatial_error.svg'
plt.savefig(out, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
