# TWIGL 基准题初始稳态快中子与热中子通量分布。
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# ── 加载稳态通量 ──
data_path = '2D-PINN/data-processed/Y_val_raw.npy'
Y = np.load(data_path) if os.path.exists(data_path) else None
if Y is None:
    raise FileNotFoundError(f'未找到数据: {data_path}')

fast  = Y[0, 0, :400].reshape(20, 20)
thermal = Y[0, 0, 400:].reshape(20, 20)

# ── 绘图 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

# 公共参数
kw = dict(origin='lower', cmap='viridis', interpolation='bilinear',
          extent=[0, 80, 0, 80], aspect='equal')

im0 = axes[0].imshow(fast, **kw)
im1 = axes[1].imshow(thermal, **kw)

# 色条
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
for cb in [cbar0, cbar1]:
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.tick_params(labelsize=16)

# 坐标轴标注
for ax, label in zip(axes, ['(a) 快中子通量', '(b) 热中子通量']):
    ax.set_xlabel('x (cm)', fontsize=20)
    ax.set_ylabel('y (cm)', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.text(0.02, 0.96, label, transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

plt.tight_layout(pad=1.2)
out_path = 'fig/fig3_steady_flux.svg'
plt.savefig(out_path, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
