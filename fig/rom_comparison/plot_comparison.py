# 四种降阶方法横向对比：POD / PCA / DMD / VAE 重构场与误差场（Case 0, t=50, 快群）。
import os, numpy as np, yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
PATHS = config['paths']
PHYSICS = config['physics']

CASE, TIME = 0, 50
OFFSET = 0
NX = NY = int(np.sqrt(PHYSICS['num_nodes_per_group']))

Y_true = np.load(os.path.join(PATHS['processed_dir'], 'Y_test_raw.npy'))
true_1d = Y_true[CASE, TIME, OFFSET:OFFSET + NX * NY]
true_2d = true_1d.reshape(NY, NX)

METHODS = [
    ('POD', os.path.join(PATHS['rom_dir'], 'Y_test_recon_pod.npy')),
    ('PCA', os.path.join(PATHS['rom_dir'], 'Y_test_recon_pca.npy')),
    ('DMD', os.path.join(PATHS['rom_dir'], 'Y_test_recon_dmd.npy')),
    ('VAE', os.path.join(PATHS['rom_dir'], 'Y_test_recon_vae.npy')),
]

# ── 4 行 × 3 列大图 ──
fig, axes = plt.subplots(4, 3, figsize=(18, 20))

kw = dict(origin='lower', interpolation='bilinear', extent=[0, 80, 0, 80])

for row, (method_name, recon_path) in enumerate(METHODS):
    Y_pred = np.load(recon_path)
    pred_1d = Y_pred[CASE, TIME, OFFSET:OFFSET + NX * NY]
    pred_2d = pred_1d.reshape(NY, NX)
    error_2d = np.abs(true_2d - pred_2d)

    vmin = min(true_2d.min(), pred_2d.min())
    vmax = max(true_2d.max(), pred_2d.max())

    for col, (data, title, cmap) in enumerate([
        (true_2d, '高保真参考场', 'viridis'),
        (pred_2d, f'{method_name} 重构场', 'viridis'),
        (error_2d, '绝对误差', 'inferno'),
    ]):
        ax = axes[row, col]
        v = (vmin, vmax) if col < 2 else None
        im = ax.imshow(data, cmap=cmap,
                       vmin=v[0] if v else None, vmax=v[1] if v else None, **kw)
        if row == 0:
            ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel('x (cm)', fontsize=16)
        ax.set_ylabel('y (cm)', fontsize=16)
        ax.tick_params(labelsize=14)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=12)

    # 行首标注方法名
    axes[row, 0].set_ylabel(f'{method_name}\ny (cm)', fontsize=20, fontweight='bold')

plt.tight_layout(pad=1.5)
out_path = 'fig/rom_comparison/all_methods_comparison.svg'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Max errors — POD: {np.abs(true_2d - np.load(METHODS[0][1])[CASE,TIME,OFFSET:OFFSET+NX*NY].reshape(NY,NX)).max():.2e} | PCA: {np.abs(true_2d - np.load(METHODS[1][1])[CASE,TIME,OFFSET:OFFSET+NX*NY].reshape(NY,NX)).max():.2e} | DMD: {np.abs(true_2d - np.load(METHODS[2][1])[CASE,TIME,OFFSET:OFFSET+NX*NY].reshape(NY,NX)).max():.2e} | VAE: {np.abs(true_2d - np.load(METHODS[3][1])[CASE,TIME,OFFSET:OFFSET+NX*NY].reshape(NY,NX)).max():.2e}')
print(f'Saved: {out_path}')
