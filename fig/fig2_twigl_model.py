# TWIGL 基准题四分之一堆芯几何模型：区域划分、边界条件与尺寸标注。
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# ── 加载稳态通量场作为背景 ──
data_path = '2D-PINN/data-processed/Y_val_raw.npy'
if os.path.exists(data_path):
    Y = np.load(data_path)
    # 快群 t=0, reshape 为 20×20, 上下翻转使堆芯中心对齐左上角
    flux = np.flipud(Y[0, 0, :400].reshape(20, 20))
else:
    flux = None

# ── 配色 ──
C_EDGE = '#263238'
C_GRID = '#607D8B'
C_REG1 = '#F4B183'
C_REG2 = '#93C0D5'
C_REG3 = '#D0CEC7'
C_BC   = '#37474F'

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-5, 85)
ax.set_ylim(-5, 85)

# ── 0. 背景：稳态快中子通量场 ──
if flux is not None:
    ax.imshow(flux, extent=[0, 80, 0, 80], cmap='viridis', origin='upper',
              alpha=0.45, zorder=0, interpolation='bilinear')

# ── 1. 填充区域（半透明，露出背景通量轮廓） ──
alpha_fill = 0.42
# 区域 3: 左上角 + 右侧 + 底部
ax.add_patch(patches.Rectangle((56, 0), 24, 80, facecolor=C_REG3,
    edgecolor='none', alpha=alpha_fill, zorder=1))
ax.add_patch(patches.Rectangle((0, 0), 56, 24, facecolor=C_REG3,
    edgecolor='none', alpha=alpha_fill, zorder=1))
ax.add_patch(patches.Rectangle((0, 56), 24, 24, facecolor=C_REG3,
    edgecolor='none', alpha=alpha_fill, zorder=1))
# 区域 2
ax.add_patch(patches.Rectangle((24, 56), 32, 24, facecolor=C_REG2,
    edgecolor='none', alpha=alpha_fill, zorder=1))
ax.add_patch(patches.Rectangle((0, 24), 24, 32, facecolor=C_REG2,
    edgecolor='none', alpha=alpha_fill, zorder=1))
ax.add_patch(patches.Rectangle((24, 24), 32, 32, facecolor=C_REG2,
    edgecolor='none', alpha=alpha_fill, zorder=1))
# 区域 1
ax.add_patch(patches.Rectangle((24, 24), 32, 32, facecolor=C_REG1,
    edgecolor='none', alpha=alpha_fill, zorder=1))

# ── 2. 外边界框 ──
ax.add_patch(patches.Rectangle((0, 0), 80, 80, linewidth=1.8,
    edgecolor=C_EDGE, facecolor='none', zorder=3))

# ── 3. 内部区域分割线 ──
lw = 1.2
ax.plot([24, 24], [24, 80], color=C_GRID, linewidth=lw, zorder=3, linestyle='-')
ax.plot([56, 56], [24, 80], color=C_GRID, linewidth=lw, zorder=3, linestyle='-')
ax.plot([0, 56], [56, 56], color=C_GRID, linewidth=lw, zorder=3, linestyle='-')
ax.plot([0, 56], [24, 24], color=C_GRID, linewidth=lw, zorder=3, linestyle='-')

# ── 4. 区域标签 ──
FS = 20
ax.text(40, 40, '区域 1\n扰动种子区', fontsize=FS, ha='center', va='center',
        fontweight='bold', color=C_EDGE, zorder=5)
ax.text(40, 68, '区域 2\n非扰动种子区', fontsize=FS, ha='center', va='center',
        fontweight='bold', color=C_EDGE, zorder=5)
ax.text(12, 40, '区域 2\n非扰动种子区', fontsize=FS, ha='center', va='center',
        fontweight='bold', color=C_EDGE, zorder=5)
ax.text(12, 68, '区域 3\n增殖区', fontsize=FS, ha='center', va='center',
        fontweight='bold', color=C_EDGE, zorder=5)
ax.text(68, 68, '区域 3\n增殖区', fontsize=FS, ha='center', va='center',
        fontweight='bold', color=C_EDGE, zorder=5)
ax.text(12, 12, '区域 3\n增殖区', fontsize=FS, ha='center', va='center',
        fontweight='bold', color=C_EDGE, zorder=5)
ax.text(68, 12, '区域 3\n增殖区', fontsize=FS, ha='center', va='center',
        fontweight='bold', color=C_EDGE, zorder=5)

# ── 5. 尺寸标注 ──
TFS = 16
for x, label in [(0, '0'), (24, '24'), (56, '56'), (80, '80')]:
    ax.text(x, 80.8, label, fontsize=TFS, ha='center', va='bottom', color='#757575')
for y, label in [(80, '0'), (56, '24'), (24, '56'), (0, '80')]:
    ax.text(-2.5, y, label, fontsize=TFS, ha='right', va='center', color='#757575')

# X/Y 轴单位
ax.text(40, 83, 'cm', fontsize=TFS, ha='center', va='bottom', color='#9E9E9E')
ax.text(-4.5, 40, 'cm', fontsize=TFS, ha='center', va='center',
        color='#9E9E9E', rotation=90)

# ── 6. 边界条件 ──
MFS = 20
# 左
ax.text(-1.5, 40, r'$\partial\phi/\partial x = 0$', fontsize=MFS,
        ha='right', va='center', color=C_BC, rotation=90)
# 顶
ax.text(40, 81.5, r'$\partial\phi/\partial y = 0$', fontsize=MFS,
        ha='center', va='bottom', color=C_BC)
# 右
ax.text(82.5, 40, r'$\phi = 0$', fontsize=MFS,
        ha='left', va='center', color=C_BC, rotation=270)
# 底
ax.text(40, -1.5, r'$\phi = 0$', fontsize=MFS,
        ha='center', va='top', color=C_BC)

bfs = 16
bbox_opts = dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.75)
ax.text(1, 28, '反射边界', fontsize=bfs, ha='center', va='center',
        color=C_BC, rotation=90, bbox=bbox_opts, zorder=5)
ax.text(28, 79, '反射边界', fontsize=bfs, ha='center', va='center',
        color=C_BC, bbox=bbox_opts, zorder=5)
ax.text(79, 28, '真空边界', fontsize=bfs, ha='center', va='center',
        color=C_BC, rotation=270, bbox=bbox_opts, zorder=5)
ax.text(28, 1, '真空边界', fontsize=bfs, ha='center', va='center',
        color=C_BC, bbox=bbox_opts, zorder=5)

# ── 7. 堆芯中心标记 ──
ax.plot(0, 80, marker='+', color=C_EDGE, markersize=10,
        markeredgewidth=1.5, zorder=5)
ax.text(2.5, 78, '堆芯中心', fontsize=bfs, ha='left', va='top', color=C_EDGE)

# ── 8. 收尾 ──
ax.axis('off')
ax.set_aspect('equal')
plt.tight_layout(pad=0.2)
plt.savefig('fig/fig2_twigl_model.svg', bbox_inches='tight', facecolor='white')
print('Saved: fig/fig2_twigl_model.svg')
