# 图 5-7: 消融实验贡献度柱状图 — 各变体验证集/外推集 L2 误差对比
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

variants = ['基线\n(no_cumsum)', 'full\n(含 cumsum)', 'no_amp_shape', 'no_decay', 'no_symlog']
val_l2   = [0.18, 0.70, 0.77, 0.79, 8.57]
ext_l2   = [1.24, 2.72, 10.26, 4.65, 10.91]
multipliers = [1.0, 2.2, 8.3, 3.8, 8.8]  # ext / baseline ext

x = np.arange(len(variants))
width = 0.32
colors_val = '#1565C0'
colors_ext = '#E53935'

fig, ax = plt.subplots(figsize=(13, 7))

bars1 = ax.bar(x - width/2, val_l2, width, color=colors_val, edgecolor='white',
               linewidth=0.8, label='验证集 L2', zorder=3)
bars2 = ax.bar(x + width/2, ext_l2, width, color=colors_ext, edgecolor='white',
               linewidth=0.8, label='外推集 L2', zorder=3)

# Baseline reference line
ax.axhline(y=1.24, color='#757575', linestyle='--', linewidth=1.0, alpha=0.6)
ax.text(0.15, 1.5, '基线外推 1.24%', fontsize=16, color='#757575')

# Annotate bars with values
for bar, val in zip(bars1, val_l2):
    y_pos = bar.get_height() + 0.15
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}%',
            ha='center', fontsize=16, fontweight='bold', color=colors_val)

for i, (bar, val, mult) in enumerate(zip(bars2, ext_l2, multipliers)):
    y_pos = bar.get_height() + 0.25
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}%',
            ha='center', fontsize=16, fontweight='bold', color=colors_ext)
    if i >= 2:  # annotate multiplier for ablations
        ax.text(bar.get_x() + bar.get_width()/2, y_pos + 0.65,
                f'×{mult:.1f}', ha='center', fontsize=16, fontweight='bold', color='#E65100')

ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=18)
ax.set_ylabel('L2 相对误差 (%)', fontsize=20)
ax.set_title('消融实验贡献度 — 验证集/外推集 L2 相对误差', fontsize=22, fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, axis='y', linestyle=':', alpha=0.4)

# Separator line between "baseline/full" and "ablations"
ax.axvline(x=1.5, color='#BDBDBD', linestyle='-', linewidth=1.2, alpha=0.6)
ax.text(1.5, 11.5, ' ← 对比参照    |    消融变体 →',
        fontsize=16, color='#9E9E9E', ha='center')

# Annotation box
ax.text(0.02, 0.96, '橙色数值 = 相对基线的外推误差放大倍数\n'
        'SymLog 和双流分解是不可移除的结构性组件',
        transform=ax.transAxes, fontsize=16, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor='#F9A825', lw=0.8))

plt.tight_layout(pad=0.8)
out = 'fig/fig_ch5_ablation_bars.svg'
plt.savefig(out, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
