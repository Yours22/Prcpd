# 图 5-10: 三种退化模式对比 — no_amp_shape / no_decay / no_symlog 外推逐模态误差
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

modes = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
x = np.arange(len(modes))
width = 0.25

# Data from ablation logs (extrapolation per-mode)
datasets = [
    {
        'title': 'no_amp_shape: 剧烈分化',
        'color': '#E53935',
        'baseline': [4.04, 4.46, 6.46, 4.80],
        'ablation': [33.35, 47.64, 19.85, 59.68],
        'desc': 'Mode 1/3 可控\nMode 2/4 崩溃'
    },
    {
        'title': 'no_decay: 均匀退化',
        'color': '#F4A261',
        'baseline': [4.04, 4.46, 6.46, 4.80],
        'ablation': [12.04, 12.40, 11.62, 12.09],
        'desc': '全模态 ~12%\n无一幸免'
    },
    {
        'title': 'no_symlog: 选择性存活',
        'color': '#7B1FA2',
        'baseline': [4.04, 4.46, 6.46, 4.80],
        'ablation': [13.49, 99.66, 101.56, 103.63],
        'desc': '仅 Mode 1 存活\n其余 ~100%'
    },
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

for ax, ds in zip(axes, datasets):
    ax.bar(x - width/2, ds['baseline'], width, color='#1565C0', edgecolor='white',
           linewidth=0.6, label='基线 (no_cumsum)', zorder=3)
    ax.bar(x + width/2, ds['ablation'], width, color=ds['color'], edgecolor='white',
           linewidth=0.6, label='消融变体', zorder=3)

    # 100% line for no_symlog panel
    if 'symlog' in ds['title']:
        ax.axhline(y=100, color='#757575', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.text(3.3, 101, '100%', fontsize=14, color='#757575')

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=16)
    ax.set_title(ds['title'], fontsize=20, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    if ax == axes[0]:
        ax.set_ylabel('L2 相对误差 (%)', fontsize=20)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=16)

    # Description box
    ax.text(0.98, 0.94, ds['desc'], transform=ax.transAxes, fontsize=16,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FAFAFA',
                      edgecolor=ds['color'], lw=0.8, alpha=0.9))

fig.suptitle('三种退化模式的逐模态对比 — 外推集',
             fontsize=22, fontweight='bold', y=1.02)
plt.tight_layout(pad=1.0)
out = 'fig/fig_ch5_three_patterns.svg'
plt.savefig(out, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
