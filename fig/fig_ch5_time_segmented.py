# 图 5-1: 消融变体外推时间分段误差折线图
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

# Data: extrapolation time-segmented L2 errors
variants = ['基线 (no_cumsum)', 'full (含 cumsum)', 'no_amp_shape', 'no_decay', 'no_symlog']
early  = [0.28, 1.67, 1.39, 2.01, 8.92]
mid    = [2.98, 5.38, 16.09, 7.73, 11.24]
late   = [4.03, 9.12, 34.92, 11.65, 15.56]
overall = [1.24, 2.72, 10.26, 4.65, 10.91]

colors = ['#1565C0', '#78909C', '#E53935', '#F4A261', '#7B1FA2']
markers = ['o', 's', 'D', '^', 'v']
x_labels = ['前期 (t=0–30)', '中期 (t=30–70)', '后期 (t=70–100)']
x_pos = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(13, 7))

for i, (name, e, m, l, c, mk) in enumerate(
    zip(variants, early, mid, late, colors, markers)):
    vals = [e, m, l]
    ax.plot(x_pos, vals, color=c, marker=mk, linewidth=2.2, markersize=11,
            label=f'{name}  (整体 {overall[i]:.2f}%)', zorder=3)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylabel('L2 相对误差 (%)', fontsize=20)
ax.set_title('各消融变体外推集分时段 L2 相对误差', fontsize=22, fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9, ncol=1)
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_ylim(0, 38)

# Annotations for key observations
ax.annotate('基线: 平缓上升\n(0.28%→4.03%)',
            xy=(2, 4.03), xytext=(1.3, 10),
            fontsize=15, color=colors[0],
            arrowprops=dict(arrowstyle='->', color=colors[0], lw=1.2))

ax.annotate('no_amp_shape:\n中期跃升至 34.92%',
            xy=(2, 34.92), xytext=(1.3, 29),
            fontsize=15, color=colors[2],
            arrowprops=dict(arrowstyle='->', color=colors[2], lw=1.2))

ax.annotate('no_symlog: 前期即 8.92%\n(全体最高起点)',
            xy=(0, 8.92), xytext=(0.4, 16),
            fontsize=15, color=colors[4],
            arrowprops=dict(arrowstyle='->', color=colors[4], lw=1.2))

plt.tight_layout(pad=0.8)
out = 'fig/fig_ch5_time_segmented.svg'
plt.savefig(out, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
