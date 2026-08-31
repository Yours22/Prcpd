# 数据集参数筛选前后对比：扰动截止时间与斜率分布的散点图。
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

df_orig = pd.read_csv("./fig/dataset_parameters.csv")
df_clean = pd.read_csv("./fig/dataset_parameters_cleaned.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# Color by group_changing, marker by material_changing
colors = {1: '#E41A1C', 2: '#377EB8'}  # Set1 colors
markers = {1: 'o', 3: 's'}
labels_g = {1: '快群', 2: '热群'}
labels_m = {1: '堆芯区', 3: '控制棒区'}

def plot_scatter(ax, df, label_text):
    for gc in [1, 2]:
        for mc in [1, 3]:
            mask = (df['group_changing'] == gc) & (df['material_changing'] == mc)
            subset = df[mask]
            if len(subset) > 0:
                ax.scatter(subset['cut_time'], subset['slope_up'],
                           c=colors[gc], marker=markers[mc], alpha=0.65,
                           edgecolors='none', s=40,
                           label=f'{labels_g[gc]}, {labels_m[mc]}')
    ax.set_xlabel('截止时间 $t_{cut}$ (s)', fontsize=20)
    ax.set_ylabel('扰动斜率 $s_{up}$ (s$^{-1}$)', fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.text(0.5, -0.14, label_text, transform=ax.transAxes,
            fontsize=20, ha='center', va='top')

plot_scatter(axes[0], df_orig, '(a)')
plot_scatter(axes[1], df_clean, '(b)')

# 合并图例
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels, title='扰动能群, 扰动区域',
               loc='lower right', framealpha=0.9, fontsize=20, title_fontsize=20)

plt.tight_layout(pad=2.5)
plt.savefig('fig/fig4_parameter_screening.svg', bbox_inches='tight', pad_inches=0.3)
print("Saved: fig4_parameter_screening.svg")
