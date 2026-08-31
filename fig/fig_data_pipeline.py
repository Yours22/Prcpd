# 数据清洗管道流程：三阶段筛选规则与剔除条件。
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

C_BOX   = '#CFD8DC'
C_ARROW = '#546E7A'
C_TEXT  = '#263238'
C_NUM   = '#37474F'

# ── 左列：阶段编号 ──
stages = [
    ('阶段 1', '完整性检查'),
    ('阶段 2', '数值稳定性检查'),
    ('阶段 3', '物理响应检查'),
]

# ── 右列：操作内容 ──
ops = [
    '剔除文件缺失或时间步不完整的算例',
    '剔除功率发散或出现负功率的算例',
    '剔除未发生有效物理响应的算例',
]

y_positions = [3.2, 2.0, 0.8]

for i, ((stage, check), op, y) in enumerate(zip(stages, ops, y_positions)):
    # 阶段框
    box_w, box_h = 2.0, 0.8
    ax.add_patch(patches.FancyBboxPatch((0.3, y - box_h/2), box_w, box_h,
        boxstyle='round,pad=0.1', facecolor=C_NUM, edgecolor='none', zorder=2))
    ax.text(0.3 + box_w/2, y, stage, fontsize=20, ha='center', va='center',
            color='white', fontweight='bold', zorder=3)
    ax.text(0.3 + box_w/2, y - 0.28, check, fontsize=16, ha='center', va='center',
            color='#B0BEC5', zorder=3)

    # 操作框
    op_w, op_h = 5.4, 0.8
    ax.add_patch(patches.FancyBboxPatch((3.0, y - op_h/2), op_w, op_h,
        boxstyle='round,pad=0.1', facecolor=C_BOX, edgecolor='#90A4AE',
        linewidth=0.8, zorder=2))
    ax.text(3.0 + op_w/2, y, op, fontsize=20, ha='center', va='center',
            color=C_TEXT, zorder=3)

    # 阶段间箭头
    if i < len(stages) - 1:
        next_y = y_positions[i + 1]
        ax.annotate('', xy=(0.3 + box_w/2, next_y + 0.4),
                    xytext=(0.3 + box_w/2, y - 0.4),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))

# ── 连接箭头：阶段 → 操作 ──
for y in y_positions:
    ax.annotate('', xy=(3.0, y), xytext=(0.3 + 2.0, y),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))

plt.tight_layout(pad=0.5)
plt.savefig('fig/fig_data_pipeline.svg', bbox_inches='tight', facecolor='white')
print('Saved: fig/fig_data_pipeline.svg')
