# POD 快照矩阵构建流程：从三维数据张量到二维快照矩阵的空间展开变换。
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_centered_snapshot_logic():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')

    cy = 3.5  # 垂直中心

    C_TENSOR  = '#93C0D5'
    C_T_EDGE  = '#4682B4'
    C_MATRIX  = '#C8E6C9'
    C_M_EDGE  = '#43A047'
    C_ARROW   = '#37474F'
    C_DIM     = '#757575'
    C_TEXT    = '#263238'

    # ── 1. 左侧：原始数据张量 Y ──
    tw, th = 2.8, 1.8
    bx, by = 0.6, cy - th/2 - 0.2

    for i in range(3):
        o = i * 0.3
        ax.add_patch(patches.FancyBboxPatch(
            (bx + o, by + o), tw, th,
            boxstyle='round,pad=0.08',
            linewidth=1.2, edgecolor=C_T_EDGE, facecolor=C_TENSOR, alpha=0.85, zorder=2))

    # 省略号
    ey = by + 0.6 + th + 0.1
#     ax.text(bx + 0.3 + tw/2, ey, '. . .', fontsize=20, ha='center', va='bottom', color=C_DIM)

    # 算例标签
    ax.text(bx, by - 0.1, '算例 1', fontsize=20, ha='left', va='top', color=C_TEXT)
    ax.text(bx + 0.6, ey + 0.15, '算例 K...', fontsize=20, ha='left', va='bottom', color=C_TEXT)

    # 张量名称（置于上方）
    ax.text(bx + tw/2 + 0.2, ey + 0.65, '原始数据张量 ($K \\!\\times\\! T \\!\\times\\! N$)',
            ha='center', fontsize=20, fontweight='bold', color=C_TEXT)

    # 维度箭头 — T
    ax.annotate('', xy=(bx - 0.5, by), xytext=(bx - 0.5, by + th),
                arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=0.8))
    ax.text(bx - 0.85, by + th/2, '$T$', va='center', ha='center', fontsize=20, color=C_DIM)

    # 维度箭头 — N
    ax.annotate('', xy=(bx, by - 0.5), xytext=(bx + tw, by - 0.5),
                arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=0.8))
    ax.text(bx + tw/2, by - 0.8, '$N$', ha='center', fontsize=20, color=C_DIM)

    # ── 2. 中间：空间展开箭头 ──
    ax_pad = bx + tw + 0.7
    mx_pad = 7.5

    # 箭头上方标注
    mid_x = (ax_pad + mx_pad) / 2
    ax.text(mid_x, cy + 0.55, '空间展开', ha='center', va='bottom',
            fontsize=20, fontweight='bold', color=C_TEXT)
    ax.text(mid_x, cy + 0.1, '$K \\!\\times\\! T \\!\\times\\! N \\;\\rightarrow\\; M \\!\\times\\! N$',
            ha='center', fontsize=16, color=C_DIM)

    # 箭头本体
    ax.annotate('', xy=(mx_pad - 0.1, cy - 0.15), xytext=(ax_pad + 0.1, cy - 0.15),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.0),
                zorder=3)

    # ── 3. 右侧：快照矩阵 X ──
    mw, mh = 2.4, 5.0
    mx, my = mx_pad + 0.2, cy - mh/2

    ax.add_patch(patches.FancyBboxPatch(
        (mx, my), mw, mh,
        boxstyle='round,pad=0.12',
        linewidth=1.5, edgecolor=C_M_EDGE, facecolor=C_MATRIX, alpha=0.7, zorder=2))

    # 内部虚线
    for i in range(1, 4):
        ly = my + mh - i * (mh / 5)
        ax.plot([mx, mx + mw], [ly, ly],
                color=C_M_EDGE, linestyle='--', linewidth=0.7, alpha=0.45, zorder=3)
        ax.text(mx + mw/2, ly - 0.2, f'算例 {i}', ha='center', fontsize=16,
                color=C_M_EDGE, zorder=4)

    # 矩阵名称
#     ax.text(mx + mw/2, my - 0.7, '快照矩阵 $X$  ($M \\!\\times\\! N$)',
#             ha='center', fontsize=20, fontweight='bold', color=C_TEXT)

    # 维度箭头 — M
    ax.annotate('', xy=(mx + mw + 0.45, my), xytext=(mx + mw + 0.45, my + mh),
                arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=0.8))
    ax.text(mx + mw + 0.8, cy, '$M = K \\!\\cdot\\! T$',
            rotation=270, va='center', fontsize=20, color=C_DIM)

    # 维度箭头 — N
    ax.annotate('', xy=(mx, my - 0.4), xytext=(mx + mw, my - 0.4),
                arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=0.8))
    ax.text(mx + mw/2, my - 0.7, '$N$', ha='center', va='top', fontsize=20, color=C_DIM)

    plt.tight_layout(pad=0.8)
    save_path = 'fig/fig5_snapshot_structure.svg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {save_path}')

if __name__ == '__main__':
    plot_centered_snapshot_logic()
