# 图片占位符，待后续替换为实际图片内容。
import matplotlib.pyplot as plt

# 设置字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 创建画布，尺寸设置为适合插入Word的常规长宽比 (6x4)
fig, ax = plt.subplots(figsize=(6, 4))

# 设置极度显眼的背景颜色（亮黄色）
bg_color = '#F1C40F'
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# 隐藏坐标轴边框和刻度
ax.axis('off')

# 绘制经典的占位图交叉对角线
line_color = '#2C3E50'
ax.plot([0, 1], [0, 1], color=line_color, linewidth=2, alpha=0.2, transform=ax.transAxes)
ax.plot([0, 1], [1, 0], color=line_color, linewidth=2, alpha=0.2, transform=ax.transAxes)

# 添加居中的粗体大字提示
ax.text(0.5, 0.5, '图 片 占 位 符\n[ 待后续补充 ]', 
        fontsize=28, fontweight='bold', color=line_color,
        ha='center', va='center', transform=ax.transAxes,
        bbox=dict(facecolor=bg_color, edgecolor='none', alpha=0.8)) # 添加一点背景遮罩让字更清晰

# 保存并输出
plt.savefig('fig/fig_placeholder.svg', bbox_inches='tight', facecolor=fig.get_facecolor())
