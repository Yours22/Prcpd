import matplotlib.pyplot as plt
import numpy as np

# 1. 设置学术绘图字体（兼容 Windows 和 Mac，避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# 2. 录入你提取的数据（已按年份重新排序）
years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
counts = np.array([1, 1, 1, 2, 2, 1, 4, 6, 13, 14, 22, 27, 44, 65, 67])

# 3. 创建画布：尺寸设为 10x6，DPI 设为 300（满足毕业论文打印标准）
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 4. 绘制柱状图与折线图结合（学术图表常用表现形式）
# 使用经典的学术蓝（SteelBlue）绘制柱子
bars = ax.bar(years, counts, color='#4682B4', alpha=0.85, width=0.6, edgecolor='black', linewidth=0.8, label='年度发文量')
# 叠加一条红色趋势折线，更直观地体现“爆发”态势
ax.plot(years, counts, color='#C0392B', marker='o', linestyle='-', linewidth=2, markersize=5, label='增长趋势')

# 5. 在每根柱子上方添加具体的数值标签
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 垂直向上偏移 3 个像素
                textcoords="offset points",
                ha='center', va='bottom', fontsize=20)

# 6. 【核心点睛】在 2018 年位置添加学术标注
ax.annotate('非侵入式代理模型\n(POD-NN/LSTM) 开始引入',
            xy=(2018, 13), xytext=(2011, 40),
            arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=7),
            fontsize=20, fontweight='bold', color='#333333',
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8F9F9", ec="#BDC3C7", lw=1.5))

# 7. 设置标题和坐标轴标签
ax.set_xlabel('出版年份', fontsize=20, fontweight='bold', labelpad=8)
ax.set_ylabel('文献发文量', fontsize=20, fontweight='bold', labelpad=8)
# ax.set_title('核工程领域数据驱动降阶模型研究趋势 (2010-2024)', fontsize=15, fontweight='bold', pad=15)

# 8. 图表细节美化
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45) # 年份倾斜45度防止重叠
ax.spines['top'].set_visible(False)    # 隐藏上方边框
ax.spines['right'].set_visible(False)  # 隐藏右侧边框
ax.grid(axis='y', linestyle='--', alpha=0.5) # 添加水平辅助线

# 9. 添加图例
ax.legend(loc='upper left', frameon=True, fontsize=20)

# 10. 自动紧凑布局，并保存为高清图片
plt.tight_layout()
plt.savefig('fig/fig1_publication_trend.svg', bbox_inches='tight')
print("Saved: fig1_publication_trend.svg")