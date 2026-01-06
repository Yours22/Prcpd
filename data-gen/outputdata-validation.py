import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 数据加载与检查 ---
input_file = './data-gen/inputs.npy'
output_file = './data-gen/outputs.npy'

# 检查文件 (保持原逻辑)
if not os.path.exists(input_file) or not os.path.exists(output_file):
    print(f"错误: 找不到数据文件。请确保 {input_file} 和 {output_file} 存在。")
    # 生成假数据用于演示 (格式与你描述的一致: 100步, 3200点)
    # 假设是一个指数增长的功率
    N_samples = 20
    inputdata = np.random.rand(N_samples, 27) 
    t = np.linspace(0, 0.5, 100)
    # 模拟功率随时间上升 (TWIGL Ramp 行为)
    base_trend = np.exp(1.5 * t).reshape(100, 1) 
    spatial_dist = np.random.rand(1, 3200)
    outputdata = np.zeros((N_samples, 100, 3200))
    for i in range(N_samples):
        outputdata[i] = base_trend * spatial_dist * (1 + 0.1*np.random.rand())
else:
    inputdata = np.load(input_file)
    outputdata = np.load(output_file)

# --- 2. 绘图设置 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sample_idx = 10  # 选择第 10 个样本
plt.subplots_adjust(hspace=0.3, wspace=0.3) # 调整子图间距

# === 第一块图：输入特征矩阵 (保持不变) ===
input_matrix = inputdata[sample_idx].reshape(3, 9)
im1 = axes[0, 0].imshow(input_matrix, cmap='viridis', aspect='auto')
axes[0, 0].set_title(f'Sample {sample_idx} Input Cross-sections')
axes[0, 0].set_ylabel('Material ID')
axes[0, 0].set_xlabel('XS Parameters')
plt.colorbar(im1, ax=axes[0, 0])

# =======================================================
# === 第二块图：整体功率随时间变化 (重点修改部分) ===
# =======================================================
sample_out = outputdata[sample_idx] # Shape: (100, 3200)

# 1. 计算总通量 (Total Flux) 作为功率的代理量
# 对 Axis 1 (空间点+能群) 求和，得到每一时刻的全堆总中子数
total_power_raw = np.sum(sample_out, axis=1) 

# 2. 归一化 (Normalize)
# P_rel = P(t) / P(0)
power_initial = total_power_raw[0]
if power_initial == 0:
    relative_power = total_power_raw # 避免除以0
else:
    relative_power = total_power_raw / power_initial

# 3. 构建时间轴 (假设总时长0.5s，共100步)
time_axis = np.linspace(0, 0.5, len(relative_power))

# 4. 绘图
axes[0, 1].plot(time_axis, relative_power, color='#d62728', linewidth=2.5, label='Relative Power')
axes[0, 1].set_title(f'Sample {sample_idx}: Total Power Evolution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Relative Power P(t)/P(0)')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)
axes[0, 1].legend()

# 标注最终功率值
final_p = relative_power[-1]
axes[0, 1].text(time_axis[-1], final_p, f' {final_p:.2f}', va='center', fontsize=10, color='#d62728')


# === 第三块图：对角线分布 (保持不变) ===
time_slice = 75 # 选择观察的时刻
flux_snapshot = sample_out[time_slice] 
grid_size = 40 # 假设 40x40 = 1600
fast_flux_2d = flux_snapshot[:1600].reshape(grid_size, grid_size)
thermal_flux_2d = flux_snapshot[1600:].reshape(grid_size, grid_size)

# 提取副对角线
fast_diag = [fast_flux_2d[grid_size - 1 - i, i] for i in range(grid_size)]
thermal_diag = [thermal_flux_2d[grid_size - 1 - i, i] for i in range(grid_size)]
x_diag = np.arange(grid_size)

axes[1, 0].plot(x_diag, fast_diag, label='Fast Group', color='tab:blue', alpha=0.8)
axes[1, 0].plot(x_diag, thermal_diag, label='Thermal Group', color='tab:orange', alpha=0.8)
axes[1, 0].set_title(f'Flux Profile (Diagonal) at t={time_axis[time_slice]:.3f}s')
axes[1, 0].set_xlabel('Diagonal Position')
axes[1, 0].set_ylabel('Flux')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# === 第四块图：热群通量分布热力图 (修改为热群通常更有代表性) ===
im4 = axes[1, 1].imshow(thermal_flux_2d, cmap='inferno', origin='lower') # origin='lower' 符合物理坐标习惯
axes[1, 1].set_title(f'Thermal Flux Map at t={time_axis[time_slice]:.3f}s')
plt.colorbar(im4, ax=axes[1, 1], label='Flux Magnitude')

# --- 保存与显示 ---
plt.tight_layout()
save_path = 'power_analysis_plot.png'
plt.savefig(save_path, dpi=300)
print(f"绘图完成！图片已保存至 {save_path}")
print(f"样本 {sample_idx} 的最终相对功率: {final_p:.4f}")