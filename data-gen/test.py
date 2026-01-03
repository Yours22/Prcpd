import numpy as np
import matplotlib.pyplot as plt
import os

input = './data-gen/inputs.npy'
output = './data-gen/outputs.npy'

inputdata = np.load(input)
outputdata = np.load(output)

print(f"input.npy形状: {inputdata.shape}")
print(f"output.npy形状: {outputdata.shape}")

print("\n--- 1. 数值检查 ---")
if np.isnan(inputdata).any() or np.isnan(outputdata).any():
    print("警告: 数据中发现 NaN (空值)！请检查转换逻辑。")
else:
    print("数据中无 NaN。")

if (inputdata < 0).any() or (outputdata < 0).any():
    print("警告: 数据中发现负数！物理量(截面/通量)不应为负。")
    print(f"Output Min Value: {outputdata.min()}")
else:
    print("数据全部非负。")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sample_idx = 0
input_matrix = inputdata[sample_idx].reshape(3, 9)
im1 = axes[0, 0].imshow(input_matrix, cmap='viridis', aspect='auto')
axes[0, 0].set_title(f'Sample {sample_idx} Input Features')
axes[0, 0].set_xlabel('Cross-section data')  
axes[0, 0].set_ylabel('Material Index')
plt.colorbar(im1, ax=axes[0, 0])

mid = (input_matrix.max() + input_matrix.min()) / 2
for (i, j), val in np.ndenumerate(input_matrix):
	color = 'white' if val < mid else 'black'
	axes[0, 0].text(j, i, f"{val:.3f}", ha='center', va='center', color=color, fontsize=8)

sample_out = outputdata[sample_idx] # Shape: (100, 3200)
avg_flux_time = np.mean(sample_out, axis=1) # Shape: (100,)
axes[0, 1].plot(avg_flux_time, 'b.-', label='Avg Flux')
axes[0, 1].set_title(f'Sample {sample_idx} Output: Time Consistency')
axes[0, 1].set_xlabel('Time Step (0-99)')
axes[0, 1].set_ylabel('Average Neutron Flux')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)


time_slice = 10
flux_snapshot = sample_out[time_slice] # Shape: (3200,)
fast_flux = flux_snapshot[:1600]
thermal_flux = flux_snapshot[1600:]

# 为了显示方便，只画前100个空间点的对比
axes[1, 0].plot(fast_flux[:100], label='Fast Flux', alpha=0.7)
axes[1, 0].plot(thermal_flux[:100], label='Thermal Flux', alpha=0.7)
axes[1, 0].set_title(f'Sample {sample_idx} (t={time_slice}): Fast vs Thermal (First 100 pts)')
axes[1, 0].set_xlabel('Spatial Index')
axes[1, 0].legend()
# 验证点：两者趋势应有相关性，且数值范围符合该堆型预期

# --- 图 4: 空间分布热力图 (假设1600是40x40网格) ---



grid_size = int(np.sqrt(1600)) # 假设是正方形网格
fast_flux_map = fast_flux.reshape((grid_size, grid_size),order='C')
im4 = axes[1, 1].imshow(fast_flux_map, cmap='inferno')
axes[1, 1].set_title(f'Sample {sample_idx} (t={time_slice}): Fast Flux Map ({grid_size}x{grid_size})')
plt.colorbar(im4, ax=axes[1, 1])
    
plt.tight_layout()
plt.savefig('data_validation_plots.png')