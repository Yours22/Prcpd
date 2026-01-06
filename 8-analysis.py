import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import r2_score

# ================= 配置 =================
DATA_DIR = '.\\data-gen'
INPUT_DIM = 27
R_MODES = 5  # 之前训练时保留的模态数

# ================= 1. 加载资源 =================
print(">>> Loading resources...")
# 加载数据
inputs = np.load(os.path.join(DATA_DIR, 'inputs.npy'))
outputs = np.load(os.path.join(DATA_DIR, 'outputs.npy'))
comps = np.load(os.path.join(DATA_DIR, 'pod_components.npz'))
Phi_r = comps['Phi_r']      # (320000, 5)
mean_vec = comps['mean_vec'] # (320000, 1)

# 真实的测试集系数 (从 npz 中恢复或者重新计算，这里为了方便直接取 npz)
# 注意这里重新切分测试集以确保索引对应
from sklearn.model_selection import train_test_split
_, X_test_raw, _, Y_test_raw = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

scaler_X = joblib.load(os.path.join(DATA_DIR, 'scaler_X.pkl'))
scaler_Y = joblib.load(os.path.join(DATA_DIR, 'scaler_Y.pkl'))

class POD_DNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(POD_DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = POD_DNN(INPUT_DIM, R_MODES)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'pod_dnn.pth')))
model.eval()


print(">>> Predicting and Reconstructing...")

X_test_norm = scaler_X.transform(X_test_raw)
with torch.no_grad():
    pred_coeffs_norm = model(torch.from_numpy(X_test_norm).float()).numpy()
    
pred_coeffs_real = scaler_Y.inverse_transform(pred_coeffs_norm)

# 获取真实系数 (为了对比模态精度)
# 需要重新投影: Coeffs = Phi^T * (U - Mean)
U_test_flat = Y_test_raw.reshape(Y_test_raw.shape[0], -1).T # (320000, 40)
U_test_centered = U_test_flat - mean_vec
true_coeffs_real = np.dot(Phi_r.T, U_test_centered).T # (40, 5)

# 重构物理场
# U_pred = Mean + Phi * Pred_Coeffs^T
U_pred_flat = mean_vec + np.dot(Phi_r, pred_coeffs_real.T) # (320000, 40)
U_pred_flat = U_pred_flat.T # (40, 320000)
U_true_flat = U_test_flat.T # (40, 320000)

# --- 图 1: 模态预测精度 (R2 Score) ---
# 检查每个模态，DNN 预测得准不准
print("\n[Analysis 1] Modal Accuracy (R2 Score):")
r2_values = []
for i in range(R_MODES):
    score = r2_score(true_coeffs_real[:, i], pred_coeffs_real[:, i])
    r2_values.append(score)
    print(f"  Mode {i+1}: R2 = {score:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# 画第1模态 (最重要) 的散点图
plt.scatter(true_coeffs_real[:, 0], pred_coeffs_real[:, 0], alpha=0.6, c='blue')
plt.plot([true_coeffs_real[:, 0].min(), true_coeffs_real[:, 0].max()], 
         [true_coeffs_real[:, 0].min(), true_coeffs_real[:, 0].max()], 'r--')
plt.xlabel('True Coefficient (Mode 1)')
plt.ylabel('Predicted Coefficient (Mode 1)')
plt.title(f'Mode 1 Correlation (R2={r2_values[0]:.4f})')
plt.grid(True)

plt.subplot(1, 2, 2)
# 画最后一个模态 (细节) 的散点图
last_idx = R_MODES - 1
plt.scatter(true_coeffs_real[:, last_idx], pred_coeffs_real[:, last_idx], alpha=0.6, c='green')
plt.plot([true_coeffs_real[:, last_idx].min(), true_coeffs_real[:, last_idx].max()], 
         [true_coeffs_real[:, last_idx].min(), true_coeffs_real[:, last_idx].max()], 'r--')
plt.xlabel(f'True Coefficient (Mode {R_MODES})')
plt.ylabel(f'Predicted Coefficient (Mode {R_MODES})')
plt.title(f'Mode {R_MODES} Correlation (R2={r2_values[last_idx]:.4f})')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Mode {R_MODES} Correlation (R2={r2_values[last_idx]:.4f}).png')

# --- 图 2: 关键安全参数误差 ---
# 计算每个样本的峰值功率和相对误差
peak_power_true = np.max(U_true_flat, axis=1) # (40,)
peak_power_pred = np.max(U_pred_flat, axis=1) # (40,)
rel_errors = (peak_power_pred - peak_power_true) / peak_power_true * 100

print("\n[Analysis 2] Safety Parameter Errors:")
print(f"  Max Relative Error: {np.max(np.abs(rel_errors)):.2f}%")
print(f"  Mean Relative Error: {np.mean(np.abs(rel_errors)):.2f}%")

# 找出误差最大的样本
worst_sample_idx = np.argmax(np.abs(rel_errors))
print(f"  Worst Sample Index: {worst_sample_idx} (Error: {rel_errors[worst_sample_idx]:.2f}%)")

plt.figure(figsize=(6, 4))
plt.hist(rel_errors, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('Peak Power Relative Error (%)')
plt.ylabel('Number of Samples')
plt.title('Error Distribution of Peak Power')
plt.grid(True)
plt.savefig('Error Distribution of Peak Power')

# --- 图 3: 最差样本的时间演变诊断 ---
# 还原该样本的时间序列 (假设 T=100)
T = 100
K = 3200
# Reshape: (Time, Space)
worst_true_3d = U_true_flat[worst_sample_idx].reshape(T, K)
worst_pred_3d = U_pred_flat[worst_sample_idx].reshape(T, K)

# 计算随时间变化的全堆峰值
worst_peak_t_true = np.max(worst_true_3d, axis=1)
worst_peak_t_pred = np.max(worst_pred_3d, axis=1)
time_axis = np.linspace(0, 0.5, T)

plt.figure(figsize=(10, 6))
plt.plot(time_axis, worst_peak_t_true, 'k-', linewidth=2, label='True (FEMFFUSION)')
plt.plot(time_axis, worst_peak_t_pred, 'r--', linewidth=2, label='Predicted (POD-DNN)')

# 计算局部误差
diff = worst_peak_t_pred - worst_peak_t_true
plt.fill_between(time_axis, worst_peak_t_true, worst_peak_t_pred, color='red', alpha=0.2, label='Error Area')

plt.xlabel('Time (s)')
plt.ylabel('Max Core Power (W/cm^3)')
plt.title(f'Transient Evolution of Worst Sample (Idx: {worst_sample_idx})')
plt.legend()
plt.grid(True)
plt.savefig(f'Transient Evolution of Worst Sample (Idx: {worst_sample_idx})')