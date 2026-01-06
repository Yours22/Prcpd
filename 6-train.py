import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import joblib
import os

DATA_DIR = '.\\data-gen'
INPUTS_FILE = os.path.join(DATA_DIR, 'inputs.npy')
OUTPUTS_FILE = os.path.join(DATA_DIR, 'outputs.npy')
MODEL_PATH = os.path.join(DATA_DIR, 'pod_dnn.pth')

TARGET_ENERGY = 0.999  # 能量截断阈值
EPOCHS = 3000          # 训练轮数
LR = 0.001             # 学习率

# 数据加载与预处理 需要去均值
print(">>> 1.数据处理中")

inputs = np.load(INPUTS_FILE)    # (200, 71)
outputs = np.load(OUTPUTS_FILE)  # (200, 100, 3200)

# 划分数据集 (160 Train, 40 Test)
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42
)

# 构建快照矩阵 (CTSR Format) Shape: (Features=320000, Samples=160)
U_train = Y_train_raw.reshape(Y_train_raw.shape[0], -1).T
print(f"快照矩阵均值: {U_train.shape}")

# 计算并移除均值场
mean_vec = np.mean(U_train, axis=1, keepdims=True) # (320000, 1)
U_train_centered = U_train - mean_vec

# SVD
print("2.进行SVD分界")
U_basis, Sigma, VT = randomized_svd(U_train_centered, n_components=160, random_state=42)

# 能量截断
energy = Sigma**2
cumulative_energy = np.cumsum(energy) / np.sum(energy)
r = np.searchsorted(cumulative_energy, TARGET_ENERGY) + 1
print(f" 目标能量{TARGET_ENERGY*100}% -> 保留模态数 r = {r}")

Phi_r = U_basis[:, :r]

# 计算投影系数 (这是神经网络的真值)，需要注意投影的是去均值后的数据
coeffs_train_raw = np.dot(Phi_r.T, U_train_centered).T # (160, r)

U_test = Y_test_raw.reshape(Y_test_raw.shape[0], -1).T
U_test_centered = U_test - mean_vec # 减去训练集的均值
coeffs_test_raw = np.dot(Phi_r.T, U_test_centered).T   # (40, r)

print("3.数据归一化")

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

scaler_Y = StandardScaler()
coeffs_train = scaler_Y.fit_transform(coeffs_train_raw)
coeffs_test = scaler_Y.transform(coeffs_test_raw)

joblib.dump(scaler_X, os.path.join(DATA_DIR, 'scaler_X.pkl'))
joblib.dump(scaler_Y, os.path.join(DATA_DIR, 'scaler_Y.pkl'))
np.savez(os.path.join(DATA_DIR, 'pod_components.npz'), 
         Phi_r=Phi_r, mean_vec=mean_vec, coeffs_test=coeffs_test_raw)

print("4.DNN网络训练")

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

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(coeffs_train).float()
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(coeffs_test).float()

model = POD_DNN(X_train.shape[1], r)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_hist, test_hist = [], []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()
    
    train_hist.append(loss.item())
    
    if (epoch+1) % 500 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_t), y_test_t)
        print(f"    Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

torch.save(model.state_dict(), MODEL_PATH)
print("模型已保存")

print("5.数据验证")

model.eval()
with torch.no_grad():
    # 预测归一化系数
    pred_norm = model(X_test_t).numpy()
    # 逆归一化 (还原为物理系数)
    pred_real = scaler_Y.inverse_transform(pred_norm)

# 诊断：打印第一个样本的系数对比
print("\n--- Diagnostic: Sample 0 Coefficients ---")
print(f"True: {coeffs_test_raw[0, :4]}") # 只看前4个
print(f"Pred: {pred_real[0, :4]}")

# 重构物理场
# 公式: U = Mean + Phi * Coeffs^T
U_pred_flat = mean_vec + np.dot(Phi_r, pred_real.T) # (320000, 40)
U_pred_flat = U_pred_flat.T # (40, 320000)

U_true_flat = Y_test_raw.reshape(40, -1) # 直接对比原始 FEMFFUSION 数据

# 计算误差 (排除 HZP 低功率区域的噪音)
# 只有当真实功率 > 最大功率的 1% 时才计算相对误差
max_power = np.max(U_true_flat)
mask = U_true_flat > (0.01 * max_power)

abs_error = np.abs(U_pred_flat - U_true_flat)
rel_error = np.zeros_like(abs_error)
rel_error[mask] = (abs_error[mask] / U_true_flat[mask]) * 100

rmse = np.sqrt(np.mean(abs_error**2))
max_rel = np.max(rel_error) # 只看有效区域的最大相对误差

print(f"\n========================================")
print(f"Final Validation Results:")
print(f"  RMSE: {rmse:.4f}")
print(f"  Max Relative Error (Active Region): {max_rel:.2f}%")
print(f"========================================")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_hist)
plt.title('Loss Curve')
plt.yscale('log')

plt.subplot(1,2,2)
# 画第一个样本的峰值时间步功率对比 (假设峰值在中间)
# 简单取个切片看吻合度
sample_idx = 0
plt.plot(U_true_flat[sample_idx, ::1000], label='True', alpha=0.7)
plt.plot(U_pred_flat[sample_idx, ::1000], label='Pred', linestyle='--')
plt.title('Snapshot Reconstruction (Sample 0)')
plt.legend()

plt.tight_layout()
plt.savefig('TrainResults.png')