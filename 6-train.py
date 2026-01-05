import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # 引入标准化工具

data = np.load('.\\data-gen\\pod_data.npz')
X_train_raw = data['X_train']
coeffs_train_raw = data['coeffs_train']
X_test_raw = data['X_test']
coeffs_test_raw = data['coeffs_test']

# 归一化输入 X 
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw) # 注意：测试集必须用训练集的参数归一化

# 归一化输出 Y 
scaler_Y = StandardScaler()
coeffs_train = scaler_Y.fit_transform(coeffs_train_raw)
coeffs_test = scaler_Y.transform(coeffs_test_raw)

print(f"原始系数均值: {np.mean(coeffs_train_raw):.2f}, 归一化后均值: {np.mean(coeffs_train):.2f}")
print(f"原始系数方差: {np.var(coeffs_train_raw):.2f}, 归一化后方差: {np.var(coeffs_train):.2f}")

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(coeffs_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(coeffs_test).float()

input_dim = X_train.shape[1]
output_dim = coeffs_train.shape[1]

# 模型定义
class POD_DNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(POD_DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

model = POD_DNN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 2000
train_losses = []

print(">>> 开始训练 (带归一化)...")
for epoch in range(epochs):
    model.train()
    prediction = model(X_train_tensor)
    loss = criterion(prediction, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 保存归一化器 (Scaler)
# 预测时，需要把神经网络预测出的归一化数值还原回真实物理数值
import joblib
joblib.dump(scaler_X, '.\\data-gen\\scaler_X.pkl')
joblib.dump(scaler_Y, '.\\data-gen\\scaler_Y.pkl')
print(">>> 模型与 Scaler 已保存")

plt.plot(train_losses)
plt.yscale('log')
plt.title('Normalized Training Loss')
plt.show()