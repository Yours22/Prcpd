import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class TransientSequenceDataset(Dataset):
    def __init__(self, X_npy_path, A_npy_path, X_stats=None, A_stats=None, decay_lambdas=[0.1, 1.0, 10.0], dt=0.005):
        raw_X = np.load(X_npy_path)
        self.A = torch.tensor(np.load(A_npy_path), dtype=torch.float32)
        
        num_cases, num_steps, _ = raw_X.shape
        p_t = raw_X[:, :, 2]
        
        # 指数衰减积分 (物理记忆)
        decay_features = []
        # ... 前面提取 p_t 和计算 decay_features 的代码保持不变 ...
        for lam in decay_lambdas:
            integral = np.zeros((num_cases, num_steps))
            for t in range(1, num_steps):
                integral[:, t] = integral[:, t-1] * np.exp(-lam * dt) + p_t[:, t] * dt
            decay_features.append(integral[:, :, np.newaxis])
            
        # ================= 新增：加上简单累积积分特征 =================
        # np.cumsum 计算累加，乘以 dt 就是物理意义上的简单积分
        simple_integral = np.cumsum(p_t, axis=1) * dt
        decay_features.append(simple_integral[:, :, np.newaxis])
        # ==============================================================
        
        # 拼接原始特征与所有积分特征 -> 此时变成了 8 维！
        combined_X = np.concatenate([raw_X] + decay_features, axis=-1)
        self.X = torch.tensor(combined_X, dtype=torch.float32)

        # ... 后面的 SymLog 转换和标准化代码完全保持不变 ...
        # 【核心保留】：对 A 进行 SymLog 转换，防止网络饱和
        self.A = torch.sign(self.A) * torch.log1p(torch.abs(self.A))

        # 标准化 X
        if X_stats is None:
            self.X_mean = self.X.mean(dim=(0, 1), keepdim=True)
            self.X_std = self.X.std(dim=(0, 1), keepdim=True)
            self.X_std[self.X_std == 0] = 1e-5 
        else:
            self.X_mean, self.X_std = X_stats
        self.X = (self.X - self.X_mean) / self.X_std

        # 标准化 A (此时 A 已在 SymLog 空间)
        if A_stats is None:
            self.A_mean = self.A.mean(dim=(0, 1), keepdim=True)
            self.A_std = self.A.std(dim=(0, 1), keepdim=True)
            self.A_std[self.A_std == 0] = 1e-5
        else:
            self.A_mean, self.A_std = A_stats
        self.A = (self.A - self.A_mean) / self.A_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx]


class POD_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(POD_LSTM, self).__init__()
        
        self.lstm_macro = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_micro = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc_mode1_delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.fc_mode1_residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),  
            nn.Linear(hidden_dim // 2, 1)
        )

        self.fc_higher_modes = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim - 1)
        )

    def forward(self, x):
        # --- 宏观干道提取特征 ---
        lstm_out_macro, _ = self.lstm_macro(x) 
        
        # 1. 计算宏观趋势 (积分器)
        delta_m1 = self.fc_mode1_delta(lstm_out_macro)
        pred_m1_trend = torch.cumsum(delta_m1, dim=1)
        
        # ================= 新增：计算残差并硬拼接 =================
        # 直接由 LSTM 状态映射出瞬时残差 (不经过 cumsum！)
        m1_residual = self.fc_mode1_residual(lstm_out_macro)
        
        # 最终的宏观振幅 = 苦力趋势 + 狙击手残差
        pred_m1_scaled = pred_m1_trend + m1_residual
        # ==========================================================

        # --- 微观干道 ---
        lstm_out_micro, _ = self.lstm_micro(x) 
        pred_R = self.fc_higher_modes(lstm_out_micro)  
        
        return torch.cat([pred_m1_scaled, pred_R], dim=2)