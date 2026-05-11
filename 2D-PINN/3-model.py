import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)
_ABLATION = _cfg.get('ablation', {})

P_T_IDX = 4


def symlog_inverse(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def get_ablation():
    return _ABLATION.get('mode', 'full')


def get_output_root():
    """消融输出根目录"""
    label = _ABLATION.get('label', get_ablation())
    return f"2D-PINN/ablation/{label}"


class TransientSequenceDataset(Dataset):
    def __init__(self, X_npy_path, A_npy_path, X_stats=None, A_stats=None,
                 decay_lambdas=None, dt=0.005, ablation_mode=None):
        if decay_lambdas is None:
            decay_lambdas = [0.1, 1.0, 10.0]
        if ablation_mode is None:
            ablation_mode = get_ablation()

        no_decay = (ablation_mode == 'no_decay')
        no_symlog = (ablation_mode == 'no_symlog')

        raw_X = np.load(X_npy_path)
        self.A = torch.tensor(np.load(A_npy_path), dtype=torch.float32)
        num_cases, num_steps, _ = raw_X.shape

        if no_decay:
            combined_X = raw_X
        else:
            p_t = raw_X[:, :, P_T_IDX]
            decay_features = []
            for lam in decay_lambdas:
                integral = np.zeros((num_cases, num_steps))
                for t in range(1, num_steps):
                    integral[:, t] = integral[:, t - 1] * np.exp(-lam * dt) + p_t[:, t] * dt
                decay_features.append(integral[:, :, np.newaxis])
            simple_integral = np.cumsum(p_t, axis=1) * dt
            decay_features.append(simple_integral[:, :, np.newaxis])
            combined_X = np.concatenate([raw_X] + decay_features, axis=-1)

        self.X = torch.tensor(combined_X, dtype=torch.float32)
        self.input_dim = combined_X.shape[-1]

        # SymLog
        if not no_symlog:
            self.A = torch.sign(self.A) * torch.log1p(torch.abs(self.A))

        # 标准化 X
        if X_stats is None:
            self.X_mean = self.X.mean(dim=(0, 1), keepdim=True)
            self.X_std = self.X.std(dim=(0, 1), keepdim=True)
            self.X_std[self.X_std == 0] = 1e-5
        else:
            self.X_mean, self.X_std = X_stats
        self.X = (self.X - self.X_mean) / self.X_std

        # 标准化 A
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, ablation_mode=None):
        super(POD_LSTM, self).__init__()
        if ablation_mode is None:
            ablation_mode = get_ablation()
        self.ablation_mode = ablation_mode
        self.output_dim = output_dim

        if ablation_mode == 'no_amp_shape':
            # 消融 1: 单 LSTM 直接预测所有模态
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc_all = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            # 完整/其他消融: 双流架构
            self.lstm_macro = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.lstm_micro = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

            if ablation_mode == 'no_cumsum':
                # Mode 1 直接回归
                self.fc_mode1_direct = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
            else:
                # 完整: cumsum + residual
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
        if self.ablation_mode == 'no_amp_shape':
            lstm_out, _ = self.lstm(x)
            return self.fc_all(lstm_out)

        lstm_out_macro, _ = self.lstm_macro(x)
        lstm_out_micro, _ = self.lstm_micro(x)

        if self.ablation_mode == 'no_cumsum':
            pred_m1_scaled = self.fc_mode1_direct(lstm_out_macro)
        else:
            delta_m1 = self.fc_mode1_delta(lstm_out_macro)
            pred_m1_trend = torch.cumsum(delta_m1, dim=1)
            m1_residual = self.fc_mode1_residual(lstm_out_macro)
            pred_m1_scaled = pred_m1_trend + m1_residual

        pred_R = self.fc_higher_modes(lstm_out_micro)
        return torch.cat([pred_m1_scaled, pred_R], dim=2)
