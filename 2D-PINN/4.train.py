import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
from importlib import import_module

model_module = import_module("3-model")
TransientSequenceDataset = model_module.TransientSequenceDataset
POD_LSTM = model_module.POD_LSTM

with open("config.yaml", "r", encoding="utf-8") as f: 
    config = yaml.safe_load(f)
PATHS, TRAIN, POD = config['paths'], config['training'], config['pod']

os.makedirs(PATHS['model_save_dir'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 核心重构：物理空间绝对误差损失 =================
class PhysicalSpaceLoss(nn.Module):
    def __init__(self, a_mean, a_std, device):
        super(PhysicalSpaceLoss, self).__init__()
        # 提取统计参数用于微分逆还原
        self.a_mean = a_mean.view(1, 1, -1).to(device)
        self.a_std = a_std.view(1, 1, -1).to(device)

    def forward(self, pred_scaled, true_scaled):
        # 1. 逆标准化 (保持张量的计算图和梯度)
        pred_symlog = pred_scaled * self.a_std + self.a_mean
        true_symlog = true_scaled * self.a_std + self.a_mean

        # 2. 逆 SymLog 还原到绝对物理空间
        pred_phys = torch.sign(pred_symlog) * torch.expm1(torch.abs(pred_symlog))
        true_phys = torch.sign(true_symlog) * torch.expm1(torch.abs(true_symlog))

        # 3. 在真实物理尺度上计算 L1 损失 (MAE)
        # 物理空间的误差极大 (如几万)，使用 L1 Loss 可以提供稳定的线性梯度，防止 MSE 的平方效应导致梯度爆炸
        return torch.nn.functional.l1_loss(pred_phys, true_phys)

def main():
    train_ds = TransientSequenceDataset(
        os.path.join(PATHS['processed_dir'], "X_train.npy"),
        os.path.join(PATHS['processed_dir'], "A_train.npy")
    )
    val_ds = TransientSequenceDataset(
        os.path.join(PATHS['processed_dir'], "X_val.npy"),
        os.path.join(PATHS['processed_dir'], "A_val.npy"),
        X_stats=(train_ds.X_mean, train_ds.X_std),
        A_stats=(train_ds.A_mean, train_ds.A_std)
    )

    train_loader = DataLoader(train_ds, batch_size=TRAIN['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN['batch_size'], shuffle=False)

    model = POD_LSTM(TRAIN['input_dim'], TRAIN['hidden_dim'], POD['r_fast'] + POD['r_thermal'], TRAIN.get('num_layers', 2)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN['learning_rate'], weight_decay=1e-4)
    
    # 实例化物理空间损失函数
    a_mean_tensor = train_ds.A_mean.clone().detach()
    a_std_tensor = train_ds.A_std.clone().detach()
    criterion = PhysicalSpaceLoss(a_mean_tensor, a_std_tensor, device)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(TRAIN['epochs']):
        model.train()
        train_loss_ep = 0.0
        for b_X, b_A in train_loader:
            b_X, b_A = b_X.to(device), b_A.to(device)
            optimizer.zero_grad()
            
            pred_A = model(b_X) 
            # 损失现在是在物理空间计算的
            loss = criterion(pred_A, b_A)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_ep += loss.item() * b_X.size(0)
            
        model.eval()
        val_loss_ep = 0.0
        with torch.no_grad():
            for b_X, b_A in val_loader:
                b_X, b_A = b_X.to(device), b_A.to(device)
                pred_A = model(b_X)
                loss = criterion(pred_A, b_A)
                val_loss_ep += loss.item() * b_X.size(0)

        train_loss_ep /= len(train_ds)
        val_loss_ep /= len(val_ds)
        scheduler.step(val_loss_ep)

        if val_loss_ep < best_val_loss:
            best_val_loss = val_loss_ep
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': train_ds.X_mean, 'X_std': train_ds.X_std,
                'A_mean': train_ds.A_mean, 'A_std': train_ds.A_std
            }, os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth"))
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{TRAIN['epochs']}] | Train Phys-MAE: {train_loss_ep:.2f} | Val Phys-MAE: {val_loss_ep:.2f}")

        if epochs_no_improve >= TRAIN['patience']:
            print(f"触发早停，结束于第 {epoch+1} 轮。")
            break

if __name__ == "__main__":
    main()