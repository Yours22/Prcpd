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
class HybridModalLoss(nn.Module):
    def __init__(self, a_mean, a_std, device, lambda_higher=100.0):
        super(HybridModalLoss, self).__init__()
        self.a_mean = a_mean.view(1, 1, -1).to(device)
        self.a_std = a_std.view(1, 1, -1).to(device)
        self.l1 = nn.L1Loss(reduction='none')
        # 用于平衡绝对误差(百量级)与归一化误差(零点几量级)的缩放因子
        self.lambda_higher = lambda_higher 

    def forward(self, pred_scaled, true_scaled):
        # 1. 还原至绝对物理空间
        pred_symlog = pred_scaled * self.a_std + self.a_mean
        true_symlog = true_scaled * self.a_std + self.a_mean

        pred_phys = torch.sign(pred_symlog) * torch.expm1(torch.abs(pred_symlog))
        true_phys = torch.sign(true_symlog) * torch.expm1(torch.abs(true_symlog))

        # 2. 计算基础 L1 绝对误差矩阵 (Batch, 101, 16)
        abs_err = self.l1(pred_phys, true_phys)
        
        # 3. 分支 A：Mode 1 采用纯绝对误差，死保全局能量基准
        loss_m1 = abs_err[:, :, 0].mean()
        
        # 4. 分支 B：Mode 2~16 采用归一化误差，提升高频空间形变精度
        # 提取高阶模态并计算标准差进行归一化 (+1.0 防止除零)
        higher_true_phys = true_phys[:, :, 1:]
        higher_abs_err = abs_err[:, :, 1:]
        
        mode_scale = torch.std(higher_true_phys, dim=(0, 1), keepdim=True).detach() + 1.0
        normalized_higher_err = higher_abs_err / mode_scale
        
        loss_higher = normalized_higher_err.mean()
        
        # 5. 组合总损失
        return loss_m1 + self.lambda_higher * loss_higher
    

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
    criterion = HybridModalLoss(a_mean_tensor, a_std_tensor, device)
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