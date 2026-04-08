import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import csv
import time
from importlib import import_module

model_module = import_module("3-model")
TransientSequenceDataset = model_module.TransientSequenceDataset
POD_LSTM = model_module.POD_LSTM

with open("config.yaml", "r", encoding="utf-8") as f: 
    config = yaml.safe_load(f)
PATHS, TRAIN, POD = config['paths'], config['training'], config['pod']

os.makedirs(PATHS['model_save_dir'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 冻结/解冻参数的辅助函数
def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

class DualStageRatioLoss(nn.Module):
    def __init__(self, a_mean, a_std, device, seq_len=101, end_weight=50.0):
        super(DualStageRatioLoss, self).__init__()
        self.a_mean = a_mean.view(1, 1, -1).to(device)
        self.a_std = a_std.view(1, 1, -1).to(device)
        self.l1_loss = nn.L1Loss(reduction='none')
        
        # 1. 保留指数级时间加权，按住末端的爆炸尾巴
        exponent = torch.linspace(0, np.log(end_weight), steps=seq_len).view(1, -1, 1).to(device)
        weights = torch.exp(exponent)
        
        # ================= 终极补丁：浴缸加权 (锚定初始条件) =================
        # 强行给前 5 个时间步施加与末端同等（甚至更高）的极限权重！
        # 这将逼迫网络绝对不准在起点乱跳，必须从物理原点老老实实出发。
        weights[:, 0:5, :] = end_weight * 1.5  
        # ====================================================================
        
        self.time_weights = weights / weights.mean()

    def forward(self, pred_out, true_scaled, stage):
        # ... (这里的 forward 代码完全保持上一次的逻辑，无需改动) ...
        if stage == 1:
            pred_m1_scaled = pred_out[:, :, 0:1]
            true_m1_scaled = true_scaled[:, :, 0:1]
            mag_weights = torch.abs(true_m1_scaled) + 1.0
            base_loss = self.l1_loss(pred_m1_scaled, true_m1_scaled)
            weighted_loss = base_loss * self.time_weights * mag_weights
            return weighted_loss.mean()
            
        elif stage == 2:
            true_symlog = true_scaled * self.a_std + self.a_mean
            true_phys = torch.sign(true_symlog) * torch.expm1(torch.abs(true_symlog))
            pred_R = pred_out[:, :, 1:]
            true_R = true_phys[:, :, 1:] / (torch.abs(true_phys[:, :, 0:1]) + 1.0).detach()
            # 依然使用 L1 保证收敛的锋利度
            base_loss_higher = self.l1_loss(pred_R * 1000.0, true_R * 1000.0)
            weighted_loss_higher = base_loss_higher * self.time_weights
            return weighted_loss_higher.mean()
        
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
    
    # 物理与对数异构损失函数
    a_mean_tensor = train_ds.A_mean.clone().detach()
    a_std_tensor = train_ds.A_std.clone().detach()
    criterion = DualStageRatioLoss(a_mean_tensor, a_std_tensor, device)

    # ================= 1. 初始化日志文件 =================
    log_file_path = os.path.join(PATHS['output_dir'], 'training_log.csv')
    log_headers = [
        'Epoch', 'Stage', 'Train_Loss', 'Val_Phys_RMSE',
        'M1_MAE', 'M1_RelErr(%)', 
        'M2_MAE', 'M2_RelErr(%)', 
        'M3_MAE', 'M3_RelErr(%)',
        'Time_Cost(s)'
    ]
    with open(log_file_path, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(log_headers)
    print(f">>> 启动两阶段微调，日志实时保存至: {log_file_path}\n")

    # ================= 2. 训练阶段规划 =================
    # 假设总 Epochs 为 100，前 40 轮算阶段 1，后 60 轮算阶段 2
    total_epochs = TRAIN['epochs']
    stage1_epochs = int(total_epochs * 0.3) 
    
    # 定义两套独立的优化器
    optimizer_macro = optim.AdamW(
        list(model.lstm_macro.parameters()) + list(model.fc_mode1_delta.parameters()), 
        lr=TRAIN['learning_rate'], weight_decay=1e-4
    )
    optimizer_micro = optim.AdamW(
        list(model.lstm_micro.parameters()) + list(model.fc_higher_modes.parameters()), 
        lr=TRAIN['learning_rate'], weight_decay=1e-4
    )
    
    scheduler_macro = optim.lr_scheduler.ReduceLROnPlateau(optimizer_macro, mode='min', factor=0.5, patience=10)
    scheduler_micro = optim.lr_scheduler.ReduceLROnPlateau(optimizer_micro, mode='min', factor=0.5, patience=10)

    best_val_loss_s1 = float('inf')
    best_val_loss_s2 = float('inf')
    current_stage = 1

    # ================= 3. 主循环 =================
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        
        # 阶段切换探测与动作执行
        if epoch == 0:
            print(f"\n========== 进入 Stage 1：锚定宏观能量演化 (锁定 Mode 1) ==========")
            set_requires_grad(model.lstm_micro, False)
            set_requires_grad(model.fc_higher_modes, False)
            set_requires_grad(model.lstm_macro, True)
            set_requires_grad(model.fc_mode1_delta, True)
            
        elif epoch == stage1_epochs:
            current_stage = 2
            print(f"\n========== 进入 Stage 2：精雕微观空间形变 (冻结 Mode 1，激活 Mode 2~16) ==========")
            # 锁定已训练好的宏观干道
            set_requires_grad(model.lstm_macro, False)
            set_requires_grad(model.fc_mode1_delta, False)
            # 唤醒微观干道
            set_requires_grad(model.lstm_micro, True)
            set_requires_grad(model.fc_higher_modes, True)

        model.train()
        train_loss_ep = 0.0
        
        for b_X, b_A in train_loader:
            b_X, b_A = b_X.to(device), b_A.to(device)
            
            if current_stage == 1:
                optimizer_macro.zero_grad()
                pred_A = model(b_X) 
                loss = criterion(pred_A, b_A, stage=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.lstm_macro.parameters(), max_norm=1.0)
                optimizer_macro.step()
                
            elif current_stage == 2:
                optimizer_micro.zero_grad()
                pred_A = model(b_X)
                loss = criterion(pred_A, b_A, stage=2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.lstm_micro.parameters(), max_norm=1.0)
                optimizer_micro.step()
                
            train_loss_ep += loss.item() * b_X.size(0)
            
        # ================= 4. 全维度指标测算 (保持不变的透视镜) =================
        model.eval()
        val_mae_m1, val_mae_m2, val_mae_m3 = 0.0, 0.0, 0.0
        val_rel_m1, val_rel_m2, val_rel_m3 = 0.0, 0.0, 0.0
        val_sse_total = 0.0
        stage_val_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for b_X, b_A in val_loader:
                b_X, b_A = b_X.to(device), b_A.to(device)
                pred_out = model(b_X)
                
                # 计算 Loss
                stage_loss = criterion(pred_out, b_A, stage=current_stage)
                batch_size = b_X.size(0)
                stage_val_loss += stage_loss.item() * batch_size
                total_samples += batch_size
                
                # --- 物理场透视诊断重构 ---
                # 1. 真实值正常还原
                true_symlog = b_A * train_ds.A_std.to(device) + train_ds.A_mean.to(device)
                true_phys = torch.sign(true_symlog) * torch.expm1(torch.abs(true_symlog))
                
                # 2. 预测值进行“时空组装”
                pred_m1_scaled = pred_out[:, :, 0:1]
                pred_m1_symlog = pred_m1_scaled * train_ds.A_std[:, :, 0:1].to(device) + train_ds.A_mean[:, :, 0:1].to(device)
                pred_m1_phys = torch.sign(pred_m1_symlog) * torch.expm1(torch.abs(pred_m1_symlog))
                
                pred_R = pred_out[:, :, 1:]
                # 【组装】：形状函数 * 振幅函数 -> 高阶模态物理绝对值！
                pred_higher_phys = pred_R * pred_m1_phys
                
                # 将主模态与组装好的高阶模态重新拼合
                pred_phys = torch.cat([pred_m1_phys, pred_higher_phys], dim=2)
                
                # --- 误差计算（和以前完全一样） ---
                abs_err = torch.abs(pred_phys - true_phys)
                val_mae_m1 += abs_err[:, :, 0].mean().item() * batch_size
                val_mae_m2 += abs_err[:, :, 1].mean().item() * batch_size
                val_mae_m3 += abs_err[:, :, 2].mean().item() * batch_size
                
                rel_err = abs_err / (torch.abs(true_phys) + 1e-5)
                val_rel_m1 += rel_err[:, :, 0].mean().item() * batch_size
                val_rel_m2 += rel_err[:, :, 1].mean().item() * batch_size
                val_rel_m3 += rel_err[:, :, 2].mean().item() * batch_size
                
                val_sse_total += torch.sum((pred_phys - true_phys)**2).item()

        # 计算 Epoch 均值
        stage_val_loss /= total_samples
        val_mae_m1 /= total_samples
        val_mae_m2 /= total_samples
        val_mae_m3 /= total_samples
        val_rel_m1 /= total_samples
        val_rel_m2 /= total_samples
        val_rel_m3 /= total_samples
        val_rmse_phys = np.sqrt(val_sse_total / (total_samples * b_X.size(1) * b_A.size(2)))
        train_loss_ep /= len(train_ds)
        
        epoch_time = time.time() - epoch_start_time

        # ================= 5. 分阶段独立保存模型 =================
        saved_flag = ""
        if current_stage == 1:
            scheduler_macro.step(stage_val_loss)
            if stage_val_loss < best_val_loss_s1:
                best_val_loss_s1 = stage_val_loss
                saved_flag = "[S1 Saved]"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'X_mean': train_ds.X_mean, 'X_std': train_ds.X_std,
                    'A_mean': train_ds.A_mean, 'A_std': train_ds.A_std
                }, os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth"))
                
        elif current_stage == 2:
            scheduler_micro.step(stage_val_loss)
            if stage_val_loss < best_val_loss_s2:
                best_val_loss_s2 = stage_val_loss
                saved_flag = "[S2 Saved]"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'X_mean': train_ds.X_mean, 'X_std': train_ds.X_std,
                    'A_mean': train_ds.A_mean, 'A_std': train_ds.A_std
                }, os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth"))

        # 写入 CSV 日志
        log_row = [
            epoch + 1, current_stage,
            f"{train_loss_ep:.4f}", f"{val_rmse_phys:.4e}",
            f"{val_mae_m1:.4e}", f"{val_rel_m1*100:.2f}",
            f"{val_mae_m2:.4e}", f"{val_rel_m2*100:.2f}",
            f"{val_mae_m3:.4e}", f"{val_rel_m3*100:.2f}",
            f"{epoch_time:.1f}"
        ]
        with open(log_file_path, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(log_row)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == stage1_epochs:
            print(f"[{epoch+1:03d}/{TRAIN['epochs']}] Stage {current_stage} | M1_Rel: {val_rel_m1*100:.2f}% | M2_Rel: {val_rel_m2*100:.2f}% {saved_flag}")

if __name__ == "__main__":
    main()