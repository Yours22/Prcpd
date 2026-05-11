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
symlog_inverse = model_module.symlog_inverse
get_ablation = model_module.get_ablation
get_output_root = model_module.get_output_root

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
PATHS, TRAIN, POD, PHYSICS = config['paths'], config['training'], config['pod'], config['physics']
ABLATION = config.get('ablation', {})
ABLATION_MODE = ABLATION.get('mode', 'full')

# 消融输出目录
ROOT = get_output_root()
MODEL_DIR = os.path.join(ROOT, 'models')
LOG_DIR = os.path.join(ROOT, 'log')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


class DualStageRatioLoss(nn.Module):
    def __init__(self, a_mean, a_std, device, seq_len=101, end_weight=50.0, ratio_scale=1000.0):
        super(DualStageRatioLoss, self).__init__()
        self.a_mean = a_mean.view(1, 1, -1).to(device)
        self.a_std = a_std.view(1, 1, -1).to(device)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ratio_scale = ratio_scale
        exponent = torch.linspace(0, np.log(end_weight), steps=seq_len).view(1, -1, 1).to(device)
        weights = torch.exp(exponent)
        weights[:, 0:5, :] = end_weight * 0.8
        self.time_weights = weights / weights.mean()

    def forward(self, pred_out, true_scaled, stage):
        if stage in (1, 3):
            pred_m1_scaled = pred_out[:, :, 0:1]
            true_m1_scaled = true_scaled[:, :, 0:1]
            mag_weights = torch.abs(true_m1_scaled) + 1.0
            base_loss = self.l1_loss(pred_m1_scaled, true_m1_scaled)
            weighted_loss = base_loss * self.time_weights * mag_weights
            return weighted_loss.mean()
        elif stage == 2:
            true_symlog = true_scaled * self.a_std + self.a_mean
            true_phys = symlog_inverse(true_symlog)
            pred_R = pred_out[:, :, 1:]
            true_R = true_phys[:, :, 1:] / (torch.abs(true_phys[:, :, 0:1]) + 1.0).detach()
            base_loss_higher = self.l1_loss(pred_R * self.ratio_scale, true_R * self.ratio_scale)
            weighted_loss_higher = base_loss_higher * self.time_weights
            return weighted_loss_higher.mean()


class SimpleWeightedL1Loss(nn.Module):
    """用于 no_amp_shape 消融：对所有模态做幅值加权的 L1"""
    def __init__(self, device, seq_len=101, end_weight=50.0):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        exponent = torch.linspace(0, np.log(end_weight), steps=seq_len).view(1, -1, 1).to(device)
        weights = torch.exp(exponent)
        weights[:, 0:5, :] = end_weight * 0.8
        self.time_weights = weights / weights.mean()

    def forward(self, pred, true):
        mag_weights = torch.abs(true) + 1.0
        base_loss = self.l1_loss(pred, true)
        return (base_loss * self.time_weights * mag_weights).mean()


def _train_step(model, criterion, optimizer, clip_params, batch, stage, device, grad_clip_norm):
    b_X, b_A = batch
    b_X, b_A = b_X.to(device), b_A.to(device)
    optimizer.zero_grad()
    pred_A = model(b_X)
    if isinstance(criterion, SimpleWeightedL1Loss):
        loss = criterion(pred_A, b_A)
    else:
        loss = criterion(pred_A, b_A, stage=stage)
    loss.backward()
    if clip_params is not None:
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=grad_clip_norm)
    optimizer.step()
    return loss.item() * b_X.size(0)


def main():
    ablation_mode = ABLATION_MODE
    no_symlog = (ablation_mode == 'no_symlog')
    no_amp = (ablation_mode == 'no_amp_shape')

    train_ds = TransientSequenceDataset(
        os.path.join(PATHS['processed_dir'], "X_train.npy"),
        os.path.join(PATHS['processed_dir'], "A_train.npy"),
        ablation_mode=ablation_mode,
    )
    val_ds = TransientSequenceDataset(
        os.path.join(PATHS['processed_dir'], "X_val.npy"),
        os.path.join(PATHS['processed_dir'], "A_val.npy"),
        X_stats=(train_ds.X_mean, train_ds.X_std),
        A_stats=(train_ds.A_mean, train_ds.A_std),
        ablation_mode=ablation_mode,
    )

    input_dim = train_ds.input_dim
    output_dim = POD['r_fast'] + POD['r_thermal']
    print(f"[Ablation: {ablation_mode}] input_dim={input_dim}, output_dim={output_dim}")

    train_loader = DataLoader(train_ds, batch_size=TRAIN['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN['batch_size'], shuffle=False)

    model = POD_LSTM(input_dim, TRAIN['hidden_dim'], output_dim,
                     TRAIN.get('num_layers', 2), ablation_mode=ablation_mode).to(device)

    # 损失函数
    a_mean_tensor = train_ds.A_mean.clone().detach()
    a_std_tensor = train_ds.A_std.clone().detach()

    if no_amp:
        criterion = SimpleWeightedL1Loss(device, end_weight=TRAIN['end_weight'])
    else:
        criterion = DualStageRatioLoss(
            a_mean_tensor, a_std_tensor, device,
            end_weight=TRAIN['end_weight'],
            ratio_scale=TRAIN['ratio_loss_scale']
        )

    # 日志
    log_file_path = os.path.join(LOG_DIR, 'training_log.csv')
    log_headers = ['Epoch', 'Stage', 'Train_Loss', 'Val_Phys_RMSE',
                   'M1_MAE', 'M1_RelErr(%)', 'M2_MAE', 'M2_RelErr(%)',
                   'M3_MAE', 'M3_RelErr(%)', 'Time_Cost(s)']
    with open(log_file_path, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(log_headers)
    print(f">>> Training log: {log_file_path}\n")

    total_epochs = TRAIN['epochs']
    grad_clip_norm = TRAIN['grad_clip_norm']

    if no_amp:
        # 单阶段训练
        optimizer = optim.AdamW(model.parameters(), lr=TRAIN['learning_rate'],
                                weight_decay=TRAIN['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=TRAIN['scheduler_factor'],
                                                         patience=TRAIN['scheduler_patience'])
        stage_train_cfg = {1: (optimizer, model.parameters())}
        stage_val_cfg = {1: {"scheduler": scheduler}}
        stage_transitions = {}
        current_stage = 1
        best_val_loss = {1: float('inf')}
        n_modes = output_dim
    else:
        # 三阶段训练（原逻辑）
        stage1_epochs = int(total_epochs * TRAIN['stage1_ratio'])
        stage2_epochs = int(total_epochs * TRAIN['stage2_ratio'])
        wd = TRAIN['weight_decay']

        if ablation_mode == 'no_cumsum':
            optimizer_macro = optim.AdamW(
                list(model.lstm_macro.parameters()) + list(model.fc_mode1_direct.parameters()),
                lr=TRAIN['learning_rate'], weight_decay=wd
            )
            stage_train_cfg = {
                1: (optimizer_macro, model.lstm_macro.parameters()),
                2: (optimizer_micro := optim.AdamW(
                    list(model.lstm_micro.parameters()) + list(model.fc_higher_modes.parameters()),
                    lr=TRAIN['learning_rate'], weight_decay=wd
                ), model.lstm_micro.parameters()),
            }
            stage_val_cfg = {
                1: {"scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_macro, mode='min', factor=TRAIN['scheduler_factor'],
                    patience=TRAIN['scheduler_patience'])},
                2: {"scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_micro, mode='min', factor=TRAIN['scheduler_factor'],
                    patience=TRAIN['scheduler_patience'])},
            }
            stage_transitions = {
                0: (1, "Stage 1: Direct Mode-1 regression", [model.lstm_macro, model.fc_mode1_direct]),
                stage1_epochs: (2, "Stage 2: Higher-mode shape ratios", [model.lstm_micro, model.fc_higher_modes]),
            }
        else:
            optimizer_macro = optim.AdamW(
                list(model.lstm_macro.parameters()) + list(model.fc_mode1_delta.parameters()),
                lr=TRAIN['learning_rate'], weight_decay=wd
            )
            optimizer_micro = optim.AdamW(
                list(model.lstm_micro.parameters()) + list(model.fc_higher_modes.parameters()),
                lr=TRAIN['learning_rate'], weight_decay=wd
            )
            optimizer_residual = optim.AdamW(
                model.fc_mode1_residual.parameters(),
                lr=TRAIN['residual_lr'], weight_decay=wd * 0.1
            )
            scheduler_macro = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_macro, mode='min', factor=TRAIN['scheduler_factor'],
                patience=TRAIN['scheduler_patience'])
            scheduler_micro = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_micro, mode='min', factor=TRAIN['scheduler_factor'],
                patience=TRAIN['scheduler_patience'])

            stage_train_cfg = {
                1: (optimizer_macro, model.lstm_macro.parameters()),
                2: (optimizer_micro, model.lstm_micro.parameters()),
                3: (optimizer_residual, model.fc_mode1_residual.parameters()),
            }
            stage_val_cfg = {
                1: {"scheduler": scheduler_macro},
                2: {"scheduler": scheduler_micro},
                3: {"scheduler": None},
            }
            stage_transitions = {
                0: (1, "Stage 1: Mode-1 cumsum integrator", [model.lstm_macro, model.fc_mode1_delta]),
                stage1_epochs: (2, "Stage 2: Higher-mode shape ratios", [model.lstm_micro, model.fc_higher_modes]),
                stage2_epochs: (3, "Stage 3: Mode-1 residual refinement", [model.fc_mode1_residual]),
            }

        current_stage = 1
        best_val_loss = {1: float('inf'), 2: float('inf'), 3: float('inf')}
        n_modes = output_dim

    # ==================== 主循环 ====================
    for epoch in range(total_epochs):
        epoch_start_time = time.time()

        # 阶段切换
        if epoch in stage_transitions:
            new_stage, msg, unfreeze_modules = stage_transitions[epoch]
            current_stage = new_stage
            print(f">> {msg}")
            set_requires_grad(model, False)
            for m in unfreeze_modules:
                set_requires_grad(m, True)

        # 训练
        model.train()
        train_loss_ep = 0.0
        opt, clip_params = stage_train_cfg[current_stage]
        for batch in train_loader:
            train_loss_ep += _train_step(model, criterion, opt, clip_params, batch,
                                         current_stage, device, grad_clip_norm)

        # 验证
        model.eval()
        val_mae_m1, val_mae_m2, val_mae_m3 = 0.0, 0.0, 0.0
        val_rel_m1, val_rel_m2, val_rel_m3 = 0.0, 0.0, 0.0
        val_sse_total = 0.0
        stage_val_loss = 0.0
        total_samples = 0

        A_std_dev = train_ds.A_std.to(device)
        A_mean_dev = train_ds.A_mean.to(device)

        with torch.no_grad():
            for b_X, b_A in val_loader:
                b_X, b_A = b_X.to(device), b_A.to(device)
                pred_out = model(b_X)
                batch_size = b_X.size(0)

                if no_amp:
                    stage_loss = criterion(pred_out, b_A)
                    stage_val_loss += stage_loss.item() * batch_size
                    total_samples += batch_size
                    # 物理场还原
                    if no_symlog:
                        pred_phys = pred_out * A_std_dev + A_mean_dev
                        true_phys = b_A * A_std_dev + A_mean_dev
                    else:
                        pred_symlog = pred_out * A_std_dev + A_mean_dev
                        pred_phys = symlog_inverse(pred_symlog)
                        true_symlog = b_A * A_std_dev + A_mean_dev
                        true_phys = symlog_inverse(true_symlog)
                else:
                    stage_loss = criterion(pred_out, b_A, stage=current_stage)
                    stage_val_loss += stage_loss.item() * batch_size
                    total_samples += batch_size

                    if no_symlog:
                        true_phys = b_A * A_std_dev + A_mean_dev
                        pred_m1_scaled = pred_out[:, :, 0:1]
                        pred_m1_phys = pred_m1_scaled * A_std_dev[:, :, 0:1] + A_mean_dev[:, :, 0:1]
                        pred_higher_phys = pred_out[:, :, 1:] * pred_m1_phys
                        pred_phys = torch.cat([pred_m1_phys, pred_higher_phys], dim=2)
                    else:
                        true_symlog = b_A * A_std_dev + A_mean_dev
                        true_phys = symlog_inverse(true_symlog)
                        pred_m1_scaled = pred_out[:, :, 0:1]
                        pred_m1_symlog = pred_m1_scaled * A_std_dev[:, :, 0:1] + A_mean_dev[:, :, 0:1]
                        pred_m1_phys = symlog_inverse(pred_m1_symlog)
                        pred_higher_phys = pred_out[:, :, 1:] * pred_m1_phys
                        pred_phys = torch.cat([pred_m1_phys, pred_higher_phys], dim=2)

                # 误差
                abs_err = torch.abs(pred_phys - true_phys)
                n_m = pred_phys.shape[-1]
                val_mae_m1 += abs_err[:, :, 0].mean().item() * batch_size
                if n_m >= 2:
                    val_mae_m2 += abs_err[:, :, 1].mean().item() * batch_size
                if n_m >= 3:
                    val_mae_m3 += abs_err[:, :, 2].mean().item() * batch_size
                rel_err = abs_err / (torch.abs(true_phys) + 1e-5)
                val_rel_m1 += rel_err[:, :, 0].mean().item() * batch_size
                if n_m >= 2:
                    val_rel_m2 += rel_err[:, :, 1].mean().item() * batch_size
                if n_m >= 3:
                    val_rel_m3 += rel_err[:, :, 2].mean().item() * batch_size
                val_sse_total += torch.sum((pred_phys - true_phys) ** 2).item()

        stage_val_loss /= total_samples
        val_mae_m1 /= total_samples
        val_mae_m2 /= total_samples if n_m >= 2 else 1
        val_mae_m3 /= total_samples if n_m >= 3 else 1
        val_rel_m1 /= total_samples
        val_rel_m2 /= total_samples if n_m >= 2 else 1
        val_rel_m3 /= total_samples if n_m >= 3 else 1
        val_rmse_phys = np.sqrt(val_sse_total / (total_samples * PHYSICS['num_time_steps'] * n_modes))
        train_loss_ep /= len(train_ds)

        epoch_time = time.time() - epoch_start_time

        # 保存
        saved_flag = ""
        meta = stage_val_cfg.get(current_stage, {})
        if meta.get("scheduler") is not None:
            meta["scheduler"].step(stage_val_loss)

        if stage_val_loss < best_val_loss[current_stage]:
            best_val_loss[current_stage] = stage_val_loss
            saved_flag = f"[S{current_stage} Saved]"
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': train_ds.X_mean, 'X_std': train_ds.X_std,
                'A_mean': train_ds.A_mean, 'A_std': train_ds.A_std,
                'ablation_mode': ablation_mode,
            }, os.path.join(MODEL_DIR, "best_pod_lstm.pth"))

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

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch in stage_transitions:
            print(f"[{epoch+1:03d}/{total_epochs}] Stage {current_stage} | "
                  f"M1_Rel: {val_rel_m1*100:.2f}% | M2_Rel: {val_rel_m2*100:.2f}% {saved_flag}")


if __name__ == "__main__":
    main()
