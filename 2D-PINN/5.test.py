import os
import torch
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
from importlib import import_module

model_module = import_module("3-model")
POD_LSTM = model_module.POD_LSTM

with open("config.yaml", "r", encoding="utf-8") as f: 
    config = yaml.safe_load(f)

PATHS = config['paths']
TRAIN = config['training']
POD = config['pod']
PHYSICS = config['physics']

os.makedirs(PATHS['output_dir'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_trajectories(A_true, A_pred, case_idx=0):
    time_steps = np.arange(PHYSICS['num_time_steps'])
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for i in range(4):
        axes[i].plot(time_steps, A_true[case_idx, :, i], 'k-', linewidth=2, label='True (POD)')
        axes[i].plot(time_steps, A_pred[case_idx, :, i], 'r--', linewidth=2, label='Predicted (LSTM)')
        axes[i].set_ylabel(f'Mode {i+1}')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        if i == 0: axes[i].legend(loc='best')
            
    axes[-1].set_xlabel('Time Step')
    plt.suptitle(f"Principal Component Trajectories - Case {case_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['output_dir'], f"trajectory_case_{case_idx}.png"), dpi=300)
    print(f"轨迹对比图已保存至 {PATHS['output_dir']}/trajectory_case_{case_idx}.png")

def main():
    X_test_raw = np.load(os.path.join(PATHS['processed_dir'], "X_test.npy"))
    A_test_true = np.load(os.path.join(PATHS['processed_dir'], "A_test.npy")) 
    Y_test_raw = np.load(os.path.join(PATHS['processed_dir'], "Y_test_raw.npy")) 
    
    num_cases, num_steps, _ = X_test_raw.shape
    
    # ================= 核心修改 1：特征索引对齐 =================
    # 在独热编码下：[is_reg1, is_reg2, is_fast, is_thermal, p_t, t]
    p_t = X_test_raw[:, :, 4] 
    
    dt = PHYSICS['dt']
    decay_lambdas = PHYSICS['decay_constants']
    
    decay_features = []
    for lam in decay_lambdas:
        integral = np.zeros((num_cases, num_steps))
        for t in range(1, num_steps):
            integral[:, t] = integral[:, t-1] * np.exp(-lam * dt) + p_t[:, t] * dt
        decay_features.append(integral[:, :, np.newaxis])
        
    simple_integral = np.cumsum(p_t, axis=1) * dt
    decay_features.append(simple_integral[:, :, np.newaxis])
    
    # 拼接后变为 10 维
    X_test_enhanced = np.concatenate([X_test_raw] + decay_features, axis=-1)

    ckpt = torch.load(os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth"), map_location=device)
    X_mean, X_std = ckpt['X_mean'].cpu().numpy(), ckpt['X_std'].cpu().numpy()
    A_mean, A_std = ckpt['A_mean'].cpu().numpy(), ckpt['A_std'].cpu().numpy()

    X_test_scaled = (X_test_enhanced - X_mean) / X_std
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    output_dim = POD['r_fast'] + POD['r_thermal']
    # 强制 input_dim=10 以匹配增强后的特征
    model = POD_LSTM(10, TRAIN['hidden_dim'], output_dim, TRAIN.get('num_layers', 2)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        pred_out_raw = model(X_test_tensor).cpu().numpy()
        
    # ================= 核心修改 2：时空组装逻辑 (Amplitude-Shape) =================
    pred_m1_scaled = pred_out_raw[:, :, 0:1]
    pred_R = pred_out_raw[:, :, 1:]
    
    # 1. 主振幅 P(t) 逆向对数还原
    pred_m1_symlog = pred_m1_scaled * A_std[:, :, 0:1] + A_mean[:, :, 0:1]
    pred_m1_phys = np.sign(pred_m1_symlog) * np.expm1(np.abs(pred_m1_symlog))
    
    # 2. 物理组装：形状比值 R(t) * 绝对振幅 P(t)
    pred_higher_phys = pred_R * pred_m1_phys
    
    # 3. 拼合出完整的高维系数矩阵
    A_pred = np.concatenate([pred_m1_phys, pred_higher_phys], axis=2)
    # ==============================================================================
    
    svd_fast = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl'))
    svd_thermal = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_thermal.pkl'))
    
    N, T, _ = A_pred.shape
    A_pred_flat = A_pred.reshape(N * T, -1)
    
    Y_pred_fast = svd_fast.inverse_transform(A_pred_flat[:, :POD['r_fast']])
    Y_pred_thermal = svd_thermal.inverse_transform(A_pred_flat[:, POD['r_fast']:])
    
    Y_pred = np.concatenate([Y_pred_fast, Y_pred_thermal], axis=1).reshape(N, T, -1)
    
    abs_error = np.abs(Y_pred - Y_test_raw)
    rel_error_per_sample = np.linalg.norm(Y_pred - Y_test_raw, axis=(1,2)) / np.linalg.norm(Y_test_raw, axis=(1,2))
    
    print(f"\n========== 测试集推理完成 (详细诊断) ==========")
    print(f"全局最大绝对误差 (Max Error): {np.max(abs_error):.6e}")
    print(f"全局平均绝对误差 (Mean Error): {np.mean(abs_error):.6e}")
    print(f"样本平均相对误差 (L2 Norm): {np.mean(rel_error_per_sample)*100:.4f}%\n")

    norm_diff_t = np.linalg.norm(Y_pred - Y_test_raw, axis=(0, 2))
    norm_true_t = np.linalg.norm(Y_test_raw, axis=(0, 2))
    rel_err_t = norm_diff_t / (norm_true_t + 1e-10) 
    
    print(f"【时间分段相对误差】")
    print(f"-> 前期 (t=00~30): {np.mean(rel_err_t[:30])*100:.4f}%")
    print(f"-> 中期 (t=30~70): {np.mean(rel_err_t[30:70])*100:.4f}%")
    print(f"-> 后期 (t=70~100): {np.mean(rel_err_t[70:])*100:.4f}%\n")

    print(f"【POD 独立误差】")
    for i in range(min(4, A_pred.shape[-1])): 
        true_mode = A_test_true[:, :, i]
        pred_mode = A_pred[:, :, i]
        mode_rel_err = np.linalg.norm(pred_mode - true_mode) / (np.linalg.norm(true_mode) + 1e-10)
        print(f"-> Mode {i+1} 相对误差 = {mode_rel_err*100:.4f}%")

    np.save(os.path.join(PATHS['output_dir'], "Y_test_pred.npy"), Y_pred)
    np.save(os.path.join(PATHS['output_dir'], "A_test_pred.npy"), A_pred)
    
    plot_trajectories(A_test_true, A_pred, case_idx=0)
    if N > 10:
        for idx in [10, 20, 30, 40]:
            if idx < N: plot_trajectories(A_test_true, A_pred, case_idx=idx)

if __name__ == "__main__":
    main()