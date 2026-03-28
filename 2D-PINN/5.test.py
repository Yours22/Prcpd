import os
import torch
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
from importlib import import_module

# 动态导入模型类 (回归基础的 POD_LSTM)
model_module = import_module("3-model")
POD_LSTM = model_module.POD_LSTM

# 读取配置
with open("config.yaml", "r", encoding="utf-8") as f: 
    config = yaml.safe_load(f)

PATHS = config['paths']
TRAIN = config['training']
POD = config['pod']
PHYSICS = config['physics']

os.makedirs(PATHS['output_dir'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_trajectories(A_true, A_pred, case_idx=0):
    """绘制主模态随时间的演化轨迹对比图"""
    time_steps = np.arange(PHYSICS['num_time_steps'])
    
    # 取前四个主导模态进行绘制
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for i in range(4):
        axes[i].plot(time_steps, A_true[case_idx, :, i], 'k-', linewidth=2, label='True (POD)')
        axes[i].plot(time_steps, A_pred[case_idx, :, i], 'r--', linewidth=2, label='Predicted (LSTM)')
        axes[i].set_ylabel(f'Mode {i+1}')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            axes[i].legend(loc='best')
            
    axes[-1].set_xlabel('Time Step')
    plt.suptitle(f"Principal Component Trajectories - Case {case_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['output_dir'], f"trajectory_case_{case_idx}.png"), dpi=300)
    print(f"轨迹对比图已保存至 {PATHS['output_dir']}/trajectory_case_{case_idx}.png")

def main():
    # ================= 1. 加载测试数据 =================
    X_test_raw = np.load(os.path.join(PATHS['processed_dir'], "X_test.npy"))
    A_test_true = np.load(os.path.join(PATHS['processed_dir'], "A_test.npy")) # 真实的绝对物理量级
    Y_test_raw = np.load(os.path.join(PATHS['processed_dir'], "Y_test_raw.npy")) 
    
    num_cases, num_steps, _ = X_test_raw.shape
    p_t = X_test_raw[:, :, 2]
    dt = PHYSICS['dt']
    decay_lambdas = PHYSICS['decay_constants']
    
    # ================= 2. 构造 8 维物理特征 =================
    decay_features = []
    
    # 特征 5-7: 指数衰减积分 (模拟缓发中子滞后)
    for lam in decay_lambdas:
        integral = np.zeros((num_cases, num_steps))
        for t in range(1, num_steps):
            integral[:, t] = integral[:, t-1] * np.exp(-lam * dt) + p_t[:, t] * dt
        decay_features.append(integral[:, :, np.newaxis])
        
    # 特征 8: 简单累积积分 (提供线性增长趋势斜率)
    simple_integral = np.cumsum(p_t, axis=1) * dt
    decay_features.append(simple_integral[:, :, np.newaxis])
    
    # 拼接形成最终的 8 维输入张量
    X_test_enhanced = np.concatenate([X_test_raw] + decay_features, axis=-1)

    # ================= 3. 加载模型与标准化参数 =================
    ckpt = torch.load(os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth"), map_location=device)
    X_mean, X_std = ckpt['X_mean'].cpu().numpy(), ckpt['X_std'].cpu().numpy()
    
    # 注意：这里的 A_mean 和 A_std 是网络在 SymLog(对数) 空间下统计出来的
    A_mean, A_std = ckpt['A_mean'].cpu().numpy(), ckpt['A_std'].cpu().numpy()

    # 输入特征标准化
    X_test_scaled = (X_test_enhanced - X_mean) / X_std
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    # ================= 4. 初始化模型并推理 =================
    output_dim = POD['r_fast'] + POD['r_thermal']
    # 强制 input_dim=8 以匹配增强后的特征
    model = POD_LSTM(TRAIN['input_dim'], TRAIN['hidden_dim'], output_dim, TRAIN.get('num_layers', 2)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        # 模型输出的是标准化状态下的 SymLog 预测值
        A_pred_scaled = model(X_test_tensor).cpu().numpy()
        
    # ================= 5. 双重逆向还原 (核心解密) =================
    # 第一重：逆 Z-score 标准化，得到真实的 SymLog 空间轨迹
    A_pred_symlog = A_pred_scaled * A_std + A_mean
    
    # 第二重：逆 SymLog (expm1)，将对数线性的轨迹，还原成物理空间中真实的指数爆炸轨迹！
    A_pred = np.sign(A_pred_symlog) * np.expm1(np.abs(A_pred_symlog))
    
    # ================= 6. POD 三维重构 =================
    svd_fast = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl'))
    svd_thermal = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_thermal.pkl'))
    
    N, T, _ = A_pred.shape
    A_pred_flat = A_pred.reshape(N * T, -1)
    
    Y_pred_fast = svd_fast.inverse_transform(A_pred_flat[:, :POD['r_fast']])
    Y_pred_thermal = svd_thermal.inverse_transform(A_pred_flat[:, POD['r_fast']:])
    
    Y_pred = np.concatenate([Y_pred_fast, Y_pred_thermal], axis=1).reshape(N, T, -1)
    
# ================= 7. 误差计算与多维诊断 =================
    # 物理场绝对误差与相对误差
    abs_error = np.abs(Y_pred - Y_test_raw)
    rel_error_per_sample = np.linalg.norm(Y_pred - Y_test_raw, axis=(1,2)) / np.linalg.norm(Y_test_raw, axis=(1,2))
    
    print(f"\n========== 测试集推理完成 (详细诊断) ==========")
    print(f"【全局汇总】")
    print(f"全局最大绝对误差 (Max Error): {np.max(abs_error):.6e}")
    print(f"全局平均绝对误差 (Mean Error): {np.mean(abs_error):.6e}")
    print(f"样本平均相对误差 (L2 Norm): {np.mean(rel_error_per_sample)*100:.4f}%")

    # 诊断 1：时间分段误差 (暴露后期漂移问题)
    # 按时间步计算所有样本和空间的平均相对误差
    norm_diff_t = np.linalg.norm(Y_pred - Y_test_raw, axis=(0, 2))
    norm_true_t = np.linalg.norm(Y_test_raw, axis=(0, 2))
    rel_err_t = norm_diff_t / (norm_true_t + 1e-10) # 避免除 0
    
    print(f"\n【物理场·时间分段相对误差】")
    print(f"-> 瞬态前期 (t=00~30): {np.mean(rel_err_t[:30])*100:.4f}%")
    print(f"-> 瞬态中期 (t=30~70): {np.mean(rel_err_t[30:70])*100:.4f}%")
    print(f"-> 瞬态后期 (t=70~100): {np.mean(rel_err_t[70:])*100:.4f}%")

    # 诊断 2：核心 POD 模态独立误差 (暴露能量分配问题)
    print(f"\n【核心 POD 模态系数独立误差】")
    for i in range(min(4, A_pred.shape[-1])): # 仅输出前 4 个主模态
        true_mode = A_test_true[:, :, i]
        pred_mode = A_pred[:, :, i]
        
        mode_abs_err = np.mean(np.abs(pred_mode - true_mode))
        mode_rel_err = np.linalg.norm(pred_mode - true_mode) / (np.linalg.norm(true_mode) + 1e-10)
        
        print(f"-> Mode {i+1}: 平均绝对误差 = {mode_abs_err:.4e} | 相对误差 = {mode_rel_err*100:.4f}%")

    # 诊断 3：极端值定位
    max_err_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
    print(f"\n【极端值定位】")
    print(f"-> 全局最大误差爆发点: Case {max_err_idx[0]}, 时间步 t = {max_err_idx[1]}, 空间节点 = {max_err_idx[2]}")
    
    # 保存结果
    np.save(os.path.join(PATHS['output_dir'], "Y_test_pred.npy"), Y_pred)
    np.save(os.path.join(PATHS['output_dir'], "A_test_pred.npy"), A_pred)
    
    # 8. 绘制轨迹验证图
    plot_trajectories(A_test_true, A_pred, case_idx=0)
    if N > 10:
        plot_trajectories(A_test_true, A_pred, case_idx=10)
        plot_trajectories(A_test_true, A_pred, case_idx=20)
        plot_trajectories(A_test_true, A_pred, case_idx=30)
        plot_trajectories(A_test_true, A_pred, case_idx=40)

if __name__ == "__main__":
    main()