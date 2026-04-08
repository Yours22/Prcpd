import os
import numpy as np
import torch
import joblib
import yaml
import matplotlib.pyplot as plt
from importlib import import_module

# 动态导入你的模型定义
model_module = import_module("3-model")
POD_LSTM = model_module.POD_LSTM
TransientSequenceDataset = model_module.TransientSequenceDataset

def main():
    print(">>> 开始生成总功率演化对比图...")

    # 1. 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    TRAIN = config['training']
    POD = config['pod']
    os.makedirs(PATHS['output_dir'], exist_ok=True)
    device = torch.device("cpu") # 绘图时使用 CPU 即可

    # 2. 加载 SVD 模型
    svd_fast = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl'))
    svd_thermal = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_thermal.pkl'))
    
    # 3. 加载测试集原始真实物理场 Y_true (用于计算真实的功率)
    Y_test_true = np.load(os.path.join(PATHS['processed_dir'], "Y_test_raw.npy"))
    
    # 4. 加载测试集特征并初始化 Dataset
    # 注意：需要借用 train_ds 的统计量进行标准化，这里我们直接从保存的 checkpoint 中提取
    checkpoint_path = os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    X_mean = checkpoint['X_mean'].numpy()
    X_std = checkpoint['X_std'].numpy()
    A_mean = checkpoint['A_mean'].numpy()
    A_std = checkpoint['A_std'].numpy()

    test_ds = TransientSequenceDataset(
        os.path.join(PATHS['processed_dir'], "X_test.npy"),
        os.path.join(PATHS['processed_dir'], "A_test.npy"),
        X_stats=(torch.tensor(X_mean), torch.tensor(X_std)),
        A_stats=(torch.tensor(A_mean), torch.tensor(A_std))
    )

    # 5. 加载并初始化模型
    model = POD_LSTM(
        input_dim=TRAIN['input_dim'], 
        hidden_dim=TRAIN['hidden_dim'], 
        output_dim=POD['r_fast'] + POD['r_thermal'], 
        num_layers=TRAIN.get('num_layers', 2)
    ).to(device)    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 6. 选择一个测试样本 (例如测试集中的第 0 个 Case)
    case_idx = 40
    X_input = test_ds.X[case_idx].unsqueeze(0).to(device) # Shape: (1, 101, 8)
    
    # 获取对应 Case 的真实物理场
    Y_true_case = Y_test_true[case_idx] # Shape: (101, N_total_nodes)
    
    with torch.no_grad():
        # 模型预测 (输出为标准化的 SymLog 空间)
        pred_A_scaled = model(X_input).squeeze(0).cpu().numpy() # Shape: (101, 16)
        
        # (1) 反标准化
        pred_A_symlog = pred_A_scaled * A_std.squeeze() + A_mean.squeeze()
        # (2) 物理空间逆变换 (Inverse SymLog)
        pred_A_phys = np.sign(pred_A_symlog) * np.expm1(np.abs(pred_A_symlog))
        
        # (3) SVD 逆变换重构高维物理场 Y_pred
        r_fast = POD['r_fast']
        Y_fast_pred = svd_fast.inverse_transform(pred_A_phys[:, :r_fast])
        Y_thermal_pred = svd_thermal.inverse_transform(pred_A_phys[:, r_fast:])
        Y_pred_case = np.concatenate([Y_fast_pred, Y_thermal_pred], axis=1) # Shape: (101, N_total_nodes)

    # 7. 计算全堆总功率 (对所有空间节点的通量求和)
    # 真实的瞬态功率演化
    power_true = np.sum(Y_true_case, axis=1)
    # 预测的瞬态功率演化
    power_pred = np.sum(Y_pred_case, axis=1)
    
    time_steps = np.arange(len(power_true))

    # ================= 8. 绘制对比图 =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 子图 1：全局完整视角 ---
    ax1.plot(time_steps, power_true, 'k-', linewidth=3, label='FEMFFUSION (Ground Truth)')
    ax1.plot(time_steps, power_pred, 'r--', linewidth=2.5, label='POD-LSTM (Predicted)')
    ax1.set_title("Total Reactor Power Evolution (Full Time)", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Time Step", fontsize=14)
    ax1.set_ylabel("Total Sum of Flux (a.u.)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=12)
    
    # --- 子图 2：末端指数爆发局部放大视角 ---
    # 选取瞬态演化最剧烈的最后 30 个步长
    zoom_start = int(len(power_true) * 0.7)
    ax2.plot(time_steps[zoom_start:], power_true[zoom_start:], 'k-', linewidth=3, label='FEMFFUSION')
    ax2.plot(time_steps[zoom_start:], power_pred[zoom_start:], 'r--', linewidth=2.5, label='POD-LSTM')
    ax2.set_title("Zoomed-in: Exponential Burst Phase", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Time Step", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=12)
    
    # 添加误差标注
    max_error = np.max(np.abs(power_true - power_pred) / power_true) * 100
    ax2.text(0.05, 0.85, f"Max Relative Error: {max_error:.2f}%", 
             transform=ax2.transAxes, fontsize=14, color='darkred',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    save_path = os.path.join(PATHS['output_dir'], f"power_conservation_case{case_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> 绘图完成！总功率演化图已保存至: {save_path}")

if __name__ == "__main__":
    main()