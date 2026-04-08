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
PATHS, TRAIN, POD, PHYSICS = config['paths'], config['training'], config['pod'], config['physics']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PATHS['output_dir'], exist_ok=True)

def restore_and_visualize():
    print("========== 开始执行物理场还原与可视化流程 ==========")
    
    try:
        X_test_base = np.load(os.path.join(PATHS['processed_dir'], "X_test.npy"))
        ckpt = torch.load(os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth"), map_location=device)
    except FileNotFoundError as e:
        print(f"错误: 找不到必要文件，请确保已重新运行数据处理与训练。\n详情: {e}")
        return

    print("正在计算积分项，将基础特征增强至 10 维...")
    num_cases, num_steps, _ = X_test_base.shape
    dt = PHYSICS.get('dt', 0.01)
    decay_lambdas = PHYSICS.get('decay_constants', [0.1, 0.5, 1.0]) 
    
    # 索引变更：独热编码后的第 5 列才是微扰幅值 p(t)
    p_t = X_test_base[:, :, 4] 
    
    decay_features = []
    for lam in decay_lambdas:
        integral = np.zeros((num_cases, num_steps))
        for t in range(1, num_steps):
            integral[:, t] = integral[:, t-1] * np.exp(-lam * dt) + p_t[:, t] * dt
        decay_features.append(integral[:, :, np.newaxis])
        
    simple_integral = np.cumsum(p_t, axis=1) * dt
    decay_features.append(simple_integral[:, :, np.newaxis])
    
    X_test_raw = np.concatenate([X_test_base] + decay_features, axis=-1)

    X_mean, X_std = ckpt['X_mean'].cpu().numpy(), ckpt['X_std'].cpu().numpy()
    A_mean, A_std = ckpt['A_mean'].cpu().numpy(), ckpt['A_std'].cpu().numpy()

    output_dim = POD['r_fast'] + POD['r_thermal']
    # 强制 input_dim=10
    model = POD_LSTM(10, TRAIN['hidden_dim'], output_dim, TRAIN.get('num_layers', 2)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    X_test_scaled = (X_test_raw - X_mean) / (X_std + 1e-8)
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_out_raw = model(X_tensor).cpu().numpy()

    # --- 振幅-形状 物理组装 ---
    pred_m1_scaled = pred_out_raw[:, :, 0:1]
    pred_R = pred_out_raw[:, :, 1:]
    
    pred_m1_symlog = pred_m1_scaled * A_std[:, :, 0:1] + A_mean[:, :, 0:1]
    pred_m1_phys = np.sign(pred_m1_symlog) * np.expm1(np.abs(pred_m1_symlog))
    
    pred_higher_phys = pred_R * pred_m1_phys
    A_pred_phys = np.concatenate([pred_m1_phys, pred_higher_phys], axis=2)

    svd_fast = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl'))
    svd_thermal = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_thermal.pkl'))
    
    A_flat = A_pred_phys.reshape(num_cases * num_steps, -1)
    Y_fast_flat = svd_fast.inverse_transform(A_flat[:, :POD['r_fast']])
    Y_thermal_flat = svd_thermal.inverse_transform(A_flat[:, POD['r_fast']:])
    
    Y_pred = np.concatenate([Y_fast_flat, Y_thermal_flat], axis=-1).reshape(num_cases, num_steps, -1)
    
    total_nodes = Y_pred.shape[-1]
    N_NODES_PER_GROUP = total_nodes // 2
    NX = NY = int(np.sqrt(N_NODES_PER_GROUP))
    print(f"物理场向量还原完成，自适应识别网格尺寸: {NX}x{NY}")

    # ================= 二维热力图对比绘图 (代码维持极简逻辑) =================
    try:
        Y_test_raw = np.load(os.path.join(PATHS['processed_dir'], "Y_test_raw.npy"))
    except FileNotFoundError:
        Y_test_raw = None

    case_idx = 0
    time_steps = [0, 40, 80, 99] 
    
    for group in ["thermal", "fast"]:
        if group == "fast":
            start_idx, end_idx = 0, N_NODES_PER_GROUP
            title_pfix = "Fast Flux"
        else:
            start_idx, end_idx = N_NODES_PER_GROUP, N_NODES_PER_GROUP * 2
            title_pfix = "Thermal Flux"

        true_case = Y_test_raw[case_idx] if Y_test_raw is not None else None
        pred_case = Y_pred[case_idx]
        
        num_snaps = len(time_steps)
        nrows = 3 if true_case is not None else 1
        fig, axes = plt.subplots(nrows, num_snaps, figsize=(5 * num_snaps, nrows * 3.7 + 0.5), squeeze=False)
        
        for i, t in enumerate(time_steps):
            raw_pred = pred_case[t, start_idx:end_idx]
            grid_pred = raw_pred.reshape(NY, NX)
            plot_vmin, plot_vmax = grid_pred.min(), grid_pred.max()

            if true_case is not None:
                raw_true = true_case[t, start_idx:end_idx]
                grid_true = raw_true.reshape(NY, NX)
                grid_err = np.abs(grid_true - grid_pred)
                plot_vmin, plot_vmax = grid_true.min(), grid_true.max()

                im0 = axes[0, i].imshow(grid_true, cmap='jet', origin='lower', vmin=plot_vmin, vmax=plot_vmax)
                axes[0, i].set_title(f"True {title_pfix}\nt={t}", fontsize=11)
                plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

                curr_pred_ax = axes[1, i]
                
                im2 = axes[2, i].imshow(grid_err, cmap='hot', origin='lower')
                axes[2, i].set_title(f"Abs Error\nt={t}", fontsize=11)
                plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
            else:
                curr_pred_ax = axes[0, i]

            im1 = curr_pred_ax.imshow(grid_pred, cmap='jet', origin='lower', vmin=plot_vmin, vmax=plot_vmax)
            curr_pred_ax.set_title(f"Pred {title_pfix}\nt={t}", fontsize=11)
            plt.colorbar(im1, ax=curr_pred_ax, fraction=0.046, pad=0.04)
            
        plt.suptitle(f"TWIGL {NX}x{NY} Reconstruction - Case {case_idx} ({title_pfix})", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path = os.path.join(PATHS['output_dir'], f"reconstruction_{group}_case_{case_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f">>> {title_pfix} 可视化完成，保存至: {save_path}")

if __name__ == "__main__":
    restore_and_visualize()