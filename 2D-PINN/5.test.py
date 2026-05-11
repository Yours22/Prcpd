import os
import torch
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
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

PLOT_CASE_INDICES = [0, 10, 20, 30, 40]


def log_print(log_file, *args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        pass
    print(*args, **kwargs, file=log_file)


def plot_trajectories(A_true, A_pred, case_idx, out_dir, log_file):
    time_steps = np.arange(PHYSICS['num_time_steps'])
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
    plt.savefig(os.path.join(out_dir, f"trajectory_case_{case_idx}.png"), dpi=300)
    plt.close()
    log_print(log_file, f"  轨迹对比图已保存: trajectory_case_{case_idx}.png")


def evaluate_test_set(model, ckpt, data_prefix, log_dir):
    """
    data_prefix: 'val' → 同分布验证集, 'test' → 外推测试集
    返回日志文件路径
    """
    # --- 输出子目录 ---
    if data_prefix == 'val':
        output_name = 'val'
        set_label = '验证集 (同分布)'
    else:
        output_name = 'test_extrap'
        set_label = '外推测试集'

    out_dir = os.path.join(PATHS['output_dir'], output_name)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_log_{output_name}_{timestamp}.txt")
    log_file = open(log_path, 'w', encoding='utf-8')

    log_print(log_file, f"评测日志 — {timestamp}")
    log_print(log_file, f"数据集: {set_label}")
    log_print(log_file, f"模型: {PATHS['model_save_dir']}/best_pod_lstm.pth\n")

    # --- 加载数据 ---
    X_raw = np.load(os.path.join(PATHS['processed_dir'], f"X_{data_prefix}.npy"))
    A_true = np.load(os.path.join(PATHS['processed_dir'], f"A_{data_prefix}.npy"))
    Y_raw = np.load(os.path.join(PATHS['processed_dir'], f"Y_{data_prefix}_raw.npy"))

    num_cases, num_steps, _ = X_raw.shape
    log_print(log_file, f"数据规模: {num_cases} 算例, {num_steps} 时间步")

    # --- 特征工程 ---
    p_t = X_raw[:, :, 4]
    dt = PHYSICS['dt']
    decay_lambdas = PHYSICS['decay_constants']

    decay_features = []
    for lam in decay_lambdas:
        integral = np.zeros((num_cases, num_steps))
        for t_step in range(1, num_steps):
            integral[:, t_step] = integral[:, t_step - 1] * np.exp(-lam * dt) + p_t[:, t_step] * dt
        decay_features.append(integral[:, :, np.newaxis])

    simple_integral = np.cumsum(p_t, axis=1) * dt
    decay_features.append(simple_integral[:, :, np.newaxis])

    X_enhanced = np.concatenate([X_raw] + decay_features, axis=-1)

    # --- 标准化 + 推理 ---
    X_mean, X_std = ckpt['X_mean'].cpu().numpy(), ckpt['X_std'].cpu().numpy()
    A_mean, A_std = ckpt['A_mean'].cpu().numpy(), ckpt['A_std'].cpu().numpy()

    X_scaled = (X_enhanced - X_mean) / X_std
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_out_raw = model(X_tensor).cpu().numpy()

    # --- 时空组装 (Amplitude-Shape) ---
    pred_m1_scaled = pred_out_raw[:, :, 0:1]
    pred_R = pred_out_raw[:, :, 1:]

    pred_m1_symlog = pred_m1_scaled * A_std[:, :, 0:1] + A_mean[:, :, 0:1]
    pred_m1_phys = np.sign(pred_m1_symlog) * np.expm1(np.abs(pred_m1_symlog))

    pred_higher_phys = pred_R * pred_m1_phys
    A_pred = np.concatenate([pred_m1_phys, pred_higher_phys], axis=2)

    # --- SVD 逆变换重建物理场 ---
    svd_fast = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl'))
    svd_thermal = joblib.load(os.path.join(PATHS['pod_save_dir'], 'svd_thermal.pkl'))

    N, T, _ = A_pred.shape
    A_pred_flat = A_pred.reshape(N * T, -1)

    Y_pred_fast = svd_fast.inverse_transform(A_pred_flat[:, :POD['r_fast']])
    Y_pred_thermal = svd_thermal.inverse_transform(A_pred_flat[:, POD['r_fast']:])

    Y_pred = np.concatenate([Y_pred_fast, Y_pred_thermal], axis=1).reshape(N, T, -1)

    # --- 全局误差 ---
    abs_error = np.abs(Y_pred - Y_raw)
    rel_error_per_sample = np.linalg.norm(Y_pred - Y_raw, axis=(1, 2)) / np.linalg.norm(Y_raw, axis=(1, 2))

    log_print(log_file, f"\n========== 全局误差 ==========")
    log_print(log_file, f"全局最大绝对误差 (Max Error): {np.max(abs_error):.6e}")
    log_print(log_file, f"全局平均绝对误差 (Mean Error): {np.mean(abs_error):.6e}")
    log_print(log_file, f"样本平均相对误差 (L2 Norm): {np.mean(rel_error_per_sample)*100:.4f}%")

    # --- 时间分段误差 ---
    norm_diff_t = np.linalg.norm(Y_pred - Y_raw, axis=(0, 2))
    norm_true_t = np.linalg.norm(Y_raw, axis=(0, 2))
    rel_err_t = norm_diff_t / (norm_true_t + 1e-10)

    log_print(log_file, f"\n========== 时间分段相对误差 ==========")
    log_print(log_file, f"前期 (t=00~30): {np.mean(rel_err_t[:30])*100:.4f}%")
    log_print(log_file, f"中期 (t=30~70): {np.mean(rel_err_t[30:70])*100:.4f}%")
    log_print(log_file, f"后期 (t=70~100): {np.mean(rel_err_t[70:])*100:.4f}%")

    # --- POD 逐模态误差 ---
    log_print(log_file, f"\n========== POD 逐模态相对误差 ==========")
    for i in range(min(4, A_pred.shape[-1])):
        true_mode = A_true[:, :, i]
        pred_mode = A_pred[:, :, i]
        mode_rel_err = np.linalg.norm(pred_mode - true_mode) / (np.linalg.norm(true_mode) + 1e-10)
        log_print(log_file, f"Mode {i+1}: {mode_rel_err*100:.4f}%")

    # --- 按物理参数分类评估 ---
    is_reg1 = X_raw[:, 0, 0]
    is_fast = X_raw[:, 0, 2]

    region = np.where(is_reg1 == 1, "Core", "Rod")
    group = np.where(is_fast == 1, "Fast", "Thermal")

    categories = {}
    for r in ["Core", "Rod"]:
        for g in ["Fast", "Thermal"]:
            name = f"{r} + {g}"
            mask = (region == r) & (group == g)
            categories[name] = mask

    log_print(log_file, f"\n========== 按物理参数分类评估 ==========")
    log_print(log_file, f"{'类别':<20} {'样本数':>6} {'平均RelErr':>10} {'前期':>8} {'中期':>8} {'后期':>8}")
    log_print(log_file, f"{'-'*60}")

    for cat_name, mask in categories.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        group_rel = rel_error_per_sample[idx]
        norm_diff_g = np.linalg.norm(Y_pred[idx] - Y_raw[idx], axis=(0, 2))
        norm_true_g = np.linalg.norm(Y_raw[idx], axis=(0, 2))
        rel_err_g = norm_diff_g / (norm_true_g + 1e-10)

        log_print(log_file,
                  f"{cat_name:<20} {len(idx):>6} "
                  f"{np.mean(group_rel)*100:>9.2f}% "
                  f"{np.mean(rel_err_g[:30])*100:>7.2f}% "
                  f"{np.mean(rel_err_g[30:70])*100:>7.2f}% "
                  f"{np.mean(rel_err_g[70:])*100:>7.2f}%")

    # --- 保存预测结果 ---
    np.save(os.path.join(out_dir, "Y_pred.npy"), Y_pred)
    np.save(os.path.join(out_dir, "A_pred.npy"), A_pred)
    log_print(log_file, f"\n预测结果已保存: {out_dir}/{{Y_pred,A_pred}}.npy")

    # --- 轨迹图 ---
    log_print(log_file, f"\n========== 轨迹图 ==========")
    for idx in PLOT_CASE_INDICES:
        if idx < N:
            plot_trajectories(A_true, A_pred, case_idx=idx, out_dir=out_dir, log_file=log_file)
        else:
            log_print(log_file, f"  [跳过] Case {idx} 超出范围 (共 {N} 个算例)")

    log_file.close()
    print(f"  Log saved to: {log_path}")
    return log_path


def main():
    log_dir = PATHS['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # --- 加载模型（只加载一次） ---
    ckpt = torch.load(
        os.path.join(PATHS['model_save_dir'], "best_pod_lstm.pth"),
        map_location=device
    )

    output_dim = POD['r_fast'] + POD['r_thermal']
    model = POD_LSTM(10, TRAIN['hidden_dim'], output_dim, TRAIN.get('num_layers', 2)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # --- 评测两个集 ---
    for data_prefix in ['val', 'test']:
        label = 'Val (in-distribution)' if data_prefix == 'val' else 'Test (extrapolation)'
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")
        evaluate_test_set(model, ckpt, data_prefix, log_dir)


if __name__ == "__main__":
    main()
