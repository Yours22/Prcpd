import os
import time
import numpy as np
import joblib
import yaml
from sklearn.decomposition import TruncatedSVD

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PATHS = config['paths']
PHYSICS = config['physics']
POD = config['pod']

os.makedirs(PATHS['pod_save_dir'], exist_ok=True)

# 日志
log_dir = os.path.join("2D-PINN", "log")
os.makedirs(log_dir, exist_ok=True)
log_file = open(os.path.join(log_dir, "pod_processing_log.txt"), 'w', encoding='utf-8')

def log(msg):
    print(msg)
    log_file.write(msg + '\n')

log(f"POD 处理日志")
log(f"非中心化 POD (TruncatedSVD, 不减均值)")
log(f"快群模态数: {POD['r_fast']}, 热群模态数: {POD['r_thermal']}")
log(f"")

def flatten_tensor(Y_3d): return Y_3d.reshape(Y_3d.shape[0] * Y_3d.shape[1], Y_3d.shape[2])
def unflatten_tensor(A_flat, N, T): return A_flat.reshape(N, T, -1)

# ==================== 1. 训练集拟合 ====================
t0 = time.time()

Y_train = np.load(os.path.join(PATHS['processed_dir'], "Y_train_raw.npy"))
N_train, T, _ = Y_train.shape
log(f"[训练集] 原始物理场: {N_train} 算例 × {T} 时间步 × {PHYSICS['total_nodes']} 节点")

Y_train_flat = flatten_tensor(Y_train)
log(f"  展平后: {Y_train_flat.shape[0]} 样本 × {Y_train_flat.shape[1]} 特征")

Y_train_fast = Y_train_flat[:, :PHYSICS['num_nodes_per_group']]
Y_train_thermal = Y_train_flat[:, PHYSICS['num_nodes_per_group']:]
log(f"  快群: {Y_train_fast.shape}, 热群: {Y_train_thermal.shape}")

t1 = time.time()
svd_fast = TruncatedSVD(n_components=POD['r_fast'], random_state=42)
svd_thermal = TruncatedSVD(n_components=POD['r_thermal'], random_state=42)

A_train_fast = svd_fast.fit_transform(Y_train_fast)
A_train_thermal = svd_thermal.fit_transform(Y_train_thermal)
t_svd = time.time() - t1

fast_var = svd_fast.explained_variance_ratio_
thermal_var = svd_thermal.explained_variance_ratio_
log(f"\n[SVD 拟合耗时] {t_svd:.2f}s")
log(f"[快群方差占比] " + " | ".join([f"M{m+1}: {v*100:.2f}%" for m, v in enumerate(fast_var)]))
log(f"  累计: {fast_var.sum()*100:.2f}%")
log(f"[热群方差占比] " + " | ".join([f"M{m+1}: {v*100:.2f}%" for m, v in enumerate(thermal_var)]))
log(f"  累计: {thermal_var.sum()*100:.2f}%")

joblib.dump(svd_fast, os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl'))
joblib.dump(svd_thermal, os.path.join(PATHS['pod_save_dir'], 'svd_thermal.pkl'))

A_train_flat = np.concatenate([A_train_fast, A_train_thermal], axis=1)
A_train = unflatten_tensor(A_train_flat, N_train, T)
np.save(os.path.join(PATHS['processed_dir'], "A_train.npy"), A_train)
log(f"\n[训练集 POD 系数] {A_train.shape} → 已保存 A_train.npy")

# ==================== 2. 验证集与测试集变换 ====================
for split in ['val', 'test']:
    Y_path = os.path.join(PATHS['processed_dir'], f"Y_{split}_raw.npy")
    if not os.path.exists(Y_path):
        log(f"\n[{split}] 文件不存在，跳过")
        continue

    Y_split = np.load(Y_path)
    N_split = Y_split.shape[0]
    log(f"\n[{split}] 原始物理场: {Y_split.shape}")
    Y_flat = flatten_tensor(Y_split)

    t2 = time.time()
    A_fast = svd_fast.transform(Y_flat[:, :PHYSICS['num_nodes_per_group']])
    A_thermal = svd_thermal.transform(Y_flat[:, PHYSICS['num_nodes_per_group']:])
    t_transform = time.time() - t2

    A_split = unflatten_tensor(np.concatenate([A_fast, A_thermal], axis=1), N_split, T)
    np.save(os.path.join(PATHS['processed_dir'], f"A_{split}.npy"), A_split)
    log(f"  变换耗时: {t_transform:.2f}s, POD 系数: {A_split.shape} → 已保存 A_{split}.npy")

t_total = time.time() - t0
log(f"\n总耗时: {t_total:.2f}s")
log_file.close()
print(f"\n日志已保存至: {log_dir}/pod_processing_log.txt")