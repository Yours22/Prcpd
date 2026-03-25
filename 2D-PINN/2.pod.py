import os
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

def flatten_tensor(Y_3d): return Y_3d.reshape(Y_3d.shape[0] * Y_3d.shape[1], Y_3d.shape[2])
def unflatten_tensor(A_flat, N, T): return A_flat.reshape(N, T, -1)

# 1. 对训练集进行拟合与转换
Y_train = np.load(os.path.join(PATHS['processed_dir'], "Y_train_raw.npy"))
N_train, T, _ = Y_train.shape
Y_train_flat = flatten_tensor(Y_train)

Y_train_fast = Y_train_flat[:, :PHYSICS['num_nodes_per_group']]
Y_train_thermal = Y_train_flat[:, PHYSICS['num_nodes_per_group']:]

svd_fast = TruncatedSVD(n_components=POD['r_fast'], random_state=42)
svd_thermal = TruncatedSVD(n_components=POD['r_thermal'], random_state=42)

A_train_fast = svd_fast.fit_transform(Y_train_fast)
A_train_thermal = svd_thermal.fit_transform(Y_train_thermal)

joblib.dump(svd_fast, os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl'))
joblib.dump(svd_thermal, os.path.join(PATHS['pod_save_dir'], 'svd_thermal.pkl'))

A_train_flat = np.concatenate([A_train_fast, A_train_thermal], axis=1)
A_train = unflatten_tensor(A_train_flat, N_train, T)
np.save(os.path.join(PATHS['processed_dir'], "A_train.npy"), A_train)

# 2. 对验证集与测试集仅进行转换 (Transform)
for split in ['val', 'test']:
    Y_path = os.path.join(PATHS['processed_dir'], f"Y_{split}_raw.npy")
    if not os.path.exists(Y_path): continue
    
    Y_split = np.load(Y_path)
    N_split = Y_split.shape[0]
    Y_flat = flatten_tensor(Y_split)
    
    A_fast = svd_fast.transform(Y_flat[:, :PHYSICS['num_nodes_per_group']])
    A_thermal = svd_thermal.transform(Y_flat[:, PHYSICS['num_nodes_per_group']:])
    
    A_split = unflatten_tensor(np.concatenate([A_fast, A_thermal], axis=1), N_split, T)
    np.save(os.path.join(PATHS['processed_dir'], f"A_{split}.npy"), A_split)

print("POD 降维处理完毕。")