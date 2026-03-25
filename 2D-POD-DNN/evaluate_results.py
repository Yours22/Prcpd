import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from models import POD_DNN

CONFIG = {
    'data_dir': '2D-POD-DNN/data',
    'model_dir': '2D-POD-DNN/model',
    'image_dir': '2D-POD-DNN/image/new',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    os.makedirs(CONFIG['image_dir'], exist_ok=True)
    
    # 1. 加载数据与模型组件
    X_test = np.load(os.path.join(CONFIG['data_dir'], 'X_test.npy'))
    Y_test_raw = np.load(os.path.join(CONFIG['data_dir'], 'Y_test_raw.npy'))
    
    scaler_X = joblib.load(os.path.join(CONFIG['model_dir'], 'scaler_X.pkl'))
    scaler_A = joblib.load(os.path.join(CONFIG['model_dir'], 'scaler_A.pkl'))
    pod_models = joblib.load(os.path.join(CONFIG['model_dir'], 'pod_models.pkl'))
    
    pod_fast, pod_therm = pod_models['fast'], pod_models['therm']
    
    # 2. 模型推理
    model = POD_DNN(X_test.shape[1], pod_fast.r + pod_therm.r).to(CONFIG['device'])
    model.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'pod_dnn_weights.pth')))
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32).to(CONFIG['device'])
        A_pred_s = model(X_tensor).cpu().numpy()
        
    A_pred = scaler_A.inverse_transform(A_pred_s)
    Y_pred_fast = pod_fast.inverse_transform(A_pred[:, :pod_fast.r])
    Y_pred_therm = pod_therm.inverse_transform(A_pred[:, pod_fast.r:])
    Y_pred_raw = np.hstack((Y_pred_fast, Y_pred_therm))
    
    # 3. 寻找最大误差发生的时间步和样本
    abs_errors = np.abs(Y_pred_raw - Y_test_raw)
    max_err_idx = np.unravel_index(np.argmax(abs_errors, axis=None), abs_errors.shape)
    worst_sample_idx = max_err_idx[0]
    worst_node_idx = max_err_idx[1]
    
    print(f"最大误差 ({np.max(abs_errors):.2f}) 发生在测试集第 {worst_sample_idx} 个切片, 空间节点 {worst_node_idx}")
    
    # 4. 绘图 1：最差时刻的空间场分布对比
    plt.figure(figsize=(14, 5))
    plt.suptitle(f"Spatial Distribution at Worst Snapshot (Index: {worst_sample_idx})", fontsize=12)
    
    plt.subplot(1, 2, 1)
    plt.plot(Y_test_raw[worst_sample_idx, :400], label='True Fast', alpha=0.8)
    plt.plot(Y_pred_raw[worst_sample_idx, :400], label='Pred Fast', linestyle='--', alpha=0.8)
    plt.title("Fast Flux")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(Y_test_raw[worst_sample_idx, 400:], label='True Thermal', alpha=0.8)
    plt.plot(Y_pred_raw[worst_sample_idx, 400:], label='Pred Thermal', linestyle='--', alpha=0.8)
    plt.title("Thermal Flux")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['image_dir'], 'worst_spatial_distribution.png'), dpi=300)
    plt.close()
    
    # 5. 绘图 2：POD 系数潜空间对比 (Parity Plot)
    plt.figure(figsize=(10, 5))
    plt.suptitle("Latent Space (POD Coefficients) Prediction Accuracy")
    
    A_true = scaler_A.inverse_transform(scaler_A.transform(A_pred)) # 仅占位获取真实维度, 此处A_true需从Y_test反推
    A_true_fast = pod_fast.transform(Y_test_raw[:, :400])
    A_true_therm = pod_therm.transform(Y_test_raw[:, 400:])
    A_true = np.hstack((A_true_fast, A_true_therm))
    
    plt.subplot(1, 2, 1)
    plt.scatter(A_true[:, 0], A_pred[:, 0], alpha=0.5, s=2)
    plt.plot([A_true[:, 0].min(), A_true[:, 0].max()], [A_true[:, 0].min(), A_true[:, 0].max()], 'r--')
    plt.title("Mode 1 (Dominant Shape)")
    plt.xlabel("True Coefficient")
    plt.ylabel("Predicted Coefficient")
    
    plt.subplot(1, 2, 2)
    plt.scatter(A_true[:, -1], A_pred[:, -1], alpha=0.5, s=2)
    plt.plot([A_true[:, -1].min(), A_true[:, -1].max()], [A_true[:, -1].min(), A_true[:, -1].max()], 'r--')
    plt.title("Mode 15 (High-Frequency Perturbation)")
    plt.xlabel("True Coefficient")
    plt.ylabel("Predicted Coefficient")
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['image_dir'], 'coefficient_parity_plot.png'), dpi=300)
    plt.close()
    
    print("图像已保存至 image 文件夹。")

if __name__ == "__main__":
    main()