import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

from pod_utils import PODReducer
from models import POD_DNN

CONFIG = {
    'data_dir': '.\\data-gen',
    'target_energy': 0.999,
    'epochs': 3000,
    'lr': 0.001,
    'test_size': 0.2,
    'random_state': 42
}
img_save_path='./image'

def main():
    inputs_file = os.path.join(CONFIG['data_dir'], 'inputs.npy')
    outputs_file = os.path.join(CONFIG['data_dir'], 'outputs.npy')
    
    print(">>> 1. Loading Data...")
    inputs = np.load(inputs_file)
    outputs = np.load(outputs_file) 
    
    # 展平 Y: (Samples, Time*Space)
    Y_flat = outputs.reshape(outputs.shape[0], -1)

    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
        inputs, Y_flat, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state']
    )

    # POD 降维 (只在训练集上拟合)
    print(">>> 2. Performing POD Reduction...")
    pod = PODReducer(target_energy=CONFIG['target_energy'])
    pod.fit(Y_train_raw) # 计算基向量和均值
    
    # 获取降维后的系数
    coeffs_train_raw = pod.transform(Y_train_raw)
    coeffs_test_raw = pod.transform(Y_test_raw)
    
    # 保存 POD 组件
    pod.save(os.path.join(CONFIG['data_dir'], 'pod_components.npz'))

    # 数据归一化
    print(">>> 3. Normalizing Data...")
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler() # 对 POD 系数进行归一化
    
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)
    
    coeffs_train = scaler_Y.fit_transform(coeffs_train_raw)
    coeffs_test = scaler_Y.transform(coeffs_test_raw)
    
    # 保存归一化器
    joblib.dump(scaler_X, os.path.join(CONFIG['data_dir'], 'scaler_X.pkl'))
    joblib.dump(scaler_Y, os.path.join(CONFIG['data_dir'], 'scaler_Y.pkl'))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(coeffs_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(coeffs_test, dtype=torch.float32)

    print(">>> 4. Training DNN...")
    input_dim = X_train.shape[1]
    output_dim = pod.r # 输出维度等于保留的模态数
    
    model = POD_DNN(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    train_hist = []
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        train_hist.append(loss.item())
        
        if (epoch+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test_t), y_test_t)
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

    torch.save(model.state_dict(), os.path.join(CONFIG['data_dir'], 'pod_dnn.pth'))
    print("模型训练完成并保存。")
    
    # 快速验证
    validate_results(model, X_test_t, Y_test_raw, scaler_Y, pod, train_hist)

def validate_results(model, X_test_t, Y_test_raw, scaler_Y, pod, train_hist):
    print("\n>>> 5. Validation...")
    model.eval()
    with torch.no_grad():
        pred_norm = model(X_test_t).numpy()
    
    # 逆归一化 (Norm Coeffs -> Real Coeffs)
    pred_coeffs = scaler_Y.inverse_transform(pred_norm)
    
    # POD 重构 (Real Coeffs -> Physical Field)
    Y_pred_flat = pod.inverse_transform(pred_coeffs)
    
    # 计算误差
    mse = np.mean((Y_pred_flat - Y_test_raw)**2)
    rmse = np.sqrt(mse)
    print(f"Test Set RMSE: {rmse:.6f}")
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_hist)
    plt.title("Training Loss")
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    # 随机取一个样本的一个点随时间的变化 (这里简化为取前100个数据点示意)
    sample_idx = 0
    plt.plot(Y_test_raw[sample_idx, :200], label='True', alpha=0.7)
    plt.plot(Y_pred_flat[sample_idx, :200], label='Pred', linestyle='--')
    plt.title(f"Reconstruction Sample {sample_idx} (First 200 points)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{img_save_path}/training_report.png')
    print("验证图表已保存至 training_report.png")

if __name__ == "__main__":
    main()