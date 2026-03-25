import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

from models import POD_DNN

# ================= 配置参数 =================
CONFIG = {
    'data_dir': '2D-POD-DNN/data',
    'model_dir': '2D-POD-DNN/model',  # 新增：模型与字典的独立保存路径
    'image_dir': '2D-POD-DNN/image',
    'epochs': 500,
    'lr': 0.001,
    'batch_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    # 确保目录存在
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    os.makedirs(CONFIG['image_dir'], exist_ok=True)

    print("\n===== 1. 加载特征 (X) 与系数标签 (A) =====")
    X_train = np.load(os.path.join(CONFIG['data_dir'], 'X_train.npy'))
    X_val   = np.load(os.path.join(CONFIG['data_dir'], 'X_val.npy'))
    
    A_train = np.load(os.path.join(CONFIG['data_dir'], 'A_train.npy'))
    A_val   = np.load(os.path.join(CONFIG['data_dir'], 'A_val.npy'))

    print("\n===== 2. 数据标准化 =====")
    scaler_X = StandardScaler()
    scaler_A = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)

    A_train_s = scaler_A.fit_transform(A_train)
    A_val_s   = scaler_A.transform(A_val)

    # 保存标准化器到 model 文件夹
    joblib.dump(scaler_X, os.path.join(CONFIG['model_dir'], 'scaler_X.pkl'))
    joblib.dump(scaler_A, os.path.join(CONFIG['model_dir'], 'scaler_A.pkl'))

    print("\n===== 3. 构建 PyTorch DataLoader =====")
    train_dataset = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), 
                                  torch.tensor(A_train_s, dtype=torch.float32))
    val_dataset   = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), 
                                  torch.tensor(A_val_s, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print("\n===== 4. 初始化 POD-DNN 模型 =====")
    input_dim = X_train_s.shape[1]   
    output_dim = A_train_s.shape[1]  
    
    model = POD_DNN(input_dim, output_dim).to(CONFIG['device'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)

    train_loss_history = []
    val_loss_history = []

    print("\n===== 5. 开始训练 =====")
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:04d}/{CONFIG['epochs']} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    print("\n===== 6. 保存网络权重与 Loss 曲线 =====")
    # 保存权重到 model 文件夹
    torch.save(model.state_dict(), os.path.join(CONFIG['model_dir'], 'pod_dnn_weights.pth'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train MSE Loss', color='blue', alpha=0.8)
    plt.plot(val_loss_history, label='Validation MSE Loss', color='red', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (Log Scale)')
    plt.title('POD-DNN Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    
    loss_pic_path = os.path.join(CONFIG['image_dir'], 'loss_curve.png')
    plt.savefig(loss_pic_path, dpi=300)
    plt.close()
    print(f"训练完成，文件已分别保存至 {CONFIG['model_dir']} 和 {CONFIG['image_dir']} 目录。")

if __name__ == "__main__":
    main()