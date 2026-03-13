import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 路径定义
TRAIN_CSV = 'data-split/dataset_train_processed.csv'
VAL_CSV = 'data-split/dataset_val_processed.csv'
NPY_DIR = 'data-processed/power_series_log/'
TEST_CSV = 'data-split/dataset_test_extrapolation_processed.csv' # 新增外推集路径

# ==========================================
# 1. 数据装载器 (Custom Dataset)
# ==========================================
class TWIGLDataset(Dataset):
    def __init__(self, csv_file, npy_dir, seq_len=101):
        """
        csv_file: 经过特征工程处理后的参数表 (如 dataset_train_processed.csv)
        npy_dir: 存放对数功率序列的文件夹
        seq_len: 时间步长度 (默认 101 步)
        """
        self.df = pd.read_csv(csv_file)
        self.npy_dir = npy_dir
        self.seq_len = seq_len
        
        # 提取 case_id 用于寻找对应的 .npy 文件
        self.case_ids = self.df['case_id'].astype(str).str.replace('case_', '').astype(int).values        
        # 提取所有的输入特征 (剔除 case_id 列)
        feature_cols = [col for col in self.df.columns if col != 'case_id']
        self.features = self.df[feature_cols].values # shape: (num_samples, num_features)
        
        # 预先生成归一化后的时间步向量 [0, 1]，shape: (seq_len, 1)
        self.t_steps = np.linspace(0, 1, self.seq_len).reshape(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 获取静态物理特征
        static_params = self.features[idx]
        
        # 2. 读取对应的目标序列 (真实 ln(P))
        case_id = self.case_ids[idx]
        npy_path = os.path.join(self.npy_dir, f"power_{case_id:04d}.npy")
        true_log_power = np.load(npy_path) # shape: (seq_len, 1)
        
        # 3. 构造 LSTM 的动态输入
        # 复制静态参数 seq_len 次: shape (seq_len, num_features)
        static_params_seq = np.tile(static_params, (self.seq_len, 1))
        # 拼上时间步: shape (seq_len, num_features + 1)
        x_seq = np.hstack([static_params_seq, self.t_steps])
        
        # 转为 PyTorch 张量
        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(true_log_power, dtype=torch.float32)
        
        return x_tensor, y_tensor

# ==========================================
# 2. LSTM 网络架构设计
# ==========================================
class ReactorPowerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(ReactorPowerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True 表示张量格式为 (batch_size, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # 全连接层：将 LSTM 在每个时间步的隐藏状态映射为 1 个数值 (功率)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # 初始化隐状态 h0 和细胞状态 c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 前向传播
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # 通过全连接层预测最终结果
        # predictions shape: (batch_size, seq_len, 1)
        predictions = self.fc(out)
        
        return predictions

# ==========================================
# 3. 训练引擎 (Training Loop)
# ==========================================
def train_model():
    # --- 超参数配置 ---
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 200
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 正在使用设备: {device}")

    # --- 准备数据加载器 ---
    train_dataset = TWIGLDataset(TRAIN_CSV, NPY_DIR)
    val_dataset = TWIGLDataset(VAL_CSV, NPY_DIR)
    test_dataset = TWIGLDataset(TEST_CSV, NPY_DIR) # 新增外推集
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # 新增外推 DataLoader
    
    # 自动推断 input_size
    sample_x, _ = train_dataset[0]
    INPUT_SIZE = sample_x.shape[1] 
    print(f"网络的 Input Size 自动判定为: {INPUT_SIZE}")

    # --- 实例化模型、损失函数和优化器 ---
    model = ReactorPowerLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 开始训练 ---
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # 打印进度并保存最优模型
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_lstm_model.pth')

    print(f"\n🎉 训练结束！最优验证集 Loss: {best_val_loss:.6f}")
    print(f"✅ 最优模型已保存至: models/best_lstm_model.pth")

    # 加载刚刚保存的最优模型权重 (确保用的是巅峰状态，而不是过拟合的最后状态)
    model.load_state_dict(torch.load('models/best_lstm_model.pth'))
    model.eval() # 切换到评估模式
    
    test_loss = 0.0
    with torch.no_grad(): # 绝对不计算梯度
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播预测
            outputs = model(batch_x)
            
            # 计算误差
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            
    test_loss /= len(test_loader.dataset)
    rmse_loss = test_loss ** 0.5
    
    print(f"🎯 外推测试集 (Extrapolation) 最终 MSE Loss : {test_loss:.6f}")
    print(f"🎯 外推测试集 (Extrapolation) 最终 RMSE      : {rmse_loss:.6f}")
    print("==========================================\n")

if __name__ == "__main__":
    train_model()