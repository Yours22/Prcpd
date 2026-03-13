import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 路径配置（全局变量） ---
TEST_CSV = 'data-split/dataset_test_extrapolation_processed.csv'
NPY_DIR = 'data-processed/power_series_log/'
MODEL_PATH = 'models/best_lstm_model.pth'

# 结果输出相关
OUT_DIR = '1D-power-prediction'                     # 保存图像的目录
OUT_PLOT_NAME = 'extrapolation_results.png'         # 图片文件名
OUT_PLOT_PATH = os.path.join(OUT_DIR, OUT_PLOT_NAME)  # 最终保存路径


# ==========================================
# 1. 重新定义模型结构 (用于加载权重)
# ==========================================
class ReactorPowerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(ReactorPowerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        predictions = self.fc(out)
        return predictions

def evaluate_and_plot():

    SEQ_LEN = 101

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 正在使用设备: {device} 进行推理")

    # --- 读取外推集数据 ---
    df_test = pd.read_csv(TEST_CSV)
    case_ids = df_test['case_id'].astype(str).str.replace('case_', '').astype(int).values        

    feature_cols = [col for col in df_test.columns if col != 'case_id']
    features = df_test[feature_cols].values
    
    INPUT_SIZE = len(feature_cols) + 1 # 特征列 + 1个时间列
    
    # --- 加载训练好的模型 ---
    model = ReactorPowerLSTM(INPUT_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # 极其重要：关闭 Dropout，固定 Batch Norm
    print("✅ 模型权重加载成功！")

    # --- 随机挑选 6 个外推工况进行展示 ---
    # 你也可以把 random.sample 换成特定的 case_id 列表，比如 [0, 10, 50...] 去看最极端的几个
    sample_indices = random.sample(range(len(case_ids)), min(6, len(case_ids)))
    
    # 设置画布大小 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # 预生成时间步 t_steps
    t_steps = np.linspace(0, 1, SEQ_LEN).reshape(-1, 1)

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            case_id = case_ids[idx]
            static_params = features[idx]
            
            # 读取真实的功率曲线
            npy_path = os.path.join(NPY_DIR, f"power_{case_id:04d}.npy")
            true_log_power = np.load(npy_path).flatten() # 展平为 1D
            
            # 构造输入张量
            static_params_seq = np.tile(static_params, (SEQ_LEN, 1))
            x_seq = np.hstack([static_params_seq, t_steps])
            
            # 转换为 Tensor，增加 batch 维度 (1, SEQ_LEN, INPUT_SIZE)
            x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 模型预测
            pred_log_power = model(x_tensor)
            # 移回 CPU 并转为 numpy 数组展平
            pred_log_power = pred_log_power.squeeze().cpu().numpy() 
            
            # --- 开始画图 ---
            ax = axes[i]
            x_axis = np.arange(SEQ_LEN)
            
            # 画真实值与预测值 (对数功率)
            ax.plot(x_axis, true_log_power, label='True ln(P) [TWIGL]', color='black', linewidth=2.5, linestyle='-')
            ax.plot(x_axis, pred_log_power, label='Predicted ln(P) [LSTM]', color='red', linewidth=2, linestyle='--')
            
            # 计算当前 Case 的 RMSE
            rmse = np.sqrt(np.mean((true_log_power - pred_log_power)**2))
            
            # 图表美化
            ax.set_title(f"Extrapolation Case {case_id:04d}\nRMSE: {rmse:.4f}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel("ln(Total Power)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='best')

    plt.suptitle("LSTM Extrapolation Test: True vs Predicted Power Trajectories", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存到: {OUT_PLOT_PATH}")

if __name__ == "__main__":
    evaluate_and_plot()