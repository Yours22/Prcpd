import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 全局路径配置 (常量大写)
# ==========================================
VAL_CSV = '1D-power-prediction/dataset_val_processed.csv'
EXTRAPOL_CSV = '1D-power-prediction/dataset_test_extrapolation_processed.csv'
RAW_CSV = 'dataset_parameters_cleaned.csv' 
NPY_DIR = 'data-processed/power_series_log/'
MODEL_PATH = 'models/best_lstm_model.pth' 
PLOT_SAVE_DIR = '1D-power-prediction/plots/' 
SEQ_LEN = 101

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

# ==========================================
# 2. 推理与绘图核心函数
# ==========================================
def evaluate_and_plot(dataset_type='val'):
    # 自动创建图片保存目录
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    
    if dataset_type == 'val':
        target_csv = VAL_CSV
        title_prefix = "Validation (Interpolation) Set"
        save_filename = "plot_validation_results.png"
    elif dataset_type == 'extrapol':
        target_csv = EXTRAPOL_CSV
        title_prefix = "Extrapolation Test Set"
        save_filename = "plot_extrapolation_results.png"
    else:
        raise ValueError("dataset_type 只能是 'val' 或 'extrapol'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 正在读取 [{title_prefix}] 进行推理和绘图...")

    # --- 1. 读取处理后的特征表 ---
    df_test = pd.read_csv(target_csv)
    if 'case_id' not in df_test.columns:
        raise ValueError("CSV 文件中找不到 'case_id' 列。")
    case_ids = df_test['case_id'].astype(str).str.replace('case_', '').astype(int).values
    
    feature_cols = [col for col in df_test.columns if col != 'case_id']
    features = df_test[feature_cols].values
    
    # --- 2. 读取原始物理参数表 (用于计算动态扰动) ---
    df_raw = pd.read_csv(RAW_CSV)
    df_raw['case_id_clean'] = df_raw['case_id'].astype(str).str.replace('case_', '').astype(int)
    df_raw = df_raw.set_index('case_id_clean')
    
    # 【更新】：输入维度 = 静态特征 + 1个归一化时间步 + 1个动态反应性
    INPUT_SIZE = len(feature_cols) + 2 
    
    # --- 3. 加载训练好的模型 ---
    model = ReactorPowerLSTM(INPUT_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- 4. 随机挑选 6 个工况进行展示 ---
    sample_indices = random.sample(range(len(case_ids)), min(6, len(case_ids)))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # 预生成双时间轴
    t_steps_norm = np.linspace(0, 1, SEQ_LEN).reshape(-1, 1) # 归一化时间
    t_phys = np.linspace(0, 0.5, SEQ_LEN) # TWIGL 物理时间 [0, 0.5s]

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            case_id = case_ids[idx]
            static_params = features[idx]

            raw_slope = df_raw.loc[case_id, 'slope_up']
            raw_cut = df_raw.loc[case_id, 'cut_time']
            
            dynamic_reactivity = np.zeros((SEQ_LEN, 1))
            for j, t in enumerate(t_phys):
                if t <= raw_cut:
                    dynamic_reactivity[j, 0] = raw_slope * t
                else:
                    dynamic_reactivity[j, 0] = raw_slope * raw_cut
            
            # 读取真实的功率曲线
            npy_path = os.path.join(NPY_DIR, f"power_{case_id:04d}.npy")
            true_log_power = np.load(npy_path).flatten()
            
            # 横向拼接所有张量：静态特征 + 时间特征 + 动态物理特征
            static_params_seq = np.tile(static_params, (SEQ_LEN, 1))
            x_seq = np.hstack([static_params_seq, t_steps_norm, dynamic_reactivity])
            
            x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 模型预测
            pred_log_power = model(x_tensor).squeeze().cpu().numpy() 
            
            # --- 开始画图 ---
            ax = axes[i]
            x_axis = np.arange(SEQ_LEN)
            
            ax.plot(x_axis, true_log_power, label='True ln(P) [TWIGL]', color='black', linewidth=2.5, linestyle='-')
            ax.plot(x_axis, pred_log_power, label='Predicted ln(P) [LSTM]', color='blue', linewidth=2, linestyle='--')
            
            rmse = np.sqrt(np.mean((true_log_power - pred_log_power)**2))
            
            ax.set_title(f"Case {case_id:04d}\nRMSE: {rmse:.4f}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel("ln(Total Power)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='best')

    plt.suptitle(f"LSTM {title_prefix}: True vs Predicted Power", fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_path = os.path.join(PLOT_SAVE_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已成功生成并保存至: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    evaluate_and_plot(dataset_type='val') 
    evaluate_and_plot(dataset_type='extrapol')