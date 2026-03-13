import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==========================================
# 路径配置
# ==========================================
CSV_FILE = 'dataset_parameters_cleaned.csv'
NPY_DIR = 'data-processed/power_series_log/'

def verify_npy_files(csv_file, npy_dir):
    print("🔍 开始自动化体检...\n")
    
    df = pd.read_csv(csv_file)
    case_ids = df['case_id'].astype(str).str.replace('case_', '').astype(int).values
    
    total_files = len(case_ids)
    missing_files = []
    bad_shape_files = []
    nan_inf_files = []
    
    lengths = []
    valid_case_ids = []

    for case_id in case_ids:
        npy_path = os.path.join(npy_dir, f"power_{case_id:04d}.npy")
        
        # 1. 检查文件是否存在
        if not os.path.exists(npy_path):
            missing_files.append(case_id)
            continue
            
        try:
            # 读取数据
            data = np.load(npy_path)
            valid_case_ids.append(case_id)
            
            # 2. 检查数据形状 (应该是 二维的 [Seq_Len, 1])
            if len(data.shape) != 2 or data.shape[1] != 1:
                bad_shape_files.append((case_id, data.shape))
            else:
                lengths.append(data.shape[0])
                
            # 3. 检查是否有 NaN 或 Inf
            if np.isnan(data).any() or np.isinf(data).any():
                nan_inf_files.append(case_id)
                
        except Exception as e:
            print(f"读取 Case {case_id} 时发生错误: {e}")

    # ==========================================
    # 打印体检报告
    # ==========================================
    print("📊 【数据体检报告】 📊")
    print(f"总计检查文件: {total_files} 个")
    
    if missing_files:
        print(f"❌ 丢失文件数: {len(missing_files)} (例如: {missing_files[:5]}...)")
    else:
        print("✅ 没有丢失任何文件。")
        
    if bad_shape_files:
        print(f"❌ 形状异常数: {len(bad_shape_files)} (预期应为 [N, 1])")
    else:
        print("✅ 所有文件形状符合 LSTM 输入标准 [Seq_Len, 1]。")
        
    if nan_inf_files:
        print(f"❌ 包含 NaN/Inf 的文件数: {len(nan_inf_files)} (严重错误，会毁掉训练！)")
    else:
        print("✅ 数据极其健康，没有发现 NaN 或 Inf 脏数据。")
        
    if lengths:
        unique_lengths = set(lengths)
        if len(unique_lengths) == 1:
            print(f"✅ 所有时间序列长度完美一致，均为 {unique_lengths.pop()} 步。")
        else:
            print(f"⚠️ 警告！序列长度不一致，存在以下几种长度: {unique_lengths}")
            print("如果长度不一致，在构建 PyTorch DataLoader 时必须进行 Padding（补齐）或截断处理！")

    # ==========================================
    # 视觉抽查 (画图)
    # ==========================================
    if valid_case_ids:
        print("\n🎨 正在随机抽取 5 组数据绘制曲线图，请查看弹出的图表...")
        plot_random_cases(valid_case_ids, npy_dir, num_plots=5)

def plot_random_cases(valid_case_ids, npy_dir, num_plots=5):
    # 随机挑选几个 Case
    sample_cases = random.sample(valid_case_ids, min(num_plots, len(valid_case_ids)))
    
    plt.figure(figsize=(10, 6))
    for case_id in sample_cases:
        npy_path = os.path.join(npy_dir, f"power_{case_id:04d}.npy")
        data = np.load(npy_path)
        
        # 将二维 [N, 1] 展平为一维 [N] 方便画图
        y_values = data.flatten() 
        x_values = np.arange(len(y_values))
        
        plt.plot(x_values, y_values, label=f"Case {case_id:04d}", linewidth=2)
        
    plt.title("Random Sample of log(Power) Trajectories", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("ln(Total Power)", fontsize=12) # 注意这里是 ln(P)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # 如果你在没有图形界面的服务器上跑，请把 plt.show() 换成 plt.savefig('check_plot.png')
    plt.show() 

if __name__ == "__main__":
    verify_npy_files(CSV_FILE, NPY_DIR)