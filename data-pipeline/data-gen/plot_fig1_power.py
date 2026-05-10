import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 运行配置区
# ==========================================
RAW_DATA_DIR = "data-raw/2D_TWIGL_diff"  # 存放 case_xxx 文件夹的总目录
NUM_SAMPLES = 10                         # 你希望随机抽取的算例数量
OUTPUT_IMG = "data-check-plots/Fig1_Power_Curves.png"     # 输出图片的名称

def extract_time_power(filepath):
    """核心解析函数：从 FEMFFUSION 的 .out 文件中提取时间和功率向量"""
    times = []
    powers = []
    mode = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 状态机：识别数据块的开始与结束
            if "Time vector" in line:
                mode = "time"
                continue
            elif "Total Power vector" in line:
                mode = "power"
                continue
            elif "Error estimation" in line or "CPU Time" in line:
                mode = None
                continue
                
            # 根据当前模式提取浮点数
            if mode == "time":
                try:
                    times.extend([float(x) for x in line.split()])
                except ValueError:
                    mode = None
            elif mode == "power":
                try:
                    powers.extend([float(x) for x in line.split()])
                except ValueError:
                    mode = None
                    
    return times, powers

def main():
    print(f"🔍 正在从 '{RAW_DATA_DIR}' 中寻找算例...")
    
    # 获取所有的 .out 文件路径
    all_out_files = glob.glob(os.path.join(RAW_DATA_DIR, "case_*", "*.out"))
    
    if not all_out_files:
        print("❌ 未找到任何 .out 文件，请检查路径设置是否正确！")
        return
        
    total_found = len(all_out_files)
    print(f"✅ 共找到 {total_found} 个算例。")
    
    # 随机抽取指定数量的算例 (如果不放回抽样)
    draw_count = min(NUM_SAMPLES, total_found)
    selected_files = np.random.choice(all_out_files, size=draw_count, replace=False)
    
    print(f"🎲 随机抽取了 {draw_count} 个算例进行绘制...")
    
    # 初始化画布
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 遍历抽中的文件，解析并画图
    for file in selected_files:
        case_name = os.path.basename(file).split('.')[0] # 获取算例名称，例如 'case_000'
        times, powers = extract_time_power(file)
        
        if times and powers:
            # 数据截断对齐（防止 FEMFFUSION 输出被强行中断导致长短不一）
            min_len = min(len(times), len(powers))
            times = times[:min_len]
            powers = powers[:min_len]
            
            # --- 核心归一化逻辑 ---
            # 强行以时刻0的功率值 (P_0) 为基准，计算 P(t) / P_0
            p0 = powers[0]
            normalized_powers = [p / p0 for p in powers]
            
            # 绘制曲线
            plt.plot(times, normalized_powers, alpha=0.8, linewidth=2, label=case_name)
        else:
            print(f"⚠️ 警告: 算例 {case_name} 数据解析失败或为空。")

    # 完善图表美学设置
    plt.title(f"Normalized Total Power Transient Response (Random {draw_count} Cases)", fontsize=16, fontweight='bold')
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("Relative Power $P(t)/P_0$", fontsize=14)
    
    # 刻度线与网格
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.minorticks_on()
    
    # 显示图例 (自动寻找最佳位置)
    plt.legend(loc='best', fontsize=10, ncol=2)
    
    # 强制让 Y 轴起点从稍低于 1.0 的地方开始，以突出初始值
    plt.ylim(bottom=0.95)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    plt.close()
    
    print(f"🎉 绘图成功！已保存为当前目录下的: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()