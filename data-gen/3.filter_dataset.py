import os
import shutil
import pandas as pd

# ==========================================
# 1. 运行配置与物理阈值
# ==========================================
RAW_DATA_DIR = "data-raw/2D_TWIGL_diff"
ORIGINAL_CSV = "data-gen/dataset_parameters.csv"
CLEANED_CSV  = "data-gen/dataset_parameters_cleaned.csv"

# 物理与数值稳定性判定标准
MAX_POWER_RATIO = 50.0   # 阈值：相对功率如果超过初始值的 50 倍，判定为数值发散 (爆炸)
MIN_POWER_RATIO = 0.0    # 阈值：相对功率不能为负数
NO_CHANGE_TOLERANCE = 1e-5 # 阈值：如果瞬态结束时，功率变化小于此值，判定为未发生物理响应

def extract_power(filepath):
    """从 .out 文件中提取功率向量"""
    powers = []
    mode = None
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 状态机读取功率块
            if "Total Power vector" in line:
                mode = "power"
                continue
            elif "Error estimation" in line or "CPU Time" in line:
                if mode == "power":
                    break # 功率读取完毕，提前退出循环
                mode = None
                continue
                
            if mode == "power":
                try:
                    powers.extend([float(x) for x in line.split()])
                except ValueError:
                    mode = None
    return powers

def main():
    print("==================================================")
    print("🧹 开始执行数据集物理与数值稳定性清洗...")
    print("==================================================")
    
    if not os.path.exists(ORIGINAL_CSV):
        print(f"❌ 找不到参数记录文件: {ORIGINAL_CSV}")
        return

    # 读取原始参数列表
    df = pd.read_csv(ORIGINAL_CSV)
    total_cases = len(df)
    print(f"📄 原始数据集包含 {total_cases} 个算例。")
    
    indices_to_drop = []
    reason_counts = {"diverged": 0, "no_change": 0, "missing_or_crashed": 0}

    # 逐一检查算例
    for index, row in df.iterrows():
        case_id = row['case_id']
        case_dir = os.path.join(RAW_DATA_DIR, case_id)
        out_file = os.path.join(case_dir, f"{case_id}.out")
        
        powers = extract_power(out_file)
        
        # 1. 检查是否缺失或程序中途崩溃(无数据)
        if not powers or len(powers) < 2:
            print(f"  [移除] {case_id}: 文件缺失或未计算完成。")
            indices_to_drop.append(index)
            reason_counts["missing_or_crashed"] += 1
            continue
            
        p0 = powers[0]
        # 规避分母为0的极端情况
        if p0 == 0: 
            p0 = 1e-10 
            
        normalized_powers = [p / p0 for p in powers]
        max_p = max(normalized_powers)
        min_p = min(normalized_powers)
        end_p = normalized_powers[-1]
        
        is_bad = False
        
        # 2. 检查是否数值发散 (例如达到了 10^57)
        if max_p > MAX_POWER_RATIO or min_p < MIN_POWER_RATIO:
            print(f"  [移除] {case_id}: 数值发散 (最大相对功率: {max_p:.2e})。")
            is_bad = True
            reason_counts["diverged"] += 1
            
        # 3. 检查是否没有发生物理响应 (功率一直是一条直线)
        elif abs(end_p - 1.0) < NO_CHANGE_TOLERANCE:
            print(f"  [移除] {case_id}: 未发生物理响应 (相对功率无变化)。")
            is_bad = True
            reason_counts["no_change"] += 1
            
        # 物理删除坏数据的文件夹以释放空间
        if is_bad:
            indices_to_drop.append(index)
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir) # 危险操作：彻底删除该算例的文件夹及所有 vtk
                
    # 从 DataFrame 中剔除坏算例
    df_cleaned = df.drop(indices_to_drop)
    
    # 保存干净的 CSV 供神经网络使用
    df_cleaned.to_csv(CLEANED_CSV, index=False)
    
    print("\n==================================================")
    print("✅ 清洗完成！统计报告：")
    print(f"   - 原始算例数: {total_cases}")
    print(f"   - 发散/爆炸剔除: {reason_counts['diverged']}")
    print(f"   - 无响应剔除:   {reason_counts['no_change']}")
    print(f"   - 缺失/崩溃剔除: {reason_counts['missing_or_crashed']}")
    print(f"   ----------------------------------")
    print(f"   - 最终保留的高质量算例数: {len(df_cleaned)}")
    print("==================================================")
    print(f"📁 新的特征表格已保存至: {CLEANED_CSV}")
    print(f"🗑️ 对应的损坏文件夹已被彻底删除，释放了硬盘空间。")

if __name__ == "__main__":
    main()