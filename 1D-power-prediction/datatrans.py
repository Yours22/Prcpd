import os
import re
import numpy as np
import pandas as pd

# ==========================================
# 全局路径配置 (常量大写)
# ==========================================
CSV_FILE = 'dataset_parameters_cleaned.csv'
RAW_OUT_DIR = 'data-raw/2D_TWIGL_diff/'
NPY_SAVE_DIR = 'data-processed/power_series_log/'

def extract_power_to_npy(csv_file, raw_out_dir, npy_save_dir):
    """
    从 TWIGL .out 文件中提取 Total Power vector，
    取对数 ln(P) 后保存为 .npy 格式
    """
    os.makedirs(npy_save_dir, exist_ok=True)
    
    print(f"正在读取参数表: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 获取 case_id，并确保它是整数格式，防止前置 0 丢失
    if 'case_id' not in df.columns:
        raise ValueError("CSV 文件中找不到 'case_id' 列，请检查你的表格列名。")
    case_ids = df['case_id'].astype(str).str.replace('case_', '').astype(int).values

    success_count = 0
    error_cases = []

    print(f"开始提取功率序列，总计 {len(case_ids)} 组...")

    for case_id in case_ids:
        # 严格按照你的文件结构拼接路径: case_0000/case_0000.out
        folder_name = f"case_{case_id:04d}"
        file_name = f"case_{case_id:04d}.out"
        out_file_path = os.path.join(raw_out_dir, folder_name, file_name)
        
        if not os.path.exists(out_file_path):
            print(f"❌ 找不到文件: {out_file_path}")
            error_cases.append(case_id)
            continue
            
        try:
            with open(out_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. 使用正则非贪婪模式，匹配 'Total Power vector' 和 'Error estimation' 之间的所有文本
            # re.DOTALL 允许 '.' 匹配换行符
            block_pattern = r"Total Power vector\s*\n(.*?)\n\s*Error estimation"
            match = re.search(block_pattern, content, re.DOTALL)
            
            if not match:
                print(f"⚠️ Case {case_id}: 未能在文件中找到 'Total Power vector' 数据块！")
                error_cases.append(case_id)
                continue
                
            # 获取匹配到的那一大坨多行数字字符串
            power_block_str = match.group(1)
            
            # 2. 从这个字符串中提取出所有的浮点数 (科学计数法)
            # \d+\.\d+e[\+\-]\d+ 可以精准匹配 1.0068886877e+00 这种格式
            float_pattern = r"\d+\.\d+e[\+\-]\d+"
            power_str_list = re.findall(float_pattern, power_block_str)
            
            if len(power_str_list) == 0:
                print(f"⚠️ case {case_id}: 数据块内未解析出任何浮点数！")
                error_cases.append(case_id)
                continue
                
            # 3. 转换为 float 类型的 numpy 数组
            power_array = np.array(power_str_list, dtype=np.float32)
            
            # 【物理预处理】：取自然对数 ln(P)
            log_power_array = np.log(power_array + 1e-10)
            
            # 转换为二维列向量 shape: (Seq_Len, 1)，这是 LSTM 标准输入格式
            log_power_array = log_power_array.reshape(-1, 1)
            
            # 保存为二进制 .npy 文件
            save_path = os.path.join(npy_save_dir, f"power_{case_id:04d}.npy")
            np.save(save_path, log_power_array)
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理 Case {case_id} 时发生不可预知的错误: {e}")
            error_cases.append(case_id)

    print(f"\n✅ 提取完成！成功: {success_count} 组，失败: {len(error_cases)} 组。")
    print(f"✅ 序列文件已保存至: {npy_save_dir}")
    if error_cases:
        print(f"⚠️ 失败的 case_id 列表: {error_cases}")

# ==========================================
# 执行入口
# ==========================================
if __name__ == "__main__":
    # 调用函数时，直接传入上面定义好的全局常量
    extract_power_to_npy(CSV_FILE, RAW_OUT_DIR, NPY_SAVE_DIR)