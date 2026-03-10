import os
import re
import csv
import numpy as np
from scipy.stats import qmc

# =====================================================================
# [1] 运行时需调整的全局配置参数 (Runtime Configurations)
# =====================================================================
NUM_CASES = 250  # 计划生成的算例总数

# 连续物理参数 LHS 采样范围
SLOPE_BOUNDS = [-0.3, -0.01]  # 扰动斜率 (Slope_Up) 的范围
CUT_TIME_BOUNDS = [0.1, 0.4]  # 扰动截止时间 (Cut_Time) 的范围

# 离散物理参数选项
MAT_CHOICES = [1, 3]          # 扰动区域：1(核心区) 或 3(控制棒动作区)
GROUP_CHOICES = [1, 2]        # 扰动能群：1(快群) 或 2(热群)

# 路径配置
TEMPLATE_FILE = "data-gen/2D_TWIGL/twigl_diff_ramp_quarter.prm" # 模板文件路径
PRM_OUTPUT_DIR = "data-gen/prm_cases"                           # 生成的 PRM 文件存放目录
RAW_DATA_BASE_DIR = "data-raw/2D_TWIGL_diff"                    # FEMFFUSION 运行结果存放总目录
METADATA_FILE = "data-gen/dataset_parameters.csv"               # 记录生成参数的记录文件 (用于后期训练)
# =====================================================================

def generate_prm_dataset():
    """
    基于 TWIGL 模板批量生成具有不同扰动参数的 PRM 文件，并记录参数。
    """
    # 创建统一的基础目录
    os.makedirs(PRM_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_BASE_DIR, exist_ok=True)

    # 读取模板文件
    if not os.path.exists(TEMPLATE_FILE):
        raise FileNotFoundError(f"找不到模板文件: {TEMPLATE_FILE}")
        
    with open(TEMPLATE_FILE, "r") as f:
        template = f.read()

    # 进行 LHS 采样并缩放到目标物理区间
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=NUM_CASES)
    scaled_samples = qmc.scale(sample, 
                               [SLOPE_BOUNDS[0], CUT_TIME_BOUNDS[0]], 
                               [SLOPE_BOUNDS[1], CUT_TIME_BOUNDS[1]])
    
    slopes = scaled_samples[:, 0]
    cut_times = scaled_samples[:, 1]

    print(f"🚀 开始生成 {NUM_CASES} 个 TWIGL 训练算例...")

    # 准备写入 CSV 参数记录文件
    with open(METADATA_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头，这正是你以后神经网络的输入特征名
        csv_writer.writerow(['case_id', 'material_changing', 'group_changing', 'slope_up', 'cut_time'])

        for i in range(NUM_CASES):
            # 1. 参数赋值
            mat_change = np.random.choice(MAT_CHOICES)  
            group_change = np.random.choice(GROUP_CHOICES) 
            slope = slopes[i]
            cut_time = cut_times[i]
            case_id = f"case_{i:03d}"
            
            # 2. 为当前算例创建专属的 FEMFFUSION 输出文件夹
            case_out_dir = os.path.join(RAW_DATA_BASE_DIR, case_id)
            os.makedirs(case_out_dir, exist_ok=True)
            
            # 3. 复制模板内容进行正则替换
            content = template
            
            # 强制修改 VTK 输出细分度为 0，防止训练集数据量爆炸
            content = re.sub(r'set Out_Refinements\s*=\s*\d+', 'set Out_Refinements     = 0', content)
            
            # 替换物理扰动参数
            content = re.sub(r'set Slope_Up\s*=\s*[-.\d]+', f'set Slope_Up = {slope:.5f}', content)
            content = re.sub(r'set Cut_Time\s*=\s*[-.\d]+', f'set Cut_Time = {cut_time:.4f}', content)
            content = re.sub(r'set Material_Changing\s*=\s*\d+', f'set Material_Changing = {mat_change}', content)
            content = re.sub(r'set Group_Changing\s*=\s*\d+', f'set Group_Changing = {group_change}', content)
            
            # 替换输出路径，指向刚才创建的专属文件夹
            out_path = f"{case_out_dir}/{case_id}.out"
            content = re.sub(r'set Output_Filename\s*=\s*\S+', f'set Output_Filename     = {out_path}', content)
            
            # 4. 将新内容写入到对应的 .prm 文件中
            new_prm_path = os.path.join(PRM_OUTPUT_DIR, f"{case_id}.prm")
            with open(new_prm_path, "w") as f:
                f.write(content)
                
            # 5. 将当前算例的参数记录到 CSV 文件中
            csv_writer.writerow([case_id, mat_change, group_change, f"{slope:.5f}", f"{cut_time:.4f}"])

    print(f"✅ 成功生成 {NUM_CASES} 个输入卡，保存在 '{PRM_OUTPUT_DIR}/'。")
    print(f"✅ 训练参数记录已保存至 '{METADATA_FILE}'。")
    print(f"✅ 输出文件夹架构已在 '{RAW_DATA_BASE_DIR}/' 创建完毕。")

if __name__ == "__main__":
    generate_prm_dataset()