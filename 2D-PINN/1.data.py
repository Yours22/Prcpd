import os
import numpy as np
import pandas as pd
import pyvista as pv
import yaml

# 读取配置
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PATHS = config['paths']
PHYSICS = config['physics']

os.makedirs(PATHS['processed_dir'], exist_ok=True)

def extract_time_vector(out_file_path):
    time_vector = []
    with open(out_file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Time vector" in line:
                for j in range(i+1, len(lines)):
                    parts = lines[j].strip().split()
                    if not parts: break
                    try: time_vector.extend([float(x) for x in parts])
                    except ValueError: break
                break
    return np.array(time_vector)

def extract_flux(vtk_file_path):
    mesh = pv.read(vtk_file_path)
    fast_flux_raw = mesh.point_data['Fast_Flux'].copy()
    thermal_flux_raw = mesh.point_data['Thermal_Flux'].copy()
    mesh.clear_data() # 防止 PyVista 内存泄漏
    
    def reorder_to_image(raw_data):
        n_blocks_y, n_blocks_x = 10, 10
        block_size = 2  # 2x2 = 4 个节点/组件
        
        # 1. 恢复有限元组件结构 (10, 10, 2, 2)
        reshaped = raw_data.reshape(n_blocks_y, n_blocks_x, block_size, block_size)
        
        # 2. 张量轴交换：将组件行与节点行合并，组件列与节点列合并
        transposed = reshaped.transpose(0, 2, 1, 3)
        
        # 3. 展平为连续的一维向量，完美对应 20x20 的行优先图像
        return transposed.reshape(-1)

    fast_flux = reorder_to_image(fast_flux_raw)
    thermal_flux = reorder_to_image(thermal_flux_raw)
    
    return np.hstack((fast_flux, thermal_flux))

def process_and_save(csv_file, prefix):
    csv_path = os.path.join(PATHS['split_data_dir'], csv_file)
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    num_cases = len(df)
    
    # 【核心重构：将基础特征从 4 维扩展至 6 维】
    # 特征顺序: [is_reg1, is_reg2, is_fast, is_thermal, p_t, t]
    X_matrix = np.zeros((num_cases, PHYSICS['num_time_steps'], 6), dtype=np.float32)
    Y_matrix = np.zeros((num_cases, PHYSICS['num_time_steps'], PHYSICS['total_nodes']), dtype=np.float32)
    
    valid_indices = []
    
    for i, row in df.iterrows():
        raw_case_id = str(row['case_id'])
        case_id = raw_case_id if raw_case_id.startswith('case_') else f"case_{int(float(raw_case_id)):04d}"
        
        # 1. 解析分类变量并进行物理独热编码 (One-Hot)
        # 彻底废除 1.0/2.0 的标量大小压制，改为平等的空间开关
        mat_changing = int(row['material_changing'])
        grp_changing = int(row['group_changing'])
        
        is_reg1 = 1.0 if mat_changing == 1 else 0.0
        is_reg2 = 1.0 if mat_changing == 2 else 0.0
        is_fast = 1.0 if grp_changing == 1 else 0.0
        is_thermal = 1.0 if grp_changing == 2 else 0.0
        
        slope_up = float(row['slope_up'])
        cut_time = float(row['cut_time'])
        
        case_dir = os.path.join(PATHS['raw_data_dir'], case_id)
        out_file = os.path.join(case_dir, f"{case_id}.out")
        if not os.path.exists(out_file): continue
            
        time_vector = extract_time_vector(out_file)
        if len(time_vector) != PHYSICS['num_time_steps']: continue
        
        is_valid = True
        for k, t in enumerate(time_vector):
            vtk_file = os.path.join(case_dir, f"{case_id}.out{k}.vtk")
            if not os.path.exists(vtk_file): 
                is_valid = False
                break
                
            # 2. 计算连续的物理驱动力 (微扰幅值)
            p_t = slope_up * t if t <= cut_time else slope_up * cut_time
            
            # 3. 组装全新的无歧义特征向量
            X_matrix[i, k, :] = [is_reg1, is_reg2, is_fast, is_thermal, p_t, t]
            Y_matrix[i, k, :] = extract_flux(vtk_file)
            
        if is_valid: valid_indices.append(i)

    X_matrix = X_matrix[valid_indices]
    Y_matrix = Y_matrix[valid_indices]
    
    np.save(os.path.join(PATHS['processed_dir'], f"X_{prefix}.npy"), X_matrix)
    np.save(os.path.join(PATHS['processed_dir'], f"Y_{prefix}_raw.npy"), Y_matrix)
    print(f"{prefix} 集处理完成: 全新特征 X 形状 {X_matrix.shape}, 物理场 Y 形状 {Y_matrix.shape}")

if __name__ == "__main__":
    process_and_save('dataset_train.csv', 'train')
    process_and_save('dataset_val.csv', 'val')
    process_and_save('dataset_test_extrapolation.csv', 'test')