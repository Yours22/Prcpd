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
    
    # 1. 提取真实的物理几何坐标
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    
    # 2. 生成基于坐标的绝对排序索引
    # 使用 np.round 消除有限元网格可能的浮点数精度误差
    # np.lexsort 接收的键是逆序的：传入 (x, y) 代表优先按 y 排序，再按 x 排序
    # 这将生成标准的行优先（从下到上，从左到右）的一维索引
    sort_idx = np.lexsort((np.round(x, 5), np.round(y, 5)))
    
    # 3. 提取通量数据
    fast_flux_raw = mesh.point_data['Fast_Flux'].copy()
    thermal_flux_raw = mesh.point_data['Thermal_Flux'].copy()
    mesh.clear_data() # 防止 PyVista 内存泄漏
    
    # 4. 应用排序索引，纠正拓扑顺序
    fast_flux_sorted = fast_flux_raw[sort_idx]
    thermal_flux_sorted = thermal_flux_raw[sort_idx]
    
    return np.hstack((fast_flux_sorted, thermal_flux_sorted))

def process_and_save(csv_file, prefix):
    csv_path = os.path.join(PATHS['split_data_dir'], csv_file)
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    num_cases = len(df)
    
    # 【内存优化】: 预分配全零 NumPy 数组，避免动态 append
    X_matrix = np.zeros((num_cases, PHYSICS['num_time_steps'], 4), dtype=np.float32)
    Y_matrix = np.zeros((num_cases, PHYSICS['num_time_steps'], PHYSICS['total_nodes']), dtype=np.float32)
    
    valid_indices = []
    
    for i, row in df.iterrows():
        raw_case_id = str(row['case_id'])
        case_id = raw_case_id if raw_case_id.startswith('case_') else f"case_{int(float(raw_case_id)):04d}"
        
        mat_changing = row['material_changing']
        grp_changing = row['group_changing']
        slope_up = row['slope_up']
        cut_time = row['cut_time']
        
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
                
            p_t = slope_up * t if t <= cut_time else slope_up * cut_time
            X_matrix[i, k, :] = [mat_changing, grp_changing, p_t, t]
            Y_matrix[i, k, :] = extract_flux(vtk_file)
            
        if is_valid: valid_indices.append(i)

    X_matrix = X_matrix[valid_indices]
    Y_matrix = Y_matrix[valid_indices]
    
    np.save(os.path.join(PATHS['processed_dir'], f"X_{prefix}.npy"), X_matrix)
    np.save(os.path.join(PATHS['processed_dir'], f"Y_{prefix}_raw.npy"), Y_matrix)
    print(f"{prefix} 集处理完成: X形状 {X_matrix.shape}, Y形状 {Y_matrix.shape}")

if __name__ == "__main__":
    process_and_save('dataset_train.csv', 'train')
    process_and_save('dataset_val.csv', 'val')
    process_and_save('dataset_test_extrapolation.csv', 'test')