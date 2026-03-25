import os
import numpy as np
import pandas as pd
import pyvista as pv

# ================= 路径配置 =================
RAW_DATA_DIR = "data-raw/2D_TWIGL_diff"
SPLIT_DATA_DIR = "data-split"
NPY_SAVE_DIR = "2D-POD-DNN/data"

os.makedirs(NPY_SAVE_DIR, exist_ok=True)

# ================= 辅助函数 =================
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
    fast_flux = mesh.point_data['Fast_Flux']
    thermal_flux = mesh.point_data['Thermal_Flux']
    return np.hstack((fast_flux, thermal_flux))

def process_and_save(csv_file, prefix):
    print(f"正在处理 {csv_file} ...")
    df = pd.read_csv(os.path.join(SPLIT_DATA_DIR, csv_file))
    
    X_list, Y_list = [], []
    
    for _, row in df.iterrows():
        case_id = row['case_id']
        
        # 提取特征参数用于计算
        mat_changing = row['material_changing']
        grp_changing = row['group_changing']
        slope_up = row['slope_up']
        cut_time = row['cut_time']
        
        case_dir = os.path.join(RAW_DATA_DIR, case_id)
        out_file = os.path.join(case_dir, f"{case_id}.out")
        
        if not os.path.exists(out_file): 
            continue
            
        time_vector = extract_time_vector(out_file)
        for k, t in enumerate(time_vector):
            vtk_file = os.path.join(case_dir, f"{case_id}.out{k}.vtk")
            if os.path.exists(vtk_file):
                # 显式计算当前时刻的微扰状态 p(t)
                if t <= cut_time:
                    p_t = slope_up * t
                else:
                    p_t = slope_up * cut_time
                
                # 新的特征向量 (4维): [材料, 能群, 当前微扰量, 当前时间]
                features = [mat_changing, grp_changing, p_t, t]
                
                X_list.append(features)
                Y_list.append(extract_flux(vtk_file))
                
    X_matrix = np.array(X_list)
    Y_matrix = np.array(Y_list) # 这里是原始高维数据，不再做 SVD
    
    # 保存数据 (确保 NPY_SAVE_DIR 在全局已定义)
    np.save(os.path.join(NPY_SAVE_DIR, f"X_{prefix}.npy"), X_matrix)
    np.save(os.path.join(NPY_SAVE_DIR, f"Y_{prefix}_raw.npy"), Y_matrix)
    print(f" -> {prefix} 集处理完成: X形状 {X_matrix.shape}, Y形状 {Y_matrix.shape}")

if __name__ == "__main__":
    process_and_save('dataset_train.csv', 'train')
    process_and_save('dataset_val.csv', 'val')
    process_and_save('dataset_test_extrapolation.csv', 'test')
    print("所有原始数据提取完毕！")