import pyvista as pv
import numpy as np
import os
import tqdm

num_samples = 200    # 组数
num_timesteps = 100  # 每组时间步数
num_nodes = 1600     # 节点数
num_features = 2     # 特征数，即快，热中子群

dataset_matrix = np.zeros((num_samples, num_timesteps, num_nodes * num_features))

print("开始处理数据...")
data_dir = '.\\dataset_raw\\outputs'

def read_vtk_file(filename):
    mesh = pv.read(filename)

    fast_flux = mesh.point_data['Fast_Flux']
    thermal_flux = mesh.point_data['Thermal_Flux']
    
    data = np.hstack((fast_flux, thermal_flux)) 
    
    return data


def build_dataset():
    print("开始处理数据...")
    for sample_id in tqdm.tqdm(range(num_samples),desc="Processing samples"):
        for t in tqdm.tqdm(range(num_timesteps),desc="Processing timesteps"):
            filename = os.path.join(data_dir, f"sample_{sample_id}.out{t}.vtk")
            if not os.path.exists(filename):
                raise FileNotFoundError(f"文件不存在: {filename}")
            try:
                data = read_vtk_file(filename)
                dataset_matrix[sample_id, t, :] = data
            except Exception as e:
                print(f"读取文件失败: {filename} -> {e}")
                return False
    return True

def main():
    success = build_dataset()
    if not success:
        print("数据处理未完成，存在错误。")
    else:
        print("数据处理完成。")

    np.save('.\\data-gen\\reactor_data_200x100.npy', dataset_matrix)
    print("数据已保存")

if __name__ == "__main__":
    main()