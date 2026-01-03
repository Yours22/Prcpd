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

    # 1. 获取原始数据 (长度 1600)
    fast_flux_raw = mesh.point_data['Fast_Flux']
    thermal_flux_raw = mesh.point_data['Thermal_Flux']
    
    # 2. 定义重排逻辑：从 (块, 块内) -> (行, 列)
    # 假设结构是 10x10 个块，每个块 4x4 个点
    n_blocks_y = 10
    n_blocks_x = 10
    block_size = 4  # 4x4 points per block
    
    def reorder_to_image(raw_data):
        # 步骤 1: 恢复块结构 (10, 10, 4, 4)
        # 注意：这里假设块的扫描顺序是行优先，如果图不对，尝试交换 n_blocks_y 和 n_blocks_x
        reshaped = raw_data.reshape(n_blocks_y, n_blocks_x, block_size, block_size)
        
        # 步骤 2: 交换轴，把 "块行" 和 "块内行" 放到一起
        # 变换前: (BlockRow, BlockCol, InnerRow, InnerCol) -> (0, 1, 2, 3)
        # 变换后: (BlockRow, InnerRow, BlockCol, InnerCol) -> (0, 2, 1, 3)
        transposed = reshaped.transpose(0, 2, 1, 3)
        
        # 步骤 3: 展平为图像 (40, 40) 然后再展平回 (1600,)
        # 这样以后 reshape(40, 40) 就能得到正确的图像
        return transposed.reshape(-1)

    # 3. 应用重排
    fast_flux_ordered = reorder_to_image(fast_flux_raw)
    thermal_flux_ordered = reorder_to_image(thermal_flux_raw)
    
    # 4. 堆叠 (保持你原有的结构 [Fast, Thermal])
    data = np.hstack((fast_flux_ordered, thermal_flux_ordered))
    
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

    np.save('.\\data-gen\\outputs.npy', dataset_matrix)
    print("outputs.npy 已保存")

if __name__ == "__main__":
    main()