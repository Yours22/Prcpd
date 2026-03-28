import pyvista as pv
import matplotlib.pyplot as plt

def diagnose_vtk_physical_field(vtk_file_path):
    # 1. 读取单文件
    mesh = pv.read(vtk_file_path)
    
    # 2. 提取物理几何坐标和通量数据
    # mesh.points 是一个 (N, 3) 的数组，包含 x, y, z 坐标
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    
    fast_flux = mesh.point_data['Fast_Flux']
    thermal_flux = mesh.point_data['Thermal_Flux']
    
    # 3. 绕过任何 reshape，直接基于真实坐标映射颜色
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 快中子散点图
    sc1 = axes[0].scatter(x, y, c=fast_flux, cmap='jet', marker='s', s=100)
    axes[0].set_title("Fast Flux (Based on True Coordinates)")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    plt.colorbar(sc1, ax=axes[0])
    
    # 热中子散点图
    sc2 = axes[1].scatter(x, y, c=thermal_flux, cmap='jet', marker='s', s=100)
    axes[1].set_title("Thermal Flux (Based on True Coordinates)")
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Y coordinate")
    plt.colorbar(sc2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("vtk_diagnostic_true_coordinates.png", dpi=300)
    print("基于真实物理坐标的诊断图已保存为: vtk_diagnostic_true_coordinates.png")

if __name__ == "__main__":
    # 请将此处替换为你的 raw_data_dir 中实际存在的一个 VTK 文件路径
    # 例如: ".\\dataset_raw\\outputs\\case_0000\\case_0000.out99.vtk"
    test_vtk_path = ".\\data-raw\\2D_TWIGL_diff\case_0000\\case_0000.out99.vtk"
    diagnose_vtk_physical_field(test_vtk_path)