import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

# ==========================================
# 配置路径
# ==========================================
RAW_DATA_DIR = "data-raw/2D_TWIGL_diff"
CSV_FILE = "data-gen/dataset_parameters.csv"
OUTPUT_IMG_DIR = "data-check-plots"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
pv.set_plot_theme('document')
pv.global_theme.show_scalar_bar = True

def plot_parameter_space():
    """图 4：绘制 LHS 采样参数空间散点图"""
    print("正在绘制 图4：参数空间分布...")
    if not os.path.exists(CSV_FILE):
        print(f"找不到 {CSV_FILE}，跳过参数图绘制。")
        return
        
    df = pd.read_csv(CSV_FILE)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['cut_time'], df['slope_up'], 
                          c=df['material_changing'], cmap='coolwarm', 
                          alpha=0.8, edgecolors='w', s=80)
    
    plt.title("LHS Parameter Space Coverage", fontsize=14)
    plt.xlabel("Cut Time [s]", fontsize=12)
    plt.ylabel("Slope Up (Reactivity Insertion Rate)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Material Changing (1: Core, 3: Rod)', fontsize=10)
    
    out_path = os.path.join(OUTPUT_IMG_DIR, "Fig4_Parameter_Space.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> 已保存至 {out_path}")

def plot_power_curves(num_samples=10):
    """图 1：绘制归一化总功率随时间演化曲线"""
    print(f"正在绘制 图1：总功率演化曲线 (抽取 {num_samples} 个算例)...")
    
    out_files = glob.glob(os.path.join(RAW_DATA_DIR, "case_*", "*.out"))
    if not out_files:
        print("找不到任何 .out 文件，请检查路径。")
        return
        
    np.random.shuffle(out_files)
    selected_files = out_files[:num_samples]
    
    plt.figure(figsize=(10, 6))
    
    for file in selected_files:
        times = []
        powers = []
        case_name = os.path.basename(file).split('.')[0]
        
        mode = None # 记录当前读取的状态
        
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 状态机：遇到特定的表头，切换读取模式
                if "Time vector" in line:
                    mode = "time"
                    continue
                elif "Total Power vector" in line:
                    mode = "power"
                    continue
                elif "Error estimation" in line or "CPU Time" in line:
                    mode = None # 遇到其他无关表头，停止读取
                    continue
                
                # 根据当前模式提取数据
                if mode == "time":
                    try:
                        vals = [float(x) for x in line.split()]
                        times.extend(vals)
                    except ValueError:
                        mode = None
                elif mode == "power":
                    try:
                        vals = [float(x) for x in line.split()]
                        powers.extend(vals)
                    except ValueError:
                        mode = None
        
        if times and powers:
            # 确保时间数组和功率数组长度匹配
            min_len = min(len(times), len(powers))
            times = times[:min_len]
            powers = powers[:min_len]
            
            p0 = powers[0]
            normalized_powers = [p / p0 for p in powers]
            plt.plot(times, normalized_powers, alpha=0.7, linewidth=2, label=case_name)

    plt.title("Normalized Total Power Transient Response", fontsize=14)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Relative Power P(t)/P(0)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_path = os.path.join(OUTPUT_IMG_DIR, "Fig1_Power_Curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> 已保存至 {out_path}")

def plot_flux_vtk(vtk_file, out_name, title_text):
    """通用的 VTK 云图绘制函数 (图2 和 图3 使用)"""
    if not os.path.exists(vtk_file):
        print(f"找不到 {vtk_file}，跳过云图绘制。")
        return
        
    mesh = pv.read(vtk_file)
    array_name = None
    valid_keys = mesh.point_data.keys()
    
    # 按照优先级搜索热群通量 (Thermal Flux / Group 2 Flux)
    if 'Thermal_Flux' in valid_keys:
        array_name = 'Thermal_Flux'
    elif 'phi_g2_eig_1' in valid_keys:
        array_name = 'phi_g2_eig_1'
    else:
        for key in valid_keys:
            if 'g2' in key.lower() or 'thermal' in key.lower():
                array_name = key
                break
            
    if array_name is None:
        print(f"在 {vtk_file} 中找不到热群通量数据。包含的数据有: {list(valid_keys)}")
        return

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, scalars=array_name, cmap='jet', show_edges=False)
    plotter.view_xy()
    plotter.add_text(title_text, font_size=12, position='upper_edge')
    
    out_path = os.path.join(OUTPUT_IMG_DIR, out_name)
    plotter.screenshot(out_path)
    plotter.close()
    print(f" -> 已保存至 {out_path} (使用的变量名: {array_name})")

def plot_spatial_snapshots():
    """图 2 & 图 3：绘制稳态和瞬态末期的二维通量分布"""
    print("正在绘制 图2 & 图3：二维空间通量快照...")
    
    case_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "case_*")))
    if not case_dirs:
        print("找不到任何算例目录。")
        return
        
    case_dir = case_dirs[0] # 取第一个算例作为展示
    
    vtk_files = []
    # 兼容两种文件后缀命名机制 (例如 case_000.out.vtk 和 case_000.out0.vtk)
    for ext in ("*.out.vtk", "*.out[0-9]*.vtk", "*.vtk"):
        files = glob.glob(os.path.join(case_dir, ext))
        if files:
            vtk_files.extend(files)
            
    # 去重并排序，确保取到第一帧和最后一帧
    vtk_files = sorted(list(set(vtk_files)))
    
    if len(vtk_files) < 2:
        print(f"在 {case_dir} 中找到的 .vtk 文件不足以绘制对比图。")
        return

    # 第一帧
    plot_flux_vtk(vtk_files[0], "Fig2_Initial_Flux_t0.png", "Initial Thermal Flux (t=0s)")
    # 最后一帧
    plot_flux_vtk(vtk_files[-1], "Fig3_Final_Flux_t_end.png", "Final Thermal Flux")

if __name__ == "__main__":
    print("==================================================")
    print("开始执行数据可视化校验流水线...")
    print("==================================================")
    
    plot_parameter_space()
    plot_power_curves(num_samples=8)
    plot_spatial_snapshots()
    
    print("==================================================")
    print("✅ 所有绘图任务完成！请前往 'data-check-plots/' 文件夹查看结果。")
    print("==================================================")