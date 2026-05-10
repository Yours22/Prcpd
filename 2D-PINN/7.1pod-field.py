import os
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt

def main():
    print(">>> 开始生成 POD 空间模态与时间系数解析图...")
    
    # 1. 加载配置与路径
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    os.makedirs(PATHS['output_dir'], exist_ok=True)
    
    # 2. 加载 SVD 模型
    svd_fast_path = os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl')
    svd_fast = joblib.load(svd_fast_path)
    components = svd_fast.components_  # (r_fast, N_NODES)
    
    N_NODES = components.shape[1]
    NX = NY = int(np.sqrt(N_NODES))
    
    # 3. 加载训练集并计算平均场
    Y_train = np.load(os.path.join(PATHS['processed_dir'], "Y_test_raw.npy"))
    # Y_train shape: (num_cases, num_time_steps, total_nodes)
    Y_train_fast = Y_train[:, :, :N_NODES].reshape(-1, N_NODES)
    mean_field = Y_train_fast.mean(axis=0).reshape(NY, NX)
    
    # ---------------------------------------------------------
    # 新增核心逻辑：提取时间系数 (以 Case 0 为例)
    # ---------------------------------------------------------
    TARGET_CASE = 0 
    case_data_fast = Y_train[TARGET_CASE, :, :N_NODES]  # 提取单个工况的快群数据 (T, Nodes)
    
    # 通过 SVD 模型将高维空间场投影到低维潜空间，得到时间演化系数 A(t)
    time_coeffs = svd_fast.transform(case_data_fast)    # 形状: (T, r_fast)
    time_steps = np.arange(time_coeffs.shape[0])

    # ================= 4. 开始绘图 =================
    # 创建 2行4列 的画布 (上下对齐：上图空间，下图时间)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), gridspec_kw={'height_ratios': [1.2, 1]})
    
    # ---------------- 第一行：空间模态 ----------------
    # 图 1-0：平均场
    im0 = axes[0, 0].imshow(mean_field, cmap='jet', origin='lower')
    axes[0, 0].set_title("Mean Field\n(Average Base Flux)", fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar0.formatter.set_powerlimits((0, 0))
    
    # 图 1-1, 1-2, 1-3：POD 空间模态 1, 2, 3
    for i in range(3):
        ax = axes[0, i+1]
        mode_data = components[i].reshape(NY, NX)
        vmax = np.max(np.abs(mode_data))
        im = ax.imshow(mode_data, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        
        variance_ratio = svd_fast.explained_variance_ratio_[i] * 100
        if i == 0:
            title = f"Spatial Mode {i+1} (Base Shape)\nEnergy: {variance_ratio:.2f}%"
        else:
            title = f"Spatial Mode {i+1} (Deformation)\nEnergy: {variance_ratio:.4f}%"
            
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))

    # ---------------- 第二行：时间演化系数 ----------------
    # 图 2-0：左下角留白或者放一些说明文字
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, f"Temporal Evolution\n(Case {TARGET_CASE})", 
                    fontsize=18, fontweight='bold', ha='center', va='center')

    # 图 2-1, 2-2, 2-3：对应模态的时间系数曲线
    for i in range(3):
        ax = axes[1, i+1]
        # 提取第 i 阶模态的时间序列
        coeff_series = time_coeffs[:, i]
        
        # 绘制曲线
        ax.plot(time_steps, coeff_series, color='black', linewidth=2.5)
        
        ax.set_title(f"Amplitude of Mode {i+1}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel(f"Coefficient $a_{i+1}(t)$", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 科学计数法格式化 Y 轴，因为量级差距极大
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # 为了更直观地看出涨落，如果是 Mode 2 和 3，通常会在 y=0 画一条红色虚线参考线
        if i > 0:
            ax.axhline(0, color='red', linestyle=':', linewidth=1.5)

    # 全局排版
    plt.suptitle("Spatiotemporal Decomposition of Fast Flux via POD", fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(PATHS['output_dir'], "pod_spatiotemporal_modes.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> 绘图完成！图片已保存至: {save_path}")

if __name__ == "__main__":
    main()