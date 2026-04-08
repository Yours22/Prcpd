import os
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt

def main():
    print(">>> 开始生成 POD 空间模态解析图...")
    
    # 1. 加载配置与路径
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    os.makedirs(PATHS['output_dir'], exist_ok=True)
    
    # 2. 加载 SVD 模型 (以快中子为例)
    svd_fast_path = os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl')
    svd_fast = joblib.load(svd_fast_path)
    components = svd_fast.components_  # 形状: (r_fast, N_NODES)
    
    # 推断网格尺寸 (NX * NY = N_NODES)
    N_NODES = components.shape[1]
    NX = NY = int(np.sqrt(N_NODES))
    
    # 3. 加载训练集计算真实的“平均场 (Mean Field)”
    # 只取快中子部分计算平均值
    Y_train = np.load(os.path.join(PATHS['processed_dir'], "Y_train_raw.npy"))
    Y_train_fast = Y_train[:, :, :N_NODES].reshape(-1, N_NODES)
    mean_field = Y_train_fast.mean(axis=0).reshape(NY, NX)
    
    # ================= 4. 开始绘图 =================
    # 创建 1行4列 的画布，适合 PPT 宽屏展示
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # --- 图 1：平均场 (Mean Field) ---
    # 使用 jet 色带，因为通量全为正值
    im0 = axes[0].imshow(mean_field, cmap='jet', origin='lower')
    axes[0].set_title("Mean Field\n(Average Base Flux)", fontsize=16, fontweight='bold')
    axes[0].axis('off') # 关闭坐标轴显得更干净
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.formatter.set_powerlimits((0, 0))
    
    # --- 图 2~4：POD 模态 1, 2, 3 ---
    # 模态通常有正负交替，使用冷暖色带 RdBu_r，并强制把 0 设为白色(中心)
    for i in range(3):
        ax = axes[i+1]
        mode_data = components[i].reshape(NY, NX)
        
        # 获取绝对值的最大值，确保色阶关于 0 绝对对称
        vmax = np.max(np.abs(mode_data))
        
        im = ax.imshow(mode_data, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        
        # 获取该模态的能量贡献占比
        variance_ratio = svd_fast.explained_variance_ratio_[i] * 100
        
        # 标注
        if i == 0:
            title = f"Mode {i+1} (Principal Shape)\nEnergy: {variance_ratio:.2f}%"
        else:
            title = f"Mode {i+1} (Spatial Deformation)\nEnergy: {variance_ratio:.4f}%"
            
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))

    # 全局排版
    plt.suptitle("Physical Interpretation of Fast Flux POD Modes", fontsize=22, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(PATHS['output_dir'], "pod_spatial_modes.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> 绘图完成！图片已保存至: {save_path}")

if __name__ == "__main__":
    main()