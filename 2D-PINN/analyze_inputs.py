import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

def main():
    print(">>> 开始提取并分析 Case 的完整输入参数矩阵...")
    
    # 1. 加载路径配置
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        PATHS = config['paths']
    except FileNotFoundError:
        print("未找到 config.yaml，请确保在项目根目录下运行。")
        return

    # 2. 加载输入特征矩阵 (X_test)
    x_test_path = os.path.join(PATHS['processed_dir'], "X_test.npy")
    if not os.path.exists(x_test_path):
        print(f"找不到文件: {x_test_path}")
        return
        
    X_test_raw = np.load(x_test_path)
    
    # 指定要分析的 Case
    TARGET_CASE = 0
    case_inputs = X_test_raw[TARGET_CASE]  # 提取单工况，形状期望为 (TimeSteps, Features)
    
    # 3. 按独热编码特征顺序拆解参数
    # 顺序依据: [is_reg1, is_reg2, is_fast, is_thermal, p_t, t]
    is_reg1 = case_inputs[:, 0]
    is_reg2 = case_inputs[:, 1]
    is_fast = case_inputs[:, 2]
    is_thermal = case_inputs[:, 3]
    p_t = case_inputs[:, 4]
    t_steps = case_inputs[:, 5]  # 使用第6列作为物理时间或时间步轴

    # 4. 终端打印统计信息 (用于严谨的数值核对)
    print(f"\n========== Case {TARGET_CASE} 完整特征统计 ==========")
    print(f"{'特征名称':<15} | {'最小值':<10} | {'最大值':<10} | {'属性判定'}")
    print("-" * 60)
    
    def analyze_feature(name, data):
        min_v, max_v = np.min(data), np.max(data)
        prop = "静态常量" if min_v == max_v else "动态变量"
        if set(np.unique(data)).issubset({0.0, 1.0}): prop += " (布尔/独热)"
        print(f"{name:<15} | {min_v:<10.4f} | {max_v:<10.4f} | {prop}")

    analyze_feature("1. is_reg1", is_reg1)
    analyze_feature("2. is_reg2", is_reg2)
    analyze_feature("3. is_fast", is_fast)
    analyze_feature("4. is_thermal", is_thermal)
    analyze_feature("5. p_t", p_t)
    analyze_feature("6. t", t_steps)

    # 5. 绘制全景特征图 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Complete Input Features Overview - Case {TARGET_CASE}", fontsize=20, fontweight='bold', y=1.02)

    # 定义一个辅助画图函数
    def plot_sub(ax, x_data, y_data, title, color, is_bool=False):
        ax.plot(x_data, y_data, color=color, linewidth=2.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (t)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if is_bool:
            ax.set_ylim(-0.2, 1.2) # 布尔值特征固定Y轴范围，避免画成一条无法分辨的直线
            ax.set_yticks([0, 1])

    # 依次绘制 6 个特征
    plot_sub(axes[0, 0], t_steps, is_reg1, "Feature 0: is_reg1", "gray", True)
    plot_sub(axes[0, 1], t_steps, is_reg2, "Feature 1: is_reg2", "gray", True)
    plot_sub(axes[0, 2], t_steps, is_fast, "Feature 2: is_fast", "orange", True)
    plot_sub(axes[1, 0], t_steps, is_thermal, "Feature 3: is_thermal", "orange", True)
    
    # 核心扰动参数 p_t
    axes[1, 1].plot(t_steps, p_t, 'b-', linewidth=3)
    axes[1, 1].fill_between(t_steps, p_t, alpha=0.2, color='blue')
    axes[1, 1].set_title("Feature 4: Driving Perturbation (p_t)", fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel("Time (t)", fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 时间自身特征
    axes[1, 2].plot(t_steps, t_steps, 'g-', linewidth=2.5)
    axes[1, 2].set_title("Feature 5: Physical Time (t)", fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel("Time Step Index", fontsize=12)
    axes[1, 2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(PATHS['output_dir'], f"all_inputs_overview_case_{TARGET_CASE}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n>>> 完整特征全景图已保存至: {save_path}")

if __name__ == "__main__":
    main()