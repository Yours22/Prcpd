# POD 空间模态与时间系数解析：快中子通量场的时空分解。
import os
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt

def main():
    print(">>> Generating POD spatiotemporal mode figure...")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    PATHS = config['paths']
    os.makedirs(PATHS['output_dir'], exist_ok=True)

    svd_fast_path = os.path.join(PATHS['pod_save_dir'], 'svd_fast.pkl')
    svd_fast = joblib.load(svd_fast_path)
    components = svd_fast.components_

    N_NODES = components.shape[1]
    NX = NY = int(np.sqrt(N_NODES))

    Y_train = np.load(os.path.join(PATHS['processed_dir'], "Y_test_raw.npy"))
    Y_train_fast = Y_train[:, :, :N_NODES].reshape(-1, N_NODES)
    mean_field = Y_train_fast.mean(axis=0).reshape(NY, NX)

    TARGET_CASE = 0
    case_data_fast = Y_train[TARGET_CASE, :, :N_NODES]
    time_coeffs = svd_fast.transform(case_data_fast)
    time_steps = np.arange(time_coeffs.shape[0])

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), gridspec_kw={'height_ratios': [1.15, 1]})

    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # ── 第一行：空间模态 ──
    # (a) 平均场
    ax = axes[0, 0]
    im0 = ax.imshow(mean_field, cmap='viridis', origin='lower')
    ax.set_title('平均场', fontsize=20, fontweight='bold')
    ax.axis('off')
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar0.formatter.set_powerlimits((0, 0))
    cbar0.ax.tick_params(labelsize=16)
    ax.text(0.02, 0.96, sub_labels[0], transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    # (b)(c)(d) 空间模态 1-3
    for i in range(3):
        ax = axes[0, i + 1]
        mode_data = components[i].reshape(NY, NX)
        vmax = np.max(np.abs(mode_data))
        im = ax.imshow(mode_data, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        ratio = svd_fast.explained_variance_ratio_[i] * 100
        ax.set_title(f'空间模态 {i + 1}（{ratio:.4f}%）', fontsize=20, fontweight='bold')
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.tick_params(labelsize=16)
        ax.text(0.02, 0.96, sub_labels[i + 1], transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top')

    # ── 第二行：时间演化系数 ──
    # (e) 说明面板
    ax = axes[1, 0]
    ax.axis('off')
    ax.text(0.5, 0.5, f'快群时间演化\n（算例 {TARGET_CASE}）',
            fontsize=22, fontweight='bold', ha='center', va='center', color='#37474F')
    ax.text(0.02, 0.96, sub_labels[4], transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    # (f)(g)(h) 时间系数曲线
    for i in range(3):
        ax = axes[1, i + 1]
        coeff_series = time_coeffs[:, i]
        ax.plot(time_steps, coeff_series, color='#263238', linewidth=1.8)
        ax.set_title(f'模态 {i + 1} 时间系数', fontsize=20, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=20)
        ax.set_ylabel(f'$a_{{{i + 1}}}(t)$', fontsize=20)
        ax.tick_params(labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if i > 0:
            ax.axhline(0, color='#E53935', linestyle=':', linewidth=1.2)
        ax.text(0.02, 0.96, sub_labels[i + 5], transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top')

    plt.tight_layout(pad=1.5)
    save_path = 'fig/fig_pod_spatiotemporal.svg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f">>> Saved: {save_path}")

if __name__ == "__main__":
    main()
