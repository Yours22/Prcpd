import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
from importlib import import_module

_model_mod = import_module("3-model")
get_output_root = _model_mod.get_output_root

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
PATHS = config['paths']
PHYSICS = config['physics']

N_NODES = PHYSICS['num_nodes_per_group']  # 400
NX = NY = int(np.sqrt(N_NODES))  # 20

ROOT = get_output_root()
RESULT_DIR = os.path.join(ROOT, 'results')
LOG_DIR = os.path.join(ROOT, 'log')
os.makedirs(LOG_DIR, exist_ok=True)

PLOT_TIME_STEPS = [10, 50, 90]

# TWIGL quarter-reactor geometry: boundary left+top = Reflection (core center),
# right+bottom = Albedo (outer). core at (row=19, col=0) in origin='lower' heatmap.
_MAT_10X10 = np.array([
    [1,1,1,1,1,1,1,1,1,1],  # block_y=0 (bottom, outer)
    [1,1,1,1,1,1,1,1,1,1],  # block_y=1
    [1,1,1,1,1,1,1,1,1,1],  # block_y=2
    [2,2,2,3,3,3,3,1,1,1],  # block_y=3
    [2,2,2,3,3,3,3,1,1,1],  # block_y=4
    [2,2,2,3,3,3,3,1,1,1],  # block_y=5
    [2,2,2,3,3,3,3,1,1,1],  # block_y=6
    [1,1,1,2,2,2,2,1,1,1],  # block_y=7
    [1,1,1,2,2,2,2,1,1,1],  # block_y=8
    [1,1,1,2,2,2,2,1,1,1],  # block_y=9 (top, near core center)
], dtype=np.int32)
_ZONE_20X20 = np.repeat(np.repeat(_MAT_10X10, 2, axis=0), 2, axis=1)
ZONE_MASK = _ZONE_20X20.flatten()  # 400 elements: 1=Core fuel, 2=Blanket fuel, 3=Control rod

ZONE_NAMES = {1: 'Z1 (Core fuel)', 2: 'Z2 (Blanket)', 3: 'Z3 (Control rod)'}


def log_print(log_file, *args):
    msg = ' '.join(str(a) for a in args)
    try:
        print(msg)
    except UnicodeEncodeError:
        pass
    log_file.write(msg + '\n')


def spatial_stats(err_grid, label=''):
    """返回空间误差网格的统计摘要字符串"""
    mean_val = np.mean(err_grid)
    max_val = np.max(err_grid)
    max_idx = np.unravel_index(np.argmax(err_grid), err_grid.shape)
    # Top-5 热点
    flat_idx = np.argsort(err_grid.ravel())[-5:][::-1]
    hotspots = []
    for fi in flat_idx:
        r, c = np.unravel_index(fi, err_grid.shape)
        hotspots.append((r, c, err_grid[r, c]))
    lines = [
        f"  {label} mean={mean_val:.4e}  max={max_val:.4e} at (row={max_idx[0]}, col={max_idx[1]})",
        f"  {label} Top-5 hotspots:",
    ]
    for i, (r, c, v) in enumerate(hotspots, 1):
        lines.append(f"    #{i}: ({r:2d},{c:2d}) = {v:.4e}")
    return '\n'.join(lines)


def plot_rep_case(Y_pred, Y_true, case_idx, group_name, out_dir):
    """为代表性 case 绘制空间误差图: 2行(快群/热群) x 3列(时间步)"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, (flux_name, offset) in enumerate([("Fast", 0), ("Thermal", N_NODES)]):
        for col, t in enumerate(PLOT_TIME_STEPS):
            ax = axes[row, col]
            err = np.abs(Y_pred[case_idx, t, offset:offset + N_NODES] -
                         Y_true[case_idx, t, offset:offset + N_NODES])
            grid = err.reshape(NY, NX)
            im = ax.imshow(grid, cmap='inferno', origin='lower', aspect='equal')
            ax.set_title(f"{flux_name} Flux  t={t}")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Spatial Abs Error — {group_name}  Case {case_idx}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"spatial_error_case{case_idx}_{group_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def analyze(data_prefix, out_subdir):
    """
    data_prefix: 'val' or 'test'
    out_subdir:  'val' or 'test_extrap'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"spatial_error_{out_subdir}_{timestamp}.txt")
    log_file = open(log_path, 'w', encoding='utf-8')

    set_label = 'Val (in-distribution)' if data_prefix == 'val' else 'Test (extrapolation)'
    log_print(log_file, f"Spatial Error Analysis — {timestamp}")
    log_print(log_file, f"Dataset: {set_label}\n")

    out_dir = os.path.join(RESULT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    Y_pred = np.load(os.path.join(out_dir, "Y_pred.npy"))
    Y_true = np.load(os.path.join(PATHS['processed_dir'], f"Y_{data_prefix}_raw.npy"))
    X_raw = np.load(os.path.join(PATHS['processed_dir'], f"X_{data_prefix}.npy"))

    N, T, _ = Y_pred.shape
    log_print(log_file, f"Cases: {N}  Time steps: {T}")

    is_reg1 = X_raw[:, 0, 0]
    is_fast = X_raw[:, 0, 2]
    region = np.where(is_reg1 == 1, "Core", "Rod")
    group = np.where(is_fast == 1, "Fast", "Thermal")

    # 计算每个 case 的 L2 误差(用于选代表性 case)
    rel_l2 = np.linalg.norm(Y_pred - Y_true, axis=(1, 2)) / (np.linalg.norm(Y_true, axis=(1, 2)) + 1e-10)

    abs_err = np.abs(Y_pred - Y_true)  # [N, T, 800]

    categories = [
        ("Core", "Fast"),
        ("Core", "Thermal"),
        ("Rod", "Fast"),
        ("Rod", "Thermal"),
    ]

    # ==================== 1. 收集数据 ====================
    summary_rows = []
    for pert_region, pert_group in categories:
        mask = (region == pert_region) & (group == pert_group)
        idx = np.where(mask)[0]
        n_cases = len(idx)
        if n_cases == 0:
            continue
        cat_name = f"{pert_region} + {pert_group}"
        err_subset = abs_err[idx]
        for flux_name, offset in [("Fast", 0), ("Thermal", N_NODES)]:
            flux_err = err_subset[:, :, offset:offset + N_NODES]
            summary_rows.append({
                'cat': cat_name, 'n': n_cases, 'flux': flux_name,
                'early': flux_err[:, :30, :].mean(),
                'mid':   flux_err[:, 30:70, :].mean(),
                'late':  flux_err[:, 70:, :].mean(),
                'avg':   flux_err.mean(),
            })

    zone_rows = []
    for pert_region, pert_group in categories:
        mask = (region == pert_region) & (group == pert_group)
        idx = np.where(mask)[0]
        n_cases = len(idx)
        if n_cases == 0:
            continue
        cat_name = f"{pert_region} + {pert_group}"
        err_subset = abs_err[idx]
        for zone_id in [1, 2, 3]:
            z_mask = ZONE_MASK == zone_id
            for flux_name, offset in [("Fast", 0), ("Thermal", N_NODES)]:
                flux_err = err_subset[:, :, offset:offset + N_NODES]
                zone_flux_err = flux_err[:, :, z_mask]
                zone_rows.append({
                    'cat': cat_name, 'n': n_cases,
                    'zone': ZONE_NAMES[zone_id], 'flux': flux_name,
                    'early': zone_flux_err[:, :30, :].mean().item(),
                    'mid':   zone_flux_err[:, 30:70, :].mean().item(),
                    'late':  zone_flux_err[:, 70:, :].mean().item(),
                    'avg':   zone_flux_err.mean().item(),
                })

    header_sci = (f"{'Category':<20} {'n':>4}  "
                  f"{'Flux':<8} {'Early(0-30)':>12} {'Mid(30-70)':>12} {'Late(70-100)':>12} {'Time-avg':>12}")
    header_dec = header_sci
    header_z  = (f"{'Category':<20} {'n':>4}  {'Zone':<18} {'Flux':<8} "
                 f"{'Early(0-30)':>12} {'Mid(30-70)':>12} {'Late(70-100)':>12} {'Time-avg':>12}")

    # --- B: Per-category decimal (top, most readable) ---
    log_print(log_file, f"\n{'='*100}")
    log_print(log_file, f"SUMMARY B: Per-category spatial error (decimal)")
    log_print(log_file, f"[Unit: same as neutron flux]")
    log_print(log_file, f"{'='*100}")
    log_print(log_file, header_dec)
    log_print(log_file, '-' * 100)
    for r in summary_rows:
        log_print(log_file, f"{r['cat']:<20} {r['n']:>4}  "
                  f"{r['flux']:<8} {r['early']:>12.4f} {r['mid']:>12.4f} {r['late']:>12.4f} {r['avg']:>12.4f}")
    log_print(log_file, f"{'='*100}")

    # --- D: Zone x Category decimal ---
    log_print(log_file, f"\n{'='*100}")
    log_print(log_file, f"SUMMARY D: Zone x Category spatial error (decimal)")
    log_print(log_file, f"[Zone: Z1=Core fuel, Z2=Blanket, Z3=Control rod | unit = neutron flux]")
    log_print(log_file, f"{'='*100}")
    log_print(log_file, header_z)
    log_print(log_file, '-' * 100)
    for zr in zone_rows:
        log_print(log_file, f"{zr['cat']:<20} {zr['n']:>4}  {zr['zone']:<18} {zr['flux']:<8} "
                  f"{zr['early']:>12.4f} {zr['mid']:>12.4f} {zr['late']:>12.4f} {zr['avg']:>12.4f}")
    log_print(log_file, f"{'='*100}")

    # --- A: Per-category scientific (for precision) ---
    log_print(log_file, f"\n{'='*100}")
    log_print(log_file, f"SUMMARY A: Per-category spatial error (scientific)")
    log_print(log_file, f"[Unit: same as neutron flux]")
    log_print(log_file, f"{'='*100}")
    log_print(log_file, header_sci)
    log_print(log_file, '-' * 100)
    for r in summary_rows:
        log_print(log_file, f"{r['cat']:<20} {r['n']:>4}  "
                  f"{r['flux']:<8} {r['early']:>12.4e} {r['mid']:>12.4e} {r['late']:>12.4e} {r['avg']:>12.4e}")
    log_print(log_file, f"{'='*100}")

    # --- C: Zone x Category scientific ---
    log_print(log_file, f"\n{'='*100}")
    log_print(log_file, f"SUMMARY C: Zone x Category spatial error (scientific)")
    log_print(log_file, f"[Zone: Z1=Core fuel, Z2=Blanket, Z3=Control rod | unit = neutron flux]")
    log_print(log_file, f"{'='*100}")
    log_print(log_file, header_z)
    log_print(log_file, '-' * 100)
    for zr in zone_rows:
        log_print(log_file, f"{zr['cat']:<20} {zr['n']:>4}  {zr['zone']:<18} {zr['flux']:<8} "
                  f"{zr['early']:>12.4e} {zr['mid']:>12.4e} {zr['late']:>12.4e} {zr['avg']:>12.4e}")
    log_print(log_file, f"{'='*100}\n")

    # ==================== 3. 逐类详细分析 ====================
    for pert_region, pert_group in categories:
        mask = (region == pert_region) & (group == pert_group)
        idx = np.where(mask)[0]
        n_cases = len(idx)
        if n_cases == 0:
            continue

        cat_name = f"{pert_region} + {pert_group}"
        log_print(log_file, f"\n{'='*60}")
        log_print(log_file, f"Category: {cat_name}  (n={n_cases})")
        log_print(log_file, f"{'='*60}")

        err_subset = abs_err[idx]  # [n, T, 800]

        for flux_name, offset in [("Fast", 0), ("Thermal", N_NODES)]:
            flux_err = err_subset[:, :, offset:offset + N_NODES]  # [n, T, 400]

            # 时间平均
            avg_grid = flux_err.mean(axis=(0, 1)).reshape(NY, NX)
            # 前期 / 后期
            early_grid = flux_err[:, :30, :].mean(axis=(0, 1)).reshape(NY, NX)
            late_grid = flux_err[:, 70:, :].mean(axis=(0, 1)).reshape(NY, NX)

            log_print(log_file, f"\n  --- {flux_name} Flux (absolute error) ---")
            log_print(log_file, spatial_stats(avg_grid, label="Time-avg"))
            log_print(log_file, spatial_stats(early_grid, label="Early(0-30)"))
            log_print(log_file, spatial_stats(late_grid, label="Late(70-100)"))

        # 选代表性 case（组内 L2 中位数最接近）
        group_l2 = rel_l2[idx]
        median_l2 = np.median(group_l2)
        rep_local_idx = np.argmin(np.abs(group_l2 - median_l2))
        rep_case_idx = idx[rep_local_idx]

        log_print(log_file, f"\n  Representative case: {rep_case_idx} (L2 rel err = {group_l2[rep_local_idx]*100:.2f}%)")
        png_path = plot_rep_case(Y_pred, Y_true, rep_case_idx, cat_name, out_dir)
        log_print(log_file, f"  Plot saved: {png_path}")

    log_file.close()
    print(f"Log saved to: {log_path}")
    return log_path


def main():
    for prefix, subdir in [('val', 'val'), ('test', 'test_extrap')]:
        print(f"\n{'='*60}")
        print(f"Analyzing: {subdir}")
        print(f"{'='*60}")
        analyze(prefix, subdir)


if __name__ == "__main__":
    main()
