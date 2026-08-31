"""
Visualize gradient flow from training logs.

Reads training_log.csv (must include Grad_Total and Grad_MacroMicro_Ratio columns,
added by grad_monitor.py integration) and produces:
  1. Gradient norm vs epoch (diagnose vanishing/exploding)
  2. Macro/Micro gradient ratio over training stages
  3. Gradient norm distribution across modules (histogram at stage boundaries)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'figure.dpi': 150, 'savefig.bbox': 'tight',
})


def plot_gradient_flow(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    grad_col = 'Grad_Total'
    ratio_col = 'Grad_MacroMicro_Ratio'

    if grad_col not in df.columns:
        print(f"ERROR: '{grad_col}' not found in CSV. Ensure grad_monitor is enabled in training.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ── Panel 1: Gradient total norm + loss ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Find stage boundaries
    stage_changes = df[df['Stage'].diff().fillna(0) != 0].index

    ax1.semilogy(df['Epoch'], df[grad_col], color='#1565C0', linewidth=0.8, alpha=0.85)
    ax1.set_ylabel('Max Gradient L2 Norm', color='#1565C0', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Mark stage boundaries
    for idx in stage_changes:
        ax1.axvline(x=df.loc[idx, 'Epoch'], color='#E65100', linestyle='--', linewidth=1.2, alpha=0.7)
    ax1.set_title('Gradient Flow During Training', fontweight='bold', fontsize=13)

    # Loss on twin y
    ax1b = ax1.twinx()
    ax1b.plot(df['Epoch'], df['Train_Loss'], color='#C62828', linewidth=0.6, alpha=0.6)
    ax1b.set_ylabel('Train Loss', color='#C62828', fontsize=11)
    ax1b.tick_params(axis='y', labelcolor='#C62828')

    # ── Panel 2: Macro/Micro gradient ratio ──
    ax2.plot(df['Epoch'], df[ratio_col], color='#2E7D32', linewidth=0.8)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='Balanced (ratio=1)')
    ax2.set_ylabel('Macro / Micro Grad Ratio', fontsize=11)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.4)

    for idx in stage_changes:
        ax2.axvline(x=df.loc[idx, 'Epoch'], color='#E65100', linestyle='--', linewidth=1.2, alpha=0.7)

    # Stage annotation
    stages = df['Stage'].unique()
    stage_colors = {1: '#BBDEFB', 2: '#C8E6C9', 3: '#FFF9C4'}
    stage_labels = {1: 'S1: Macro', 2: 'S2: Micro', 3: 'S3: Residual'}
    y_bottom = ax2.get_ylim()[0]
    for s in stages:
        s_df = df[df['Stage'] == s]
        if len(s_df) > 0:
            mid = (s_df['Epoch'].min() + s_df['Epoch'].max()) / 2
            ax2.axvspan(s_df['Epoch'].min(), s_df['Epoch'].max(),
                        alpha=0.08, color=list(stage_colors.values())[s-1] if s <= 3 else '#EEEEEE')
            ax2.text(mid, y_bottom + 0.02, stage_labels.get(s, f'S{s}'),
                     ha='center', fontsize=9, color='#757575')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'gradient_flow.png'), facecolor='white')
    plt.close(fig)
    print(f'Saved: {os.path.join(out_dir, "gradient_flow.png")}')

    # ── Summary statistics ──
    print(f"\nGradient Flow Summary:")
    print(f"  {'Stage':<8s} {'Grad Mean':>12s} {'Grad Std':>12s} {'Ratio Mean':>12s}")
    for s in sorted(df['Stage'].unique()):
        s_df = df[df['Stage'] == s]
        g_mean = s_df[grad_col].mean()
        g_std = s_df[grad_col].std()
        r_mean = s_df[ratio_col].mean() if ratio_col in df.columns and s_df[ratio_col].notna().any() else 0
        print(f"  Stage {s:<3d} {g_mean:>12.2e} {g_std:>12.2e} {r_mean:>12.4f}")

    # Stage 1 vs Stage 2 gradient transfer
    for i in range(len(stage_changes)):
        idx = stage_changes[i]
        before = df.loc[idx-1, grad_col] if idx > 0 else None
        after = df.loc[min(idx+1, len(df)-1), grad_col]
        if before is not None:
            ratio = after / (before + 1e-12)
            print(f"  Stage transition at epoch {df.loc[idx, 'Epoch']:.0f}: "
                  f"grad {before:.2e} -> {after:.2e} (ratio={ratio:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Visualize gradient flow from training log')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to training_log.csv')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    # Default: find latest ablation log
    if args.csv is None:
        candidates = []
        for root, dirs, files in os.walk('2D-PINN/ablation'):
            for f in files:
                if f == 'training_log.csv':
                    candidates.append(os.path.join(root, f))
        if not candidates:
            # Fallback to main log
            fallback = '2D-PINN/log/training_log.csv'
            if os.path.exists(fallback):
                candidates = [fallback]
        if not candidates:
            print("No training_log.csv found. Specify --csv path.")
            return
        # Pick most recently modified
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        args.csv = candidates[0]
        print(f"Using: {args.csv}")

    out_dir = args.out or os.path.dirname(args.csv)
    plot_gradient_flow(args.csv, out_dir)


if __name__ == '__main__':
    main()
