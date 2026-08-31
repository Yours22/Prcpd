# POD-LSTM 双流振幅-形状分解网络架构图（no_cumsum 消融版：Mode 1 直接回归，两阶段训练）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Songti SC', 'Arial'],
    'font.family': 'sans-serif',
    'font.size': 25,
    'axes.titlesize': 25,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'mathtext.fontset': 'stix',
})

fig, ax = plt.subplots(1, 1, figsize=(30, 18))
ax.set_xlim(0, 15)
ax.set_ylim(0, 11.5)
ax.axis('off')

# ── Colors ──
C_INPUT   = '#E8EAF6'
C_RAW      = '#E3F2FD'
C_MEM      = '#F3E5F5'
C_MACRO   = '#BBDEFB';  C_MACRO_D = '#1565C0'
C_MICRO   = '#C8E6C9';  C_MICRO_D = '#2E7D32'
C_FC      = '#FFF9C4'
C_OP      = '#FFCCBC';  C_OP_D    = '#E65100'
C_OUTPUT  = '#F3E5F5'
C_EDGE    = '#37474F'
C_TEXT    = '#212121'
C_ARROW   = '#546E7A'
C_MUL     = '#7B1FA2'
C_CONCAT  = '#FF8F00'


def box(ax, x, y, w, h, text, color, fs=20, bold=False, ec=None, lw=1.5, tc=None):
    if ec is None: ec = C_EDGE
    if tc is None: tc = C_TEXT
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.22",
                       facecolor=color, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=tc, weight='bold' if bold else 'normal', zorder=4)


def group(ax, x, y, w, h, label, color, alpha=0.10):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.44",
                       facecolor=color, edgecolor=color, linewidth=2.0,
                       linestyle='--', alpha=alpha, zorder=0)
    ax.add_patch(b)
    ax.text(x - w/2 + 0.5, y + h/2 - 0.35, label, fontsize=18,
            color=color, weight='bold', va='top', ha='left', zorder=1, alpha=0.88)


def arr(ax, x1, y1, x2, y2, c=None, lw=1.5):
    if c is None: c = C_ARROW
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle='arc3,rad=0'), zorder=2)


def poly_arr(ax, pts, c=None, lw=1.5):
    if c is None: c = C_ARROW
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=c, lw=lw, zorder=2)
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle='arc3,rad=0'), zorder=2)


def circle_node(ax, x, y, symbol, color, r=0.44):
    c = plt.Circle((x, y), r, facecolor='white', edgecolor=color,
                    linewidth=2.8, zorder=3)
    ax.add_patch(c)
    ax.text(x, y, symbol, fontsize=38, ha='center', va='center',
            color=color, weight='bold', zorder=4)


def label(ax, x, y, text, c, fs=20, bold=True, ha='center', va='center'):
    ax.text(x, y, text, fontsize=fs, color=c, weight='bold' if bold else 'normal',
            ha=ha, va=va, zorder=5)


# ═══════════════════════════════════════════════════════════════
# LAYOUT GRID (×1.25 放大)
# ═══════════════════════════════════════════════════════════════

CENTER = 7.5
LX_M, LX_U = 3.8, 11.2

# Row y-positions (top → bottom)
Y_SEQ    = 10.81
Y_FEAT   = 10.13
Y_XT     = 9.13
Y_LSTM   = 7.69
Y_HID    = 6.94
Y_FC     = 5.81
Y_MUL    = 3.94
Y_OUT    = 2.31
Y_POD    = 1.00

# Sizes (×1.25)
SW, SH = 3.75, 0.65
IW, IH = 5.75, 0.75
LW, LH = 3.5, 0.94
FW, FH = 3.5, 0.73
OW, OH = 4.25, 0.65
PW, PH = 5.63, 0.70

# ═══════════════════════════════════════════════════════════════
# GROUP BOXES
# ═══════════════════════════════════════════════════════════════
group(ax, LX_M, (Y_LSTM + Y_FC) / 2, 6.25, Y_LSTM - Y_FC + 2.5,
      '宏观流 — 模态 1（振幅）', C_MACRO_D)
group(ax, LX_U, (Y_LSTM + Y_FC) / 2, 5.63, Y_LSTM - Y_FC + 2.5,
      '微观流 — 模态 2–8（形状）', C_MICRO_D)

# ═══════════════════════════════════════════════════════════════
# INPUT
# ═══════════════════════════════════════════════════════════════

label(ax, CENTER, Y_SEQ, '输入序列:  t = 0, 0.005, 0.01, ..., 0.5 s  (共 101 步)',
      C_TEXT, fs=11, bold=False)

box(ax, CENTER - 2.8, Y_FEAT, SW, SH,
    '原始特征 (6 维)\nis_reg1/2, is_fast/thermal, p(t), t',
    C_RAW, fs=9, ec='#90A4AE')
box(ax, CENTER + 2.8, Y_FEAT, SW, SH,
    '记忆特征 (4 维)\n∫p·e^{-λ(t-τ)}dτ (λ=0.1,1,10) + ∫p dτ',
    C_MEM, fs=9, ec='#90A4AE')

arr(ax, CENTER - 2.8, Y_FEAT - SH/2, CENTER - 0.19, Y_XT + IH/2 + 0.02,
    c=C_CONCAT, lw=1.6)
arr(ax, CENTER + 2.8, Y_FEAT - SH/2, CENTER + 0.19, Y_XT + IH/2 + 0.02,
    c=C_CONCAT, lw=1.6)

label(ax, CENTER, Y_FEAT - SH/2 - 0.28, '每个时间步拼接',
      C_CONCAT, fs=9, bold=True)

box(ax, CENTER, Y_XT, IW, IH,
    'x(t) ∈ R^10   (拼接后送入两个 LSTM 的输入门)',
    C_INPUT, bold=True, fs=12, ec=C_CONCAT, lw=2.2)

# Input detail panel (top-left)
feat = ("特征构成 (每时间步):\n"
        "  时不变 (4 维):  is_reg1, is_reg2,\n"
        "    is_fast, is_thermal\n"
        "  时变 (6 维):  p(t), t,\n"
        "    I_λ(t)×3, I_cum(t)")
ax.text(0.25, 10.81, feat, fontsize=14, color=C_TEXT, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.44', facecolor='#FAFAFA',
                  edgecolor='#BDBDBD', lw=1.1),
        zorder=1)

# ═══════════════════════════════════════════════════════════════
# DUAL LSTMs
# ═══════════════════════════════════════════════════════════════
box(ax, LX_M, Y_LSTM, LW, LH,
    'LSTM_macro\n2-layer, hidden_dim=H',
    C_MACRO, bold=True, fs=11, ec=C_MACRO_D)
box(ax, LX_U, Y_LSTM, LW, LH,
    'LSTM_micro\n2-layer, hidden_dim=H',
    C_MICRO, bold=True, fs=11, ec=C_MICRO_D)

arr(ax, CENTER - IW/2 + 0.38, Y_XT - IH/2, LX_M, Y_LSTM + LH/2 + 0.01)
arr(ax, CENTER + IW/2 - 0.38, Y_XT - IH/2, LX_U, Y_LSTM + LH/2 + 0.01)

label(ax, CENTER - IW/2 + 0.75, Y_XT - IH/2 - 0.44, '同一 x(t) 同时送入两路',
      C_ARROW, fs=9, bold=False)
label(ax, CENTER - IW/2 - 1.0, Y_LSTM + LH/2 + 0.56, '每个 t',
      C_CONCAT, fs=9, bold=True)
label(ax, CENTER + IW/2 + 1.0, Y_LSTM + LH/2 + 0.56, '每个 t',
      C_CONCAT, fs=9, bold=True)

label(ax, LX_M, Y_HID, 'h_macro(t)', C_MACRO_D, fs=10, bold=False)
label(ax, LX_U, Y_HID, 'h_micro(t)', C_MICRO_D, fs=10, bold=False)

# ═══════════════════════════════════════════════════════════════
# MACRO BRANCH: fc_direct → pred_m1
# ═══════════════════════════════════════════════════════════════

box(ax, LX_M, Y_FC, FW, FH,
    'fc_direct:  Linear → SiLU → Linear  →  pred_m1(t)',
    C_FC, fs=10)
arr(ax, LX_M, Y_LSTM - LH/2, LX_M, Y_FC + FH/2 + 0.01)

label(ax, LX_M - 1.94, Y_MUL, 'pred_m1(t)', C_MACRO_D, fs=11)
arr(ax, LX_M, Y_FC - FH/2, LX_M, Y_MUL + 0.45)

# ═══════════════════════════════════════════════════════════════
# MICRO BRANCH
# ═══════════════════════════════════════════════════════════════

box(ax, LX_U, Y_FC, FW + 0.25, FH,
    'fc_shape:  Linear → SiLU → Dropout(0.1) → Linear  →  R(t)',
    C_FC, fs=10)
arr(ax, LX_U, Y_LSTM - LH/2, LX_U, Y_FC + FH/2 + 0.01)

label(ax, LX_U + FW/2 + 0.88, Y_FC, 'R(t) ∈ R^7', C_MICRO_D, fs=11, ha='left')

circle_node(ax, CENTER, Y_MUL, '×', C_MUL, r=0.45)

poly_arr(ax, [
    (LX_U, Y_FC - FH/2),
    (LX_U, Y_MUL),
    (CENTER - 0.45, Y_MUL),
], c=C_MUL)

arr(ax, LX_M + 1.63, Y_MUL, CENTER - 0.45, Y_MUL, c=C_MUL)

label(ax, CENTER + 2.63, Y_MUL,
      'pred_high(t) = R(t) × pred_m1(t)', C_MUL, fs=11, ha='left')

# ═══════════════════════════════════════════════════════════════
# CONCATENATE & OUTPUT
# ═══════════════════════════════════════════════════════════════

ax.text(CENTER, Y_MUL - 0.69, 'concat', fontsize=20, color=C_TEXT,
        ha='center', va='center', style='italic', zorder=3)

arr(ax, LX_M, Y_MUL - 0.45, LX_M, Y_OUT + OH/2)
arr(ax, CENTER, Y_MUL - 0.45, CENTER, Y_OUT + OH/2)

box(ax, CENTER, Y_OUT, OW, OH,
    'a(t) ∈ R^8   [1 amplitude + 7 shape coefficients]',
    C_OUTPUT, bold=True, fs=11)

# ═══════════════════════════════════════════════════════════════
# POD RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════
arr(ax, CENTER, Y_OUT - OH/2, CENTER, Y_POD + PH/2 + 0.02, c=C_MUL)

box(ax, CENTER, Y_POD, PW, PH,
    'a(t) = mean + Σ a_k(t) · v_k    (POD reconstruction)',
    C_OUTPUT, bold=True, fs=11, ec=C_MUL)

label(ax, CENTER + PW/2 + 1.0, Y_POD,
      '∈ R^800  (400 fast + 400 thermal)', C_MUL, fs=10, bold=False, ha='left')

# ═══════════════════════════════════════════════════════════════
# BOTTOM SUMMARY
# ═══════════════════════════════════════════════════════════════
ax.text(0.5, 0.23,
        '训练策略：两阶段冻结梯度训练 → 阶段 1 (0–30%): lstm_macro + fc_direct | 阶段 2 (30–100%): lstm_micro + fc_shape',
        fontsize=18, color='#9E9E9E', ha='left', va='center')

# ── Save ──
plt.tight_layout(pad=0.3)
out_path = 'fig/fig_network_architecture.svg'
fig.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out_path}')
