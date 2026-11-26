"""Generate the JPG figures used by README.zh-CN.md.

Usage:  python docs/make_figures.py
Needs:  matplotlib, pillow, and a CJK font (WenQuanYi Zen Hei by default).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

C_BOX = "#E8F0FE"  # light blue
C_EDGE = "#3B6CB4"
C_ACT = "#E6F4EA"  # light green
C_ACT_EDGE = "#3E884F"
C_LAD = "#FDE7CE"  # light orange
C_LAD_EDGE = "#C47F17"
C_BAND = "#F5F5F5"


def new_canvas(w, h, fw, fh):
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fc=C_BOX, ec=C_EDGE, fs=10, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.6",
            fc=fc,
            ec=ec,
            lw=1.4,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(ax, x1, y, x2, label=None, label_dy=2.2):
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6),
    )
    if label:
        ax.text((x1 + x2) / 2, y + label_dy, label, ha="center", fontsize=8.5,
                color="#444")


def banner(ax, w, y, h, text, fs=10.5):
    ax.add_patch(
        FancyBboxPatch((2, y), w - 4, h, boxstyle="round,pad=0.4",
                       fc=C_BAND, ec="#999", lw=1)
    )
    ax.text(w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def fig_ffn():
    fig, ax = new_canvas(100, 34, 11, 3.6)
    y = 21
    ax.text(5, y, "x\n(p 维)", ha="center", va="center", fontsize=10.5)
    box(ax, 25, y, 20, 12, "线性层 $W_1$\np → 4p", fs=10.5)
    box(ax, 50, y, 13, 12, "GELU\n激活", fc=C_ACT, ec=C_ACT_EDGE, fs=10.5)
    box(ax, 75, y, 20, 12, "线性层 $W_2$\n4p → p", fs=10.5)
    ax.text(95, y, "输出\n(p 维)", ha="center", va="center", fontsize=10.5)
    arrow(ax, 9, y, 14, "p 维")
    arrow(ax, 36, y, 43, "4p 维")
    arrow(ax, 57, y, 64, "4p 维")
    arrow(ax, 86, y, 90.5, "p 维")
    ax.text(25, 10.5, "权重 4p×p = 4p² 个", ha="center", fontsize=9.5, color="#B23B3B")
    ax.text(50, 10.5, "0 参数", ha="center", fontsize=9.5, color="#666")
    ax.text(75, 10.5, "权重 p×4p = 4p² 个", ha="center", fontsize=9.5, color="#B23B3B")
    banner(ax, 100, 1.5, 6,
           "参数量 = 4p² + 4p² = 8p² = 2 · 4 · p²   "
           "（2 个矩阵 × 扩展因子 α=4 × p²）")
    fig.savefig(os.path.join(HERE, "ffn.jpg"), format="jpg", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_fng():
    fig, ax = new_canvas(140, 38, 13.5, 3.9)
    y = 24
    ax.text(4.5, y, "x\n(p 维)", ha="center", va="center", fontsize=10.5)
    box(ax, 19, y, 17, 14, "线性层\np → 2p", fs=10.5)
    box(ax, 41, y, 21, 14, "切半 + 门控\nz = a ⊙ silu(b)\n仍是 p 维", fs=10.5)
    box(ax, 64, y, 19, 14, "线性层\np → (d+1)·L", fs=10.5)
    box(ax, 90, y, 25, 14, "L 条连分数梯子\n每条: a$_k$ = w$_k^T$z\n各吐 1 个数", fc=C_LAD,
        ec=C_LAD_EDGE, fs=10.5, weight="bold")
    box(ax, 116, y, 15, 14, "线性层\nL → p", fs=10.5)
    ax.text(135.5, y, "输出\n(p 维)", ha="center", va="center", fontsize=10.5)
    arrow(ax, 8, y, 10, None)
    arrow(ax, 28.5, y, 30, None)
    arrow(ax, 52.5, y, 54, None)
    arrow(ax, 74.5, y, 77, "(d+1)·L 个数", label_dy=2.6)
    arrow(ax, 103.5, y, 108, "L 个数", label_dy=2.6)
    arrow(ax, 124.5, y, 131, None)
    ax.text(19, 12.5, "2p² 参数", ha="center", fontsize=9.5, color="#B23B3B")
    ax.text(41, 12.5, "0 参数（不扩展 = 省参数的关键）", ha="center", fontsize=9.5,
            color="#666")
    ax.text(64, 12.5, "L·p·(d+1) 参数", ha="center", fontsize=9.5, color="#B23B3B")
    ax.text(90, 12.5, "详见 ladder.jpg", ha="center", fontsize=9.5, color="#666")
    ax.text(116, 12.5, "L·p 参数", ha="center", fontsize=9.5, color="#B23B3B")
    banner(ax, 140, 1.5, 7,
           "合计 = 2p² + L·p·(d+1) + L·p；p=768, L=5, d=5 时 ≈ 1.2M，"
           "是 vanilla FFN（4.7M）的 0.26x")
    fig.savefig(os.path.join(HERE, "fng.jpg"), format="jpg", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_ladder():
    fig, ax = new_canvas(130, 48, 12.5, 4.8)

    # left: one ladder
    box(ax, 24, 41, 30, 8, "输入 z（p 维）", fs=10.5)
    ax.annotate("", xy=(24, 34.5), xytext=(24, 36.8),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6))
    box(ax, 24, 30, 36, 8, "d+1 个内积:  a$_k$ = w$_k^T$ z  (k=0..d)", fs=10.5)
    ax.annotate("", xy=(24, 23.5), xytext=(24, 25.8),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6))
    box(ax, 24, 13, 44, 20, "", fc=C_LAD, ec=C_LAD_EDGE)
    ax.text(24, 17.5,
            r"$a_0+\dfrac{1}{a_1+\dfrac{1}{a_2+\dfrac{1}{a_3+"
            r"\dfrac{1}{a_4}}}}$",
            ha="center", va="center", fontsize=17)
    ax.text(24, 6.5, "1 条梯子（d=4 示例）：吃进 z，吐出 1 个数",
            ha="center", fontsize=9.5, color="#666")

    # right: the two tricks
    box(ax, 90, 40, 72, 11,
        "直接算的麻烦：要 d 次除法（慢）；\n分母 ≈ 0 时函数值和梯度爆炸（极点 pole）",
        fc="#FDECEA", ec="#C5221F", fs=10.5)
    box(ax, 90, 24.5, 72, 13,
        "技巧 1：continuant 递推（论文 eqs. 3–6）\n"
        r"$\tilde{f}\,=\,K_{d-1}\,/\,K_d$，$K_k = a_{d-k+1}K_{k-1} + K_{k-2}$"
        "\nd 次除法 → 只剩 1 次",
        fs=10.5)
    box(ax, 90, 8.5, 72, 12,
        "技巧 2：log/exp 改写（论文 eq. 8）\n"
        r"$\mathrm{sgn}\cdot\mathrm{sgn}\cdot\exp(\log|num|-\log\max(|den|,\epsilon))$"
        "\n除法消失，d>1 直接训，无需增量训练",
        fc=C_ACT, ec=C_ACT_EDGE, fs=10.5)
    fig.savefig(os.path.join(HERE, "ladder.jpg"), format="jpg", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_ffn()
    fig_fng()
    fig_ladder()
    print("wrote docs/ffn.jpg, docs/fng.jpg, docs/ladder.jpg")
