"""Figure 1, integrated: access levels, the private-information example and the measured advantage in one connected figure."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = "../results"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "mathtext.fontset": "dejavusans"})
INK = "#1d2533"; GREY = "#6b7280"
C_PUB, C_BEH, C_WB, C_TR = "#a9c6ea", "#3b6fb6", "#e07b39", "#4e9a6b"

W, H = 7.2, 3.15
fig = plt.figure(figsize=(W, H))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 100); ov.set_ylim(0, 100); ov.axis("off")


def box(x, y, w, h, c, lw=1.2, alpha=0.10):
    ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec=c, lw=lw, zorder=1))
    ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=c, alpha=alpha, ec="none", zorder=1))


def step(x, y, n):
    ov.text(x, y, str(n), fontsize=7.5, fontweight="bold", color="white", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.25", fc=INK, ec="none"))


# ---------------- 1. nested levels of access
levels = [(1.0, 9, 40, 80, C_TR, "Trace", "training data S"),
          (3.5, 13, 35, 61, C_WB, "White box", "representation R"),
          (6.0, 17, 30, 40, C_BEH, "Behaviour", "unlimited queries to $b_R$"),
          (8.5, 21, 25, 16, C_PUB, "Public", "input $x$ only")]
for x, y, w, h, c, t, s in levels:
    box(x, y, w, h, c)
    ov.text(x + 1.2, y + h - 2.0, t, fontsize=7.5, fontweight="bold", color=c if c != C_PUB else "#4a78b0", va="top", zorder=3)
    ov.text(x + 1.2, y + h - 8.0, s, fontsize=6.2, color=GREY, va="top", zorder=3)
ov.text(37.5, 71.5, "introspective\nadvantage (Thm 1)", fontsize=6.0, color=C_WB, ha="right", va="top", linespacing=1.1, zorder=3)
ov.text(40.0, 87.0, "ceiling (Thm 2)", fontsize=6.0, color=C_TR, ha="right", va="top", zorder=3)
ov.text(21, 3.2, r"$A^*(\mathcal{P})\leq A^*(\mathcal{B})\leq A^*(\mathcal{W})\leq A^*(\mathcal{T})$", ha="center", fontsize=6.8, color=INK)
ov.text(21, 95, "What a monitor of correctness can see", ha="center", fontsize=7.5, fontweight="bold", color=INK)
step(1.8, 95, 1)

# ---------------- 2. what lives in the gap: two leaves
ov.text(54.5, 95, "What the gap can hold", ha="center", fontsize=7.5, fontweight="bold", color=INK)
step(44.3, 95, 2)
def leaf(y0, n, k, cols, ms, title, sub):
    box(46.5, y0, 16, 30, INK, lw=0.9, alpha=0.0)
    ov.text(54.5, y0 + 27, title, ha="center", fontsize=6.8, fontweight="bold", color=INK, va="top")
    rows = int(np.ceil(n / cols)); gx = 12.0 / cols; gy = 4.2
    xs, ys, fc = [], [], []
    for i in range(n):
        rr, cc = divmod(i, cols)
        xs.append(48.5 + gx * (cc + 0.5)); ys.append(y0 + 17.5 + gy * (rows - 1) / 2 - gy * rr)
        fc.append(C_BEH if i < k else "white")
    ov.scatter(xs, ys, s=ms, c=fc, edgecolors=INK, linewidths=0.5, zorder=4)
    ov.text(54.5, y0 + 6.5, "score 2/3", ha="center", fontsize=6.4, color=C_BEH)
    ov.text(54.5, y0 + 2.2, sub, ha="center", fontsize=6.4, color=C_WB)
leaf(56, 3, 2, 3, 60, "leaf A: 3 cases", "P(correct) = 0.60")
leaf(19, 30, 20, 10, 6, "leaf B: 30 cases", "P(correct) = 0.66")
ov.text(54.5, 13.5, "same answer outside,\ndifferent support inside", ha="center", fontsize=6.0, color=GREY, va="top", linespacing=1.15)

# connector from the gap (between behaviour and white box) to the leaves
con1 = FancyArrowPatch((36.6, 60), (46.3, 70), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=7,
                       color=C_WB, lw=1.0, ls=(0, (3, 2)), zorder=5)
ov.add_patch(con1)
con1b = FancyArrowPatch((36.6, 60), (46.3, 34), connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=7,
                        color=C_WB, lw=1.0, ls=(0, (3, 2)), zorder=5)
ov.add_patch(con1b)

# ---------------- 3. measured advantage
axC = fig.add_axes([0.775, 0.19, 0.21, 0.63])
ov.text(86.5, 95, "How much survives queries", ha="center", fontsize=7.5, fontweight="bold", color=INK)
step(70.3, 95, 3)
Q = [0, 6, 24, 96, 384]
def series(path, model, fam="digits-partition"):
    rows = json.load(open(path))
    d = {r["level"]: r for r in rows if r["family"] == fam and r["model"] == model}
    return [d[f"adv_vs_beh_q{q}"] for q in Q]
mlp = series(os.path.join(BASE, "results_label/summary.json"), "mlp")
knn = series(os.path.join(BASE, "results_knn_label/summary.json"), "knn")
x = np.arange(len(Q))
for s, c, m, name in [(mlp, C_BEH, "o", "network (identifiable)"), (knn, C_WB, "s", "nearest neighbour (stores data)")]:
    y = [r["auroc"] for r in s]; lo = [r["lo"] for r in s]; hi = [r["hi"] for r in s]
    axC.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)
    axC.plot(x, y, color=c, marker=m, ms=3.2, lw=1.3, label=name)
axC.axhline(0, color=GREY, lw=0.6)
axC.set_xticks(x, [str(q) for q in Q])
axC.set_xlabel("queries allowed to the behavioural monitor", color=C_BEH, fontsize=6.4)
axC.set_ylabel("introspective advantage\n(type-2 AUROC)", color=C_WB, fontsize=6.4)
axC.spines[["top", "right"]].set_visible(False)
axC.set_ylim(-0.005, 0.13)
axC.text(0.1, 0.008, "computational only", color=C_BEH, fontsize=6, ha="left")
axC.text(4, 0.043, "privileged", color=C_WB, fontsize=6, ha="right")
axC.legend(frameon=False, fontsize=5.8, loc="upper right", handlelength=1.5, bbox_to_anchor=(1.08, 1.06))

# connector from the leaves to the measured advantage
con2 = FancyArrowPatch((62.8, 50), (67.2, 50), arrowstyle="-|>", mutation_scale=7, color=C_WB, lw=1.0,
                       ls=(0, (3, 2)), zorder=5)
ov.add_patch(con2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join("../figures", f"fig_overview_integrated.{ext}"), dpi=300)
print("ok")
