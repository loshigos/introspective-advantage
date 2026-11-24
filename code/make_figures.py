"""Figures and tables for the manuscript, from results_*/summary.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "../results"
OUT = "../figures"
FAM = ["teacher-net", "teacher-tree", "digits-partition"]
FAM_LABEL = {"teacher-net": "Teacher network", "teacher-tree": "Teacher tree",
             "digits-partition": "Digit partitions (real inputs)"}
MOD = ["mlp", "tree", "forest", "knn"]
MOD_LABEL = {"mlp": "tanh network", "tree": "decision tree", "forest": "random forest",
             "knn": "15-nearest neighbours"}
COL = {"mlp": "#2a78d6", "tree": "#eb6834", "forest": "#1baf7a", "knn": "#eda100"}
MARK = {"mlp": "o", "tree": "s", "forest": "^", "knn": "D"}
Q = [0, 6, 24, 96, 384]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def load(channel):
    rows = json.load(open(os.path.join(BASE, f"results_{channel}", "summary.json")))
    kp = os.path.join(BASE, f"results_knn_{channel}", "summary.json")
    if os.path.exists(kp):
        rows = [r for r in rows if r["model"] != "knn"] + json.load(open(kp))
    d = {}
    for r in rows:
        d[(r["family"], r["model"], r["level"])] = r
    return d


def fig_advantage(D):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True)
    for i, ch in enumerate(["label", "score"]):
        for j, fam in enumerate(FAM):
            ax = axes[i, j]
            ax.axhline(0, color="#999", lw=0.6)
            for m in MOD:
                y = [D[ch][(fam, m, f"adv_vs_beh_q{q}")]["auroc"] for q in Q]
                lo = [D[ch][(fam, m, f"adv_vs_beh_q{q}")]["lo"] for q in Q]
                hi = [D[ch][(fam, m, f"adv_vs_beh_q{q}")]["hi"] for q in Q]
                x = np.arange(len(Q))
                ax.fill_between(x, lo, hi, color=COL[m], alpha=0.15, lw=0)
                ax.plot(x, y, color=COL[m], marker=MARK[m], ms=4, lw=1.5, label=MOD_LABEL[m])
            ax.set_xticks(range(len(Q)), [str(q) for q in Q])
            if i == 0:
                ax.set_title(FAM_LABEL[fam], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(("Label channel" if ch == "label" else "Score channel")
                              + "\nwhite-box minus behavioural\ntype-2 AUROC")
            if i == 1:
                ax.set_xlabel("queries per input")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig_advantage.pdf"))
    fig.savefig(os.path.join(OUT, "fig_advantage.png"), dpi=300)


def fig_decomp(D, ch="label"):
    comps = [("public", "#9ec5f4", "difficulty (public)"),
             ("beh", "#2a78d6", "behaviour (384 queries)"),
             ("adv", "#eb6834", "representation only"),
             ("forg", "#1baf7a", "forgotten by learning")]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    labels, y = [], 0
    yt = []
    for fam in FAM:
        for m in MOD:
            g = lambda lv: D[ch][(fam, m, lv)]["auroc"]
            vals = {"public": g("public") - 0.5, "beh": g("beh_q384") - g("public"),
                    "adv": g("white") - g("beh_q384"), "forg": g("trace") - g("white")}
            left = 0.5
            for key, c, _ in comps:
                v = vals[key]
                ax.barh(y, max(v, 0), left=left, color=c, height=0.7, edgecolor="white", lw=1)
                left += max(v, 0)
            labels.append(f"{FAM_LABEL[fam].split(' (')[0]} / {MOD_LABEL[m]}")
            yt.append(y); y += 1
        y += 0.6
    ax.set_yticks(yt, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("type-2 AUROC (0.5 = chance)")
    from matplotlib.patches import Patch
    ax.legend([Patch(color=c) for _, c, _ in comps], [t for *_, t in comps], ncol=2,
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_decomposition_{ch}.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_decomposition_{ch}.png"), dpi=300)


def table(D):
    lines = []
    for ch in ["label", "score"]:
        for fam in FAM:
            for m in MOD:
                g = lambda lv: D[ch].get((fam, m, lv))
                a0 = g("adv_vs_beh_q0"); a384 = g("adv_vs_beh_q384")
                tw = g("adv_vs_twin"); tr = g("trace_minus_white")
                f = lambda r: "--" if r is None else f"{r['auroc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
                lines.append(dict(channel=ch, family=fam, model=m,
                                  acc=g("white")["acc"], public=g("public")["auroc"],
                                  beh0=g("beh_q0")["auroc"], beh384=g("beh_q384")["auroc"],
                                  white=g("white")["auroc"], trace=g("trace")["auroc"],
                                  adv0=f(a0), adv384=f(a384), twin=f(tw), forg=f(tr),
                                  ratio=(a384["auroc"] / a0["auroc"]) if a0["auroc"] > 0.01 else np.nan))
    json.dump(lines, open(os.path.join(BASE, "table_main.json"), "w"), indent=1)
    for l in lines:
        print(l)


if __name__ == "__main__":
    D = {ch: load(ch) for ch in ["label", "score"]}
    fig_advantage(D); fig_decomp(D, "label"); fig_decomp(D, "score"); table(D)
