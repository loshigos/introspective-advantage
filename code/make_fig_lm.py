import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
rows = json.load(open("../results/results_lm/summary.json"))
Q = [0, 4, 16, 64, 256]; x = np.arange(len(Q))
fig, ax = plt.subplots(figsize=(3.6, 2.5))
ax.axhline(0, color="#999", lw=0.6)
for fam, c, m, lab in [("facts", "#eb6834", "s", "memorized facts"), ("arithmetic", "#2a78d6", "o", "modular addition")]:
    d = {r["level"]: r for r in rows if r["family"] == fam}
    y = [d[f"adv_vs_beh_q{q}"]["auroc"] for q in Q]; lo = [d[f"adv_vs_beh_q{q}"]["lo"] for q in Q]; hi = [d[f"adv_vs_beh_q{q}"]["hi"] for q in Q]
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0); ax.plot(x, y, color=c, marker=m, ms=4, lw=1.5, label=lab)
ax.set_xticks(x, [str(q) for q in Q]); ax.set_xlabel("sampled answers available to the observer")
ax.set_ylabel("white-box minus behavioural\ntype-2 AUROC"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("../figures/fig_lm.pdf"); fig.savefig("../figures/fig_lm.png", dpi=300)
