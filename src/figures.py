import pandas as pd, numpy as np, json, joblib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
ml = joblib.load("ml_model.joblib"); df = pd.read_csv("../results/all_results.csv"); st = pd.read_csv("../results/stress_results.csv")

# 1) decision boundary
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
xx, yy = np.meshgrid(np.linspace(-1.6, 4.6, 400), np.linspace(-1.6, 4.6, 400))
P = ml.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
ax.contourf(xx, yy, P, levels=[0, .5, 1], colors=["#dce9f8", "#fbe1d6"], alpha=.9)
ax.contour(xx, yy, P, levels=[.5], colors=[INK], linewidths=2)
rng = np.random.default_rng(1)
def jit(v): return v + rng.uniform(-.18, .18, len(v))
for split, mk, size in [("train", "o", 70), ("test", "s", 80)]:
    for lab, col, name in [(0, BLUE, "support"), (1, ORANGE, "sales")]:
        d = df[(df.split == split) & (df.label == lab)]
        ax.scatter(jit(d.support_score), jit(d.sales_score), marker=mk, s=size, c=col, edgecolors="white", linewidths=1.5,
                   label=f"{name} ({split})", zorder=3)
wrong = df[df.ml_pred != df.label]
ax.scatter(wrong.support_score, wrong.sales_score, s=330, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=4, label="misrouted by ML (mixed intent)")
ax.text(3.9, -1.25, "support region (flag 0)", ha="right", color=BLUE, fontsize=12, weight="bold")
ax.text(-1.4, 4.2, "sales region (flag 1)", color="#b8451b", fontsize=12, weight="bold")
ax.set_xlabel("support_score  (problem cues)"); ax.set_ylabel("sales_score  (purchase cues)")
ax.set_xlim(-1.6, 4.6); ax.set_ylim(-1.6, 4.6)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.set_title("Logistic regression (degree-2) decision boundary", loc="left", fontsize=15, color=INK)
fig.tight_layout(); fig.savefig("../results/fig/boundary.png"); plt.close(fig)

# 2) accuracy comparison
res = json.load(open("../results/results.json"))
groups = [("Test set\n(20 messages)", "test20"), ("Mixed intent\n(4 training messages)", "mixed4"), ("New hard cases\n(8 messages)", "stress8")]
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
x = np.arange(len(groups)); w = .34
for i, (key, col, name) in enumerate([("ml_acc", BLUE, "ML classifier"), ("dm_acc", ORANGE, "Decision model")]):
    vals = [res[g][key] * 100 for _, g in groups]; ns = [20, 4, 8]
    bars = ax.bar(x + (i - .5) * (w + .02), vals, w, color=col, label=name, zorder=3)
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{round(v*n/100)}/{n}", ha="center", color=INK, fontsize=12)
ax.set_xticks(x, [g for g, _ in groups]); ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_ylabel("Correct routes (%)")
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.12))
fig.tight_layout(); fig.savefig("../results/fig/accuracy.png"); plt.close(fig)
print("ok")
