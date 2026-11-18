"""[MOD] Figure 4 and Table 4: the embedding space before and after RAG fusion.

Question these two figures answer: what does RAG fusion do to the embedding
space, and which part of the fused vector carries the discriminative signal.

Two representations are compared:

    query   plain ESM C embedding of the peptide
    fused   ESM C + RAG fused features (what the model is trained on)

All numbers are computed on the independent test set.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

from common import ROOT, FIGDIR, PALETTE, masked_mean_pool

RNG = 0
D = os.path.join(ROOT, "dataset", "ESMC")


print("loading independent-test embeddings ...")
# [MOD] each entry: (positive file, negative file, mask source or None).
# The fused files are pooled over the positions where the RAW query has real
# residues - see masked_mean_pool for why that matters.
SOURCES = {
    "query": (f"{D}/Ind_pos.npy", f"{D}/Ind_neg.npy", None),
    "fused": (f"{D}/RAG/Ind_pos_fused_1_1.npy", f"{D}/RAG/Ind_neg_fused_1_1.npy",
              (f"{D}/Ind_pos.npy", f"{D}/Ind_neg.npy")),
}
X, y = {}, None
for key, (pos, neg, mask) in SOURCES.items():
    mp_, mn_ = (mask if mask else (None, None))
    xp, xn = masked_mean_pool(pos, mp_), masked_mean_pool(neg, mn_)
    X[key] = np.vstack([xp, xn])
    if y is None:
        y = np.r_[np.ones(len(xp)), np.zeros(len(xn))]

KEYS = ["query", "fused"]
LABEL = {
    "query":     "ESM C, no fusion",
    "fused":     "ESM C + RAG fused features",
}
SHORT = {
    "query":     "ESM C\n(query)",
    "fused":     "RAG fused\nfeatures",
}
COLOUR = {"query": PALETTE[0], "fused": PALETTE[1]}

print({k: X[k].shape for k in KEYS}, "positives =", int(y.sum()))

# ------------------------------------------------------------------ Figure 4
print("\nFigure 4: t-SNE")
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
sil = {}
for ax, k in zip(axes, KEYS):
    Zs = StandardScaler().fit_transform(X[k])
    emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
               random_state=RNG).fit_transform(Zs)
    sil[k] = silhouette_score(Zs, y)
    for lab, col, name in [(0, PALETTE[0], "Non-epitope"), (1, PALETTE[1], "Epitope")]:
        m = y == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=4, c=col, alpha=0.5, linewidths=0, label=name)
    ax.set_title(f"{LABEL[k]}\nsilhouette by label = {sil[k]:+.3f}",
                 loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("0.7")
axes[0].legend(loc="upper left", markerscale=3)
fig.suptitle("t-SNE of independent-test embeddings (mean-pooled, 1152 dimensions)",
             x=0.009, ha="left", fontsize=11, fontweight="bold")
fig.subplots_adjust(top=0.78, wspace=0.07)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGDIR, f"fig4_tsne.{ext}"))
plt.close(fig)
print("  -> analysis/figures/fig4_tsne.png / .pdf")
print("  silhouette:", {k: round(float(v), 4) for k, v in sil.items()})

# ------------------------------------------------------------- Table 4 numbers
print("\nProbe measurements for Table 4")


def probe_auc(Z):
    """AUC of a 5-fold logistic-regression probe fitted on the representation.

    This is a measurement of how much class-discriminative structure a
    representation contains, not a prediction result.
    """
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return cross_val_score(pipe, Z, y, cv=5, scoring="roc_auc").mean()


def cohen_d(Z):
    """|Cohen's d| per dimension between the two classes."""
    a, b = Z[y == 1], Z[y == 0]
    s = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-12
    return np.abs(a.mean(0) - b.mean(0)) / s


probe = {k: probe_auc(X[k]) for k in KEYS}
dvals = {k: cohen_d(X[k]) for k in KEYS}
print("  probe AUC:", {k: round(float(v), 4) for k, v in probe.items()})
print("  median |Cohen's d|:", {k: round(float(np.median(v)), 3) for k, v in dvals.items()})

# [MOD] Geometry of each representation, quoted in section 5.1 of the report.
# ratio = distance between class centroids divided by the pooled within-class
# RMS distance from the own centroid, both in the same units, so it says how
# far apart the classes are relative to their own scatter. frac_d_gt_02 is the
# share of dimensions whose own |Cohen's d| exceeds 0.2 - the quantity a linear
# probe can accumulate over but a Euclidean projection cannot.
geom = {}
for k in KEYS:
    a, b = X[k][y == 1], X[k][y == 0]
    dmu = float(np.linalg.norm(a.mean(0) - b.mean(0)))
    within = float(np.sqrt(np.mean([
        np.mean(np.sum((a - a.mean(0)) ** 2, 1)),
        np.mean(np.sum((b - b.mean(0)) ** 2, 1))])))
    geom[k] = {"centroid_distance": dmu, "within_class_rms": within,
               "ratio": dmu / within,
               "frac_dims_d_gt_02": float(np.mean(dvals[k] > 0.2)),
               "frac_dims_d_gt_03": float(np.mean(dvals[k] > 0.3))}
print("  geometry:", {k: round(v["ratio"], 3) for k, v in geom.items()})

# [MOD] merge instead of overwrite: retrieval_check.py writes its own section
# into the same file and must not be clobbered by a rerun of this script.
STATS_PATH = os.path.join(ROOT, "analysis", "analysis_stats.json")
stats = json.load(open(STATS_PATH)) if os.path.exists(STATS_PATH) else {}
stats.update({"silhouette": {k: float(v) for k, v in sil.items()},
              "probe_auc": {k: float(v) for k, v in probe.items()},
              "cohen_d_median": {k: float(np.median(v)) for k, v in dvals.items()},
              "cohen_d_p99": {k: float(np.percentile(v, 99)) for k, v in dvals.items()},
              "geometry": geom})
json.dump(stats, open(STATS_PATH, "w"), indent=2)
print("\nstats -> analysis/analysis_stats.json")
