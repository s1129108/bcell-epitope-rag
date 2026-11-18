"""[MOD] Figures 5-6 and Tables 5-6: what the ESM C representation is actually
keying on when it separates B-cell epitopes from non-epitopes.

Two questions are answered separately.

  (1) Is the signal concentrated in a few of the 1152 ESM C dimensions, or
      spread across many? Measured with per-dimension effect sizes and with a
      nested feature-selection curve: inside every cross-validation fold the
      dimensions are ranked on the training part only and a probe is fitted on
      the top-k, so the curve is not inflated by selecting on the test data.

  (2) What do the discriminating dimensions correspond to biologically? Each
      peptide is scored on a panel of sequence-derived physicochemical
      properties, including the propensity scales classically used for B-cell
      epitope prediction (Parker hydrophilicity, Kolaskar-Tongaonkar
      antigenicity, Chou-Fasman beta-turn, Karplus-Schulz flexibility). The top
      dimensions are then correlated against that panel, and the panel is also
      compared directly between the two classes.

Only sequence-derived properties are used. No IEDB provenance field (host,
assay, source organism) enters this analysis: those fields are predictive of
the label by themselves through the way the benchmark was assembled, so any
"importance" they showed would be a property of the dataset, not of B-cell
epitopes.

All numbers are computed on the independent test set (n = 2518).
"""
import json
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from common import ROOT, FIGDIR, PALETTE, masked_mean_pool

RNG = 0
D = os.path.join(ROOT, "dataset", "ESMC")
AA = "ACDEFGHIKLMNPQRSTVWY"

# --------------------------------------------------------------- propensity scales
# Classical residue-level scales. The first four are the ones used by the
# standard B-cell epitope propensity methods.
SCALES = {
    "Parker hydrophilicity": dict(zip("ARNDCQEGHILKMFPSTWYV", [
        2.1, 4.2, 7.0, 10.0, 1.4, 6.0, 7.8, 5.7, 2.1, -8.0,
        -9.2, 5.7, -4.2, -9.2, 2.1, 6.5, 5.2, -10.0, -1.9, -3.7])),
    "Kolaskar antigenicity": dict(zip("ARNDCQEGHILKMFPSTWYV", [
        1.064, 0.873, 0.776, 0.866, 1.412, 1.015, 0.851, 0.874, 1.105, 1.152,
        1.250, 0.930, 0.826, 1.091, 1.064, 1.012, 0.909, 0.893, 1.161, 1.383])),
    "Chou-Fasman beta-turn": dict(zip("ARNDCQEGHILKMFPSTWYV", [
        0.66, 0.95, 1.56, 1.46, 1.19, 0.98, 0.74, 1.56, 0.95, 0.47,
        0.59, 1.01, 0.60, 0.60, 1.52, 1.43, 0.96, 0.96, 1.14, 0.50])),
    "Karplus flexibility": dict(zip("ARNDCQEGHILKMFPSTWYV", [
        1.041, 1.038, 1.117, 1.033, 0.960, 1.165, 1.094, 1.142, 0.982, 1.002,
        0.967, 1.093, 0.947, 0.930, 1.055, 1.169, 1.073, 0.925, 0.961, 0.982])),
    "Kyte-Doolittle hydropathy": dict(zip("ARNDCQEGHILKMFPSTWYV", [
        1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
        3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2])),
    "Grantham polarity": dict(zip("ARNDCQEGHILKMFPSTWYV", [
        8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2,
        4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6, 5.4, 6.2, 5.9])),
    "Zimmerman bulkiness": dict(zip("ARNDCQEGHILKMFPSTWYV", [
        11.50, 14.28, 12.82, 11.68, 13.46, 14.45, 13.57, 3.40, 13.69, 21.40,
        21.40, 15.71, 16.25, 19.80, 17.43, 9.47, 15.77, 21.67, 18.03, 21.57])),
}
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
CLASSES = {
    "Charged fraction": set("DEKRH"),
    "Aromatic fraction": set("FWY"),
    "Polar fraction": set("STNQCY"),
    "Tiny fraction": set("AGSCP"),
}


def read_fasta(path):
    seqs = []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            seqs.append("")
        elif line:
            seqs[-1] += line
    return seqs


def properties(seq):
    """Per-peptide property vector. Residues outside the standard 20 are
    ignored by the scale averages rather than mapped to an arbitrary value."""
    clean = [c for c in seq.upper() if c in AA]
    n = max(len(clean), 1)
    out = {}
    for name, table in SCALES.items():
        out[name] = float(np.mean([table[c] for c in clean])) if clean else np.nan
    out["Net charge (pH 7)"] = float(sum(CHARGE.get(c, 0.0) for c in clean))
    for name, members in CLASSES.items():
        out[name] = sum(c in members for c in clean) / n
    out["Length"] = float(len(seq))
    return out


# --------------------------------------------------------------- load
print("loading independent-test sequences and ESM C embeddings ...")
pos_seq = read_fasta(os.path.join(ROOT, "dataset", "Ind_pos.fasta"))
neg_seq = read_fasta(os.path.join(ROOT, "dataset", "Ind_neg.fasta"))
X = np.vstack([masked_mean_pool(f"{D}/Ind_pos.npy"),
               masked_mean_pool(f"{D}/Ind_neg.npy")])
y = np.r_[np.ones(len(pos_seq)), np.zeros(len(neg_seq))]
seqs = pos_seq + neg_seq
assert len(seqs) == len(X), f"{len(seqs)} sequences vs {len(X)} embeddings"
print(f"  {len(X)} peptides, {int(y.sum())} epitopes, {X.shape[1]} dimensions")

PROPS = list(properties(seqs[0]).keys())
P = np.array([[properties(s)[k] for k in PROPS] for s in seqs], dtype=np.float64)

# --------------------------------------------------------------- per-dimension stats
a, b = X[y == 1], X[y == 0]
pooled_sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-12
d_dim = (a.mean(0) - b.mean(0)) / pooled_sd                   # signed
absd = np.abs(d_dim)
auc_dim = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
order = np.argsort(-absd)
TOP = order[:20]

probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
probe.fit(X, y)
coef = probe.named_steps["logisticregression"].coef_.ravel()

oof = cross_val_predict(make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=5000)),
                        X, y, cv=5, method="decision_function")
print(f"  full-embedding probe AUC (out-of-fold) = {roc_auc_score(y, oof):.4f}")

# --------------------------------------------------------------- nested selection curve
# Ranking dimensions inside each training fold only; nothing is selected using
# the fold being scored.
KS = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800, X.shape[1]]
curve = []
skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
splits = list(skf.split(X, y))
for k in KS:
    scores = np.empty(len(y))
    for tr, te in splits:
        at, bt = X[tr][y[tr] == 1], X[tr][y[tr] == 0]
        sd = np.sqrt((at.var(0, ddof=1) + bt.var(0, ddof=1)) / 2) + 1e-12
        rank = np.argsort(-np.abs((at.mean(0) - bt.mean(0)) / sd))[:k]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        m.fit(X[tr][:, rank], y[tr])
        scores[te] = m.decision_function(X[te][:, rank])
    curve.append(roc_auc_score(y, scores))
    print(f"  top {k:>4} dimensions -> nested probe AUC {curve[-1]:.4f}", flush=True)

# --------------------------------------------------------------- biology
prop_stats = []
for j, name in enumerate(PROPS):
    va, vb = P[y == 1, j], P[y == 0, j]
    sd = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2) + 1e-12
    u, pv = stats.mannwhitneyu(va, vb, alternative="two-sided")
    prop_stats.append({"property": name, "mean_epitope": float(va.mean()),
                       "mean_non_epitope": float(vb.mean()),
                       "cohen_d": float((va.mean() - vb.mean()) / sd),
                       "p": float(pv),
                       "spearman_with_probe": float(stats.spearmanr(P[:, j], oof).statistic)})
# Benjamini-Hochberg across the property panel
pv = np.array([r["p"] for r in prop_stats])
rk = stats.rankdata(pv)
q = np.minimum.accumulate((pv * len(pv) / rk)[np.argsort(-pv)])[np.argsort(np.argsort(-pv))]
for r_, qq in zip(prop_stats, q):
    r_["q"] = float(min(qq, 1.0))
prop_stats.sort(key=lambda r_: -abs(r_["cohen_d"]))
for r_ in prop_stats:
    print(f"  {r_['property']:<26} d = {r_['cohen_d']:+.3f}  q = {r_['q']:.2e}  "
          f"rho(probe) = {r_['spearman_with_probe']:+.3f}")

# amino-acid composition, per class
comp = {}
for lab, name in [(1, "epitope"), (0, "non_epitope")]:
    c = Counter()
    tot = 0
    for s in (np.array(seqs, dtype=object)[y == lab]):
        for ch in s.upper():
            if ch in AA:
                c[ch] += 1
                tot += 1
    comp[name] = np.array([c[ch] / tot for ch in AA])
comp_lr = np.log2(comp["epitope"] / comp["non_epitope"])

# correlation of the top dimensions with the property panel
rho = np.zeros((len(TOP), len(PROPS)))
for i, j in enumerate(TOP):
    for m_, _ in enumerate(PROPS):
        rho[i, m_] = stats.spearmanr(X[:, j], P[:, m_]).statistic

# --------------------------------------------------------------- Figure 5
print("\nFigure 5: dimension-level importance")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.8, 4.3))

ax1.plot(np.arange(1, len(absd) + 1), absd[order], color=PALETTE[0], lw=1.4)
ax1.fill_between(np.arange(1, 21), absd[order][:20], color=PALETTE[1], alpha=0.35)
ax1.set_yscale("log")
ax1.set_xlabel("ESM C dimension, ranked")
ax1.set_ylabel("|Cohen's d| (epitope vs non-epitope)")
ax1.set_title("a   No dominant dimension", loc="left", fontweight="bold")
ax1.annotate(f"strongest: dim {order[0]}\n|d| = {absd[order[0]]:.2f}",
             xy=(1, absd[order[0]]), xytext=(60, absd[order[0]] * 0.9),
             fontsize=8, color="0.3",
             arrowprops=dict(arrowstyle="-", lw=0.7, color="0.5"))
ax1.grid(alpha=0.18, lw=0.6)

ax2.plot(KS, curve, marker="o", ms=4.5, color=PALETTE[2])
ax2.axhline(curve[-1], ls=(0, (4, 4)), color="0.6", lw=1)
ax2.text(1.2, curve[-1] + 0.004, f"all {X.shape[1]} dims = {curve[-1]:.3f}",
         fontsize=7.5, color="0.45")
ax2.set_xscale("log")
ax2.set_xlabel("Number of top-ranked dimensions used")
ax2.set_ylabel("Nested probe AUC")
ax2.set_title("b   A few dimensions already suffice", loc="left", fontweight="bold")
ax2.grid(alpha=0.18, lw=0.6)

ax3.scatter(absd, np.abs(coef), s=6, c="0.7", linewidths=0, label="all dimensions")
ax3.scatter(absd[TOP], np.abs(coef)[TOP], s=22, c=PALETTE[1], linewidths=0,
            label="top 20 by |d|")
ax3.set_xlabel("|Cohen's d| of the dimension")
ax3.set_ylabel("|weight| in the linear probe")
ax3.set_title("c   Effect size vs probe weight", loc="left", fontweight="bold")
ax3.legend(fontsize=7.5)
ax3.grid(alpha=0.18, lw=0.6)

fig.subplots_adjust(wspace=0.3)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(FIGDIR, f"fig5_dimension_importance.{e}"))
plt.close(fig)
print("  -> analysis/figures/fig5_dimension_importance.png / .pdf")

# --------------------------------------------------------------- Figure 6
print("\nFigure 6: biological character of the signal")
fig = plt.figure(figsize=(15.0, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.42)
axA, axB, axC = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                 fig.add_subplot(gs[2]))

names = [r_["property"] for r_ in prop_stats][::-1]
ds = [r_["cohen_d"] for r_ in prop_stats][::-1]
qs = [r_["q"] for r_ in prop_stats][::-1]
cols = [PALETTE[1] if v > 0 else PALETTE[0] for v in ds]
axA.barh(range(len(ds)), ds, color=cols, edgecolor="white", linewidth=0.6)
for i, (v, qq) in enumerate(zip(ds, qs)):
    if qq < 0.05:
        axA.text(v + (0.012 if v > 0 else -0.012), i, "*", va="center",
                 ha="left" if v > 0 else "right", fontsize=11, color="0.3")
axA.axvline(0, color="0.4", lw=0.8)
axA.set_yticks(range(len(names)))
axA.set_yticklabels(names, fontsize=8)
axA.set_xlabel("Cohen's d  (epitope - non-epitope)")
axA.set_title("a   Sequence properties by class", loc="left", fontweight="bold")
axA.grid(axis="x", alpha=0.18, lw=0.6)

v = np.abs(rho).max()
im = axB.imshow(rho, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
axB.set_xticks(range(len(PROPS)))
axB.set_xticklabels(PROPS, rotation=42, ha="right", fontsize=7.5)
axB.set_yticks(range(len(TOP)))
axB.set_yticklabels([f"dim {j}" for j in TOP], fontsize=7)
axB.set_title("b   Top dimensions vs properties (Spearman)", loc="left",
              fontweight="bold")
fig.colorbar(im, ax=axB, fraction=0.035, pad=0.02).ax.tick_params(labelsize=7)

ordc = np.argsort(comp_lr)
axC.barh(range(20), comp_lr[ordc],
         color=[PALETTE[1] if comp_lr[i] > 0 else PALETTE[0] for i in ordc],
         edgecolor="white", linewidth=0.6)
axC.axvline(0, color="0.4", lw=0.8)
axC.set_yticks(range(20))
axC.set_yticklabels([AA[i] for i in ordc], fontsize=8)
axC.set_xlabel("log2 (epitope / non-epitope) composition")
axC.set_title("c   Amino-acid composition", loc="left", fontweight="bold")
axC.grid(axis="x", alpha=0.18, lw=0.6)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(FIGDIR, f"fig6_biology.{e}"))
plt.close(fig)
print("  -> analysis/figures/fig6_biology.png / .pdf")

# --------------------------------------------------------------- stats out
best_prop = []
for i, j in enumerate(TOP):
    m_ = int(np.argmax(np.abs(rho[i])))
    best_prop.append({"dim": int(j), "cohen_d": float(d_dim[j]),
                      "univariate_auc": float(auc_dim[j]),
                      "probe_weight": float(coef[j]),
                      "best_property": PROPS[m_], "rho": float(rho[i, m_])})

STATS_PATH = os.path.join(ROOT, "analysis", "analysis_stats.json")
stats_all = json.load(open(STATS_PATH)) if os.path.exists(STATS_PATH) else {}
stats_all["dimension"] = {
    "n_dims": int(X.shape[1]),
    "max_abs_d": float(absd.max()),
    "median_abs_d": float(np.median(absd)),
    "frac_d_gt_02": float(np.mean(absd > 0.2)),
    "best_univariate_auc": float(np.max(np.abs(auc_dim - 0.5)) + 0.5),
    "selection_curve": {"k": KS, "auc": [float(v) for v in curve]},
    "full_probe_auc_oof": float(roc_auc_score(y, oof)),
    "top_dims": best_prop,
}
stats_all["biology"] = {
    "n": int(len(y)), "n_epitope": int(y.sum()),
    "properties": prop_stats,
    "aa_log2_ratio": {AA[i]: float(comp_lr[i]) for i in range(20)},
}
json.dump(stats_all, open(STATS_PATH, "w"), indent=2)
print("\nstats -> analysis/analysis_stats.json  (keys 'dimension', 'biology')")
