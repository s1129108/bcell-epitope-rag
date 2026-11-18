"""[MOD] Shared helpers for the figures and tables in analysis/.

Every number produced here comes from the INDEPENDENT TEST set
(Ind_pos + Ind_neg = 1110 + 1408 = 2518 peptides).

Nothing is re-computed in this module: ROC / PR curves are read back from the
pickles written by MCNN.py at experiment time (PKL/) and from the .npz files
written by Seq_emb_experiment/run.py.
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# colour-blind-safe qualitative palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10.5,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "legend.frameon": False, "figure.autolayout": False,
})


def load_pkl_curve(tag, pkl_dir=os.path.join(ROOT, "PKL")):
    """Read the <tag>_AUCROC.pkl / <tag>_PRROC.pkl pair written by MCNN.py."""
    with open(os.path.join(pkl_dir, f"{tag}_AUCROC.pkl"), "rb") as f:
        roc = pickle.load(f)
    with open(os.path.join(pkl_dir, f"{tag}_PRROC.pkl"), "rb") as f:
        pr = pickle.load(f)
    return {"fpr": roc["fpr"], "tpr": roc["tpr"], "auc": float(roc["AUC"]),
            "precision": pr["precision"], "recall": pr["recall"], "pr_auc": float(pr["AUC"])}


def load_npz_curve(path):
    """Read an .npz curve file written by Seq_emb_experiment/run.py."""
    d = np.load(path)
    return {"fpr": d["fpr"], "tpr": d["tpr"], "auc": float(d["auc"]),
            "precision": d["precision"], "recall": d["recall"], "pr_auc": float(d["pr_auc"])}


def roc_pr_panel(curves, title_roc, title_pr, outname, baseline_pos_rate=None,
                 highlight=None, figsize=(9.4, 4.3), pr_legend="lower left"):
    """Draw a two-panel figure: ROC on the left, precision-recall on the right.

    curves    list of (label, curve dict)
    highlight label drawn with a heavier stroke and on top
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for i, (label, c) in enumerate(curves):
        col = PALETTE[i % len(PALETTE)]
        lw = 2.4 if (highlight and label == highlight) else 1.5
        z = 5 if (highlight and label == highlight) else 2
        ax1.plot(c["fpr"], c["tpr"], color=col, lw=lw, zorder=z,
                 label=f"{label} (AUC = {c['auc']:.4f})")
        ax2.plot(c["recall"], c["precision"], color=col, lw=lw, zorder=z,
                 label=f"{label} (AP = {c['pr_auc']:.4f})")

    ax1.plot([0, 1], [0, 1], ls=(0, (4, 4)), color="0.6", lw=1, zorder=1)
    ax1.set_xlabel("False positive rate (1 - specificity)")
    ax1.set_ylabel("True positive rate (sensitivity)")
    ax1.set_title(title_roc, loc="left", fontweight="bold")

    if baseline_pos_rate is not None:
        ax2.axhline(baseline_pos_rate, ls=(0, (4, 4)), color="0.6", lw=1, zorder=1)
        ax2.text(0.015, baseline_pos_rate + 0.012, f"random = {baseline_pos_rate:.3f}",
                 fontsize=7.5, color="0.45")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(title_pr, loc="left", fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.legend(loc="lower right" if ax is ax1 else pr_legend)
        ax.grid(alpha=0.18, lw=0.6)

    fig.subplots_adjust(wspace=0.28)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{outname}.{ext}"))
    plt.close(fig)
    print(f"  -> analysis/figures/{outname}.png / .pdf")


def metrics_from_curve(c):
    """Only AUC and AP are taken from the curves. Threshold-dependent metrics
    are read from the Excel reports instead, so that no metric is silently
    recomputed at a threshold other than the one used during the run."""
    return {"AUC": c["auc"], "AP": c["pr_auc"]}


def masked_mean_pool(path, mask_path=None):
    """[MOD] Mean-pool over the residue axis, counting only real residues.

    The arrays are zero-padded to maxseq = 25. A plain .mean(1) therefore
    divides by 25 regardless of peptide length, which scales every dimension by
    length/25 and injects peptide length into the whole representation - an
    artefact of the padding, not something ESM C encodes.

    mask_path names the file whose zero-padding defines the real positions.
    For a fused embedding this must be the corresponding RAW file: after
    fusion a padded position is 0.5 * 0 + 0.5 * neighbour, which is not zero
    whenever a retrieved neighbour is longer than the query.
    """
    a = np.load(path, mmap_mode="r")
    a = np.asarray(a).astype(np.float32).reshape(len(a), a.shape[2], a.shape[3])
    src = a if mask_path is None else np.asarray(
        np.load(mask_path, mmap_mode="r")).astype(np.float32).reshape(a.shape)
    real = (np.abs(src).sum(-1) > 0)                      # (N, maxseq)
    n = np.maximum(real.sum(1, keepdims=True), 1)
    return (a * real[:, :, None]).sum(1) / n
