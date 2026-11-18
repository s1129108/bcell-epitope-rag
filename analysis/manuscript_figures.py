"""[MOD] Figures 7 and 8 for the manuscript.

Figure 7  RAG + KG context on the independent test set: the two evaluation
          directions, against ESM C alone and against RAG without context
Figure 8  the held-out breast-cancer antigen set: the same configurations plus
          the iBCE-EL reproduction

Both read curves written at experiment time; nothing is re-trained here.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from common import ROOT, FIGDIR, PALETTE, load_pkl_curve, roc_pr_panel

TAG = "MCNN_25_{}_w2-9-11-25_f64_h128x3_d0.7_lr2e-03_ce_None"
POS_RATE = 1110 / (1110 + 1408)
BC_POS_RATE = 107 / (107 + 162)

# ---------------------------------------------------------------- Figure 7
print("Figure 7: RAG + KG context, two evaluation directions")
c7 = [
    ("ESM C alone", load_pkl_curve(TAG.format("ESMC_RAW"))),
    ("RAG r = 0.3, test fused", load_pkl_curve(TAG.format("ESMC_R03"))),
    ("RAG + KG r = 0.3, test fused", load_pkl_curve(TAG.format("ESMC_RAGKG_R03"))),
    ("RAG + KG r = 0.3, test raw", load_pkl_curve(TAG.format("ESMC_RAGKG_R03_RAWIND"))),
]
roc_pr_panel(c7, "a   ROC - independent test", "b   Precision-recall",
             "fig7_ragkg", baseline_pos_rate=POS_RATE,
             highlight="RAG + KG r = 0.3, test fused")

# ---------------------------------------------------------------- Figure 8
print("Figure 8: held-out breast-cancer antigen set")
c8 = [
    ("ESM C + MCNN", load_pkl_curve(TAG.format("BC_ESMC_RAW"))),
    ("RAG + KG, test fused", load_pkl_curve(TAG.format("BC_RAGKG_FUSED"))),
    ("RAG + KG, test raw", load_pkl_curve(TAG.format("BC_RAGKG_RAWIND"))),
]
roc_pr_panel(c8, "a   ROC - breast-cancer antigen set", "b   Precision-recall",
             "fig8_breast_cancer", baseline_pos_rate=BC_POS_RATE,
             highlight="ESM C + MCNN", pr_legend="upper right")

# ------------------------------------------- Figure 8c: the summary bar chart
ind = pd.read_excel(os.path.join(ROOT, "MCNN_ind_report.xlsx"), sheet_name="Independent")
ibce = pd.read_csv(os.path.join(ROOT, "ibce_repro", "ibce_bc.csv"))

bars = [("iBCE-EL\n(reproduction)", ibce.AUC.mean(), ibce.AUC.std(ddof=1)),
        ("ESM C\n+ MCNN", ind[ind.feature == "BC_ESMC_RAW"].AUC.mean(),
         ind[ind.feature == "BC_ESMC_RAW"].AUC.std(ddof=1)),
        ("RAG + KG\ntest raw", ind[ind.feature == "BC_RAGKG_RAWIND"].AUC.mean(),
         ind[ind.feature == "BC_RAGKG_RAWIND"].AUC.std(ddof=1)),
        ("RAG + KG\ntest fused*", ind[ind.feature == "BC_RAGKG_FUSED"].AUC.mean(),
         ind[ind.feature == "BC_RAGKG_FUSED"].AUC.std(ddof=1))]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
cols = [PALETTE[4], PALETTE[0], PALETTE[1], "0.75"]
for i, (name, m, s) in enumerate(bars):
    ax.bar(i, m, yerr=s, color=cols[i], width=0.62, capsize=4, lw=0)
    ax.text(i, m + s + 0.015, f"{m:.3f}", ha="center", fontsize=8.5)
ax.axhline(0.5, ls=(0, (4, 4)), color="0.6", lw=1)
ax.text(3.42, 0.515, "chance", fontsize=7.5, color="0.45", ha="right")
ax.set_xticks(range(len(bars)))
ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
ax.set_ylabel("AUC on the breast-cancer set")
ax.set_ylim(0, 1.12)
ax.set_title("c   Validation on held-out breast-cancer antigens",
             loc="left", fontweight="bold")
ax.grid(axis="y", alpha=0.18, lw=0.6)
fig.text(0.02, -0.04, "* uses the true label to choose the retrieval partition, "
                      "so it is not a deployable configuration", fontsize=7.2,
         color="0.35")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGDIR, f"fig8c_bc_summary.{ext}"), bbox_inches="tight")
plt.close(fig)
print("  -> analysis/figures/fig8c_bc_summary.png / .pdf")
print("done")
