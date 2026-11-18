"""[MOD] Figures 1-3: ROC / PR comparisons on the INDEPENDENT TEST set.

Figure 1  the five protein language models, each at its own Optuna-best MCNN
          configuration (FFN head)
Figure 2  ESM C + MCNN against traditional classifiers on mean-pooled ESM C
Figure 3  ESM C alone against ESM C + RAG fused features
"""
import os
from common import ROOT, load_pkl_curve, load_npz_curve, roc_pr_panel

POS_RATE = 1110 / (1110 + 1408)          # positive rate of the independent set


# ---------------------------------------------------------------- Figure 1
PLM = [
    ("ESM C 600M",   "MCNN_25_ESMC_RAW_w2-9-11-25_f64_h128x3_d0.7_lr2e-03_ce_None"),
    ("ESM-2 650M",   "MCNN_25_ESM2_w4-19-22-23_f256_h500x2_d0.7_lr4e-03_ce_None"),
    ("Ankh3",        "MCNN_25_Ankh3_w22-25_f32_h128x1_d0.7_lr4e-04_ce_None"),
    ("ProtT5-XL",    "MCNN_25_ProtT5_w11-17-25_f32_h500x3_d0.7_lr5e-04_focal_None"),
    ("AMPLIFY 350M", "MCNN_25_AMPLIFY_w5-14-19-25_f64_h256x3_d0.7_lr1e-04_ce_None"),
]

print("Figure 1: protein language model comparison")
c1 = [(n, load_pkl_curve(t)) for n, t in PLM]
c1.sort(key=lambda kv: -kv[1]["auc"])
roc_pr_panel(c1, "a   ROC - protein language models", "b   Precision-recall",
             "fig1_plm_comparison", baseline_pos_rate=POS_RATE,
             highlight=c1[0][0])

# ---------------------------------------------------------------- Figure 2
print("Figure 2: ESM C + MCNN vs traditional classifiers")
TRAD_DIR = os.path.join(ROOT, "PKL", "traditional_ESMC")
c2 = [("ESM C per-residue + MCNN",
       load_pkl_curve("MCNN_25_ESMC_RAW_w2-9-11-25_f64_h128x3_d0.7_lr2e-03_ce_None"))]
for clf in ["SVM", "XGBoost", "LR", "RF", "KNN"]:
    c2.append((f"ESM C mean-pooled + {clf}", load_pkl_curve(f"{clf}_ESMC", TRAD_DIR)))
roc_pr_panel(c2, "a   ROC - MCNN vs traditional classifiers",
             "b   Precision-recall", "fig2_esmc_vs_traditional",
             baseline_pos_rate=POS_RATE, highlight=c2[0][0])

# ---------------------------------------------------------------- Figure 3
# [MOD] query-ratio sweep. r is the weight on the peptide's own embedding:
#     fused = r * query + (1 - r) * mean(top-5 neighbours)
# r = 0.5 reuses the existing *_fused_1_1.npy data; r = 0.9 and r = 0.3 were
# built by analysis/make_ratio.py from the same neighbour indices.
print("Figure 3: query-ratio sweep for RAG fused features")
TAG = "MCNN_25_ESMC_{}_w2-9-11-25_f64_h128x3_d0.7_lr2e-03_ce_None"
c3 = [
    ("ESM C, no fusion", load_pkl_curve(TAG.format("RAW"))),
    ("RAG fused, r = 0.9", load_pkl_curve(TAG.format("R09"))),
    ("RAG fused, r = 0.5", load_pkl_curve(TAG.format("R05"))),
    ("RAG fused, r = 0.3", load_pkl_curve(TAG.format("R03"))),
]
roc_pr_panel(c3, "a   ROC - query-ratio sweep", "b   Precision-recall",
             "fig3_rag_fusion", baseline_pos_rate=POS_RATE,
             highlight="RAG fused, r = 0.3")
print("done")
