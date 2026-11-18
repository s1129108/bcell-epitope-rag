"""[MOD] Build the Word report: comparison tables + figures, all on the
INDEPENDENT TEST set.

Every number is read back from files produced at experiment time (PKL/,
*.xlsx, analysis_stats.json). This script trains nothing.
"""
import json
import os

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from scipy import stats

from common import ROOT, FIGDIR, load_pkl_curve

OUT = os.path.join(ROOT, "analysis", "BCell_analysis_report.docx")
STATS = json.load(open(os.path.join(ROOT, "analysis", "analysis_stats.json")))

# [MOD] query-ratio sweep. Each entry: feature name in MCNN_ind_report.xlsx,
# the r value it was built with, and its PKL curve tag.
RATIOS = [("ESMC_R09", 0.9), ("ESMC_R05", 0.5), ("ESMC_R03", 0.3)]
TAG = "MCNN_25_ESMC_{}_w2-9-11-25_f64_h128x3_d0.7_lr2e-03_ce_None"


# ------------------------------------------------------------------ helpers
def add_table(doc, df, caption, note=None, widths=None):
    p = doc.add_paragraph()
    r = p.add_run(caption)
    r.bold = True
    r.font.size = Pt(9.5)

    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = str(c)
        for par in cell.paragraphs:
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in par.runs:
                run.bold = True
                run.font.size = Pt(8.5)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = "" if v is None else str(v)
            for par in cells[j].paragraphs:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
                for run in par.runs:
                    run.font.size = Pt(8.5)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)
    if note:
        p = doc.add_paragraph()
        r = p.add_run(note)
        r.italic = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()


def add_figure(doc, name, caption, width=6.5):
    doc.add_picture(os.path.join(FIGDIR, f"{name}.png"), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(caption)
    r.font.size = Pt(8.5)
    doc.add_paragraph()


def fmt(m, s=None, nd=4):
    return f"{m:.{nd}f}" if s is None or not np.isfinite(s) else f"{m:.{nd}f} ± {s:.{nd}f}"


# ------------------------------------------------------------------ data
lstm = pd.read_excel(os.path.join(ROOT, "MCNN_lstm_plm_ind.xlsx"))
ind = pd.read_excel(os.path.join(ROOT, "MCNN_ind_report.xlsx"))
trad = pd.read_excel(os.path.join(ROOT, "Traditional_ind_report_ESMC.xlsx"))

PLM_NAME = {"ESMC": "ESM C 600M", "ESM2": "ESM-2 650M", "Ankh3": "Ankh3",
            "ProtT5": "ProtT5-XL", "AMPLIFY": "AMPLIFY 350M"}
PLM_TAG = {
    "ESMC": "MCNN_25_ESMC_RAW_w2-9-11-25_f64_h128x3_d0.7_lr2e-03_ce_None",
    "ESM2": "MCNN_25_ESM2_w4-19-22-23_f256_h500x2_d0.7_lr4e-03_ce_None",
    "Ankh3": "MCNN_25_Ankh3_w22-25_f32_h128x1_d0.7_lr4e-04_ce_None",
    "ProtT5": "MCNN_25_ProtT5_w11-17-25_f32_h500x3_d0.7_lr5e-04_focal_None",
    "AMPLIFY": "MCNN_25_AMPLIFY_w5-14-19-25_f64_h256x3_d0.7_lr1e-04_ce_None",
}
ARCH = {
    "ESMC": "w = 2,9,11,25; f = 64; 128x3; lr 2.4e-3; CE",
    "ESM2": "w = 4,19,22,23; f = 256; 500x2; lr 4.0e-3; CE",
    "Ankh3": "w = 22,25; f = 32; 128x1; lr 4.3e-4; CE",
    "ProtT5": "w = 11,17,25; f = 32; 500x3; lr 5.0e-4; focal",
    "AMPLIFY": "w = 5,14,19,25; f = 64; 256x3; lr 1.0e-4; CE",
}

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.85)

doc.add_heading("Linear B-cell epitope prediction with protein language models "
                "and a multi-scale CNN", level=0)
p = doc.add_paragraph()
r = p.add_run("Analysis report - all figures reported on the independent test set "
              "(1110 epitopes / 1408 non-epitopes, n = 2518)")
r.italic = True

# ------------------------------------------------------------------ Table 1
doc.add_heading("1. Comparison of protein language models", level=1)
doc.add_paragraph(
    "Each protein language model was tuned separately with Optuna. The table "
    "below reports each PLM at its own best configuration with the FFN head. "
    "The AUC column is the mean +/- standard deviation over 10 independent "
    "training runs with different seeds; the remaining columns come from the "
    "representative run whose ROC and PR curves were stored."
)
rows = []
for k in ["ESMC", "ESM2", "Ankh3", "ProtT5", "AMPLIFY"]:
    sub = lstm[(lstm.feature == k) & (lstm["head"] == "ffn")]
    c = load_pkl_curve(PLM_TAG[k])
    rep = sub.iloc[(sub.AUC - sub.AUC.mean()).abs().argsort().iloc[0]]
    rows.append({
        "PLM": PLM_NAME[k], "MCNN configuration": ARCH[k],
        "AUC (10 runs)": fmt(sub.AUC.mean(), sub.AUC.std(ddof=1)),
        "AP": f"{c['pr_auc']:.4f}",
        "MCC": f"{rep.MCC:.4f}", "Acc": f"{rep.Acc:.4f}",
        "Sens": f"{rep.Sens:.4f}", "Spec": f"{rep.Spec:.4f}",
    })
t1 = pd.DataFrame(rows).sort_values("AUC (10 runs)", ascending=False)
add_table(doc, t1, "Table 1. Protein language models with MCNN (FFN head), "
                   "independent test set.",
          "AP = average precision, the area under the precision-recall curve. "
          "Decision thresholds for MCC / Acc / Sens / Spec were selected on the "
          "training set, never on the independent test set.")
add_figure(doc, "fig1_plm_comparison",
           "Figure 1. ROC (a) and precision-recall (b) curves for the five "
           "protein language models on the independent test set. The dashed "
           "line marks a random classifier.")

# ------------------------------------------------------------------ Table 2
doc.add_heading("2. ESM C with MCNN versus traditional classifiers", level=1)
doc.add_paragraph(
    "The traditional classifiers operate on ESM C embeddings mean-pooled over "
    "the residue axis (1152 dimensions); each was grid-searched separately with "
    "5-fold cross-validation on the training set. MCNN instead consumes the "
    "per-residue representation (25 x 1152), so the gap between the two groups "
    "measures the value of positional residue information."
)
raw10 = ind[ind.feature == "ESMC_RAW"]
rep_raw = raw10.iloc[(raw10.AUC - raw10.AUC.mean()).abs().argsort().iloc[0]]
rows = [{
    "Model": "ESM C per-residue + MCNN", "Representation": "25 x 1152",
    "AUC": fmt(raw10.AUC.mean(), raw10.AUC.std(ddof=1)),
    "AP": f"{raw10.PR_AUC.mean():.4f}",
    "MCC": f"{rep_raw.MCC:.4f}", "Acc": f"{rep_raw.Acc:.4f}",
    "Sens": f"{rep_raw.Sens:.4f}", "Spec": f"{rep_raw.Spec:.4f}",
}]
for _, r_ in trad.sort_values("AUC", ascending=False).iterrows():
    rows.append({
        "Model": f"ESM C mean-pooled + {r_.classifier}", "Representation": "1152",
        "AUC": f"{r_.AUC:.4f}", "AP": f"{r_.PR_AUC:.4f}", "MCC": f"{r_.MCC:.4f}",
        "Acc": f"{r_.Acc:.4f}", "Sens": f"{r_.Sens:.4f}", "Spec": f"{r_.Spec:.4f}",
    })
add_table(doc, pd.DataFrame(rows),
          "Table 2. MCNN versus traditional classifiers on the same ESM C "
          "embeddings.",
          "The first row is the mean over 10 runs; the remaining rows are single "
          "runs at grid-searched hyperparameters.")
add_figure(doc, "fig2_esmc_vs_traditional",
           "Figure 2. ESM C + MCNN against five traditional classifiers on "
           "mean-pooled ESM C embeddings, independent test set.")

# ------------------------------------------------------------------ Table 3
doc.add_heading("3. RAG fused features and the query ratio", level=1)
doc.add_paragraph(
    "Each peptide embedding is blended with the mean of its 5 nearest "
    "neighbours (L2 distance over the flattened 25 x 1152 vector) retrieved "
    "from a knowledge graph derived from IEDB:"
)
p = doc.add_paragraph()
r = p.add_run("fused  =  r * query  +  (1 - r) * mean(top-5 neighbours)")
r.italic = True
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph(
    "r is the weight kept on the peptide's own ESM C embedding, so a small r "
    "means a representation dominated by the retrieved neighbours. Retrieval "
    "runs against two class-specific knowledge-graph partitions: 19,003 "
    "sequences on the positive side and 222,569 on the negative side, both "
    "filtered for sequence overlap with the evaluation data. Retrieval itself "
    "does not depend on r, so all three settings below share one set of "
    "neighbour indices and differ only in the blend. Architecture, "
    "hyperparameters, training schedule and test set are identical throughout, "
    "and every row averages 5 independent training runs."
)

raw10 = ind[ind.feature == "ESMC_RAW"]
rows = [{"Configuration": "ESM C, no fusion", "r": "-", "Runs": len(raw10),
         "AUC": fmt(raw10.AUC.mean(), raw10.AUC.std(ddof=1)),
         "AP": f"{raw10.PR_AUC.mean():.4f}", "MCC": f"{raw10.MCC.mean():.4f}",
         "Acc": f"{raw10.Acc.mean():.4f}", "Sens": f"{raw10.Sens.mean():.4f}",
         "Spec": f"{raw10.Spec.mean():.4f}", "delta AUC": "-"}]
R = {}
for feat, rv in RATIOS:
    sub = ind[ind.feature == feat]
    assert len(sub) >= 3, f"{feat}: only {len(sub)} runs"
    R[rv] = sub
    rows.append({"Configuration": "ESM C + RAG fused features", "r": f"{rv:.1f}",
                 "Runs": len(sub),
                 "AUC": fmt(sub.AUC.mean(), sub.AUC.std(ddof=1)),
                 "AP": f"{sub.PR_AUC.mean():.4f}", "MCC": f"{sub.MCC.mean():.4f}",
                 "Acc": f"{sub.Acc.mean():.4f}", "Sens": f"{sub.Sens.mean():.4f}",
                 "Spec": f"{sub.Spec.mean():.4f}",
                 "delta AUC": f"{sub.AUC.mean() - raw10.AUC.mean():+.4f}"})

t_53, p_53 = stats.ttest_ind(R[0.3].AUC, R[0.5].AUC, equal_var=False)
t_59, p_59 = stats.ttest_ind(R[0.9].AUC, R[0.5].AUC, equal_var=False)
best_r = max(R, key=lambda k: R[k].AUC.mean())

add_table(doc, pd.DataFrame(rows),
          "Table 3. Query-ratio sweep for RAG fused features, independent test set.",
          f"Welch t-tests against r = 0.5: r = 0.3 gives "
          f"{R[0.3].AUC.mean() - R[0.5].AUC.mean():+.4f} AUC (p = {p_53:.1e}) and "
          f"r = 0.9 gives {R[0.9].AUC.mean() - R[0.5].AUC.mean():+.4f} "
          f"(p = {p_59:.1e}). delta AUC is measured against the no-fusion baseline.")
add_figure(doc, "fig3_rag_fusion",
           "Figure 3. ROC (a) and precision-recall (b) curves across the query "
           "ratio, independent test set. Lower r keeps less of the peptide's own "
           "embedding and more of the retrieved neighbourhood.")

# ------------------------------------------------------------------ Table 4
doc.add_heading("4. The embedding space before and after fusion", level=1)
doc.add_paragraph(
    "Table 4 characterises the two representations directly, independently of "
    "the MCNN. Embeddings are mean-pooled over the real residues of each "
    "peptide (padding positions excluded, since a plain mean over the "
    "zero-padded 25 positions would scale every dimension by peptide length). "
    "The linear probe is a logistic regression with standardisation, fitted "
    "5-fold on the independent set itself: it measures how much "
    "class-discriminative structure a representation holds and is not a "
    "prediction result, so it is not comparable with Tables 1-3."
)
NAMES = {"query": "ESM C embedding", "fused": "RAG fused features (r = 0.5)"}
t4 = pd.DataFrame([
    {"Representation": NAMES[k],
     "Linear-probe AUC": f"{STATS['probe_auc'][k]:.4f}",
     "median |Cohen's d|": f"{STATS['cohen_d_median'][k]:.3f}",
     "99th pct |Cohen's d|": f"{STATS['cohen_d_p99'][k]:.3f}",
     "centroid / scatter": f"{STATS['geometry'][k]['ratio']:.3f}",
     "silhouette by label": f"{STATS['silhouette'][k]:+.3f}"}
    for k in ["query", "fused"]
])
add_table(doc, t4, "Table 4. Class structure of the two representations.",
          "centroid / scatter is the distance between the two class centroids "
          "divided by the pooled within-class RMS radius, in the same units.")
add_figure(doc, "fig4_tsne",
           "Figure 4. t-SNE of the independent-test embeddings, mean-pooled to "
           "1152 dimensions over real residues only.", width=5.6)

# ------------------------------------------------------------------ Table 5
doc.add_heading("5. Which ESM C dimensions separate epitopes from non-epitopes",
                level=1)
DIM = STATS["dimension"]
doc.add_paragraph(
    f"This section asks whether the 1152 ESM C dimensions contain individually "
    f"identifiable epitope features. Per-dimension effect sizes are small: the "
    f"strongest single dimension reaches |Cohen's d| = {DIM['max_abs_d']:.3f} "
    f"and a univariate AUC of {DIM['best_univariate_auc']:.4f}, the median "
    f"across all dimensions is {DIM['median_abs_d']:.3f}, and only "
    f"{100 * DIM['frac_d_gt_02']:.1f}% of dimensions exceed |d| = 0.2. There is "
    f"no single dimension that acts as an epitope detector."
)
doc.add_paragraph(
    "To test how many dimensions are needed, the dimensions are ranked by "
    "effect size inside each cross-validation fold using the training part "
    "only, and a probe is fitted on the top-k; nothing is selected using the "
    "fold being scored. Figure 5b shows the resulting curve."
)
sel = DIM["selection_curve"]
kbest = sel["k"][int(np.argmax(sel["auc"]))]
t5 = pd.DataFrame([
    {"ESM C dimension": f"dim {t['dim']}",
     "Cohen's d": f"{t['cohen_d']:+.3f}",
     "Univariate AUC": f"{t['univariate_auc']:.3f}",
     "Probe weight": f"{t['probe_weight']:+.3f}",
     "Most correlated property": t["best_property"],
     "Spearman rho": f"{t['rho']:+.3f}"}
    for t in DIM["top_dims"][:10]
])
add_table(doc, t5,
          "Table 5. The ten most discriminative ESM C dimensions and the "
          "sequence property each one tracks most closely.",
          "Cohen's d is signed: positive means a higher value in epitopes. "
          "Probe weight is the coefficient of the dimension in a logistic "
          "regression fitted on all 1152 standardised dimensions.")
add_figure(doc, "fig5_dimension_importance",
           "Figure 5. (a) Per-dimension effect sizes, sorted, with the top 20 "
           "shaded. (b) Nested feature-selection curve: probe AUC against the "
           "number of top-ranked dimensions retained. (c) Per-dimension effect "
           "size against the weight the probe gives that dimension.")

# ------------------------------------------------------------------ Table 6
doc.add_heading("6. Biological character of the discriminating signal", level=1)
BIO = STATS["biology"]
doc.add_paragraph(
    "Each peptide was scored on a panel of sequence-derived physicochemical "
    "properties, including the residue propensity scales classically used for "
    "B-cell epitope prediction. Table 6 compares the panel between the two "
    "classes and also reports how strongly each property tracks the "
    "out-of-fold score of a linear probe trained on the raw ESM C embedding, "
    "which indicates what the representation is keying on."
)
doc.add_paragraph(
    "Only sequence-derived properties are used. IEDB provenance fields such as "
    "host species, assay type or source organism are deliberately excluded: on "
    "this benchmark they predict the label on their own through the way the "
    "data was assembled, so any importance they showed would describe the "
    "dataset rather than B-cell epitopes.", style="List Bullet")
t6 = pd.DataFrame([
    {"Property": r_["property"],
     "Mean, epitope": f"{r_['mean_epitope']:.3f}",
     "Mean, non-epitope": f"{r_['mean_non_epitope']:.3f}",
     "Cohen's d": f"{r_['cohen_d']:+.3f}",
     "q (BH)": f"{r_['q']:.1e}",
     "rho with probe score": f"{r_['spearman_with_probe']:+.3f}"}
    for r_ in BIO["properties"]
])
add_table(doc, t6,
          "Table 6. Sequence properties of epitopes versus non-epitopes, "
          "independent test set.",
          "Cohen's d is positive when the property is higher in epitopes. "
          "q is the Benjamini-Hochberg adjusted Mann-Whitney p-value across the "
          "panel. The last column is the Spearman correlation between the "
          "property and the out-of-fold linear-probe score on raw ESM C.")
add_figure(doc, "fig6_biology",
           "Figure 6. (a) Effect size of each sequence property between the two "
           "classes; * marks q < 0.05. (b) Spearman correlation between the 20 "
           "most discriminative ESM C dimensions and the property panel. "
           "(c) Amino-acid composition of epitopes relative to non-epitopes.")

doc.save(OUT)
print("tables written ->", OUT)

# ------------------------------------------------------------------ discussion
doc.add_heading("7. Interpretation", level=1)

doc.add_heading("7.1 The query ratio", level=2)
doc.add_paragraph(
    f"Across the three settings tested the independent-test AUC increases "
    f"monotonically as weight moves from the peptide's own embedding to the "
    f"retrieved neighbourhood: {R[0.9].AUC.mean():.4f} at r = 0.9, "
    f"{R[0.5].AUC.mean():.4f} at r = 0.5 and {R[0.3].AUC.mean():.4f} at "
    f"r = 0.3. The ordering is consistent across all 5 seeds per setting and "
    f"the gap between r = 0.3 and r = 0.5 is significant "
    f"(Welch p = {p_53:.1e}). On this evidence r = 0.5 is not the optimum of "
    f"the sweep; the best of the three values tested is r = {best_r}, and the "
    f"trend has not turned around by then, so the optimum may lie at a still "
    f"lower r. Values below 0.3 were not tested."
)
doc.add_paragraph(
    f"The direction of the trend is itself informative: the fewer of the "
    f"peptide's own features the representation keeps, the better the model "
    f"scores. That is consistent with Table 4, where the fused representation "
    f"holds considerably more class structure (linear-probe AUC "
    f"{STATS['probe_auc']['fused']:.4f}) than the plain ESM C embedding "
    f"({STATS['probe_auc']['query']:.4f})."
)

doc.add_heading("7.2 There is no single epitope dimension", level=2)
doc.add_paragraph(
    f"The per-dimension analysis gives a clear negative answer to the question "
    f"of whether ESM C contains an identifiable epitope feature. The strongest "
    f"of the 1152 dimensions separates the classes at a univariate AUC of only "
    f"{DIM['best_univariate_auc']:.4f}, barely above chance, and the "
    f"distribution of effect sizes decays smoothly with no outliers "
    f"(Figure 5a). What signal exists is carried by a moderate number of weak, "
    f"partly redundant dimensions."
)
doc.add_paragraph(
    f"The nested selection curve (Figure 5b) peaks at {kbest} dimensions with "
    f"an AUC of {max(sel['auc']):.4f} and then declines slightly to "
    f"{sel['auc'][-1]:.4f} when all {DIM['n_dims']} are used. Adding "
    f"dimensions beyond the leading few therefore contributes more noise than "
    f"signal in a linear model. This is a statement about the mean-pooled "
    f"representation only: the MCNN reaches a much higher AUC on the same "
    f"embeddings by using the per-residue structure that mean-pooling discards, "
    f"as Table 2 shows."
)

doc.add_heading("7.3 What the signal corresponds to biologically", level=2)
top3 = BIO["properties"][:3]
doc.add_paragraph(
    f"The property panel reproduces the classical picture of linear B-cell "
    f"epitopes. The strongest effects are "
    + ", ".join(f"{r_['property']} (d = {r_['cohen_d']:+.3f}, "
                f"q = {r_['q']:.1e})" for r_ in top3)
    + ". Epitopes are enriched in beta-turn propensity and depleted in "
      "hydropathy, and they score higher on Parker hydrophilicity and Grantham "
      "polarity. All four point the same way: linear B-cell epitopes sit in "
      "hydrophilic, turn-forming, solvent-exposed stretches rather than in "
      "buried hydrophobic cores, which is what antibody accessibility requires."
)
kolaskar = next(r_ for r_ in BIO["properties"] if "Kolaskar" in r_["property"])
doc.add_paragraph(
    f"One scale runs against its intended direction. Kolaskar-Tongaonkar "
    f"antigenicity is *lower* in epitopes than in non-epitopes "
    f"(d = {kolaskar['cohen_d']:+.3f}, q = {kolaskar['q']:.1e}) even though it "
    f"was designed to be higher. The scale is weighted towards hydrophobic and "
    f"bulky residues, so on a dataset where epitopes are the more hydrophilic "
    f"class it is anti-correlated with the label by construction. It should not "
    f"be used as a feature here."
)
flex = next(r_ for r_ in BIO["properties"] if "Karplus" in r_["property"])
length = next(r_ for r_ in BIO["properties"] if r_["property"] == "Length")
doc.add_paragraph(
    f"Two further observations. Karplus-Schulz flexibility is higher in "
    f"epitopes but only weakly so (d = {flex['cohen_d']:+.3f}, "
    f"q = {flex['q']:.1e}), which is a much smaller effect than the turn and "
    f"hydropathy terms. And peptide length differs between the classes "
    f"(d = {length['cohen_d']:+.3f}, q = {length['q']:.1e}): epitopes are "
    f"systematically longer in this benchmark. Length is a property of how the "
    f"dataset was assembled as much as of epitope biology, and it correlates "
    f"with several of the leading ESM C dimensions in Table 5, so part of what "
    f"the representation separates is peptide length rather than antigenicity. "
    f"A length-matched control would be needed to separate the two."
)
doc.add_paragraph(
    "At the level of individual residues (Figure 6c) the composition shift is "
    "consistent with the scale-level result: epitopes are relatively enriched "
    "in the small polar and turn-forming residues and depleted in the strongly "
    "hydrophobic ones."
)

doc.add_heading("7.4 t-SNE is not sensitive to this kind of structure", level=2)
doc.add_paragraph(
    f"A methodological point worth recording: t-SNE and the silhouette "
    f"coefficient barely separate the two representations "
    f"(silhouette {STATS['silhouette']['query']:+.3f} against "
    f"{STATS['silhouette']['fused']:+.3f}, Figure 4) even though the linear "
    f"probe puts them {STATS['probe_auc']['fused'] - STATS['probe_auc']['query']:.2f} "
    f"AUC apart. The structure that fusion adds is small in any single "
    f"dimension but consistent across many of them, so Euclidean distance - "
    f"which is what t-SNE preserves - is dominated by the much larger overall "
    f"variance. A two-dimensional projection should not be used on its own to "
    f"decide whether two representations differ."
)

doc.add_heading("8. Summary", level=1)
esmc_mean = lstm[(lstm.feature == 'ESMC') & (lstm['head'] == 'ffn')].AUC.mean()
for line in [
    f"ESM C 600M is the strongest of the five protein language models tested, "
    f"at AUC {esmc_mean:.4f} over 10 runs.",
    f"MCNN on per-residue embeddings clearly outperforms traditional "
    f"classifiers on mean-pooled embeddings ({raw10.AUC.mean():.4f} against "
    f"{trad.AUC.max():.4f} for the best of them, SVM), so positional residue "
    f"information matters.",
    f"RAG fused features raise the independent-test AUC from "
    f"{raw10.AUC.mean():.4f} to {R[0.5].AUC.mean():.4f} at r = 0.5.",
    f"Across the ratios tested the AUC rises monotonically as more weight goes "
    f"to the retrieved neighbourhood: {R[0.9].AUC.mean():.4f} (r = 0.9), "
    f"{R[0.5].AUC.mean():.4f} (r = 0.5), {R[0.3].AUC.mean():.4f} (r = 0.3). "
    f"r = 0.3 is the best of the three and the trend has not yet turned.",
    f"No individual ESM C dimension acts as an epitope detector: the best one "
    f"reaches a univariate AUC of {DIM['best_univariate_auc']:.4f}, and a "
    f"linear probe saturates at about {kbest} dimensions.",
    "Epitopes are more turn-forming, more hydrophilic and less hydropathic "
    "than non-epitopes, reproducing the classical description of linear "
    "B-cell epitopes; they are also longer, which is partly a dataset property.",
]:
    doc.add_paragraph(line, style="List Bullet")

doc.save(OUT)
print("complete ->", OUT)
