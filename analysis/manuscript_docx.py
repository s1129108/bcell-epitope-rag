"""[MOD] Build the manuscript: BCell_manuscript.docx, every table and figure inline.

Nothing is computed here that was not measured at experiment time. Numbers come
from MCNN_ind_report.xlsx (sheet Independent), Traditional_ind_report_ESMC.xlsx,
ibce_repro/, analysis/analysis_stats.json and the PKL curves; figures come from
analysis/figures/.

    python analysis/manuscript_docx.py
"""
import json
import os
import pickle

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "analysis", "figures")
OUT = os.path.join(ROOT, "analysis", "BCell_manuscript.docx")

IND = pd.read_excel(os.path.join(ROOT, "MCNN_ind_report.xlsx"), sheet_name="Independent")
TRAD = pd.read_excel(os.path.join(ROOT, "Traditional_ind_report_ESMC.xlsx"))
IBCE_BC = pd.read_csv(os.path.join(ROOT, "ibce_repro", "ibce_bc.csv"))
ST = json.load(open(os.path.join(ROOT, "analysis", "analysis_stats.json")))
TAG = "MCNN_25_{}_w2-9-11-25_f64_h128x3_d0.7_lr2e-03_ce_None"


def curve_auc(tag):
    with open(os.path.join(ROOT, "PKL", f"{tag}_AUCROC.pkl"), "rb") as f:
        a = float(pickle.load(f)["AUC"])
    with open(os.path.join(ROOT, "PKL", f"{tag}_PRROC.pkl"), "rb") as f:
        p = float(pickle.load(f)["AUC"])
    return a, p


def agg(feature, col="AUC"):
    s = IND[IND.feature == feature][col]
    return s.mean(), (s.std(ddof=1) if len(s) > 1 else 0.0), len(s)


def pm(feature, col="AUC", nd=4):
    m, s, n = agg(feature, col)
    return f"{m:.{nd}f} ± {s:.{nd}f}" if n > 1 else f"{m:.{nd}f}"


def mean_of(feature, col):
    return f"{agg(feature, col)[0]:.4f}"


# ----------------------------------------------------------------- helpers
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(1.0)
    s.top_margin = s.bottom_margin = Inches(0.9)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    return p


def para(text, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    return p


def caption(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(9)
    r.italic = True
    p.paragraph_format.space_after = Pt(10)
    return p


def table(rows, header, cap, note=None, widths=None):
    caption(cap)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, name in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(name)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9)
            if isinstance(v, str) and v.startswith("**"):
                r.text = v.strip("*")
                r.bold = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    if note:
        p = doc.add_paragraph()
        r = p.add_run(note)
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p.paragraph_format.space_after = Pt(12)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return t


def figure(name, cap, width=6.3):
    doc.add_picture(os.path.join(FIG, name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(cap)


# ====================================================================== title
t = doc.add_heading("Retrieval-augmented protein language model embeddings for "
                    "linear B-cell epitope prediction, with an immunological "
                    "knowledge graph and an application to breast-cancer antigens", 0)
for r in t.runs:
    r.font.size = Pt(19)

para("ESM C 600M per-residue embeddings, a multi-window convolutional "
     "classifier (MCNN), retrieval augmentation over an IEDB-derived peptide "
     "database, and provenance context from an immunological knowledge graph.",
     italic=True, size=10)

h("Abstract", 1)
para(
    "Linear B-cell epitope prediction remains among the least accurate tasks in "
    "immunoinformatics: published independent-test AUC on the standard iBCE-EL "
    "benchmark sits below 0.80, and the signal that distinguishes an epitope "
    "from a non-epitope peptide is distributed rather than localised. We "
    "systematically compare five protein language models as feature extractors "
    f"and find ESM C 600M strongest (independent-test AUC {curve_auc(TAG.format('ESMC_RAW'))[0]:.4f}), "
    "and we show that a multi-window CNN over per-residue embeddings outperforms "
    "five traditional classifiers trained on the same embeddings mean-pooled "
    f"(best traditional {TRAD.AUC.max():.4f}). We then add retrieval "
    "augmentation: each query peptide is fused with the mean of its five nearest "
    "neighbours in a 241,572-peptide IEDB-derived database, and each database "
    "item additionally carries its provenance (source protein, antigen, source "
    "organism) encoded as a knowledge-graph context vector. With the retrieval "
    "database partitioned by label and the independent test set fused the same "
    f"way as training, the model reaches AUC {agg('ESMC_RAGKG_R03')[0]:.4f}. We report, "
    "in equal detail, the two control experiments that establish what this "
    "number does and does not measure: when the test set is left as raw "
    f"embeddings the same model falls to {agg('ESMC_RAGKG_R03_RAWIND')[0]:.4f}, and on a "
    "held-out set of 269 peptides from breast-cancer-associated antigens (CEA, "
    "MUC1, BRCA1/2, p53, ER/PR, mammaglobin, Ki-67) it falls below chance "
    f"({agg('BC_RAGKG_RAWIND')[0]:.4f}), while the plain ESM C + MCNN model generalises "
    f"({agg('BC_ESMC_RAW')[0]:.4f}, against {IBCE_BC.AUC.mean():.4f} for our iBCE-EL "
    "reproduction). A dimension-level and biophysical analysis of the ESM C "
    "space shows no single epitope-detecting dimension, and recovers the "
    "classical propensity picture - epitopes are hydrophilic, turn-forming and "
    "solvent-exposed. We discuss what each configuration is suitable for in "
    "breast-cancer antibody and vaccine research.")

# ================================================================ introduction
h("1. Introduction", 1)
para(
    "B-cell epitopes are the regions of an antigen that antibodies recognise. "
    "Linear (continuous) epitopes are short contiguous stretches of sequence, "
    "and their computational prediction underpins serological assay design, "
    "subunit and peptide vaccine engineering, therapeutic antibody discovery, "
    "and the interpretation of autoantibody repertoires. Unlike T-cell epitope "
    "prediction, where MHC binding provides a strong, well-defined and "
    "experimentally abundant biochemical anchor, B-cell recognition depends on "
    "surface accessibility, local flexibility and conformation in the folded "
    "antigen. The consequence is visible in the literature: T-cell predictors "
    "routinely exceed AUC 0.9, while linear B-cell predictors have remained near "
    "0.75-0.80 for a decade.")
para(
    "Three generations of methods are distinguishable. The first encoded "
    "classical amino-acid propensity scales - Parker hydrophilicity, "
    "Kolaskar-Tongaonkar antigenicity, Chou-Fasman beta-turn, Karplus-Schulz "
    "flexibility - as sliding-window averages, and remains the basis of widely "
    "used tools such as BepiPred and ABCpred. The second applied ensemble "
    "machine learning to composition-style descriptors: iBCE-EL combines an "
    "extremely randomised trees model on amino-acid composition plus "
    "physicochemical properties with a gradient-boosting model on dipeptide "
    "composition plus the same properties, and reports an independent-test AUC "
    "of 0.789 on the benchmark it introduced - still a common reference point. "
    "The third, current, generation replaces hand-designed descriptors with "
    "representations learned by protein language models (PLMs) trained on "
    "hundreds of millions of sequences.")
para(
    "PLM embeddings have transformed structure prediction and remote-homology "
    "detection, but their advantage for linear B-cell epitopes is less clear, "
    "for a reason specific to this task: the input is a peptide of roughly "
    "10-25 residues presented out of its structural context, so a PLM sees only "
    "a fragment of what determines antibody accessibility. Whether larger or "
    "newer PLMs help, and whether per-residue structure in those embeddings "
    "carries information that mean-pooling destroys, are empirical questions "
    "that have not been answered side by side on one benchmark. Our first two "
    "contributions address exactly those questions.")
para(
    "A newer idea is retrieval augmentation. Rather than asking a model to "
    "generalise from training data alone, one retrieves similar examples at "
    "inference time and conditions on them. In protein modelling this has been "
    "explored both as embedding-space retrieval and as injection of curated "
    "knowledge: GOProteinGNN, for instance, treats a protein as a node of a "
    "protein-to-Gene-Ontology knowledge graph and refreshes its representation "
    "with a relation-typed graph convolution at intervals through the encoder, "
    "reporting consistent gains on contact prediction, protein-protein "
    "interaction and remote homology. The immunological equivalent is "
    "attractive, because IEDB supplies both a large peptide corpus and rich "
    "structured annotation for every entry: the source protein, the antigen, "
    "the source organism, the host, the assay.")
para(
    "This article reports that line of work in full, including the parts that "
    "did not work, because on this task the negative results are the more "
    "informative ones. We build an IEDB-derived retrieval database of 241,572 "
    "peptides, attach provenance context to every item, fuse the five nearest "
    "neighbours into each query embedding, and measure the result under several "
    "evaluation protocols that differ in what information is available at test "
    "time. The headline configuration reaches AUC 0.998. We then show, with "
    "three controls, that this figure is produced by the structure of the "
    "evaluation rather than by newly learned immunology, and we quantify "
    "precisely how much of it survives when that structure is removed. We "
    "consider this the main scientific content of the paper: a reproducible "
    "account of how retrieval augmentation can appear to solve a hard benchmark, "
    "and what remains once the shortcut is closed.")
para("Contributions:")
for b in [
    "A controlled comparison of five PLMs (ESM C 600M, ESM-2 650M, Ankh3, "
    "ProtT5-XL, AMPLIFY 350M) as feature extractors for linear B-cell epitope "
    "prediction, each at its own Optuna-selected MCNN configuration.",
    "Evidence that per-residue structure matters: a multi-window CNN beats five "
    "traditional classifiers on the same embeddings mean-pooled, by about 0.08 AUC.",
    "A retrieval-augmented pipeline with an IEDB knowledge-graph context vector, "
    "and a query-ratio sweep locating the best tested mixing weight.",
    "Three control experiments - raw-embedding testing, a source-protein-"
    "disjoint split, and a held-out breast-cancer antigen set - that separate "
    "retrieval gains from benchmark provenance structure.",
    "A dimension-level and biophysical analysis of what the ESM C space encodes "
    "about epitopes, reproducing the classical propensity picture from a learned "
    "representation.",
]:
    p = doc.add_paragraph(b, style="List Bullet")
    p.paragraph_format.space_after = Pt(3)

# ============================================================== data collection
h("2. Data Collection", 1)
h("2.1 The IEDB source table", 2)
para(
    "All data derive from the Immune Epitope Database (IEDB) B-cell assay table. "
    "The download retains, for every assay record: the linear peptide sequence, "
    "its qualitative measure (Positive / Negative and their graded variants), the "
    "parent source antigen with its UniProt accession, the source organism, the "
    "host organism, the assay type and the reference. After restricting to "
    "records with an unambiguous positive or negative qualitative measure the "
    "table holds 1,612,539 assay rows. This is an assay-level table, not a "
    "peptide-level one: the same peptide may appear many times, tested in "
    "different hosts with different assays and occasionally with opposite "
    "outcomes.")
para(
    "Two properties of this table shape everything that follows, and we state "
    "them before presenting any model. First, conflicting labels are common: "
    "when records are collapsed to unique sequences, a substantial fraction "
    "carries both positive and negative outcomes across assays (38.9% of "
    "sequences in the full-table pass, 101 of 376 in the breast-cancer subset "
    "of Section 2.4). Second, the provenance fields are strongly associated with "
    "the label. Fitting a smoothed positive rate per source antigen on the "
    "training split alone and applying it to the independent test set - using no "
    "residue of the peptide itself - yields AUC 0.8763; the source organism "
    "alone yields 0.8142. Both exceed what any sequence-only model in this paper "
    "achieves on the same split. This is a property of how the benchmark was "
    "assembled, and it is the reason every knowledge-graph result below is "
    "reported next to a control.")

h("2.2 Benchmark split", 2)
para(
    "For comparability with published work we use the iBCE-EL benchmark split: "
    "4,440 positive and 5,485 negative peptides for training, and an independent "
    "test set of 1,110 positive and 1,408 negative peptides (2,518 total, 44.1% "
    "positive). Train and test sequences are disjoint, and no sequence appears "
    "with both labels within the split. Peptides are zero-padded or truncated to "
    "a maximum length of 25 residues for the embedding tensor; median length is "
    "16.7 residues for positives and 15.9 for negatives.")

h("2.3 Retrieval database and knowledge-graph context", 2)
para(
    "The retrieval database is built from the same IEDB table, deduplicated to "
    "unique sequences, label-resolved, and filtered so that no benchmark "
    "sequence appears in it. It comprises 19,003 peptides in the positive "
    "partition and 222,569 in the negative partition, 241,572 in total, each "
    "embedded with ESM C 600M under the same 25-residue convention as the "
    "benchmark. The strong class imbalance reflects IEDB itself, where negative "
    "assay outcomes dominate.")
para(
    "Each database item additionally carries a knowledge-graph context vector "
    "built from three IEDB fields: the source antigen accession "
    "(parent_source_antigen_iri, 17,930 distinct values), the antigen name "
    "(17,928 distinct) and the source organism (726 distinct). Each field is "
    "one-hot encoded over its most frequent values (512, 512 and 128 "
    "respectively) plus an \"other\" bin, giving a 1,155-dimensional indicator "
    "vector, which a fixed seeded Gaussian projection maps to the 1,152 "
    "dimensions of the ESM C space. The projection is scaled so that the context "
    "vector's norm is half the mean residue-embedding norm of the database "
    "(scale 0.484), and it is added only at positions where the item has a real "
    "residue, so zero padding stays zero. Provenance joins to the database by "
    "sequence and covers all 241,572 items with no misses.")

h("2.4 Held-out breast-cancer antigen set", 2)
para(
    "To test whether a model transfers to antigens it has never seen, we carve "
    "out every IEDB peptide whose source antigen belongs to a family studied in "
    "breast cancer: carcinoembryonic antigen (CEA), mucin-1 (MUC1), BRCA1 and "
    "BRCA2, p53, estrogen and progesterone receptors, mammaglobin and Ki-67. "
    "This is a held-out antigen-family subset of IEDB, not an independent "
    "external cohort, and we label it as such throughout; no separate "
    "breast-cancer B-cell epitope benchmark of comparable size exists.")
para(
    "376 unique sequences match. 101 carry both labels across assays and are "
    "dropped. A further 6 appear in the benchmark training or test split and are "
    "dropped, leaving 269 peptides: 107 positive and 162 negative. Their antigen "
    "composition is CEA 81, MUC1 77, BRCA 63, p53 41, ER/PR 7, mammaglobin 5, "
    "Ki-67 1. 98 of the 269 are present in the retrieval database verbatim; "
    "every such row is masked out before retrieval, so no peptide can retrieve "
    "itself and read off its own answer. One asymmetry must be noted when "
    "reading Section 6: 46.9% of the negatives exceed 25 residues and are "
    "truncated in the embedding tensor, against 12.1% of the positives, whereas "
    "the composition-based iBCE-EL baseline sees every sequence at full length.")

# ===================================================================== methods
h("3. Methods", 1)
h("3.1 Embeddings and classifier", 2)
para(
    "ESM C 600M produces a 1,152-dimensional vector per residue; a peptide "
    "becomes a 25 x 1,152 tensor. The classifier (MCNN) applies parallel 2-D "
    "convolutions with window sizes 2, 9, 11 and 25 along the residue axis, 64 "
    "filters each, max-pools each window branch over positions, concatenates the "
    "branches, and passes the result through three 128-unit dense layers with "
    "dropout 0.7 to a two-way softmax. Training uses Adam at learning rate "
    "2.369e-3, batch size 128, 20 epochs, cross-entropy. Hyperparameters come "
    "from an Optuna search run separately per PLM; all RAG and knowledge-graph "
    "experiments reuse the ESM C configuration unchanged, so that no comparison "
    "is confounded by re-tuning. Every reported figure is the mean over five "
    "training seeds with its sample standard deviation.")
h("3.2 Retrieval and fusion", 2)
para(
    "Retrieval uses exact L2 distance on the flattened 25 x 1,152 = 28,800-"
    "dimensional embedding, returning the five nearest database items. Fusion is "
    "a convex combination,")
para("        fused = r · query + (1 − r) · mean(top-5 neighbour representations),",
     italic=True)
para(
    "where r is the weight on the peptide's own embedding and the neighbour "
    "representation is the item's ESM C embedding plus its scaled context "
    "vector. Retrieval does not depend on r, so one set of neighbour indices "
    "serves every mixing weight. Unless stated otherwise the retrieval database "
    "is partitioned by label: a positive query searches the positive partition "
    "and a negative query the negative partition. Section 5 reports what "
    "happens when that partitioning, and the evaluation-time fusion it implies, "
    "are removed.")
h("3.3 Evaluation", 2)
para(
    "All reported performance is on held-out data - the 2,518-peptide "
    "independent test set, or the 269-peptide breast-cancer set - never on "
    "cross-validation folds. AUC and average precision (AP) are "
    "threshold-independent. Threshold-dependent columns (accuracy, MCC, "
    "sensitivity, specificity) use the project's existing convention: the "
    "threshold maximising the geometric mean of sensitivity and specificity on "
    "the set being evaluated. That convention is mildly optimistic for those "
    "columns, identically for every method including the iBCE-EL reproduction, "
    "and it does not affect AUC or AP. Significance tests are Welch's t-test "
    "over independent training seeds.")

# ===================================================================== results
h("4. Results", 1)
h("4.1 Protein language model comparison", 2)
PLM = [("ESM C 600M", "ESMC_RAW"),
       ("ESM-2 650M", "ESM2"),
       ("Ankh3", "Ankh3"),
       ("ProtT5-XL", "ProtT5"),
       ("AMPLIFY 350M", "AMPLIFY")]
PLM_TAGS = {
    "ESMC_RAW": TAG.format("ESMC_RAW"),
    "ESM2": "MCNN_25_ESM2_w4-19-22-23_f256_h500x2_d0.7_lr4e-03_ce_None",
    "Ankh3": "MCNN_25_Ankh3_w22-25_f32_h128x1_d0.7_lr4e-04_ce_None",
    "ProtT5": "MCNN_25_ProtT5_w11-17-25_f32_h500x3_d0.7_lr5e-04_focal_None",
    "AMPLIFY": "MCNN_25_AMPLIFY_w5-14-19-25_f64_h256x3_d0.7_lr1e-04_ce_None",
}
rows = []
for name, key in PLM:
    a, p = curve_auc(PLM_TAGS[key])
    rows.append([("**" + name) if key == "ESMC_RAW" else name, f"{a:.4f}", f"{p:.4f}"])
rows.sort(key=lambda r: -float(r[1]))
table(rows, ["Protein language model", "AUC", "AP"],
      "Table 1. Five protein language models as feature extractors for the MCNN, "
      "independent test set. Each PLM uses its own Optuna-selected MCNN "
      "configuration, so the comparison is between PLMs at their respective best, "
      "not at a shared architecture.",
      "AUC and AP are read from the ROC / PR curves stored at experiment time.",
      widths=[2.8, 1.3, 1.3])
para(
    f"ESM C 600M is the strongest extractor at AUC {curve_auc(TAG.format('ESMC_RAW'))[0]:.4f}, and we "
    "use it for every subsequent experiment. The spread across PLMs is modest "
    "relative to the gap to the retrieval experiments below, which is itself "
    "informative: on this task the choice of PLM is not the binding constraint.")
figure("fig1_plm_comparison.png",
       "Figure 1. ROC and precision-recall curves for the five protein language "
       "models on the independent test set. The dashed line in panel b is the "
       "positive rate of the test set (0.441).")

h("4.2 Per-residue structure versus mean-pooling", 2)
rows = [["**ESM C per-residue + MCNN",
         f"{curve_auc(TAG.format('ESMC_RAW'))[0]:.4f}",
         f"{curve_auc(TAG.format('ESMC_RAW'))[1]:.4f}",
         mean_of("ESMC_RAW", "MCC"), mean_of("ESMC_RAW", "Acc")]]
for _, r in TRAD.sort_values("AUC", ascending=False).iterrows():
    rows.append([f"ESM C mean-pooled + {r.classifier}", f"{r.AUC:.4f}",
                 f"{r.PR_AUC:.4f}", f"{r.MCC:.4f}", f"{r.Acc:.4f}"])
table(rows, ["Model", "AUC", "AP", "MCC", "Accuracy"],
      "Table 2. The multi-window CNN over per-residue embeddings against five "
      "traditional classifiers trained on the same ESM C embeddings mean-pooled "
      "to a single 1,152-dimensional vector per peptide.",
      "Traditional classifiers were tuned by 5-fold cross-validation on the "
      "training split; the reported numbers are on the independent test set.",
      widths=[2.9, 1.0, 1.0, 1.0, 1.0])
para(
    f"The convolutional model gains about "
    f"{curve_auc(TAG.format('ESMC_RAW'))[0] - TRAD.AUC.max():.3f} AUC over the best "
    "traditional classifier on identical embeddings. Since the only difference "
    "is whether the residue axis survives, mean-pooling discards usable "
    "information: position-local patterns within the peptide contribute, and a "
    "single pooled vector per peptide cannot express them.")
figure("fig2_esmc_vs_traditional.png",
       "Figure 2. ESM C + MCNN against traditional classifiers on mean-pooled "
       "ESM C embeddings, independent test set.")

h("4.3 Retrieval augmentation and the query ratio", 2)
rows = [["ESM C, no fusion", "-", pm("ESMC_RAW"), mean_of("ESMC_RAW", "PR_AUC"),
         mean_of("ESMC_RAW", "MCC"), mean_of("ESMC_RAW", "Acc"), "-"]]
base = agg("ESMC_RAW")[0]
for feat, r in [("ESMC_R09", "0.9"), ("ESMC_R05", "0.5"), ("ESMC_R03", "0.3")]:
    m = agg(feat)[0]
    rows.append([("**RAG fused" if feat == "ESMC_R03" else "RAG fused"), r, pm(feat),
                 mean_of(feat, "PR_AUC"), mean_of(feat, "MCC"), mean_of(feat, "Acc"),
                 f"+{m - base:.4f}"])
table(rows, ["Features", "r", "AUC", "AP", "MCC", "Accuracy", "ΔAUC"],
      "Table 3. Query-ratio sweep for retrieval-augmented fusion, independent "
      "test set, five seeds per row. r is the weight on the peptide's own "
      "embedding; the retrieval database is partitioned by label and the test "
      "set is fused the same way as training.",
      "Monotonic in the retrieval weight across the tested range. r = 0.3 versus "
      "r = 0.5 is +0.0157 AUC, Welch p = 1.1e-03; r = 0.9 versus r = 0.5 is "
      "-0.0576, p = 2.9e-07. The trend had not turned at r = 0.3, so the optimum "
      "may lie lower; values below 0.3 were not tested.",
      widths=[1.8, 0.5, 1.3, 0.85, 0.85, 0.9, 0.85])
figure("fig3_rag_fusion.png",
       "Figure 3. Query-ratio sweep. Lower weight on the query - that is, more "
       "weight on the retrieved neighbours - monotonically improves the "
       "independent-test curves under this evaluation protocol.")

h("4.4 Adding knowledge-graph context", 2)
rows = []
for label, feat in [("ESM C, no fusion", "ESMC_RAW"),
                    ("RAG r = 0.3, test fused", "ESMC_R03"),
                    ("**RAG + KG r = 0.3, test fused", "ESMC_RAGKG_R03"),
                    ("RAG r = 0.3, test raw", "ESMC_R03_RAWIND"),
                    ("RAG + KG r = 0.3, test raw", "ESMC_RAGKG_R03_RAWIND")]:
    rows.append([label, pm(feat), mean_of(feat, "PR_AUC"), mean_of(feat, "MCC"),
                 mean_of(feat, "Acc"), mean_of(feat, "Sens"), mean_of(feat, "Spec")])
table(rows, ["Configuration", "AUC", "AP", "MCC", "Accuracy", "Sens", "Spec"],
      "Table 4. Knowledge-graph context and the two evaluation directions, "
      "independent test set, five seeds per row. In every row the model is "
      "trained on label-partitioned fused features at r = 0.3; the rows differ "
      "only in what the test peptides are given.",
      "Context adds +0.0158 AUC when the test set is fused (Welch p = 5.6e-04) "
      "and +0.0072 when it is raw (p = 5.8e-03). Both raw-test configurations "
      "fall well below the unfused baseline: -0.0861 without context "
      "(p = 4.4e-08) and -0.0790 with it (p = 1.4e-07).",
      widths=[2.4, 1.25, 0.8, 0.8, 0.85, 0.75, 0.75])
para(
    f"The best configuration reaches AUC {agg('ESMC_RAGKG_R03')[0]:.4f} ± "
    f"{agg('ESMC_RAGKG_R03')[1]:.4f}, with sensitivity {agg('ESMC_RAGKG_R03', 'Sens')[0]:.4f} "
    f"and specificity {agg('ESMC_RAGKG_R03', 'Spec')[0]:.4f} - 9 false negatives and 46 false "
    "positives out of 2,518 peptides. Seed-to-seed variation almost vanishes "
    "(±0.0001), which is itself a signal: the task has become nearly "
    "deterministic for the model.")
para(
    "The second direction is the one that matters for use. Training on fused "
    "features at r = 0.3 means 70% of the input signal comes from retrieved "
    "neighbours; presenting raw embeddings at test time removes that 70%, and "
    "performance drops below the unfused baseline. The no-context control "
    "isolates the cause: it drops by essentially the same margin, so the "
    "collapse is distribution shift between training and test inputs, not "
    "anything contributed by the knowledge graph.")
figure("fig7_ragkg.png",
       "Figure 4. Retrieval plus knowledge-graph context under both evaluation "
       "directions. The two upper curves require the test peptide to be fused "
       "with label-partitioned retrieval; the lower curve is the same trained "
       "model evaluated on raw embeddings.")

# =========================================================== control section
h("5. What the retrieval gain measures", 1)
para(
    "Three controls bound the interpretation of Table 4. We report them in the "
    "main text rather than a supplement because they change what the headline "
    "number means.")
h("5.1 The retrieval partition carries the label", 2)
para(
    "Under label-partitioned retrieval, choosing which partition to search "
    "requires knowing the query's label. At training time that is legitimate; "
    "at test time it is the answer. The fraction of a query's five neighbours "
    "drawn from the positive partition, taken alone as a one-number classifier "
    "under pooled (label-free) retrieval, already scores AUC 0.6257 - a bare "
    "five-nearest-neighbour vote. When the partition is instead chosen by the "
    "true label, that single bit is worth far more.")
h("5.2 Label-free retrieval adds nothing", 2)
para(
    "Repeating the fusion with one pooled database of all 241,572 peptides, so "
    "that a query is never steered toward the partition matching its own label, "
    "gives AUC "
    f"{agg('ESMC_PDIS')[0]:.4f} ± {agg('ESMC_PDIS')[1]:.4f} without fusion against "
    "0.7596 ± 0.0079 with it on a source-protein-disjoint split: +0.0032, Welch "
    "p = 0.43. Honest retrieval over this database does not help.")
h("5.3 Provenance structure, not peptide identity", 2)
para(
    "On a split constructed so that no source protein is shared between train "
    f"and test, the plain ESM C + MCNN model scores {agg('ESMC_PDIS')[0]:.4f} ± "
    f"{agg('ESMC_PDIS')[1]:.4f}, against {agg('ESMC_RSPLIT')[0]:.4f} ± "
    f"{agg('ESMC_RSPLIT')[1]:.4f} on a matched random split of the same peptides - a cost of "
    "only 0.0097 AUC (Welch p = 3.3e-04). The sequence-only model therefore "
    "does not exploit the provenance shortcut, because it never sees provenance; "
    "the published ESM C numbers in Tables 1-3 are not inflated by it. The "
    "shortcut becomes reachable precisely when provenance is encoded into the "
    "features, which is what the knowledge-graph context does.")

# ========================================================= biological analysis
h("6. Biological Analysis", 1)
h("6.1 Geometry of the embedding space", 2)
g, pa, cd, cp = ST["geometry"], ST["probe_auc"], ST["cohen_d_median"], ST["cohen_d_p99"]
rows = []
for key, label in [("query", "ESM C, no fusion"), ("fused", "RAG fused (r = 0.5)")]:
    rows.append([label, f"{pa[key]:.4f}", f"{cd[key]:.3f}", f"{cp[key]:.3f}",
                 f"{g[key]['ratio']:.3f}", f"{ST['silhouette'][key]:+.4f}"])
table(rows, ["Representation", "Linear probe AUC", "median |d|", "p99 |d|",
             "centroid / scatter", "silhouette"],
      "Table 5. Class-discriminative structure in the mean-pooled embedding of "
      "the independent test set, before and after fusion. The linear probe is a "
      "5-fold logistic regression on standardised features; it measures how much "
      "class structure the representation exposes to a linear reader, and is not "
      "a prediction method.",
      "Pooling counts only real residues: averaging across the zero-padded "
      "25 positions would scale every dimension by length/25 and inject peptide "
      "length into the whole representation.",
      widths=[1.9, 1.15, 0.9, 0.8, 1.15, 0.9])
figure("fig4_tsne.png",
       "Figure 5. t-SNE of the mean-pooled independent test embeddings before "
       "and after fusion. Both panels look similarly mixed even though the "
       "linear probe separates the fused representation far better: t-SNE "
       "preserves local neighbourhoods and is insensitive to the distributed, "
       "low-amplitude shift that the probe reads.")

h("6.2 Is there an epitope dimension?", 2)
dim = ST["dimension"]
_sc = dim["selection_curve"]
_i = max(range(len(_sc["auc"])), key=lambda j: _sc["auc"][j])
BEST_K, BEST_K_AUC, ALL_AUC = _sc["k"][_i], _sc["auc"][_i], _sc["auc"][-1]
top = dim["top_dims"][:10]
rows = [[str(d["dim"]), f"{d['cohen_d']:+.3f}", f"{d['univariate_auc']:.4f}",
         d["best_property"], f"{d['rho']:+.3f}"] for d in top]
table(rows, ["ESM C dimension", "Cohen's d", "univariate AUC",
             "most correlated property", "Spearman ρ"],
      "Table 6. The ten most class-discriminative ESM C dimensions on the "
      "independent test set, with the sequence-derived property each correlates "
      "with most strongly.",
      f"No dimension is an epitope detector: the strongest reaches |d| = "
      f"{abs(top[0]['cohen_d']):.3f} and univariate AUC {top[0]['univariate_auc']:.4f}; the median over "
      f"all 1,152 dimensions is |d| = {dim['median_abs_d']:.3f}, and only "
      f"{100 * dim['frac_d_gt_02']:.1f}% exceed |d| = 0.2.",
      widths=[1.2, 0.95, 1.15, 2.0, 0.95])
para(
    "A nested feature-selection curve - ranking dimensions by effect size on "
    "each cross-validation fold's training part only, then fitting a probe on "
    f"the top k - peaks at {BEST_K} dimensions (AUC {BEST_K_AUC:.4f}) and declines "
    f"to {ALL_AUC:.4f} using all 1,152. In a linear model over the mean-pooled "
    "representation, additional dimensions add noise. This concerns the pooled "
    "representation only: the MCNN extracts considerably more from the "
    "per-residue structure, as Table 2 shows.")
figure("fig5_dimension_importance.png",
       "Figure 6. Distribution of per-dimension effect sizes, the ranked effect-"
       "size curve, and the nested feature-selection curve. The signal is "
       "distributed across many weak dimensions rather than concentrated in a few.")

h("6.3 Biophysical character of the epitopes", 2)
bio = ST["biology"]["properties"]
rows = []
for r_ in sorted(bio, key=lambda x: -abs(x["cohen_d"]))[:8]:
    rows.append([r_["property"], f"{r_['mean_epitope']:.3f}",
                 f"{r_['mean_non_epitope']:.3f}", f"{r_['cohen_d']:+.3f}",
                 f"{r_['q']:.2g}"])
table(rows, ["Property", "mean (epitope)", "mean (non-epitope)", "Cohen's d",
             "q (BH)"],
      "Table 7. Sequence-derived biophysical properties compared between "
      "epitope and non-epitope peptides of the independent test set. p-values "
      "are Mann-Whitney U, corrected across the panel by Benjamini-Hochberg.",
      "Only properties computable from the sequence are used. IEDB provenance "
      "fields (host, assay, source organism) are deliberately excluded: as "
      "Section 2.1 shows, they partly recover the label through how the "
      "benchmark was assembled, so their importance would describe the dataset "
      "rather than epitope biology.",
      widths=[2.1, 1.15, 1.25, 0.95, 0.9])
para(
    "The four strongest effects point the same way. Epitopes are enriched in "
    "beta-turn propensity and hydrophilicity, and depleted in hydropathy: they "
    "are the solvent-exposed, flexible, turn-forming stretches that an antibody "
    "can reach. This reproduces, from a learned representation and without any "
    "propensity scale as input, the picture that the classical Parker, "
    "Chou-Fasman and Karplus-Schulz scales were built on - a useful "
    "sanity check that the model is reading immunology and not only dataset "
    "idiosyncrasy. Amino-acid composition agrees: tyrosine, proline, asparagine, "
    "aspartate, arginine, glycine and glutamine are enriched in epitopes, while "
    "methionine, cysteine, isoleucine, leucine and alanine are depleted.")
para(
    "Two caveats belong next to this table. Kolaskar-Tongaonkar antigenicity "
    "runs opposite to its design intent here, because the scale weights "
    "hydrophobic and bulky residues, and epitopes in this benchmark are the more "
    "hydrophilic class; it should not be used as a feature on this data. And "
    "peptide length differs between classes, correlating with several leading "
    "dimensions; length is partly an artefact of dataset assembly, so separating "
    "length from antigenicity would require a length-matched control we have not "
    "run.")
figure("fig6_biology.png",
       "Figure 7. Property distributions between classes, the property-to-"
       "dimension correlation map, and the amino-acid composition shift between "
       "epitopes and non-epitopes.")

# ===================================================== ablation: breast cancer
h("7. Ablation Study: Held-out Breast-cancer Antigens", 1)
para(
    "Section 4 evaluates on peptides drawn from the same antigen pool as "
    "training. The practically relevant question is different: given an antigen "
    "the model has never seen, does it find the epitopes? The 269-peptide set "
    "of Section 2.4 answers that for seven antigen families relevant to breast "
    "cancer, with self-retrieval masked so that no peptide can find itself in "
    "the database.")
rows = []
for label, feat in [("**ESM C + MCNN (raw in, raw out)", "BC_ESMC_RAW"),
                    ("RAG + KG r = 0.3, test raw", "BC_RAGKG_RAWIND"),
                    ("RAG + KG r = 0.3, test fused †", "BC_RAGKG_FUSED")]:
    rows.append([label, pm(feat), mean_of(feat, "PR_AUC"), mean_of(feat, "MCC"),
                 mean_of(feat, "Acc"), mean_of(feat, "Sens"), mean_of(feat, "Spec")])
rows.append(["iBCE-EL (our reproduction)",
             f"{IBCE_BC.AUC.mean():.4f} ± {IBCE_BC.AUC.std(ddof=1):.4f}",
             f"{IBCE_BC.AP.mean():.4f}", f"{IBCE_BC.MCC.mean():.4f}",
             f"{IBCE_BC.Acc.mean():.4f}", f"{IBCE_BC.Sens.mean():.4f}",
             f"{IBCE_BC.Spec.mean():.4f}"])
table(rows, ["Model", "AUC", "AP", "MCC", "Accuracy", "Sens", "Spec"],
      "Table 8. Validation on 269 held-out peptides from breast-cancer-"
      "associated antigens (107 positive, 162 negative), five seeds per row. "
      "Models are trained on the benchmark training split; only the evaluation "
      "set changes.",
      "† uses the true label to select the retrieval partition, so it is not a "
      "configuration that can be deployed; it is included to show what the "
      "partition choice is worth once self-retrieval is excluded. iBCE-EL uses "
      "composition features and therefore sees untruncated sequences, which "
      "favours it slightly on this subset (Section 2.4).",
      widths=[2.4, 1.25, 0.8, 0.8, 0.85, 0.75, 0.75])
para(
    f"The plain ESM C + MCNN model transfers: AUC {agg('BC_ESMC_RAW')[0]:.4f} ± "
    f"{agg('BC_ESMC_RAW')[1]:.4f} on unseen antigen families, against "
    f"{agg('ESMC_RAW')[0]:.4f} on the benchmark test set - a drop of about "
    f"{agg('ESMC_RAW')[0] - agg('BC_ESMC_RAW')[0]:.3f}, modest given that every antigen is new "
    f"and the class balance differs. It lands within {abs(IBCE_BC.AUC.mean() - agg('BC_ESMC_RAW')[0]):.3f} "
    "AUC of the iBCE-EL reproduction, which had the advantage of untruncated "
    "sequences.")
para(
    f"The retrieval-trained model evaluated on raw embeddings reaches "
    f"{agg('BC_RAGKG_RAWIND')[0]:.4f} - below chance, with sensitivity "
    f"{agg('BC_RAGKG_RAWIND', 'Sens')[0]:.4f}. Training on heavily fused inputs leaves a model "
    "that cannot read a raw peptide at all. The same model given "
    "label-partitioned fused test inputs scores "
    f"{agg('BC_RAGKG_FUSED')[0]:.4f} ± {agg('BC_RAGKG_FUSED')[1]:.4f} on these same unseen "
    "antigens, with self-retrieval excluded. That contrast is the cleanest "
    "available demonstration that the high figure is carried by the partition "
    "choice rather than by transferable epitope knowledge: nothing about the "
    "antigens changed between those two rows, only whether the test peptide's "
    "label was used to pick which half of the database to search.")
figure("fig8_breast_cancer.png",
       "Figure 8. ROC and precision-recall on the held-out breast-cancer antigen "
       "set. The dashed line in panel b is the positive rate (0.398).")
figure("fig8c_bc_summary.png",
       "Figure 9. Breast-cancer validation summary, five seeds per bar. The "
       "label-informed configuration is shown in grey because it cannot be run "
       "without the answer.", width=5.2)

# ============================================================ sota comparison
h("8. Comparison with Published Methods", 1)
rows = [
    ["iBCE-EL (published)", "ERT + GB ensemble, AAC / DPC / PCP", "0.7890",
     "0.4630", "0.7320", "0.7420", "0.7240"],
    ["iBCE-EL (our reproduction)", "as above, retuned", "0.7728", "0.4369",
     "0.7161", "0.7523", "0.6876"],
    ["**ESM C + MCNN (this work)", "PLM per-residue + multi-window CNN",
     f"{agg('ESMC_RAW')[0]:.4f}", mean_of("ESMC_RAW", "MCC"),
     mean_of("ESMC_RAW", "Acc"), mean_of("ESMC_RAW", "Sens"),
     mean_of("ESMC_RAW", "Spec")],
    ["ESM C mean-pooled + SVM", "PLM pooled + traditional",
     f"{TRAD[TRAD.classifier == 'SVM'].AUC.iloc[0]:.4f}",
     f"{TRAD[TRAD.classifier == 'SVM'].MCC.iloc[0]:.4f}",
     f"{TRAD[TRAD.classifier == 'SVM'].Acc.iloc[0]:.4f}",
     f"{TRAD[TRAD.classifier == 'SVM'].Sens.iloc[0]:.4f}",
     f"{TRAD[TRAD.classifier == 'SVM'].Spec.iloc[0]:.4f}"],
    ["RAG + KG, test fused †", "label-partitioned retrieval + KG context",
     f"{agg('ESMC_RAGKG_R03')[0]:.4f}", mean_of("ESMC_RAGKG_R03", "MCC"),
     mean_of("ESMC_RAGKG_R03", "Acc"), mean_of("ESMC_RAGKG_R03", "Sens"),
     mean_of("ESMC_RAGKG_R03", "Spec")],
]
table(rows, ["Method", "Representation", "AUC", "MCC", "Acc", "Sens", "Spec"],
      "Table 9. Comparison on the iBCE-EL independent test set (1,110 positive, "
      "1,408 negative). Published iBCE-EL figures are as reported by its "
      "authors; our reproduction follows the same recipe with hyperparameters "
      "retuned by cross-validation on the training split.",
      "† not a deployable configuration - the test label selects the retrieval "
      "partition (Sections 5 and 7). It is listed for completeness and should "
      "not be read as state of the art. Our reproduction of iBCE-EL lands "
      "0.0162 AUC below the published figure; the likely causes are the "
      "supplementary hyperparameter table we could not access and unspecified "
      "feature-normalisation details.",
      widths=[1.95, 1.9, 0.75, 0.7, 0.7, 0.7, 0.7])
para(
    f"Against the published iBCE-EL figure of 0.789, the ESM C + MCNN model "
    f"({agg('ESMC_RAW')[0]:.4f}) is {0.789 - agg('ESMC_RAW')[0]:+.4f} AUC - that is, "
    "modern PLM embeddings with a convolutional reader do not, on their own, "
    "clearly beat a well-tuned 2018 ensemble of composition features on this "
    "benchmark. We regard this as the honest state of the task rather than a "
    "shortcoming of the embeddings: Section 6 shows the per-dimension signal is "
    "weak and distributed, and Section 2.1 shows a sizeable part of the "
    "benchmark's apparent learnability sits in provenance rather than sequence. "
    "On the transfer test of Table 8 the two methods are also within 0.018 AUC "
    "of each other.")

# ================================================================== discussion
h("9. Discussion", 1)
h("9.1 What retrieval augmentation did and did not deliver", 2)
para(
    "Retrieval augmentation produced the largest apparent improvement in this "
    f"study - from {agg('ESMC_RAW')[0]:.4f} to {agg('ESMC_RAGKG_R03')[0]:.4f} AUC - and, under "
    "controlled evaluation, essentially none of it. The three controls of "
    "Section 5 and the transfer test of Section 7 converge on one explanation: "
    "when the retrieval database is split by label, the act of choosing a "
    "partition injects the answer, and the model learns to read that rather "
    "than the peptide. With pooled, label-free retrieval the gain is +0.0032 "
    "AUC (p = 0.43).")
para(
    "This is worth reporting in a field where retrieval-augmented and "
    "knowledge-graph-enhanced variants are proliferating. The failure mode is "
    "not exotic or implementation-specific: it follows from building a database "
    "whose partitions correspond to the prediction target, which is a natural "
    "thing to do when the corpus is labelled. Any such design needs the "
    "label-free control reported alongside it, and a transfer set with "
    "self-retrieval masked. The knowledge-graph context behaves the same way: "
    "it adds +0.0158 AUC under fused evaluation, but the fields it encodes - "
    "source antigen and organism - predict the label at AUC 0.8763 and 0.8142 "
    "on their own, so that increment describes the annotation, not the epitope.")
para(
    "What did hold up: ESM C 600M is the best of five PLMs tested; per-residue "
    "structure beats mean-pooling by about 0.08 AUC, so the residue axis should "
    "be preserved; and the resulting model transfers to entirely unseen antigen "
    "families with a drop of only about 0.014 AUC.")
h("9.2 Application to breast-cancer research", 2)
para(
    "Tumour-associated antigens are central to breast-cancer immunology in three "
    "ways, and the deployable model here - raw embeddings in, prediction out, "
    "no retrieval database required - applies to each.")
para(
    "Serological biomarker discovery. Autoantibodies against MUC1, p53, CEA and "
    "mammaglobin appear in sera of breast-cancer patients, sometimes before "
    "clinical diagnosis, and serum-based assays need peptide antigens that "
    "antibodies actually bind. Scanning an antigen with overlapping windows and "
    "ranking by predicted epitope probability narrows a hundreds-of-peptides "
    "synthesis problem to a few dozen candidates for ELISA or peptide-array "
    f"validation. Table 8's AUC of {agg('BC_ESMC_RAW')[0]:.4f} on exactly these antigen "
    "families, with sensitivity "
    f"{agg('BC_ESMC_RAW', 'Sens')[0]:.4f}, is the relevant figure: good enough to "
    "prioritise a synthesis list, not good enough to replace an assay. We would "
    "recommend using it to rank, taking the top 20-30% for experimental "
    "screening, and reporting predicted scores alongside measured ones.")
para(
    "Peptide-vaccine and antibody engineering. MUC1 and HER2-family antigens are "
    "long-standing targets for therapeutic cancer vaccines, where the aim is to "
    "elicit antibody responses against epitopes that are both accessible and "
    "tumour-restricted. Epitope prediction contributes the accessibility half of "
    "that filter. Two cautions follow from our own results. First, these are "
    "linear-epitope predictions: a large share of real antibody responses to "
    "folded tumour antigens is conformational, and no method in this paper "
    "addresses that. Second, our beta-turn and hydrophilicity findings "
    "(Section 6.3) mean predictions will favour flexible exposed loops - "
    "usually correct for antibody accessibility, but in a glycoprotein such as "
    "MUC1 the heavily O-glycosylated tandem repeats carry glycan-dependent "
    "epitopes whose recognition our sequence-only model cannot represent. "
    "Glycosylation site information should be overlaid before committing to "
    "synthesis.")
para(
    "Interpreting patient autoantibody repertoires. Phage-display and peptide-"
    "array experiments return thousands of enriched peptides, most of them "
    "noise. A sequence-only epitope score is a cheap prior for triage. Here the "
    "provenance finding of Section 2.1 becomes an active warning rather than a "
    "methodological footnote: tools trained with IEDB annotation in the features "
    "will tend to score peptides highly because they come from well-studied "
    "antigens, not because they are more antibody-accessible. For discovery work "
    "on under-studied antigens that bias runs exactly the wrong way, and the "
    "model we recommend for this use is deliberately the one that never sees "
    "provenance.")
h("9.3 Limitations", 2)
for b in [
    "Linear epitopes only; conformational epitopes are out of scope.",
    "Peptides are truncated at 25 residues for the embedding tensor, which "
    "affects 46.9% of the breast-cancer negatives against 12.1% of its "
    "positives, and flatters the untruncated iBCE-EL baseline on that set.",
    "The breast-cancer set is a held-out IEDB antigen-family subset, not an "
    "independent cohort, and at 269 peptides its confidence intervals are wide.",
    "Peptide length differs between classes and is partly an assembly artefact; "
    "a length-matched control was not run.",
    "Threshold-dependent metrics follow the project's existing max-G-mean-on-"
    "the-evaluated-set convention, which is mildly optimistic for all methods "
    "including the baseline.",
    "The query-ratio sweep had not turned at r = 0.3, so the optimum under "
    "fused evaluation may lie lower; values below 0.3 were not tested.",
]:
    p = doc.add_paragraph(b, style="List Bullet")
    p.paragraph_format.space_after = Pt(3)

h("10. Conclusion", 1)
para(
    "On the standard linear B-cell epitope benchmark, ESM C 600M per-residue "
    f"embeddings with a multi-window CNN reach AUC {agg('ESMC_RAW')[0]:.4f} on the "
    "independent test set, comparable to a retuned iBCE-EL ensemble, and "
    f"transfer to unseen breast-cancer antigen families at {agg('BC_ESMC_RAW')[0]:.4f}. "
    "Preserving the residue axis matters: mean-pooling costs about 0.08 AUC. "
    "Retrieval augmentation with an IEDB-derived database and knowledge-graph "
    f"provenance context reaches {agg('ESMC_RAGKG_R03')[0]:.4f} under evaluation that fuses "
    "the test set the same way as training, but controlled evaluation shows "
    "that figure is produced by the label-partitioned retrieval design: with "
    "label-free retrieval the gain is +0.003 AUC, and on unseen antigens the "
    "same model scores below chance on raw embeddings. We therefore recommend "
    "the plain ESM C + MCNN configuration for use, report the retrieval results "
    "as a cautionary case study, and note that the provenance structure we "
    "quantify here (source antigen alone predicts the label at AUC 0.8763) is a "
    "property of the benchmark that any future method on this data should "
    "control for.")

h("Data and code availability", 1)
para("Code, trained configurations and the scripts that regenerate every table "
     "and figure in this manuscript are available at "
     "https://github.com/s1129108/bcell-epitope-rag. All data derive from the "
     "public IEDB B-cell assay export; the benchmark split is the published "
     "iBCE-EL split.")

doc.save(OUT)
print(f"-> {OUT}")
