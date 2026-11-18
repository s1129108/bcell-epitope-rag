"""[MOD] Score iBCE-EL (EM1) on the held-out breast-cancer antigen set.

Trained exactly as run_em1.py does - ERT on AAC+PCP, GB on DPC+PCP, with the
hyperparameters in best_params.json - on the benchmark training split, then
applied to dataset/BC_{pos,neg}.fasta. 5 seeds, mean +- sd.

iBCE-EL uses composition features, so sequences are NOT truncated at 25
residues here. The MCNN numbers on the same set are computed on embeddings
truncated to 25, which affects 13/107 positives and 76/162 negatives. The
comparison therefore flatters iBCE-EL slightly on this subset, and the
manuscript says so rather than quietly matching the truncation.

    python run_bc.py          (from ibce_repro/, any env with sklearn)
"""
import json
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, accuracy_score, confusion_matrix)

import data
import features

BEST = json.load(open("best_params.json"))
MODELS = {"ERT": ExtraTreesClassifier, "GB": GradientBoostingClassifier}


def read(path):
    seqs, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if cur is not None:
                seqs.append(cur)
            cur = ""
        elif cur is not None:
            cur += line
    if cur:
        seqs.append(cur)
    return seqs


S_tr, y_tr = data.load("train")
S_bc = read("../dataset/BC_pos.fasta") + read("../dataset/BC_neg.fasta")
y_bc = np.r_[np.ones(len(read("../dataset/BC_pos.fasta"))),
             np.zeros(len(read("../dataset/BC_neg.fasta")))]
print(f"train {len(y_tr)} | BC {len(y_bc)} ({int(y_bc.sum())} pos)")

X_tr = {k: features.encode(S_tr, k) for k in ("H7", "H15")}
X_bc = {k: features.encode(S_bc, k) for k in ("H7", "H15")}

rows = []
for seed in range(5):
    p = []
    for name in ("ERT", "GB"):
        kw = dict(BEST[name]["params"])
        if MODELS[name] is ExtraTreesClassifier:
            kw["n_jobs"] = -1
        m = MODELS[name](random_state=seed, **kw).fit(X_tr[BEST[name]["feat"]], y_tr)
        p.append(m.predict_proba(X_bc[BEST[name]["feat"]])[:, 1])
    em1 = np.mean(p, axis=0)
    # same G-mean-on-the-evaluated-set rule as code/MCNN.py, so the two are comparable
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_bc, em1)
    t = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]
    yp = (em1 >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bc, yp, labels=[0, 1]).ravel()
    rows.append(dict(seed=seed, AUC=roc_auc_score(y_bc, em1),
                     AP=average_precision_score(y_bc, em1),
                     MCC=matthews_corrcoef(y_bc, yp), Acc=accuracy_score(y_bc, yp),
                     Sens=tp / (tp + fn), Spec=tn / (tn + fp)))
    print(rows[-1])

import pandas as pd
d = pd.DataFrame(rows)
print("\niBCE-EL (EM1) on the breast-cancer set, 5 seeds:")
for c in ["AUC", "AP", "MCC", "Acc", "Sens", "Spec"]:
    print(f"  {c:5s} {d[c].mean():.4f} +- {d[c].std(ddof=1):.4f}")
d.to_csv("ibce_bc.csv", index=False)
