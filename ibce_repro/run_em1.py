"""iBCE-EL = EM1 = average(P_ERT tren AAC+PCP, P_GB tren DPC+PCP), nguong 0.35.

Protocol dung nhu bai bao:
  - 5-fold CV tren benchmarking, LAP LAI 10 LAN voi partition random khac nhau,
    lay average cua performance.
  - nguong chon bang grid search toi da hoa MCC.
  - independent test.

[MOD] Mot diem KHAC bai bao, co y: nguong duoc chon tren OOF prediction CUA TAP TRAIN,
roi AP DUNG CO DINH cho independent set. Bai bao khong noi ro ho chon nguong tren tap
nao; chon tren train roi dong bang la cach duy nhat khong ro ri. De doi chieu, ta bao
cao song song ca nguong 0.35 ma bai bao cong bo.
Luu y day la protocol SACH HON convention cua project cu (chon nguong G-Mean tren chinh
tap dang danh gia), nen so Acc/MCC o day KHONG tang boc nhu cac bang truoc.
"""
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

import data, features

BEST = json.load(open("best_params.json"))
PAPER_THR = 0.35
N_REP = 10

S_tr, y_tr = data.load("train")
S_in, y_in = data.load("ind")
X_tr = {k: features.encode(S_tr, k) for k in ("H7", "H15")}
X_in = {k: features.encode(S_in, k) for k in ("H7", "H15")}
print(f"train {len(y_tr)} pos {int(y_tr.sum())} | ind {len(y_in)} pos {int(y_in.sum())}",
      flush=True)

MODELS = {"ERT": ExtraTreesClassifier, "GB": GradientBoostingClassifier}


def fit(name, Xa, ya, seed):
    cls = MODELS[name]
    kw = dict(BEST[name]["params"])
    if cls is ExtraTreesClassifier:
        kw["n_jobs"] = 1
    return cls(random_state=seed, **kw).fit(Xa, ya)


def probs(Xa_tr, ya, Xa_in, seed):
    """Tra ve (P_ERT, P_GB) tren Xa_in."""
    out = {}
    for name in ("ERT", "GB"):
        f = BEST[name]["feat"]
        m = fit(name, Xa_tr[f], ya, seed)
        out[name] = m.predict_proba(Xa_in[f])[:, 1]
    return out


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    return dict(AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                Sens=tp / max(tp + fn, 1), Spec=tn / max(tn + fp, 1),
                Acc=(tp + tn) / len(y), MCC=matthews_corrcoef(y, yh),
                F1=2 * tp / max(2 * tp + fp + fn, 1))


def best_thr(y, p):
    grid = np.arange(0.05, 0.96, 0.01)
    m = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(m))]), float(np.max(m))


# ---------- 1. CV tren train: 10 lan x 5 fold ----------
def cv_rep(rep):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
    oof = {"ERT": np.zeros(len(y_tr)), "GB": np.zeros(len(y_tr))}
    for tr, te in skf.split(np.zeros(len(y_tr)), y_tr):
        Xa_tr = {k: X_tr[k][tr] for k in X_tr}
        Xa_te = {k: X_tr[k][te] for k in X_tr}
        pr = probs(Xa_tr, y_tr[tr], Xa_te, rep)
        for k in oof:
            oof[k][te] = pr[k]
    return oof


print("=== CV tren benchmarking, 10 lan x 5 fold ===", flush=True)
oofs = Parallel(n_jobs=10, verbose=1)(delayed(cv_rep)(r) for r in range(N_REP))

rows, thrs = [], []
for rep, oof in enumerate(oofs):
    em = (oof["ERT"] + oof["GB"]) / 2.0
    t, _ = best_thr(y_tr, em)
    thrs.append(t)
    for tag, p, thr in (("ERT", oof["ERT"], best_thr(y_tr, oof["ERT"])[0]),
                        ("GB",  oof["GB"],  best_thr(y_tr, oof["GB"])[0]),
                        ("iBCE-EL (EM1)", em, t)):
        r = metrics(y_tr, p, thr); r.update(rep=rep, model=tag, thr=thr)
        rows.append(r)

cv = pd.DataFrame(rows)
COLS = ["MCC", "Acc", "Sens", "Spec", "AUC", "PR_AUC"]
print("\n===== CV tren benchmarking (average cua 10 lan) =====")
print(cv.groupby("model", sort=False)[COLS + ["thr"]].mean().round(4).to_string())
print("\nstd giua 10 lan:")
print(cv.groupby("model", sort=False)[COLS].std().round(4).to_string())

THR_CV = float(np.median(thrs))
print(f"\nnguong EM1 chon tren OOF cua train: median {THR_CV:.3f} "
      f"(10 lan: {[round(t,2) for t in thrs]})")
print(f"nguong bai bao cong bo: {PAPER_THR}")

# ---------- 2. Independent test ----------
print("\n=== Independent test, 10 seed ===", flush=True)
pr_list = Parallel(n_jobs=10, verbose=1)(
    delayed(probs)(X_tr, y_tr, X_in, s) for s in range(N_REP))

rows = []
for s, pr in enumerate(pr_list):
    em = (pr["ERT"] + pr["GB"]) / 2.0
    for tag, p in (("ERT", pr["ERT"]), ("GB", pr["GB"]), ("iBCE-EL (EM1)", em)):
        for tname, thr in (("thr_cv", THR_CV), ("thr_0.35", PAPER_THR)):
            r = metrics(y_in, p, thr)
            r.update(seed=s, model=tag, thr_src=tname, thr=thr)
            rows.append(r)

ind = pd.DataFrame(rows)
print("\n===== Independent test (average cua 10 seed) =====")
print(ind.groupby(["model", "thr_src"], sort=False)[COLS].mean().round(4).to_string())
print("\nstd giua 10 seed:")
print(ind.groupby(["model", "thr_src"], sort=False)[COLS].std().round(4).to_string())

# ---------- 3. Doi chieu voi so cong bo ----------
PAPER = {
    "cv":  {"iBCE-EL (EM1)": dict(MCC=.454, Acc=.729, Sens=.716, Spec=.739, AUC=.782),
            "ERT": dict(MCC=.437, Acc=.718, Sens=.734, Spec=.705, AUC=.776),
            "GB":  dict(MCC=.446, Acc=.725, Sens=.712, Spec=.735, AUC=.773)},
    "ind": {"iBCE-EL (EM1)": dict(MCC=.463, Acc=.732, Sens=.742, Spec=.724, AUC=.789),
            "ERT": dict(MCC=.440, Acc=.719, Sens=.742, Spec=.703, AUC=.780),
            "GB":  dict(MCC=.445, Acc=.727, Sens=.717, Spec=.734, AUC=.776)},
}
for which, df, sel in (("cv", cv, None), ("ind", ind[ind.thr_src == "thr_0.35"], None)):
    g = df.groupby("model", sort=False)[COLS].mean()
    print(f"\n===== doi chieu {which.upper()}: ta vs bai bao =====")
    print(f"{'model':16s}{'metric':8s}{'ta':>9s}{'bai bao':>10s}{'delta':>9s}")
    for m, ref in PAPER[which].items():
        for k, v in ref.items():
            print(f"{m:16s}{k:8s}{g.loc[m, k]:9.4f}{v:10.3f}{g.loc[m, k]-v:+9.4f}")

with pd.ExcelWriter("ibce_repro.xlsx", engine="openpyxl") as w:
    cv.to_excel(w, "CV", index=False)
    ind.to_excel(w, "Independent", index=False)
print("\n-> ibce_repro.xlsx")
