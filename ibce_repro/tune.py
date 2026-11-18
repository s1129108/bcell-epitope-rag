"""Tune ERT (tren H7) va GB (tren H15) bang 5-fold CV TREN TAP TRAIN.

Table S1 cua bai bao (co hyperparameter cu the) khong co trong full text, nen phai
tu tune. Grid giu nho va quanh default cua sklearn, dung tieu chi AUC trung binh
tren 5 fold — khong he nhin independent set.
"""
import json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import data, features

S, y = data.load("train")
X = {"H7": features.encode(S, "H7"), "H15": features.encode(S, "H15")}
print("H7", X["H7"].shape, "H15", X["H15"].shape, flush=True)

GRID = {
    "ERT": ("H7", ExtraTreesClassifier, [
        dict(n_estimators=500,  max_features="sqrt", min_samples_leaf=1),
        dict(n_estimators=500,  max_features="sqrt", min_samples_leaf=2),
        dict(n_estimators=1000, max_features="sqrt", min_samples_leaf=1),
        dict(n_estimators=1000, max_features=None,   min_samples_leaf=1),
        dict(n_estimators=1000, max_features=None,   min_samples_leaf=2),
        dict(n_estimators=1000, max_features=0.5,    min_samples_leaf=1),
    ]),
    "GB": ("H15", GradientBoostingClassifier, [
        dict(n_estimators=300, learning_rate=0.1,  max_depth=3),
        dict(n_estimators=500, learning_rate=0.1,  max_depth=3),
        dict(n_estimators=500, learning_rate=0.05, max_depth=3),
        dict(n_estimators=500, learning_rate=0.1,  max_depth=5),
        dict(n_estimators=1000, learning_rate=0.05, max_depth=3),
        dict(n_estimators=500, learning_rate=0.05, max_depth=5),
    ]),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
FOLDS = list(skf.split(np.zeros(len(y)), y))


def one(name, feat, cls, kw, fi):
    tr, te = FOLDS[fi]
    extra = dict(n_jobs=1) if cls is ExtraTreesClassifier else {}
    m = cls(random_state=0, **kw, **extra).fit(X[feat][tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[feat][te])[:, 1])


best = {}
for name, (feat, cls, cfgs) in GRID.items():
    t0 = time.time()
    jobs = [(ci, fi) for ci in range(len(cfgs)) for fi in range(5)]
    res = Parallel(n_jobs=20, verbose=0)(
        delayed(one)(name, feat, cls, cfgs[ci], fi) for ci, fi in jobs)
    auc = np.array(res).reshape(len(cfgs), 5)
    print(f"--- {name} tren {feat}  ({time.time()-t0:.0f}s)", flush=True)
    for ci, cfg in enumerate(cfgs):
        print(f"   {auc[ci].mean():.4f} +- {auc[ci].std():.4f}  {cfg}", flush=True)
    bi = int(auc.mean(1).argmax())
    best[name] = dict(feat=feat, params=cfgs[bi], cv_auc=float(auc[bi].mean()))
    print(f"   => chon {cfgs[bi]}  AUC {auc[bi].mean():.4f}", flush=True)

json.dump(best, open("best_params.json", "w"), indent=2)
print(json.dumps(best, indent=2))
