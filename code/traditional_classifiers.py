#!/usr/bin/env python3

"""
Run with the TF conda env, from the B_CELL directory:
  python ./code/traditional_classifiers.py -df ESMC
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_tests as load_data
from MCNN import model_test, save_curves, append_excel_row, METRIC_NAMES

parser = argparse.ArgumentParser()
parser.add_argument("-df", "--DATA_FEATURE", default="ESMC")
parser.add_argument("-k", "--KFold", type=int, default=5)
parser.add_argument("-clf", "--classifiers", nargs="+", default=["SVM", "RF", "KNN", "XGBoost", "LR"])
parser.add_argument("-jobs", type=int, default=16, help="CPU workers for grid search")
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-limit", type=int, default=None, help="use only the first N train/test samples (smoke test)")


def classifiers(seed, jobs):
    """name -> (estimator, parameter grid, n_jobs for the grid search)."""
    return {
        "SVM": (make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed)),
                {"svc__C": [0.1, 0.5, 1, 5, 10], "svc__gamma": ["scale"]}, jobs),
        "RF": (RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=4),
               {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 3]}, max(1, jobs // 4)),
        "KNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                {"kneighborsclassifier__n_neighbors": [5, 11, 21, 41],
                 "kneighborsclassifier__weights": ["uniform", "distance"],
                 "kneighborsclassifier__metric": ["euclidean", "cosine"]}, jobs),
        "XGBoost": (XGBClassifier(n_estimators=500, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                                  device="cuda", random_state=seed, eval_metric="logloss"),
                    {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]}, 1),   # one GPU -> sequential
        "LR": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)),
               {"logisticregression__C": [0.0001, 0.001, 0.01, 0.1, 1]}, jobs),
    }


def mean_pool(x):
    """(N, 1, L, D) padded residue embeddings -> (N, D) mean over the real residues."""
    x = x[:, 0].astype(np.float32)
    real = np.any(x != 0, axis=-1, keepdims=True)
    return (x * real).sum(axis=1) / np.maximum(real.sum(axis=1), 1)


def _fold_auc(estimator, params, X, y, tr, va):
    model = clone(estimator).set_params(**params).fit(X[tr], y[tr])
    # SVC without probability=True has no predict_proba; AUC only needs a ranking score
    score = model.predict_proba(X[va])[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X[va])
    return roc_auc_score(y[va], score)


def grid_search(estimator, grid, X, y, cv, n_jobs):
    """Stratified k-fold ROC-AUC for every parameter combination -> (best params, mean, std).
    Written out instead of GridSearchCV because xgboost 2.0.3 in the TF env is not compatible
    with scikit-learn 1.6's GridSearchCV (missing __sklearn_tags__)."""
    combos = list(ParameterGrid(grid))
    folds = list(cv.split(X, y))
    scores = Parallel(n_jobs=n_jobs)(delayed(_fold_auc)(estimator, p, X, y, tr, va) for p in combos for tr, va in folds)
    scores = np.array(scores).reshape(len(combos), len(folds))
    best = int(np.argmax(scores.mean(axis=1)))
    for p, s in zip(combos, scores):
        print(f"  {p}: CV AUC {s.mean():.4f} ± {s.std():.4f}")
    return combos[best], scores[best].mean(), scores[best].std()


class ProbaModel:
    """Lets MCNN.model_test() call .predict() on a fitted sklearn classifier."""
    def __init__(self, clf):
        self.clf = clf

    def predict(self, x, verbose=0):
        return self.clf.predict_proba(x)


def mcnn_best_row(feature):
    """Best MCNN trial (by CV AUC) of the search for this feature, for the comparison sheet."""
    path = f"MCNN_search_MCNN_{feature}.xlsx"
    if not os.path.exists(path):
        return None
    t = pd.read_excel(path, sheet_name="Trials")
    best = t[t.state == "COMPLETE"].sort_values("CV_AUC", ascending=False).iloc[0]
    row = {"classifier": f"MCNN (search trial {int(best.trial)})", "CV_AUC_mean": best.CV_AUC, "CV_AUC_std": best.CV_AUC_std}
    row.update({m: best[f"IND_{m}"] for m in METRIC_NAMES})
    return row


def main(args):
    feature = args.DATA_FEATURE
    excel = f"Traditional_ind_report_{feature}.xlsx"
    pkl_dir = os.path.join("PKL", f"traditional_{feature}")

    x_train, y_train, x_test, y_test = load_data.MCNN_data_load(feature_type=feature, class_0_types=["neg"], class_1_types=["pos"])
    if args.limit:
        x_train, y_train, x_test, y_test = x_train[:args.limit], y_train[:args.limit], x_test[:args.limit], y_test[:args.limit]
    X_train, X_test = mean_pool(x_train), mean_pool(x_test)
    t_train = np.argmax(y_train, axis=1)
    print(f"{feature}: train {X_train.shape}, test {X_test.shape} (mean-pooled)")

    cv = StratifiedKFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)
    rows = []
    for name in args.classifiers:
        estimator, grid, n_jobs = classifiers(args.seed, args.jobs)[name]
        t0 = time.time()
        print(f"\n===== {name}: grid {grid}")

        best_params, cv_mean, cv_std = grid_search(estimator, grid, X_train, t_train, cv, n_jobs)
        print(f"best CV AUC {cv_mean:.4f} ± {cv_std:.4f} with {best_params}")

        # refit on the whole training set with the best parameters
        final = clone(estimator).set_params(**best_params)
        if name == "SVM":
            final.set_params(svc__probability=True)    # probabilities for the PR curve / threshold
        final.fit(X_train, t_train)

        result, curves = model_test(ProbaModel(final), X_test, y_test)
        print(result)
#         save_curves(curves, pkl_dir, f"{name}_{feature}")

#         row = {"classifier": name, "feature": f"{feature} (mean-pooled)",
#                "CV_AUC_mean": cv_mean, "CV_AUC_std": cv_std,
#                "best_params": str(best_params), **result,
#                "minutes": round((time.time() - t0) / 60, 2)}
#         append_excel_row(excel, "Independent", row)
#         rows.append(row)

#     # side-by-side table with the best MCNN of the hyperparameter search
#     mcnn = mcnn_best_row(feature)
#     for row in ([mcnn] if mcnn else []) + rows:
#         append_excel_row(excel, "Comparison", {k: row[k] for k in ["classifier", "CV_AUC_mean", "CV_AUC_std"] + METRIC_NAMES})
#     print(f"\nResults saved to {excel}, curves to {pkl_dir}/")


if __name__ == "__main__":
    main(parser.parse_args())
