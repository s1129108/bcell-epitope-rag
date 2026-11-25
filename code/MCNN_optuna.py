#!/usr/bin/env python3
"""
MCNN.py
Multi-Window Convolutional Neural Network (DeepScan) + Optuna hyperparameter search

- Window sizes searched from 2 to 25
- Finds the best combination of windows
- Nested: Optuna search inside Cross-Validation
- Final evaluation on an independent test set
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
import optuna
from optuna.samplers import TPESampler
import joblib
import warnings
import import_tests as load_data
warnings.filterwarnings("ignore")

# ============================================================
#  Global configuration (change according to your data)
# ============================================================
MAXSEQ        = 25          # must match the maxlength you used when creating embeddings
NUM_FEATURE   = 1280        # embedding dimension
NUM_CLASSES   = 2
N_FOLDS       = 5
N_TRIALS      = 10          # increase for more thorough search
SEED          = 42
OUTPUT_DIR    = "optuna_mcnn_results_ESM2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
#  Model definition
# ============================================================
class DeepScan(Model):
    def __init__(self,
                 maxseq,
                 num_feature,
                 num_classes,
                 window_sizes,
                 num_filters=256,
                 num_hidden=1000,
                 dropout=0.7):
        super().__init__()
        self.window_sizes = sorted(window_sizes)

        self.conv2d = []
        self.maxpool = []
        self.flatten = []

        for ws in self.window_sizes:
            self.conv2d.append(
                layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(1, ws),
                    activation="relu",
                    padding="valid",
                    bias_initializer=tf.constant_initializer(0.1),
                    kernel_initializer="glorot_uniform"
                )
            )
            self.maxpool.append(
                layers.MaxPooling2D(
                    pool_size=(1, maxseq - ws + 1),
                    strides=(1, maxseq),
                    padding="valid"
                )
            )
            self.flatten.append(layers.Flatten())

        self.dropout = layers.Dropout(dropout)
        self.fc1 = layers.Dense(
            num_hidden,
            activation="relu",
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer="glorot_uniform"
        )
        self.fc2 = layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, x, training=False):
        xs = []
        for i in range(len(self.window_sizes)):
            x_c = self.conv2d[i](x)
            x_m = self.maxpool[i](x_c)
            x_f = self.flatten[i](x_m)
            xs.append(x_f)
        x = tf.concat(xs, axis=1)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        return self.fc2(x)


# ============================================================
#  Helper: build & train one model
# ============================================================
def train_one_model(X_tr, y_tr, X_va, y_va, params, maxseq, num_feature, num_classes):
    model = DeepScan(
        maxseq=maxseq,
        num_feature=num_feature,
        num_classes=num_classes,
        window_sizes=params["window_sizes"],
        num_filters=params["num_filters"],
        num_hidden=params["num_hidden"],
        dropout=params["dropout"]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0
        )
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=20,
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=0
    )
    return model, history


# ============================================================
#  Optuna objective (with inner Cross-Validation)
# ============================================================
def objective(trial, X, y, maxseq, num_feature, num_classes):
    # ----- Window combination search (2 ~ 25) -----
    n_windows = trial.suggest_int("n_windows", 1, 6)

    candidate_sizes = list(range(2, 26))          # 2,3,...,25
    window_sizes = []
    for i in range(n_windows):
        ws = trial.suggest_int(f"window_{i}", 2, 25)
        window_sizes.append(ws)
    window_sizes = sorted(list(set(window_sizes)))  # unique + sorted

    # Other hyperparameters
    params = {
        "window_sizes": window_sizes,
        "num_filters":  trial.suggest_categorical("num_filters", [64, 128, 256]),
        "num_hidden":   trial.suggest_categorical("num_hidden",  [500, 1000,1500]),
        "dropout":      trial.suggest_float("dropout", 0.1, 0.7, step=0.1),
        "lr":           trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }

    # ----- Cross-validation -----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_labels)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model, history = train_one_model(
            X_tr, y_tr, X_va, y_va,
            params, maxseq, num_feature, num_classes
        )

        # Use best validation accuracy of this fold
        best_acc = max(history.history["val_accuracy"])
        fold_scores.append(best_acc)

        # Free memory
        del model
        tf.keras.backend.clear_session()

    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("window_sizes", window_sizes)
    trial.set_user_attr("fold_scores", fold_scores)
    return mean_score


# ============================================================
#  Main pipeline
# ============================================================
def main(X_train, y_train, X_test, y_test):
    """
    X_train / X_test : shape (N, 1, MAXSEQ, NUM_FEATURE) or (N, MAXSEQ, 1, NUM_FEATURE)
                       will be converted to (N, 1, MAXSEQ, NUM_FEATURE)
    y_train / y_test : shape (N,) or (N, NUM_CLASSES)
    """
    # ---------- shape normalisation ----------
    if X_train.ndim == 4 and X_train.shape[1] != 1:
        # (N, L, 1, D) → (N, 1, L, D)
        X_train = np.transpose(X_train, (0, 2, 1, 3))
        X_test  = np.transpose(X_test,  (0, 2, 1, 3))

    if y_train.ndim == 1:
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_test  = to_categorical(y_test,  NUM_CLASSES)

    print(f"X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}    y_test : {y_test.shape}")

    # ---------- Optuna search ----------
    study = optuna.create_study(
        direction="maximize",
        study_name="MCNN_window_search",
        sampler=TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train,
            MAXSEQ, NUM_FEATURE, NUM_CLASSES
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    # ---------- Save study ----------
    joblib.dump(study, os.path.join(OUTPUT_DIR, "optuna_study.pkl"))
    print("\n========== Best Trial ==========")
    best = study.best_trial
    print(f"Best CV accuracy : {best.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"  window_sizes (resolved): {best.user_attrs['window_sizes']}")

    # ---------- Retrain on full training set with best params ----------
    best_params = {
        "window_sizes": best.user_attrs["window_sizes"],
        "num_filters":  best.params["num_filters"],
        "num_hidden":   best.params["num_hidden"],
        "dropout":      best.params["dropout"],
        "lr":           best.params["lr"],
        "batch_size":   best.params["batch_size"],
    }

    # Use a small validation split from training data for early-stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED,
        stratify=np.argmax(y_train, axis=1)
    )

    final_model, _ = train_one_model(
        X_tr, y_tr, X_va, y_va,
        best_params, MAXSEQ, NUM_FEATURE, NUM_CLASSES
    )

    # ---------- Independent test ----------
    y_prob = final_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n========== Independent Test Results ==========")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_true, y_prob[:, 1]):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"MCC      : {matthews_corrcoef(y_true, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Save final model & predictions
    final_model.save_weights(os.path.join(OUTPUT_DIR, "best_model.weights.h5"))
    np.save(os.path.join(OUTPUT_DIR, "test_y_true.npy"), y_true)
    np.save(os.path.join(OUTPUT_DIR, "test_y_prob.npy"), y_prob)
    np.save(os.path.join(OUTPUT_DIR, "test_y_pred.npy"), y_pred)

    # Save best configuration in readable form
    with open(os.path.join(OUTPUT_DIR, "best_config.txt"), "w") as f:
        f.write(f"Best CV accuracy: {best.value:.4f}\n")
        f.write("Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")

    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    return study, final_model


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    # -------------------------------------------------------
    # Load your data here
    # Expected shapes after loading:
    #   X : (N, 1, MAXSEQ, NUM_FEATURE)  or  (N, MAXSEQ, 1, NUM_FEATURE)
    #   y : (N,)  or  (N, NUM_CLASSES)
    # -------------------------------------------------------
    X_train, y_train, X_test, y_test = load_data.MCNN_data_load(feature_type="ESM2", class_0_types=["neg"], class_1_types=["pos"])

    study, model = main(X_train, y_train, X_test, y_test)