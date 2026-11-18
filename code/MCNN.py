"""
MCNN (DeepScan) for B-cell epitope prediction.

[MOD] Reorganised version of the original script (backup: MCNN_original.py).
Every block that is new or changed is tagged with "[MOD]". Summary of the changes:
  * PR-AUC is now returned/reported everywhere (it was computed but dropped before).
  * Reports are written to Excel files in the working directory, one row at a time
    (so partial results survive a crash):
        cross  -> MCNN_cv_report.xlsx        (sheets: CV_folds, CV_summary)
        ind    -> MCNN_ind_report.xlsx       (sheet : Independent)
        search -> MCNN_search_<study>.xlsx   (sheets: Trials, Folds, Best)
  * ROC / PR curve PKL files are written only in independent-test mode (and for the
    best trial of a search).
  * -save_model saves the weights + a JSON config after the independent test.
  * New mode "-vm search": Optuna search over filters, window sizes, hidden size,
    number of hidden layers, dropout, learning rate, batch size and loss.
    Every trial = k-fold CV (+ independent test), logged to Excel while running.
  * Hyperparameters that were hard-coded are now flags: -lr, -nh, -bs, -loss, -runs, -seed.

How to use (run from the B_CELL directory):
  # 5-fold CV
  python ./code/MCNN.py -maxseq 25 -f 128 -w 2 -df "ESM2" -imb "None" -k 5 -vm "cross"
  # independent test, also save weights + config
  python ./code/MCNN.py -maxseq 25 -f 128 -w 2 -df "ESM2" -vm "ind" -save_model
  # hyperparameter search (resumable: re-run the same command to add trials)
  python ./code/MCNN.py -df "ESM2" -vm "search" -trials 50 -obj "AUC" -save_model
"""
import os
import gc
import json
import math
import time
import pickle
import argparse
from time import gmtime, strftime
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping

import queue
import multiprocessing as mp

import import_tests as load_data

# [MOD] allocate GPU memory on demand instead of grabbing all 24 GB at start-up
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

"----------------------------------------------------------------------------------------------------"
parser = argparse.ArgumentParser(description='Program arguments')
parser.add_argument("-maxseq", "--MAXSEQ", type=int, default=25)
parser.add_argument("-f", "--FILTER", type=int, default=256)
parser.add_argument("-w", "--WINDOW", nargs='+', type=int, default=[2, 4, 6])
parser.add_argument("-nf", "--NUM_FEATURE", type=int, help="[MOD] optional, inferred from the data")

parser.add_argument("-hi", "--HIDDEN", type=int, default=500)
parser.add_argument("-drop", "--DROPOUT", type=float, default=0.7)
parser.add_argument("-ep", "--EPOCHS", type=int, default=20)

parser.add_argument("-df", "--DATA_FEATURE", type=str, default="TrimoPGLM")
parser.add_argument("-imb", "--imbalance_mod", type=str, default="None", help="the mod for imbalance 'SMOTE','ADASYN','RANDOM'")
parser.add_argument("-csv", "--csv_path", type=str, default=None, help="[MOD] optional legacy CSV summary for cross mode")

parser.add_argument("-k", "--KFold", type=int, default=5)
parser.add_argument("-vm", "--validation_mode", type=str, default="cross", choices=["cross", "ind", "search"],
                    help="[MOD] 'cross', 'ind' or 'search'")

# [MOD] new training flags (defaults reproduce the original behaviour)
parser.add_argument("-lr", "--LR", type=float, default=1e-3, help="Adam learning rate")
parser.add_argument("-nh", "--N_HIDDEN_LAYERS", type=int, default=1, help="number of dense hidden layers")
parser.add_argument("-bs", "--BATCH_SIZE", type=int, default=64)
parser.add_argument("-loss", "--LOSS", type=str, default="auto", choices=["auto", "focal", "ce"],
                    help="auto = original behaviour: focal loss for cross, cross-entropy for ind (focal for search)")
parser.add_argument("-runs", "--RUNS", type=int, default=1, help="repeat k-fold CV this many times (cross mode)")
parser.add_argument("-seed", "--SEED", type=int, default=None)
# [MOD] classifier head. "ffn" = kiến trúc gốc (global max-pool -> dropout -> Dense).
# "lstm"/"bilstm" = thay FFN bằng recurrent head: BỎ max-pool để giữ lại trục vị trí,
# vì sau MaxPooling2D(pool_size=(1, maxseq-w+1)) chuỗi chỉ còn độ dài 1 — không còn gì
# cho LSTM đọc. Xem ghi chú ở class DeepScan.
parser.add_argument("-head", "--HEAD", type=str, default="ffn", choices=["ffn", "lstm", "bilstm"],
                    help="[MOD] ffn = Dense+dropout gốc; lstm/bilstm = thay bằng LSTM trên conv feature map")
parser.add_argument("-lstm_units", "--LSTM_UNITS", type=int, default=128,
                    help="[MOD] số unit của LSTM (bilstm: mỗi chiều lstm_units)")
parser.add_argument("-lstm_merge", "--LSTM_MERGE", type=str, default="time", choices=["time", "feature"],
                    help="[MOD] ghép nhiều window: 'time' = nối theo trục thời gian (dài "
                         "sum(maxseq-w+1)); 'feature' = conv padding='same' rồi nối theo trục "
                         "feature, mọi window cùng độ dài maxseq và khớp vị trí residue")
parser.add_argument("-mask", "--MASK_PAD", action="store_true",
                    help="[MOD] bo qua vi tri zero-pad khi max-pool (chi dung cho head ffn, "
                         "va chi khi -norm none vi mask nhan dien padding bang 'toan bo D chieu = 0')")
parser.add_argument("-norm", "--NORM", type=str, default="none", choices=["none", "zscore"],
                    help="[MOD] zscore = per-dimension standardisation fitted on the training part only "
                         "(needed for embeddings with outlier dimensions, e.g. AMPLIFY)")

# [MOD] output flags
parser.add_argument("-xlsx", "--excel_path", type=str, default=None, help="override the Excel report path")
parser.add_argument("-pkl", "--pkl_dir", type=str, default="./PKL")
parser.add_argument("-save_model", action="store_true", help="save weights + config after the independent test")
parser.add_argument("-model_dir", type=str, default="./saved_models")

# [MOD] hyperparameter search flags
parser.add_argument("-trials", "--N_TRIALS", type=int, default=30)
parser.add_argument("-obj", "--OBJECTIVE", type=str, default="AUC", choices=["AUC", "PR_AUC", "MCC", "Acc", "F1"],
                    help="CV metric (mean over folds) the search maximises")
parser.add_argument("-study", "--STUDY", type=str, default=None, help="Optuna study name (default MCNN_<feature>)")
parser.add_argument("-search_ind", action=argparse.BooleanOptionalAction, default=True,
                    help="also run the independent test for every finished trial (use --no-search_ind to skip)")
# [MOD] parse the command line only when run as a script, so "from MCNN import load_saved_model" works
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

MAXSEQ = args.MAXSEQ
DATA_FEATURE = args.DATA_FEATURE
IMBALANCE = args.imbalance_mod
MODE = args.validation_mode
K_Fold = args.KFold

NUM_CLASSES = 2
CLASS_NAMES = ['Negative', 'Positive']

# [MOD] metric names in report order (PR_AUC added)
METRIC_NAMES = ["TP", "FP", "TN", "FN", "Sens", "Spec", "Acc", "MCC", "AUC", "PR_AUC", "F1", "Prec", "Recall", "Threshold"]

# [MOD] Optuna search space - edit here to widen/narrow the search
SEARCH_SPACE = {
    "n_windows":         (1, 4),                       # how many parallel conv windows
    "window":            (2, MAXSEQ),                  # each window size
    "num_filters":       [32, 64, 128, 256, 512],
    "num_hidden":        [128, 256, 500, 1000, 1500],
    "num_hidden_layers": (1, 3),
    "dropout":           (0.1, 0.7, 0.1),              # low, high, step
    "lr":                (1e-4, 1e-2),                 # log-uniform
    "batch_size":        [32, 64, 128],
    "loss":              ["focal", "ce"],
}

"===================================================================================================="
def time_log(message):
    print(message, " : ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))


# [MOD] One dict holds every hyperparameter of a run (replaces the module-level globals)
def config_from_args(mode):
    # [MOD] mask nhan dien padding bang "toan bo D chieu = 0"; zscore lam vi tri
    # zero-pad thanh khac zero nen hai thu khong di cung nhau duoc.
    if args.MASK_PAD and args.NORM != "none":
        raise SystemExit("-mask khong dung duoc voi -norm zscore: standardisation lam "
                         "vi tri zero-pad thanh khac zero nen khong con nhan dien duoc")
    loss = args.LOSS
    if loss == "auto":
        loss = "ce" if mode == "ind" else "focal"
    return {
        "feature": DATA_FEATURE,
        "num_feature": int(NUM_FEATURE),   # set from the loaded data in main
        "maxseq": MAXSEQ,
        "window_sizes": sorted(args.WINDOW),
        "num_filters": args.FILTER,
        "num_hidden": args.HIDDEN,
        "num_hidden_layers": args.N_HIDDEN_LAYERS,
        "dropout": args.DROPOUT,
        # [MOD] recurrent head
        "head": args.HEAD,
        "lstm_units": args.LSTM_UNITS,
        "lstm_merge": args.LSTM_MERGE,
        "lr": args.LR,
        "batch_size": args.BATCH_SIZE,
        "epochs": args.EPOCHS,
        "loss": loss,
        "imbalance": IMBALANCE,
        "norm": args.NORM,
        # [MOD] masked padding
        "mask_pad": args.MASK_PAD,
        "seed": args.SEED,
    }


def config_tag(cfg):
    """[MOD] Short file-name-safe description of a configuration."""
    w = "-".join(map(str, cfg["window_sizes"]))
    # [MOD] head đi vào tên file để kết quả ffn và lstm không ghi đè nhau
    head = cfg.get("head", "ffn")
    h = (f"h{cfg['num_hidden']}x{cfg['num_hidden_layers']}_d{cfg['dropout']}" if head == "ffn"
         else f"{head}{cfg['lstm_units']}_{cfg.get('lstm_merge', 'time')}")
    # [MOD] _mask vao ten file de ket qua co/khong mask khong ghi de nhau
    m = "_mask" if cfg.get("mask_pad") else ""
    return (f"MCNN_{cfg['maxseq']}_{cfg['feature']}_w{w}_f{cfg['num_filters']}_{h}{m}"
            f"_lr{cfg['lr']:.0e}_{cfg['loss']}_{cfg['imbalance']}")


"----------------------------------------------------------------------------------------------------"
# [MOD] Excel helpers: append one row at a time and save straight away, so the file is
# always up to date while experiments are running. New columns are added automatically.
def _excel_value(v):
    if isinstance(v, (list, tuple)):
        return str(list(v))
    if isinstance(v, np.generic):
        return v.item()
    return v


def append_excel_row(path, sheet, row):
    from openpyxl import Workbook, load_workbook

    if os.path.exists(path):
        wb = load_workbook(path)
    else:
        wb = Workbook()
        wb.active.title = sheet
    ws = wb[sheet] if sheet in wb.sheetnames else wb.create_sheet(sheet)

    header = [c.value for c in ws[1]] if ws.max_row >= 1 and ws.cell(1, 1).value is not None else []
    for key in row:
        if key not in header:
            header.append(key)
            ws.cell(row=1, column=len(header), value=key)
    ws.append([_excel_value(row.get(h)) for h in header])

    tmp = path[:-5] + ".tmp.xlsx"     # write then rename, so a crash never leaves a broken file
    wb.save(tmp)
    os.replace(tmp, path)


def summarise(fold_rows, prefix=""):
    """[MOD] mean and std of every metric over a list of per-fold result dicts."""
    out = {}
    for m in METRIC_NAMES:
        vals = np.array([r[m] for r in fold_rows], dtype=float)
        out[f"{prefix}{m}_mean"] = float(np.mean(vals))
        out[f"{prefix}{m}_std"] = float(np.std(vals))
    return out


"----------------------------------------------------------------------------------------------------"
# [MOD] Curves are saved only by independent-test code paths
def SAVEROC(fpr, tpr, AUC, path):
    with open(path, "wb") as file:
        pickle.dump({"fpr": fpr, "tpr": tpr, "AUC": AUC}, file)


def SAVEPR(precision, recall, PR_AUC, path):
    with open(path, "wb") as file:
        pickle.dump({"precision": precision, "recall": recall, "AUC": PR_AUC}, file)


def save_curves(curves, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    SAVEROC(curves["fpr"], curves["tpr"], curves["AUC"], os.path.join(out_dir, f"{tag}_AUCROC.pkl"))
    SAVEPR(curves["precision"], curves["recall"], curves["PR_AUC"], os.path.join(out_dir, f"{tag}_PRROC.pkl"))
    print(f"ROC/PR curves saved to {out_dir}/{tag}_*.pkl")


"----------------------------------------------------------------------------------------------------"
# model MCNN
# [MOD] All sizes are constructor arguments (no globals), and the dense part can have
# several hidden layers (num_hidden_layers=1 is the original architecture).
class DeepScan(Model):
    def __init__(
        self,
        maxseq=25,
        window_sizes=(2, 4, 6),
        num_filters=256,
        num_hidden=1000,
        num_hidden_layers=1,
        dropout=0.7,
        num_classes=NUM_CLASSES,
        head="ffn",
        lstm_units=128,
        lstm_merge="time",
        mask_pad=False,
    ):
        super().__init__()

        # [MOD] masked padding: chi ho tro head "ffn" (head LSTM khong dung max-pool)
        if mask_pad and head != "ffn":
            raise ValueError("mask_pad chi dung duoc voi head='ffn'")
        self.mask_pad = mask_pad

        self.window_sizes = window_sizes
        # [MOD] head: "ffn" giữ nguyên kiến trúc gốc; "lstm"/"bilstm" thay phần
        # dropout+Dense bằng recurrent layer.
        #
        # Vì sao phải bỏ MaxPooling2D ở head LSTM: pool_size=(1, maxseq-w+1) thu
        # cả trục vị trí về độ dài 1, nên sau pooling mỗi window chỉ còn một vector
        # num_filters chiều — không còn chuỗi nào để LSTM đọc. Giữ max-pool rồi
        # cắm LSTM vào sẽ thành LSTM chạy trên chuỗi dài 1, tức một Dense đắt tiền.
        # Nên head LSTM dùng trực tiếp conv feature map (B, T, num_filters).
        self.head = head
        self.lstm_merge = lstm_merge

        # "feature": padding='same' để mọi window ra đúng maxseq bước thời gian,
        # nhờ vậy bước t của mọi window ứng với cùng vị trí residue, rồi nối theo
        # trục feature -> (B, maxseq, num_filters*n_window).
        # "time": giữ padding='valid' như gốc, nối theo trục thời gian ->
        # (B, sum(maxseq-w+1), num_filters).
        conv_pad = "same" if (head != "ffn" and lstm_merge == "feature") else "valid"

        self.conv2d = []
        self.maxpool = []
        self.flatten = []

        for window_size in self.window_sizes:
            self.conv2d.append(
                layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(1, window_size),
                    activation="relu",
                    padding=conv_pad,
                    bias_initializer=tf.constant_initializer(0.1),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                )
            )

            # [MOD] khi mask_pad bat, max-pool duoc lam thu cong trong _masked_pool
            # nen khong can MaxPooling2D/Flatten nua.
            if head == "ffn" and not mask_pad:
                self.maxpool.append(
                    layers.MaxPooling2D(
                        pool_size=(1, maxseq - window_size + 1),
                        strides=(1, maxseq),
                        padding="valid",
                    )
                )
                self.flatten.append(layers.Flatten())

        # Keras layer for concatenating the multi-window features
        self.concat = layers.Concatenate(axis=1)

        if head == "ffn":
            self.dropout = layers.Dropout(rate=dropout)

            self.hidden = [
                layers.Dense(
                    num_hidden,
                    activation="relu",
                    bias_initializer=tf.constant_initializer(0.1),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                )
                for _ in range(num_hidden_layers)
            ]
        else:
            # [MOD] trục nối cho head LSTM: 1 = thời gian, 2 = feature
            self.seq_concat = layers.Concatenate(axis=1 if lstm_merge == "time" else 2)
            # [MOD] LSTM KHÔNG có regularisation thì overfit nặng trên data này
            # (lần probe đầu: train acc 0.975 vs test AUC 0.714). Dropout 0.7 của
            # head FFN chính là thứ giữ nó lại, nên head LSTM dùng cùng rate đó ở
            # hai chỗ: dropout trên input của LSTM, và dropout trên output trước fc2.
            # recurrent_dropout để 0 vì bật lên là mất kernel cuDNN, chậm ~10x.
            rnn = layers.LSTM(lstm_units, dropout=dropout)
            self.rnn = layers.Bidirectional(rnn) if head == "bilstm" else rnn
            self.dropout = layers.Dropout(rate=dropout)

        self.fc2 = layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(9.180504074572426e-05),
        )

    # [MOD] Max-pool bo qua vi tri zero-pad.
    #
    # Mot vi tri residue la padding khi TOAN BO D chieu bang 0 (dung voi cach
    # pad_or_cut zero-pad, va chi dung khi -norm none).
    #
    # Mot vi tri dau ra t cua conv window w chi hop le khi CA w residue trong
    # window deu la residue that. Quy tac "co it nhat 1 residue that" la SAI:
    # no van cho so vi tri hop le phu thuoc maxseq, nen ket qua khong bat bien
    # theo maxseq (da bi test B bat loi).
    #
    # Chuoi ngan hon window lon nhat thi khong con vi tri nao hop le; luc do lui
    # ve vi tri 0 de khong max-pool tren tap rong.
    def _masked_pool(self, x_conv, real, window):
        r = tf.cast(real, x_conv.dtype)
        r = tf.reshape(r, [-1, tf.shape(r)[-1], 1])              # (B, maxseq, 1)
        # AVG pool tren r = ty le residue thuc trong window; hop le khi == 1, tuc CA w
        # residue cua window deu la residue thuc.
        frac = tf.nn.pool(r, window_shape=[window], pooling_type="AVG",
                          strides=[1], padding="VALID")          # (B, T_w, 1)
        v = tf.cast(frac > 1.0 - 1e-4, x_conv.dtype)
        none_valid = tf.cast(tf.reduce_max(v, axis=1, keepdims=True) < 0.5, x_conv.dtype)
        first = tf.concat([tf.ones_like(v[:, :1]), tf.zeros_like(v[:, 1:])], axis=1)
        v = tf.maximum(v, none_valid * first)
        v = tf.reshape(v, [-1, 1, tf.shape(v)[1], 1])
        # -1e9 o vi tri khong hop le: an toan vi relu(conv) >= 0 nen khong bao gio thang
        masked = tf.where(v > 0.5, x_conv, tf.constant(-1e9, dtype=x_conv.dtype))
        return tf.reshape(tf.reduce_max(masked, axis=2), [-1, tf.shape(x_conv)[-1]])

    def call(self, x, training=False):
        # [MOD] nhanh masked padding
        if self.mask_pad:
            real = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)   # (B, 1, maxseq)
            features = []
            for conv, w in zip(self.conv2d, self.window_sizes):
                features.append(self._masked_pool(conv(x), real, w))
            x = features[0] if len(features) == 1 else self.concat(features)
            x = self.dropout(x, training=training)
            for dense in self.hidden:
                x = dense(x)
            return self.fc2(x)

        if self.head == "ffn":
            features = []

            for conv, pool, flatten in zip(
                self.conv2d, self.maxpool, self.flatten
            ):
                x_conv = conv(x)
                x_maxp = pool(x_conv)
                x_flat = flatten(x_maxp)
                features.append(x_flat)

            if len(features) == 1:
                x = features[0]
            else:
                x = self.concat(features)

            x = self.dropout(x, training=training)
            for dense in self.hidden:
                x = dense(x)
            return self.fc2(x)

        # [MOD] head LSTM. Đầu vào (B, 1, maxseq, D); conv ra (B, 1, T_w, F) nên
        # bỏ trục height size-1 để được chuỗi (B, T_w, F).
        seqs = []
        for conv in self.conv2d:
            x_conv = conv(x)
            seqs.append(tf.squeeze(x_conv, axis=1))

        s = seqs[0] if len(seqs) == 1 else self.seq_concat(seqs)
        s = self.rnn(s, training=training)
        s = self.dropout(s, training=training)
        return self.fc2(s)

"----------------------------------------------------------------------------------------------------"
# design a custom focal loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha  # Ưu tiên lớp thiểu số (cytokine receptor)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.where(y_true == 1.0, y_pred, 1.0 - y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - pt, self.gamma)
        loss = focal_weight * ce
        return tf.reduce_mean(loss)

# model fit batch funtion
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        super().__init__()   # [MOD] required by Keras 3 PyDataset
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.data[batch_indexes], self.labels[batch_indexes]   # [MOD] vectorised slicing

"----------------------------------------------------------------------------------------------------"
# transform feature by SMOTE, ADASYN, RANDOM
def IMBALANCE_funct(IMBALANCE, x_train, y_train):
    if IMBALANCE == "None":
        return x_train, y_train

    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    samplers = {
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "RANDOM": RandomOverSampler(random_state=42),
    }
    shape = x_train.shape[1:]
    # [MOD] resample on class indices (y_train is one-hot) and restore the 4D shape from the data
    x_res, y_res = samplers[IMBALANCE].fit_resample(x_train.reshape(len(x_train), -1), np.argmax(y_train, axis=1))
    x_res = x_res.reshape((len(x_res),) + shape)
    print(f"{IMBALANCE}: {x_train.shape} -> {x_res.shape}")
    gc.collect()
    return x_res, tf.keras.utils.to_categorical(y_res, NUM_CLASSES)

"----------------------------------------------------------------------------------------------------"
# [MOD] Per-dimension z-score. Statistics come from the real (non-padded) residues of the
# training part only, then are applied to train and validation/test; padded positions stay 0.
def fit_scaler(x):
    v = x.reshape(-1, x.shape[-1])
    v = v[np.any(v != 0, axis=1)].astype(np.float32)
    return {"mean": v.mean(axis=0), "std": v.std(axis=0) + 1e-6}


def apply_scaler(x, scaler):
    pad = ~np.any(x != 0, axis=-1, keepdims=True)
    z = (x.astype(np.float32) - scaler["mean"]) / scaler["std"]
    return np.where(pad, 0.0, z).astype(np.float16)


def normalise(cfg, x_fit, *others):
    """Return (scaler, x_fit, *others) normalised as cfg['norm'] asks (scaler None = unchanged)."""
    if cfg.get("norm", "none") == "none":
        return (None, x_fit) + others
    scaler = fit_scaler(x_fit)
    return (scaler, apply_scaler(x_fit, scaler)) + tuple(apply_scaler(o, scaler) for o in others)


def scaler_path(weights_path):
    return weights_path.replace(".weights.h5", ".scaler.npz")


"----------------------------------------------------------------------------------------------------"
# [MOD] Build + train one model from a config dict (shared by cross / ind / search)
def train_model(x, y, cfg, verbose=1):
    x, y = IMBALANCE_funct(cfg["imbalance"], x, y)
    generator = DataGenerator(x, y, batch_size=cfg["batch_size"])

    model = DeepScan(
        maxseq=cfg["maxseq"],
        window_sizes=cfg["window_sizes"],
        num_filters=cfg["num_filters"],
        num_hidden=cfg["num_hidden"],
        num_hidden_layers=cfg["num_hidden_layers"],
        dropout=cfg["dropout"],
        # [MOD] recurrent head (mặc định "ffn" = kiến trúc gốc)
        head=cfg.get("head", "ffn"),
        lstm_units=cfg.get("lstm_units", 128),
        lstm_merge=cfg.get("lstm_merge", "time"),
        # [MOD] masked padding
        mask_pad=cfg.get("mask_pad", False),
    )
    loss = FocalLoss(alpha=0.85, gamma=2.0) if cfg["loss"] == "focal" else "categorical_crossentropy"
    # jit_compile=False: XLA's GEMM autotuner in TF 2.16 fails on some layer shapes
    # ("All algorithms tried for %dot ... failed"), which crashed the search at trial 6
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]), loss=loss, metrics=['accuracy'],
                  jit_compile=False)
    model(np.zeros((1,) + x.shape[1:], dtype="float32"))   # create the weights (model.build() does not in Keras 3)

    model.fit(generator, epochs=cfg["epochs"],
              callbacks=[EarlyStopping(monitor='loss', patience=10)],
              verbose=verbose, shuffle=True)
    return model


def free_model(model):
    del model
    tf.keras.backend.clear_session()
    gc.collect()


# predict
# [MOD] returns (metrics dict incl. PR_AUC, curves dict); no plotting, no file writing
def model_test(model, x_test, y_test):
    pred_test = model.predict(x_test, verbose=0)
    y_true = y_test[:, 1]
    y_score = pred_test[:, 1]

    # Compute ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUC = metrics.auc(fpr, tpr)

    # Compute G-Mean and Best Threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]

    # Apply threshold to get predictions
    y_pred = (y_score >= threshold).astype(int)

    # Compute confusion matrix values
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Compute evaluation metrics
    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc = (TP + TN) / (TP + FP + TN + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if TP + FP > 0 and FP + TN > 0 and TP + FN > 0 and TN + FN > 0 else 0.0
    F1 = 2 * TP / (2 * TP + FP + FN)
    Prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    # Compute Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    PR_AUC = metrics.auc(recall, precision)  # Compute PR AUC

    result = dict(zip(METRIC_NAMES, [int(TP), int(FP), int(TN), int(FN), Sens, Spec, Acc, MCC,
                                     AUC, PR_AUC, F1, Prec, Recall, float(min(threshold, 1.0))]))
    print(f'Best Threshold={threshold:.4f}, G-Mean={gmeans[ix]:.4f}')
    print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in result.items()) + "\n")

    curves = {"fpr": fpr, "tpr": tpr, "AUC": AUC, "precision": precision, "recall": recall, "PR_AUC": PR_AUC}
    return result, curves


"----------------------------------------------------------------------------------------------------"
# [MOD] Save weights + configuration so the model can be rebuilt later with load_saved_model()
def save_model(model, cfg, result, out_dir, tag, scaler=None):
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, f"{tag}.weights.h5")
    model.save_weights(weights_path)
    if scaler is not None:
        np.savez(scaler_path(weights_path), **scaler)
    write_model_config(cfg, result, out_dir, tag)


def write_model_config(cfg, result, out_dir, tag):
    """[MOD] JSON next to <tag>.weights.h5 with everything needed to rebuild the model."""
    weights_path = os.path.join(out_dir, f"{tag}.weights.h5")
    config_path = os.path.join(out_dir, f"{tag}.config.json")
    with open(config_path, "w") as f:
        json.dump({"config": cfg,
                   "input_shape": [1, cfg["maxseq"], cfg["num_feature"]],
                   "independent_test": {k: float(v) for k, v in result.items()},
                   "weights": os.path.basename(weights_path),
                   "scaler": os.path.basename(scaler_path(weights_path)) if os.path.exists(scaler_path(weights_path)) else None,
                   "saved_at": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Model saved: {weights_path} + {config_path}")


def load_saved_model(config_path):
    """[MOD] Rebuild a DeepScan model from a *.config.json written by save_model().
    If the model was trained with -norm zscore, saved["scaler"] holds the training statistics:
    use apply_scaler(x, saved["scaler"]) on new data before model.predict()."""
    with open(config_path) as f:
        saved = json.load(f)
    cfg = saved["config"]
    model = DeepScan(maxseq=cfg["maxseq"], window_sizes=cfg["window_sizes"], num_filters=cfg["num_filters"],
                     num_hidden=cfg["num_hidden"], num_hidden_layers=cfg["num_hidden_layers"], dropout=cfg["dropout"],
                     # [MOD] recurrent head
                     head=cfg.get("head", "ffn"), lstm_units=cfg.get("lstm_units", 128),
                     lstm_merge=cfg.get("lstm_merge", "time"),
                     # [MOD] masked padding
                     mask_pad=cfg.get("mask_pad", False))
    # Keras 3: build() alone does not create the sub-layer weights of a subclassed model, and
    # load_weights() would then silently load nothing -> run one dummy batch first
    model(np.zeros((1,) + tuple(saved["input_shape"]), dtype="float32"))
    model.load_weights(os.path.join(os.path.dirname(config_path), saved["weights"]))
    if saved.get("scaler"):
        saved["scaler"] = dict(np.load(os.path.join(os.path.dirname(config_path), saved["scaler"])))
    return model, saved


"----------------------------------------------------------------------------------------------------"
# [MOD] One k-fold CV run. Stratified folds keep the pos/neg ratio equal in every fold
# (the original used plain KFold). random_state=run, so run i always gives the same split.
def cross_validate(x, y, cfg, k, run=0, on_fold=None, verbose=1):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=run)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, np.argmax(y, axis=1)), start=1):
        time_log(f"Run {run + 1} - Fold {fold}/{k}")
        _, x_tr, x_va = normalise(cfg, x[tr_idx], x[va_idx])   # statistics from this fold's training part
        model = train_model(x_tr, y[tr_idx], cfg, verbose=verbose)
        result, _ = model_test(model, x_va, y[va_idx])
        free_model(model)
        on_fold({"run": run + 1, "fold": fold, **result})


# [MOD] Keras 3 (TF 2.16) does not give back the GPU memory of a trained model, even after
# clear_session(): ~300 MB stayed allocated per model and the search ran out of GPU memory
# after ~40 trials. Training therefore runs in a short-lived child process ("job"), whose GPU
# memory is always released when it exits. The parent only collects results and writes Excel.
def _job_worker(cfg, k, run, do_cv, do_ind, weights_path, verbose, q):
    try:
        if cfg["seed"] is not None:
            tf.keras.utils.set_random_seed(cfg["seed"] + run)
        x_train, y_train, x_test, y_test = load_data.MCNN_data_load(feature_type=cfg["feature"], class_0_types=["neg"], class_1_types=["pos"])
        if do_cv:
            cross_validate(x_train, y_train, cfg, k, run, on_fold=lambda row: q.put(("fold", row)), verbose=verbose)
        if do_ind:
            scaler, x_train, x_test = normalise(cfg, x_train, x_test)
            model = train_model(x_train, y_train, cfg, verbose=verbose)
            result, curves = model_test(model, x_test, y_test)
            if weights_path:
                model.save_weights(weights_path)
                if scaler is not None:
                    np.savez(scaler_path(weights_path), **scaler)
            q.put(("ind", (result, curves)))
        q.put(("done", None))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


def run_job(cfg, k=5, run=0, do_cv=True, do_ind=False, weights_path=None, verbose=0,
            on_fold=None, trial=None, objective=None):
    """[MOD] Run CV folds and/or the independent test in a child process.
    Calls on_fold(row) as each fold finishes; with an Optuna trial, prunes on the running mean."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_job_worker, args=(cfg, k, run, do_cv, do_ind, weights_path, verbose, q), daemon=True)
    p.start()
    fold_scores, ind = [], None
    try:
        while True:
            try:
                kind, payload = q.get(timeout=10)
            except queue.Empty:
                if not p.is_alive():
                    raise RuntimeError(f"training process died (exit code {p.exitcode})")
                continue
            if kind == "fold":
                if on_fold:
                    on_fold(payload)
                if trial is not None:
                    import optuna
                    fold_scores.append(payload[objective])
                    trial.report(float(np.mean(fold_scores)), step=len(fold_scores))
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            elif kind == "ind":
                ind = payload
            elif kind == "error":
                raise RuntimeError(payload)
            elif kind == "done":
                return ind
    finally:
        if p.is_alive():
            p.terminate()   # e.g. pruned trial: stop training straight away
        p.join()


"====================================== modes ======================================"
def run_cross(x_train, y_train):
    cfg = config_from_args("cross")
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel = args.excel_path or "MCNN_cv_report.xlsx"
    base = {"exp_id": exp_id, **cfg}

    time_log("Start cross")
    fold_rows = []

    def log_fold(row):
        fold_rows.append(row)
        append_excel_row(excel, "CV_folds", {**base, **row})

    for run in range(args.RUNS):   # one child process per CV repeat (see run_job)
        run_job(cfg, K_Fold, run=run, do_cv=True, verbose=2, on_fold=log_fold)

    summary = {**base, "k": K_Fold, "runs": args.RUNS, **summarise(fold_rows)}
    if args.RUNS > 1:
        # std across the repeated CV runs (mean of each run first), as asked in the revision
        run_means = [summarise([r for r in fold_rows if r["run"] == i + 1]) for i in range(args.RUNS)]
        for m in METRIC_NAMES:
            summary[f"{m}_std_runs"] = float(np.std([rm[f"{m}_mean"] for rm in run_means]))
    append_excel_row(excel, "CV_summary", summary)

    print("\nFinal Results (Mean ± Std over folds):")
    for m in METRIC_NAMES:
        print(f"{m}: {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}")
    print(f"\nResults saved to {excel}")

    if args.csv_path:   # legacy one-line CSV summary
        pd.DataFrame([summary]).to_csv(args.csv_path, mode='a', index=False, header=not os.path.exists(args.csv_path))


def run_ind(x_train, y_train, x_test, y_test):
    cfg = config_from_args("ind")
    excel = args.excel_path or "MCNN_ind_report.xlsx"
    tag = config_tag(cfg)

    scaler, x_train, x_test = normalise(cfg, x_train, x_test)
    time_log("Start Model Train")
    model = train_model(x_train, y_train, cfg)
    time_log("Start Model Test")
    result, curves = model_test(model, x_test, y_test)

    save_curves(curves, args.pkl_dir, tag)                    # PKL only in independent mode
    append_excel_row(excel, "Independent", {"exp_id": datetime.now().strftime("%Y%m%d_%H%M%S"), **cfg, **result})
    if args.save_model:
        save_model(model, cfg, result, args.model_dir, tag, scaler)
    free_model(model)
    print(f"Results saved to {excel}")


def run_search(x_train, y_train, x_test, y_test):
    import optuna

    study_name = args.STUDY or f"MCNN_{DATA_FEATURE}"
    excel = args.excel_path or f"MCNN_search_{study_name}.xlsx"
    out_dir = os.path.join(args.model_dir, study_name)
    obj = args.OBJECTIVE
    base = config_from_args("search")
    S = SEARCH_SPACE

    def suggest(trial):
        n_win = trial.suggest_int("n_windows", *S["n_windows"])
        windows = sorted({trial.suggest_int(f"window_{i}", *S["window"]) for i in range(n_win)})
        return {**base,
                "window_sizes": windows,
                "num_filters": trial.suggest_categorical("num_filters", S["num_filters"]),
                "num_hidden": trial.suggest_categorical("num_hidden", S["num_hidden"]),
                "num_hidden_layers": trial.suggest_int("num_hidden_layers", *S["num_hidden_layers"]),
                "dropout": trial.suggest_float("dropout", S["dropout"][0], S["dropout"][1], step=S["dropout"][2]),
                "lr": trial.suggest_float("lr", *S["lr"], log=True),
                "batch_size": trial.suggest_categorical("batch_size", S["batch_size"]),
                "loss": trial.suggest_categorical("loss", S["loss"])}

    def objective(trial):
        cfg = suggest(trial)
        t0 = time.time()
        base_row = {"study": study_name, "trial": trial.number, **cfg}
        print(f"\n===== Trial {trial.number}: {config_tag(cfg)} =====")

        # folds are collected here so a pruned trial still reports the folds it finished
        fold_rows = []

        def log_fold(row):
            fold_rows.append(row)
            append_excel_row(excel, "Folds", {**base_row, **row})

        # the trial's model is written here first and kept only if the trial becomes the best
        os.makedirs(out_dir, exist_ok=True)
        tmp_weights = os.path.join(out_dir, "_current_trial.weights.h5") if args.save_model else None

        state, ind = "COMPLETE", None
        try:
            # CV folds, then (if enabled) the independent test, in one child process
            ind = run_job(cfg, K_Fold, run=0, do_cv=True, do_ind=args.search_ind, weights_path=tmp_weights,
                          on_fold=log_fold, trial=trial, objective=obj)
        except optuna.TrialPruned:
            state = "PRUNED"
        except Exception as e:
            # a crashing configuration is logged and skipped instead of stopping the whole search
            append_excel_row(excel, "Trials", {"state": "FAILED", **base_row, "folds_done": len(fold_rows),
                                               "error": str(e)[:300]})
            raise
        score = float(np.mean([r[obj] for r in fold_rows]))

        row = {"state": state, f"CV_{obj}": score, **base_row, "folds_done": len(fold_rows),
               **summarise(fold_rows, prefix="CV_")}

        if ind is not None:
            # Independent test is only reported here; the search is driven by the CV score
            result, curves = ind
            row.update({f"IND_{k}": v for k, v in result.items()})

            completed = [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed or score > max(t.value for t in completed):
                # new best by CV -> keep its curves (and model, if asked)
                tag = f"best_{study_name}"
                save_curves(curves, out_dir, tag)
                if tmp_weights:
                    best_weights = os.path.join(out_dir, f"{tag}.weights.h5")
                    os.replace(tmp_weights, best_weights)
                    if os.path.exists(scaler_path(tmp_weights)):          # -norm zscore statistics
                        os.replace(scaler_path(tmp_weights), scaler_path(best_weights))
                    elif os.path.exists(scaler_path(best_weights)):       # stale scaler of an older best
                        os.remove(scaler_path(best_weights))
                    write_model_config({**cfg, "trial": trial.number}, result, out_dir, tag)
                row["new_best"] = True
        for leftover in ([tmp_weights, scaler_path(tmp_weights)] if tmp_weights else []):
            if os.path.exists(leftover):
                os.remove(leftover)

        row["minutes"] = round((time.time() - t0) / 60, 2)
        append_excel_row(excel, "Trials", row)
        if state == "PRUNED":
            raise optuna.TrialPruned()
        return score

    # SQLite storage -> the search can be stopped and resumed with the same command
    study = optuna.create_study(
        study_name=study_name, direction="maximize", load_if_exists=True,
        storage=f"sqlite:///{study_name}.db",
        sampler=optuna.samplers.TPESampler(seed=args.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    # catch=(Exception,): a failed trial is marked FAIL in the study and the search continues
    study.optimize(objective, n_trials=args.N_TRIALS, gc_after_trial=True, catch=(Exception,))

    best = study.best_trial
    print(f"\nBest trial {best.number}: CV {obj} = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    append_excel_row(excel, "Best", {"time": datetime.now().isoformat(timespec="seconds"),
                                     "trial": best.number, f"CV_{obj}": best.value,
                                     "n_trials_total": len(study.trials), **best.params})
    print(f"\nAll trials saved to {excel}; best curves/model in {out_dir}")


"====================================== main ======================================"
if __name__ == "__main__":
    if args.SEED is not None:
        tf.keras.utils.set_random_seed(args.SEED)

    print("FEATURE:", DATA_FEATURE, "| MODE:", MODE)

    # [MOD] load the feature set given by -df (was hard-coded to "TrimoPGLM")
    x_train, y_train, x_test, y_test = load_data.MCNN_data_load(feature_type=DATA_FEATURE, class_0_types=["neg"], class_1_types=["pos"])
    NUM_FEATURE = x_train.shape[-1]
    if args.NUM_FEATURE and args.NUM_FEATURE != NUM_FEATURE:
        print(f"WARNING: -nf {args.NUM_FEATURE} ignored, the {DATA_FEATURE} data has {NUM_FEATURE} features")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    if MODE == 'cross':
        run_cross(x_train, y_train)
    elif MODE == 'ind':
        run_ind(x_train, y_train, x_test, y_test)
    elif MODE == 'search':
        run_search(x_train, y_train, x_test, y_test)
