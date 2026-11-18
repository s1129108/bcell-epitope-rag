"""[MOD] Re-split the iBCE-EL peptides so that no source protein is shared
between train and test, and build a matched random split as the control.

Why
    On the original split, the smoothed positive rate of a peptide's source
    antigen alone predicts the label at AUC 0.8763 on the independent test set
    (code/provenance_leak_test.py), because train and test draw from the same
    source proteins. Any protein-level feature - GO annotations included -
    inherits that shortcut. Grouping the split by source protein removes it.

What this writes (nothing existing is touched)
    dataset/ESMC_PDIS/     source-protein-disjoint split
    dataset/ESMC_RSPLIT/   random split of the same pool, same sizes
        {Train,Ind}_{pos,neg}.fasta   peptides
        {Train,Ind}_{pos,neg}.npy     ESM C embeddings, re-indexed (no re-embed)
        split.json                    sizes and the provenance control AUC

The two splits differ only in how the pool is cut, so comparing a model across
them isolates the effect of the provenance shortcut from the effect of
re-splitting at all.

Groups
    parent_source_antigen_iri from dataset/iedb_bcell_raw.csv, joined by
    linear_sequence (the fasta headers carry structure_id, not bcell_id).
    A peptide whose accession is missing becomes its own singleton group, so it
    can never bind a train and a test peptide together.

    python code/make_protein_disjoint_split.py
"""
import collections
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "dataset")
EMB = os.path.join(D, "ESMC")
SPLITS = ["Train_pos", "Train_neg", "Ind_pos", "Ind_neg"]
SEED = 0
N_FOLDS = 5                      # fold 0 becomes the test set -> ~20%, as before
SMOOTH = 5.0

csv.field_size_limit(10 ** 7)

# ------------------------------------------------------------------ 1. pool
seq, lab = [], []
emb_src = []                     # (file, row) so embeddings can be re-indexed
for s in SPLITS:
    lines = [l.strip() for l in open(f"{D}/{s}.fasta") if not l.startswith(">")]
    a = np.load(f"{EMB}/{s}.npy", mmap_mode="r")
    assert len(a) == len(lines), (s, len(a), len(lines))
    for i, q in enumerate(lines):
        seq.append(q)
        lab.append(1 if s.endswith("pos") else 0)
        emb_src.append((s, i))
y = np.array(lab)
print(f"pool: {len(seq)} peptides, {y.sum()} positive, {len(set(seq))} unique")

# ------------------------------------------------- 2. group = source protein
acc_of = collections.defaultdict(collections.Counter)
with open(f"{D}/iedb_bcell_raw.csv", newline="") as f:
    for row in csv.DictReader(f):
        acc_of[row["linear_sequence"]][row["parent_source_antigen_iri"].strip()] += 1

groups, singleton = [], 0
for i, q in enumerate(seq):
    c = acc_of.get(q)
    a = c.most_common(1)[0][0] if c else ""
    if not a:                                    # unknown provenance
        a = f"__singleton_{i}"
        singleton += 1
    groups.append(a)
groups = np.array(groups)
print(f"groups: {len(set(groups))} distinct ({singleton} peptides with no accession)")

# ---------------------------------------------------------------- 3. the cuts
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
test_pdis = next(sgkf.split(seq, y, groups))[1]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
test_rand = next(skf.split(seq, y))[1]

CUTS = {"ESMC_PDIS": test_pdis, "ESMC_RSPLIT": test_rand}


def provenance_auc(tr, te):
    """AUC of the smoothed per-protein positive rate, fit on tr, scored on te."""
    prior = float(y[tr].mean())
    cnt = collections.defaultdict(lambda: [0, 0])
    for i in tr:
        cnt[groups[i]][y[i]] += 1
    seen = set(cnt)                   # [MOD] snapshot: cnt is a defaultdict, so
    score, unseen = [], 0             # reading cnt[g] below would insert g
    for i in te:
        g = groups[i]
        if g not in seen:
            unseen += 1
        neg, pos = cnt[g]
        score.append((pos + SMOOTH * prior) / (pos + neg + SMOOTH))
    return roc_auc_score(y[te], score), unseen


# ------------------------------------------------------- 4. write both splits
for name, te_idx in CUTS.items():
    out = os.path.join(D, name)
    os.makedirs(out, exist_ok=True)
    te = np.zeros(len(seq), bool)
    te[te_idx] = True
    tr_idx = np.where(~te)[0]

    shared = set(groups[te_idx]) & set(groups[tr_idx])
    auc, unseen = provenance_auc(tr_idx, te_idx)
    meta = {"seed": SEED, "n_folds": N_FOLDS,
            "proteins_shared_between_train_and_test": len(shared),
            "provenance_control_auc": round(float(auc), 4),
            "test_peptides_from_unseen_protein": f"{unseen}/{len(te_idx)}"}

    for part, idx in [("Train", tr_idx), ("Ind", te_idx)]:
        for cls, want in [("pos", 1), ("neg", 0)]:
            rows = [i for i in idx if y[i] == want]
            meta[f"{part}_{cls}"] = len(rows)

            with open(f"{out}/{part}_{cls}.fasta", "w") as f:
                for i in rows:
                    f.write(f">{'Positive' if want else 'Negative'}"
                            f"_{part.lower()}_{i}|{groups[i]}\n{seq[i]}\n")

            # re-index the existing ESM C embeddings; nothing is re-embedded
            o = np.lib.format.open_memmap(
                f"{out}/{part}_{cls}.npy", mode="w+", dtype=np.float16,
                shape=(len(rows), 1, 25, 1152))
            by_file = collections.defaultdict(list)
            for j, i in enumerate(rows):
                f_, r_ = emb_src[i]
                by_file[f_].append((j, r_))
            for f_, pairs in by_file.items():
                a = np.load(f"{EMB}/{f_}.npy", mmap_mode="r")
                for j, r_ in pairs:
                    o[j] = a[r_]
            o.flush()
            del o

    with open(f"{out}/split.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{name}: {meta}")
