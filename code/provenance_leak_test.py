"""[MOD] How much of the label can be recovered from IEDB provenance alone?

For each peptide in the iBCE-EL split we look up its source antigen
(parent_source_antigen_iri) and its source organism (source_organism_name) in
dataset/iedb_bcell_raw.csv, and use nothing else - no residue of the sequence
itself enters the score.

Procedure
    1. join Train_*/Ind_*.fasta to the raw IEDB table by linear_sequence.
       The fasta headers carry structure_id, not bcell_id (only ~2% of them
       match bcell_id), so the sequence is the only reliable join key. A
       sequence can appear in several raw rows; we take its most frequent
       accession / organism.
    2. on the TRAINING split only, count positives and negatives per key.
    3. score every INDEPENDENT-TEST peptide with the smoothed positive rate of
       its key,  (pos + m * prior) / (pos + neg + m),  m = 5.
    4. report AUC of that single score against the true label.

The training labels are the only labels used, so this is an honest predictor -
which is the point: it is learnable, and it is pure provenance. Any feature
derived from the source protein (GO annotations included) inherits it.

    python code/provenance_leak_test.py
"""
import collections
import csv
import os

from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "dataset")
RAW = os.path.join(D, "iedb_bcell_raw.csv")
SMOOTH = 5.0                                   # pseudo-counts towards the prior

csv.field_size_limit(10 ** 7)

FIELDS = [("source protein", "parent_source_antigen_iri"),
          ("source organism", "source_organism_name")]

# ---------------------------------------------------------------- 1. join
seq2key = {name: collections.defaultdict(collections.Counter) for name, _ in FIELDS}
with open(RAW, newline="") as f:
    for row in csv.DictReader(f):
        s = row["linear_sequence"]
        for name, col in FIELDS:
            seq2key[name][s][row[col].strip() or "NA"] += 1

def load(split):
    path = os.path.join(D, f"{split}.fasta")
    return [l.strip() for l in open(path) if not l.startswith(">")]

SPLITS = {s: load(s) for s in ["Train_pos", "Train_neg", "Ind_pos", "Ind_neg"]}
prior = len(SPLITS["Train_pos"]) / (len(SPLITS["Train_pos"]) + len(SPLITS["Train_neg"]))

print(f"train {len(SPLITS['Train_pos'])} pos / {len(SPLITS['Train_neg'])} neg   "
      f"ind {len(SPLITS['Ind_pos'])} pos / {len(SPLITS['Ind_neg'])} neg   "
      f"prior = {prior:.4f}")

for name, _ in FIELDS:
    table = seq2key[name]

    def key(s):
        c = table.get(s)
        return c.most_common(1)[0][0] if c else "NA"

    # -------------------------------------------------------- 2. fit on train
    cnt = collections.defaultdict(lambda: [0, 0])          # key -> [neg, pos]
    for lab, split in [(1, "Train_pos"), (0, "Train_neg")]:
        for s in SPLITS[split]:
            cnt[key(s)][lab] += 1

    # -------------------------------------------------- 3. score the test set
    y, score, unseen = [], [], 0
    for lab, split in [(1, "Ind_pos"), (0, "Ind_neg")]:
        for s in SPLITS[split]:
            k = key(s)
            if k not in cnt:
                unseen += 1
            neg, pos = cnt[k]
            y.append(lab)
            score.append((pos + SMOOTH * prior) / (pos + neg + SMOOTH))

    # ------------------------------------------------------------ 4. report
    print(f"{name:16s} AUC on independent test = {roc_auc_score(y, score):.4f}"
          f"   ({len(cnt)} distinct keys in train, {unseen}/{len(y)} test "
          f"peptides from a key never seen in train)")
