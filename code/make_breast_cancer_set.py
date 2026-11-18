"""[MOD] Build a held-out breast-cancer antigen validation set from IEDB.

This is NOT an external dataset. It is carved out of the same IEDB B-cell table
the benchmark comes from, by selecting peptides whose source antigen is one of
the antigens studied in breast cancer, and then removing everything the model
has already seen. Calling it external would be wrong; it is a held-out
antigen-family subset, and that is how the manuscript reports it.

Antigen families kept (matched on parent_source_antigen_name / source_organism_name):
    CEA / carcinoembryonic antigen, MUC1 (mucin-1), BRCA1, BRCA2, p53,
    estrogen receptor, progesterone receptor, mammaglobin, Ki-67

Filters, in order:
    1. one label per sequence; sequences carrying both labels are dropped
       (101 of 376 - the same conflicting-label problem the benchmark has)
    2. sequences appearing in Train_* or Ind_* are dropped, so nothing the
       model trained or was selected on reappears here
    3. the remaining sequences are written out with the KG rows that contain
       them listed in exclude.json, so retrieval can skip a peptide finding
       itself in the database (98 of them are in the database verbatim)

Writes dataset/BC_pos.fasta, dataset/BC_neg.fasta, dataset/BC/meta.json.
Embeddings are a separate step (code/get_ESMCEmbedding.py --sets BC_pos BC_neg).

    python code/make_breast_cancer_set.py
"""
import collections
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "dataset")
OUT = os.path.join(D, "BC")
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10 ** 7)

FAMILIES = {
    "CEA": ["cea", "carcinoembryonic"],
    "MUC1": ["mucin-1", "muc1"],
    "BRCA": ["brca1", "brca2"],
    "p53": ["p53"],
    "ER_PR": ["estrogen receptor", "progesterone receptor"],
    "Mammaglobin": ["mammaglobin"],
    "Ki67": ["ki-67"],
}

# ------------------------------------------------------------- 1. collect
lab_of = collections.defaultdict(collections.Counter)
fam_of, antigen_of = {}, {}
for row in csv.DictReader(open(f"{D}/iedb_bcell_raw.csv", newline="")):
    name = (row["parent_source_antigen_name"] + " " + row["source_organism_name"]).lower()
    fam = next((k for k, pats in FAMILIES.items() if any(p in name for p in pats)), None)
    if fam is None:
        continue
    q = row["qualitative_measure"]
    lab = 1 if q.startswith("Positive") else (0 if q.startswith("Negative") else None)
    if lab is None:
        continue
    s = row["linear_sequence"]
    lab_of[s][lab] += 1
    fam_of[s] = fam
    antigen_of[s] = row["parent_source_antigen_name"]

conflict = [s for s, c in lab_of.items() if c[0] and c[1]]
clean = {s: (1 if c[1] else 0) for s, c in lab_of.items() if not (c[0] and c[1])}
print(f"matched {len(lab_of)} unique sequences; dropped {len(conflict)} with both labels")

# --------------------------------------------- 2. drop anything already seen
def fasta_seqs(name):
    return {l.strip() for l in open(f"{D}/{name}.fasta") if not l.startswith(">")}

seen = set()
for n in ["Train_pos", "Train_neg", "Ind_pos", "Ind_neg"]:
    seen |= fasta_seqs(n)
overlap = [s for s in clean if s in seen]
for s in overlap:
    del clean[s]
print(f"dropped {len(overlap)} sequences that appear in Train_*/Ind_*")

# ------------------------------- 3. where each one sits in the KG database
kg_rows = {}
for part, name in [(0, "kg_pos_clean"), (1, "kg_neg_clean")]:
    for i, s in enumerate(l.strip() for l in open(f"{D}/{name}.fasta") if not l.startswith(">")):
        if s in clean:
            kg_rows.setdefault(s, []).append([part, i])
print(f"{len(kg_rows)} of {len(clean)} are in the KG database verbatim "
      f"-> listed in exclude.json so retrieval cannot return the query itself")

order = sorted(clean, key=lambda s: (clean[s], fam_of[s], s))
for cls, want in [("pos", 1), ("neg", 0)]:
    rows = [s for s in order if clean[s] == want]
    with open(f"{D}/BC_{cls}.fasta", "w") as f:
        for i, s in enumerate(rows):
            f.write(f">BC_{cls}_{i}|{fam_of[s]}\n{s}\n")
    print(f"  BC_{cls}: {len(rows)}")

meta = {
    "families": {k: sum(1 for s in clean if fam_of[s] == k) for k in FAMILIES},
    "n_pos": sum(clean.values()),
    "n_neg": len(clean) - sum(clean.values()),
    "dropped_conflicting": len(conflict),
    "dropped_seen_in_benchmark": len(overlap),
    "exclude": {s: kg_rows.get(s, []) for s in order},
    "antigen": {s: antigen_of[s] for s in order},
    "label": clean,
}
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f, indent=1)
print(f"-> {OUT}/meta.json")
