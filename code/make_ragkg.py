"""[MOD] RAG + KG context: attach IEDB provenance to every retrieved database
item, combine it with that item's ESM C embedding, and fuse into the query.

Retrieval is label-partitioned, as requested: a positive query retrieves from
kg_pos_clean (19,003 peptides), a negative query from kg_neg_clean (222,569).
The neighbour indices are the ones already stored in dataset/ESMC/RAG/
*_nn_idx.npy, so nothing is retrieved again.

Context attached to each database item (joined to dataset/iedb_bcell_raw.csv by
linear_sequence, which matches 100% of the KG items):
    parent_source_antigen_iri    the source protein accession
    parent_source_antigen_name   the antigen
    source_organism_name         the source organism
Each field is one-hot over its most frequent VOCAB values plus an "other" bin,
the three blocks are concatenated, and the result is projected to 1152
dimensions by a fixed seeded Gaussian projection (seed 0, so the same context
always maps to the same vector). The projection is scaled so that the context
vector's norm is BETA times the mean residue-embedding norm of the database,
then added to the item's real residue positions - zero padding stays zero, the
way the existing fusion expects.

    kg_rep = esmc_kg + BETA_scaled * context
    fused  = r * query + (1 - r) * mean(top-5 kg_rep)          r = -ratio

Written (r = 0.3 by default), into dataset/ESMC_RAGKG/:
    Train_{pos,neg}.npy   fused, label-partitioned
    Ind_{pos,neg}.npy     fused the same way as training

Direction 2 of the experiment (test on raw ESM C) needs no file: it reuses
dataset/ESMC/Ind_{pos,neg}.npy unchanged.

Note on what the provenance fields are. On this benchmark the smoothed positive
rate of a peptide's source protein alone scores AUC 0.8763 on the independent
test set (code/provenance_leak_test.py), so these three fields are not neutral
side information - they carry much of the label. That is a property of the
benchmark, recorded here so the numbers are read for what they are.

    python code/make_ragkg.py -ratio 0.3
"""
import argparse
import collections
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "dataset")
RAG = os.path.join(D, "ESMC", "RAG")
SPLITS = [("Train_pos", 0), ("Train_neg", 1), ("Ind_pos", 0), ("Ind_neg", 1)]
FIELDS = ["parent_source_antigen_iri", "parent_source_antigen_name",
          "source_organism_name"]
VOCAB = {"parent_source_antigen_iri": 512,
         "parent_source_antigen_name": 512,
         "source_organism_name": 128}
BETA = 0.5          # context norm as a fraction of the mean residue norm
PROJ_SEED = 0

ap = argparse.ArgumentParser()
ap.add_argument("-ratio", type=float, default=0.3, help="weight r on the query")
ap.add_argument("-out", default=os.path.join(D, "ESMC_RAGKG"))
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
csv.field_size_limit(10 ** 7)

# ------------------------------------------- 1. provenance of each KG item
kg_seq = {}
for part, name in [(0, "kg_pos_clean"), (1, "kg_neg_clean")]:
    rows = [l.strip() for l in open(f"{D}/{name}.fasta") if not l.startswith(">")]
    kg_seq[part] = rows
    print(f"{name}: {len(rows)} items")

meta = {}
with open(f"{D}/iedb_bcell_raw.csv", newline="") as f:
    for row in csv.DictReader(f):
        s = row["linear_sequence"]
        if s not in meta:
            meta[s] = tuple(row[c].strip() or "NA" for c in FIELDS)

missing = sum(1 for p in kg_seq for s in kg_seq[p] if s not in meta)
print(f"KG items with no provenance row: {missing}")

# ------------------------------------------------------- 2. the vocabularies
counts = [collections.Counter() for _ in FIELDS]
for p in kg_seq:
    for s in kg_seq[p]:
        v = meta.get(s)
        if v:
            for j, val in enumerate(v):
                counts[j][val] += 1

vocab, offset, dim = [], [], 0
for j, c in enumerate(FIELDS):
    keep = [k for k, _ in counts[j].most_common(VOCAB[c])]
    vocab.append({k: i for i, k in enumerate(keep)})
    offset.append(dim)
    dim += len(keep) + 1                       # + 1 for "other"/missing
    print(f"  {c}: {len(counts[j])} distinct, keeping {len(keep)}")
print(f"one-hot context dimension: {dim}")

rng = np.random.default_rng(PROJ_SEED)
P = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, 1152)).astype(np.float32)


def context_vec(seq):
    """One-hot provenance -> fixed random projection -> (1152,) float32."""
    v = np.zeros(dim, np.float32)
    m = meta.get(seq)
    for j in range(len(FIELDS)):
        val = m[j] if m else "NA"
        k = vocab[j].get(val)
        v[offset[j] + (k if k is not None else len(vocab[j]))] = 1.0
    return v @ P


# --------------------------------------------------- 3. scale of the context
kg = {0: np.load(f"{RAG}/kg_pos_clean.npy", mmap_mode="r"),
      1: np.load(f"{RAG}/kg_neg_clean.npy", mmap_mode="r")}
probe = rng.choice(kg[1].shape[0], 400, replace=False)
sample = np.asarray(kg[1][np.sort(probe), 0]).astype(np.float32)
real = np.abs(sample).sum(-1) > 0
res_norm = float(np.linalg.norm(sample[real], axis=-1).mean())
ctx_probe = np.stack([context_vec(kg_seq[1][i]) for i in probe])
ctx_norm = float(np.linalg.norm(ctx_probe, axis=-1).mean())
SCALE = BETA * res_norm / max(ctx_norm, 1e-8)
print(f"mean residue norm {res_norm:.3f}, mean context norm {ctx_norm:.3f}, "
      f"scale {SCALE:.3f} (beta = {BETA})")

# ------------------------------------------------------------- 4. fuse
r = A.ratio
for split, part in SPLITS:
    q = np.load(f"{D}/ESMC/{split}.npy", mmap_mode="r")
    idx = np.load(f"{RAG}/{split}_nn_idx.npy")
    assert len(idx) == len(q), (split, idx.shape, q.shape)
    out = np.lib.format.open_memmap(f"{A.out}/{split}.npy", mode="w+",
                                    dtype=np.float16, shape=q.shape)
    src, seqs = kg[part], kg_seq[part]
    cache = {}
    for i in range(len(q)):
        acc = np.zeros((q.shape[2], q.shape[3]), np.float32)
        for g in idx[i]:
            g = int(g)
            e = np.asarray(src[g, 0]).astype(np.float32)
            if g not in cache:
                cache[g] = context_vec(seqs[g]) * SCALE
            # context only where the item has a real residue, so that the
            # zero padding of the database item stays zero after fusion
            mask = (np.abs(e).sum(-1) > 0)[:, None]
            acc += e + cache[g] * mask
        acc /= idx.shape[1]
        out[i, 0] = (r * np.asarray(q[i, 0]).astype(np.float32)
                     + (1.0 - r) * acc).astype(np.float16)
        if i % 2000 == 0:
            print(f"  {split} {i}/{len(q)}", flush=True)
    out.flush()
    del out
    print(f"{split}: {q.shape} -> {A.out}/{split}.npy  (r = {r})")

with open(f"{A.out}/ratio.json", "w") as f:
    f.write('{"ratio": %r, "beta": %r, "proj_seed": %d, "vocab": %r}\n'
            % (r, BETA, PROJ_SEED, VOCAB))
print("done")
