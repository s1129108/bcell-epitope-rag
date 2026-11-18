"""[MOD] Retrieve and fuse the breast-cancer validation peptides.

Same retrieval as the benchmark: L2 distance on the flattened 25 x 1152 = 28,800
dimensional embedding, top-5, label-partitioned (a positive query searches
kg_pos_clean, a negative query searches kg_neg_clean), then

    kg_rep = esmc_kg + SCALE * context(provenance)       as in code/make_ragkg.py
    fused  = r * query + (1 - r) * mean(top-5 kg_rep)    r = -ratio (default 0.3)

One difference from the benchmark, and it matters: 98 of the 269 validation
peptides sit in the KG database verbatim, so a plain search returns the query
itself at distance 0 and the fusion becomes a copy of the answer. Every row
listed in dataset/BC/meta.json["exclude"] is masked out before the top-5 is
taken, so no peptide can retrieve itself.

Writes dataset/BC/{BC_pos,BC_neg}_ragkg_r03.npy and the neighbour indices.

Run with the esmc env (it has the working torch):
    /home/b1607/anaconda3/envs/esmc/bin/python code/retrieve_bc.py -ratio 0.3
"""
import argparse
import collections
import csv
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "dataset")
RAG = os.path.join(D, "ESMC", "RAG")
OUT = os.path.join(D, "BC")
FIELDS = ["parent_source_antigen_iri", "parent_source_antigen_name",
          "source_organism_name"]
VOCAB = {"parent_source_antigen_iri": 512, "parent_source_antigen_name": 512,
         "source_organism_name": 128}
BETA, PROJ_SEED, K = 0.5, 0, 5

ap = argparse.ArgumentParser()
ap.add_argument("-ratio", type=float, default=0.3)
ap.add_argument("-chunk", type=int, default=4096)
A = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
meta = json.load(open(f"{OUT}/meta.json"))
csv.field_size_limit(10 ** 7)

# ------------------------------------------------ provenance, exactly as make_ragkg
kg_seq = {p: [l.strip() for l in open(f"{D}/{n}.fasta") if not l.startswith(">")]
          for p, n in [(0, "kg_pos_clean"), (1, "kg_neg_clean")]}
prov = {}
for row in csv.DictReader(open(f"{D}/iedb_bcell_raw.csv", newline="")):
    s = row["linear_sequence"]
    if s not in prov:
        prov[s] = tuple(row[c].strip() or "NA" for c in FIELDS)

counts = [collections.Counter() for _ in FIELDS]
for p in kg_seq:
    for s in kg_seq[p]:
        v = prov.get(s)
        if v:
            for j, val in enumerate(v):
                counts[j][val] += 1
vocab, offset, dim = [], [], 0
for j, c in enumerate(FIELDS):
    keep = [k for k, _ in counts[j].most_common(VOCAB[c])]
    vocab.append({k: i for i, k in enumerate(keep)})
    offset.append(dim)
    dim += len(keep) + 1
rng = np.random.default_rng(PROJ_SEED)
P = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, 1152)).astype(np.float32)


def context_vec(seq):
    v = np.zeros(dim, np.float32)
    m = prov.get(seq)
    for j in range(len(FIELDS)):
        k = vocab[j].get(m[j] if m else "NA")
        v[offset[j] + (k if k is not None else len(vocab[j]))] = 1.0
    return v @ P


kg = {0: np.load(f"{RAG}/kg_pos_clean.npy", mmap_mode="r"),
      1: np.load(f"{RAG}/kg_neg_clean.npy", mmap_mode="r")}
probe = rng.choice(kg[1].shape[0], 400, replace=False)
sample = np.asarray(kg[1][np.sort(probe), 0]).astype(np.float32)
res_norm = float(np.linalg.norm(sample[np.abs(sample).sum(-1) > 0], axis=-1).mean())
ctx_norm = float(np.linalg.norm(np.stack([context_vec(kg_seq[1][i]) for i in probe]),
                                axis=-1).mean())
SCALE = BETA * res_norm / max(ctx_norm, 1e-8)
print(f"context scale {SCALE:.3f} (same recipe as make_ragkg.py)")

# ------------------------------------------------------------------ retrieve
for cls, part in [("pos", 0), ("neg", 1)]:
    ids, seqs = [], []
    for line in open(f"{D}/BC_{cls}.fasta"):
        (ids if line.startswith(">") else seqs).append(line.strip())
    q = np.asarray(np.load(f"{D}/ESMC/BC_{cls}.npy")).astype(np.float32)
    n = len(q)
    assert n == len(seqs)
    flatq = torch.from_numpy(q.reshape(n, -1)).to(dev)
    qsq = (flatq ** 2).sum(1, keepdim=True)

    db = kg[part]
    best_d = torch.full((n, K), float("inf"), device=dev)
    best_i = torch.zeros((n, K), dtype=torch.long, device=dev)
    # rows this query may not retrieve: itself, wherever it sits in this partition
    banned = [[r for p, r in meta["exclude"].get(s, []) if p == part] for s in seqs]

    for s0 in range(0, db.shape[0], A.chunk):
        block = np.asarray(db[s0:s0 + A.chunk, 0]).astype(np.float32)
        b = torch.from_numpy(block.reshape(len(block), -1)).to(dev)
        d = qsq + (b ** 2).sum(1)[None, :] - 2.0 * flatq @ b.T      # (n, chunk)
        for i, rows in enumerate(banned):
            for r in rows:
                if s0 <= r < s0 + len(block):
                    d[i, r - s0] = float("inf")
        cat_d = torch.cat([best_d, d], 1)
        cat_i = torch.cat([best_i, torch.arange(s0, s0 + len(block), device=dev)
                           .expand(n, -1)], 1)
        best_d, sel = torch.topk(cat_d, K, dim=1, largest=False)
        best_i = torch.gather(cat_i, 1, sel)
        del b, d, cat_d, cat_i
    idx = best_i.cpu().numpy()
    dist = np.sqrt(np.maximum(best_d.cpu().numpy(), 0))
    print(f"BC_{cls}: {n} queries, nearest distance min {dist[:, 0].min():.3f} "
          f"median {np.median(dist[:, 0]):.3f}; "
          f"{sum(len(b) for b in banned)} self-rows excluded")

    fused = np.empty_like(q, dtype=np.float16)
    cache = {}
    for i in range(n):
        acc = np.zeros(q.shape[2:], np.float32)
        for g in idx[i]:
            g = int(g)
            e = np.asarray(db[g, 0]).astype(np.float32)
            if g not in cache:
                cache[g] = context_vec(kg_seq[part][g]) * SCALE
            acc += e + cache[g] * (np.abs(e).sum(-1) > 0)[:, None]
        acc /= K
        fused[i, 0] = (A.ratio * q[i, 0] + (1 - A.ratio) * acc).astype(np.float16)

    np.save(f"{OUT}/BC_{cls}_ragkg_r{str(A.ratio).replace('.', '')}.npy", fused)
    np.save(f"{OUT}/BC_{cls}_nn_idx.npy", idx)
    np.save(f"{OUT}/BC_{cls}_nn_dist.npy", dist)
print("done")
