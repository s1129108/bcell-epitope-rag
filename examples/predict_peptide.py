#!/usr/bin/env python3
"""End-to-end demo: embed one peptide, fuse it with its retrieved neighbours and
their knowledge-graph context, and classify the result with the MCNN.

    python examples/predict_peptide.py --sequence SAPDTRPAPGSTAPPAHGVT

Step 2 is skipped unless --store is given, because the retrieval store is far too
large to ship with the repository. See the README for how to build it.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

MAXSEQ = 25
DIM = 1152
K = 5
FIELDS = ["parent_source_antigen_iri", "parent_source_antigen_name",
          "source_organism_name"]
VOCAB = {"parent_source_antigen_iri": 512,
         "parent_source_antigen_name": 512,
         "source_organism_name": 128}
BETA = 0.5
PROJ_SEED = 0


# --------------------------------------------------------------- 1. embed
def embed(sequence, model_name="esmc_600m"):
    """(L, DIM) float32 per-residue ESM C embedding, BOS/EOS removed."""
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        tokens = model.encode(ESMProtein(sequence=sequence))
        out = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
    emb = out.embeddings[0, 1:-1].float().cpu().numpy()
    assert emb.shape[0] == len(sequence), (emb.shape, len(sequence))
    return emb


def pad_or_cut(emb):
    """(L, DIM) -> (1, MAXSEQ, DIM); longer peptides truncated, shorter zero-padded."""
    x = np.zeros((1, MAXSEQ, DIM), np.float32)
    n = min(len(emb), MAXSEQ)
    x[0, :n] = emb[:n]
    return x


# ------------------------------------------------- 2. RAG + KG fusion
class ContextEncoder:
    """Provenance of a store item -> frozen random projection -> (DIM,) vector.

    The vocabularies and the projection are rebuilt deterministically from the store,
    so the encoder produced here is identical to the one used during training.
    """

    def __init__(self, store):
        import collections
        csv.field_size_limit(10 ** 7)

        self.sequences = [l.strip() for l in open(os.path.join(store, "store.fasta"))
                          if not l.startswith(">")]
        self.meta = {}
        with open(os.path.join(store, "provenance.csv"), newline="") as f:
            for row in csv.DictReader(f):
                s = row["linear_sequence"]
                if s not in self.meta:
                    self.meta[s] = tuple(row[c].strip() or "NA" for c in FIELDS)

        counts = [collections.Counter() for _ in FIELDS]
        for s in self.sequences:
            v = self.meta.get(s)
            if v:
                for j, val in enumerate(v):
                    counts[j][val] += 1

        self.vocab, self.offset, self.dim = [], [], 0
        for j, field in enumerate(FIELDS):
            keep = [k for k, _ in counts[j].most_common(VOCAB[field])]
            self.vocab.append({k: i for i, k in enumerate(keep)})
            self.offset.append(self.dim)
            self.dim += len(keep) + 1              # + 1 for "other"/missing

        rng = np.random.default_rng(PROJ_SEED)
        self.P = rng.normal(0.0, 1.0 / np.sqrt(self.dim),
                            size=(self.dim, DIM)).astype(np.float32)
        self.rng = rng

    def __call__(self, sequence):
        v = np.zeros(self.dim, np.float32)
        m = self.meta.get(sequence)
        for j in range(len(FIELDS)):
            val = m[j] if m else "NA"
            k = self.vocab[j].get(val)
            v[self.offset[j] + (k if k is not None else len(self.vocab[j]))] = 1.0
        return v @ self.P

    def scale(self, store_array):
        """Context norm set to BETA times the mean residue norm of the store."""
        probe = self.rng.choice(store_array.shape[0], 400, replace=False)
        sample = np.asarray(store_array[np.sort(probe), 0]).astype(np.float32)
        real = np.abs(sample).sum(-1) > 0
        res_norm = float(np.linalg.norm(sample[real], axis=-1).mean())
        ctx = np.stack([self(self.sequences[i]) for i in probe])
        ctx_norm = float(np.linalg.norm(ctx, axis=-1).mean())
        return BETA * res_norm / max(ctx_norm, 1e-8)


def retrieve(query, store_array, k=K):
    """Indices of the k nearest store items, exact L2 on the flattened embedding."""
    q = query.reshape(1, -1).astype(np.float32)
    best_d, best_i = None, None
    for start in range(0, store_array.shape[0], 4096):
        block = np.asarray(store_array[start:start + 4096, 0]).astype(np.float32)
        d = np.linalg.norm(block.reshape(len(block), -1) - q, axis=1)
        idx = np.arange(start, start + len(block))
        if best_d is None:
            best_d, best_i = d, idx
        else:
            best_d = np.concatenate([best_d, d])
            best_i = np.concatenate([best_i, idx])
        keep = np.argsort(best_d)[:k]
        best_d, best_i = best_d[keep], best_i[keep]
    return best_i, best_d


def fuse(query, store_array, neighbours, context, scale, ratio):
    """query and the mean of the context-carrying neighbours, mixed convexly."""
    acc = np.zeros((MAXSEQ, DIM), np.float32)
    for g in neighbours:
        e = np.asarray(store_array[int(g), 0]).astype(np.float32)
        # context is added only where the item has a real residue, so the zero
        # padding of a store item stays zero after fusion
        mask = (np.abs(e).sum(-1) > 0)[:, None]
        acc += e + context(context.sequences[int(g)]) * scale * mask
    acc /= len(neighbours)
    return (ratio * query[0] + (1.0 - ratio) * acc)[None]


# ------------------------------------------------------------ 3. classify
def classify(x, config_path):
    """Probability that the peptide is an epitope, from a saved MCNN checkpoint."""
    from MCNN import load_saved_model

    model, _ = load_saved_model(config_path)
    probs = model.predict(x[None].astype(np.float32), verbose=0)
    return float(probs[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True, help="peptide, 5-25 residues")
    ap.add_argument("--store", default=None,
                    help="retrieval store directory (store.npy, store.fasta, "
                         "provenance.csv); without it the peptide is classified "
                         "from its own embedding alone")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--model", default=os.path.join(
        ROOT, "saved_models", "MCNN_ESMC", "best_MCNN_ESMC.config.json"))
    ap.add_argument("--esmc", default="esmc_600m")
    a = ap.parse_args()

    seq = a.sequence.strip().upper()
    print(f"peptide            {seq}  ({len(seq)} residues)")

    # 1. embed
    x = pad_or_cut(embed(seq, a.esmc))
    print(f"ESM C embedding    {x.shape}")

    # 2. retrieve and fuse
    if a.store:
        store_array = np.load(os.path.join(a.store, "store.npy"), mmap_mode="r")
        context = ContextEncoder(a.store)
        scale = context.scale(store_array)
        neighbours, distances = retrieve(x, store_array)
        print(f"context scale      {scale:.3f}")
        print("nearest neighbours")
        for g, d in zip(neighbours, distances):
            print(f"  d = {d:7.3f}   {context.sequences[int(g)]}")
        x = fuse(x, store_array, neighbours, context, scale, a.ratio)
        print(f"fused embedding    {x.shape}")
    else:
        print("no --store given, classifying the unfused embedding")

    # 3. classify
    p = classify(x, a.model)
    print(f"\nP(epitope)         {p:.4f}")
    print(f"prediction         {'epitope' if p >= 0.5 else 'non-epitope'}")


if __name__ == "__main__":
    main()
