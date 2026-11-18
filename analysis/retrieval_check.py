"""[MOD] Retrieval sanity check for the RAG fused features.

Recomputes, in float64 and directly from the stored neighbour indices, the
distance between every independent-test peptide and each of its 5 retrieved
neighbours. Two things are being verified:

  1. no evaluation peptide is present in the knowledge graph (a distance of
     ~0 to its own nearest neighbour would mean it is),
  2. the distance scale of the retrieved neighbourhood.

Results are merged into analysis/analysis_stats.json under "retrieval" so the
report quotes measured numbers instead of hardcoded ones.

Run with the esmc environment (it only needs numpy):
    /home/b1607/anaconda3/envs/esmc/bin/python retrieval_check.py
"""
import json
import os

import numpy as np

from common import ROOT

D = os.path.join(ROOT, "dataset", "ESMC")
R = os.path.join(D, "RAG")
STATS_PATH = os.path.join(ROOT, "analysis", "analysis_stats.json")

out = {}
for side, kg in [("pos", "kg_pos_clean"), ("neg", "kg_neg_clean")]:
    q = np.load(f"{D}/Ind_{side}.npy", mmap_mode="r")
    idx = np.load(f"{R}/Ind_{side}_nn_idx.npy")
    kgm = np.load(f"{R}/{kg}.npy", mmap_mode="r")
    Nq, k = idx.shape
    F = q.shape[2] * q.shape[3]

    # [MOD] index rows are sorted before the memmap read, which is far faster
    # but loses the column ordering. Every statistic below is therefore
    # order-independent (per-row min or per-row mean over the k neighbours).
    d = np.empty((Nq, k))
    for i in range(Nq):
        qi = np.asarray(q[i]).astype(np.float64).reshape(-1)
        rows = np.asarray(kgm[np.sort(idx[i])]).astype(np.float64).reshape(k, F)
        d[i] = np.linalg.norm(rows - qi, axis=1)

    nn = d.min(1)
    out[side] = {
        "n_query": int(Nq), "k": int(k), "kg_size": int(kgm.shape[0]),
        "nn_min": float(nn.min()), "nn_median": float(np.median(nn)),
        "nn_max": float(nn.max()),
        "mean5_median": float(np.median(d.mean(1))),
        "n_duplicate": int((nn < 1e-3).sum()),
    }
    print(side, out[side], flush=True)

stats = json.load(open(STATS_PATH)) if os.path.exists(STATS_PATH) else {}
stats["retrieval"] = out
json.dump(stats, open(STATS_PATH, "w"), indent=2)
print("\nmerged into analysis/analysis_stats.json under 'retrieval'")
