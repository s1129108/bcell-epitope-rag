"""[MOD] Build fused embeddings at an arbitrary query ratio from the stored
neighbour indices, without re-running retrieval.

    fused = r * query + (1 - r) * mean(top-k neighbours)

Retrieval does not depend on r, so the same *_nn_idx.npy files serve every
ratio. Verified: the index files saved alongside the 9:1 and 3:7 runs are
byte-identical to the 1:1 ones.

NOTE on the pre-existing files. dataset/ESMC/RAG/Train_*_fused_9_1.npy and
Train_*_fused_3_7.npy are byte-identical to Train_*_fused_1_1.npy: they were
generated without passing -ratio, so they all carry the default r = 0.5 and
differ in filename only. This script writes to *_r09 / *_r03 instead, so the
mislabelled files are never picked up by mistake.

Usage:
    python make_ratio.py 0.9 r09
    python make_ratio.py 0.3 r03
"""
import os
import sys

import numpy as np

r = float(sys.argv[1])
tag = sys.argv[2]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "dataset", "ESMC")
R = os.path.join(D, "RAG")

KG = {"pos": np.load(f"{R}/kg_pos_clean.npy", mmap_mode="r"),
      "neg": np.load(f"{R}/kg_neg_clean.npy", mmap_mode="r")}

for split in ["Ind_pos", "Ind_neg", "Train_pos", "Train_neg"]:
    side = "pos" if split.endswith("pos") else "neg"
    q = np.load(f"{D}/{split}.npy", mmap_mode="r")
    idx = np.load(f"{R}/{split}_nn_idx.npy")
    kg = KG[side]
    assert len(idx) == len(q), f"{split}: {len(idx)} idx vs {len(q)} queries"
    out_path = f"{R}/{split}_fused_{tag}.npy"
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                    shape=q.shape)
    for i in range(len(q)):
        rows = np.asarray(kg[np.sort(idx[i])]).astype(np.float32)
        out[i] = (np.asarray(q[i]).astype(np.float32) * r
                  + rows.mean(0) * (1.0 - r)).astype(np.float16)
    out.flush()
    del out
    print(f"{split}: {q.shape}  r = {r}  -> {out_path}", flush=True)
print("done")
