"""Doc dataset iBCE-EL tu globe/B_CELL/dataset/.

KHONG cat chuoi, khong dat MAXSEQ: feature cua iBCE-EL la composition nen chap nhan
do dai bat ky. Day la diem khac voi pipeline MCNN/fusion_cl (cat o 25 residue, am tham
lam hong 34 chuoi dai trong Ind-negative).
"""
import numpy as np

DIR = "../dataset"
SPLITS = {"train": [("B-positive.txt", 1), ("B-negative.txt", 0)],
          "ind":   [("Ind-positive.txt", 1), ("Ind-negative.txt", 0)]}


def read_fasta(path):
    seqs, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if cur is not None:
                seqs.append(cur)
            cur = ""
        elif cur is not None:
            cur += line
    if cur:
        seqs.append(cur)
    return seqs


def load(split):
    S, y = [], []
    for fn, lab in SPLITS[split]:
        s = read_fasta(f"{DIR}/{fn}")
        S += s
        y += [lab] * len(s)
    return S, np.array(y)
