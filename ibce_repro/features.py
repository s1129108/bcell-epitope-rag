"""Feature construction cua iBCE-EL — chi phan can cho EM1.

EM1 = average(ERT tren H7 = AAC+PCP, GB tren H15 = DPC+PCP)
nen chi can ba block: AAC (20), DPC (400), PCP (9).

Moi cong thuc duoi day lay nguyen van tu Materials and Methods cua bai bao
(Front Immunol 2018;9:1695). Cho nao bai bao khong noi du, danh dau [MOD].
"""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# Monoisotopic-free: khoi luong residue trung binh (Da), dung de tinh molecular weight
RES_MW = dict(A=71.0788, C=103.1388, D=115.0886, E=129.1155, F=147.1766,
              G=57.0519, H=137.1411, I=113.1594, K=128.1741, L=113.1594,
              M=131.1926, N=114.1038, P=97.1167, Q=128.1307, R=156.1875,
              S=87.0782, T=101.1051, V=99.1326, W=186.2132, Y=163.1760)
H2O = 18.0153

# PCP: 5 nhom residue, dung DUNG danh sach bai bao liet ke
HYDROPHOBIC = set("FIWLVMYCA")
HYDROPHILIC = set("SQTRKNDE")
NEUTRAL     = set("HGP")
POSITIVE    = set("KHR")
NEGATIVE    = set("DE")


def aac(seq):
    """AAC — ty le % cua 20 amino acid chuan. dim 20."""
    v = np.zeros(20)
    for c in seq:
        if c in AA_IDX:
            v[AA_IDX[c]] += 1
    return v / max(len(seq), 1) * 100.0


def dpc(seq):
    """DPC — ty le % cua 400 dipeptide. dim 400.

    [MOD] Chuan hoa theo so dipeptide THUC TE co the dem duoc (n-1), khong phai n.
    Bai bao viet 'normalized by all possible dipeptide combinations' nhung mau so
    dung phai la so cua so dipeptide trong chuoi; moi cach khac deu khong cho tong 100%.
    """
    v = np.zeros(400)
    for a, b in zip(seq, seq[1:]):
        if a in AA_IDX and b in AA_IDX:
            v[AA_IDX[a] * 20 + AA_IDX[b]] += 1
    return v / max(len(seq) - 1, 1) * 100.0


def pcp(seq):
    """PCP — 9 tinh chat ly-hoa, dung DUNG 9 cong thuc bai bao liet ke. dim 9."""
    n = max(len(seq), 1)
    cnt = {a: seq.count(a) for a in AA}
    f = lambda grp: sum(cnt[a] for a in grp if a in cnt) / n
    # (vii) absolute charge per residue = |(R + K - D - E)/n - 0.03|
    abs_charge = abs((cnt["R"] + cnt["K"] - cnt["D"] - cnt["E"]) / n - 0.03)
    # (viii) molecular weight
    mw = sum(RES_MW[c] for c in seq if c in RES_MW) + H2O
    # (ix) aliphatic index = (A + 2.9V + 3.9I + 3.9L)/n
    aliph = (cnt["A"] + 2.9 * cnt["V"] + 3.9 * cnt["I"] + 3.9 * cnt["L"]) / n
    return np.array([
        f(HYDROPHOBIC),                                        # (i)
        f(HYDROPHILIC),                                        # (ii)
        f(NEUTRAL),                                            # (iii)
        f(POSITIVE),                                           # (iv)
        f(NEGATIVE),                                           # (v)
        (cnt["N"] + cnt["G"] + cnt["P"] + cnt["S"]) / n,       # (vi) turn-forming
        abs_charge,                                            # (vii)
        mw,                                                    # (viii)
        aliph,                                                 # (ix)
    ])


def encode(seqs, which):
    """which: 'H7' = AAC+PCP (29 dim, cho ERT) | 'H15' = DPC+PCP (409 dim, cho GB)."""
    if which == "H7":
        return np.array([np.concatenate([aac(s), pcp(s)]) for s in seqs])
    if which == "H15":
        return np.array([np.concatenate([dpc(s), pcp(s)]) for s in seqs])
    raise ValueError(which)
