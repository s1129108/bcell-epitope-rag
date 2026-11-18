#!/usr/bin/env python3
"""Extract ESM C per-residue embeddings for the B-cell epitope datasets.

Same layout as dataset/ESM2:
  1. one text file per peptide:  dataset/ESMC/<set>/<fasta id>.esmc   (L rows x D columns,
     BOS/EOS tokens removed, so row i = residue i)
  2. one array per set:          dataset/ESMC/<set>.npy  shape (N, 1, MAXSEQ, D), float16,
     first MAXSEQ residues kept, shorter peptides zero-padded (as in get_dataset.py)
     + dataset/ESMC/<set>.order.txt with the FASTA id of every row

Run with the "esmc" conda env (esm==3.2.3), from the B_CELL directory:
  python ./code/get_ESMCEmbedding.py                    # all four sets, esmc_600m
  python ./code/get_ESMCEmbedding.py --model esmc_300m --out_dir ./dataset/ESMC_300m
Already-extracted peptides are skipped, so an interrupted run can simply be restarted.

[MOD] Ghi file per-peptide bang ghi-tam-roi-rename (save_atomic), nen "file ton tai"
dong nghia voi "file hoan chinh". Truoc day stage 1 chi kiem tra os.path.exists: mot run
bi kill giua luc np.savetxt dang ghi se de lai file CAT DO, va run sau thay file ton tai
thi SKIP, roi stage 2 moi no ra. Da xay ra that: run 13/11 bi kill luc 16:42:05 giua file
kg_neg_clean/Negative_IEDB_356655 (294912 byte = 36 x 8192, dung dau vet mat buffer),
run 14/11 skip no, stage 2 bao "number of columns changed from 1152 to 1136 at row 27".

[MOD] Voi file sinh TRUOC khi co save_atomic, dung --verify de quet lai (xem ham verify).
"""
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

SETS = ["B-positive", "B-negative", "Ind-positive", "Ind-negative"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="./dataset")
parser.add_argument("--out_dir", default="./dataset/ESMC")
parser.add_argument("--model", default="esmc_600m", choices=["esmc_300m", "esmc_600m"])
parser.add_argument("--sets", nargs="+", default=SETS)
parser.add_argument("-maxseq", "--max_sequence", type=int, default=25)
parser.add_argument("--ext", default=".esmc")
# [MOD] --verify: truoc stage 1, quet lai cac file da co (row count vs do dai chuoi,
# so column deu nhau) va XOA file nao sai de stage 1 sinh lai. Can cho du lieu sinh
# truoc khi co save_atomic.
parser.add_argument("--verify", action="store_true",
                    help="[MOD] quet va xoa file per-peptide khong hoan chinh truoc khi chay")
# [MOD] --no-npy: chi sinh file per-peptide, bo qua stage 2. Dung khi chi can embedding
# per-residue (vi du store cho RAG retrieval) va khong can mang da pad/cut.
parser.add_argument("--no-npy", dest="make_npy", action="store_false",
                    help="[MOD] bo qua stage 2 (khong sinh <set>.npy)")


def read_fasta(path):
    """[(id, sequence)] in file order."""
    records, name, seq = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq)))
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line)
    if name is not None:
        records.append((name, "".join(seq)))
    return records


@torch.no_grad()
def embed(model, sequence):
    """(L, D) float32 per-residue embedding, BOS/EOS removed."""
    tokens = model.encode(ESMProtein(sequence=sequence))
    out = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
    emb = out.embeddings[0, 1:-1].float().cpu().numpy()
    assert emb.shape[0] == len(sequence), f"{emb.shape} vs length {len(sequence)}"
    return emb


# [MOD] Ghi atomic: np.savetxt vao file .tmp roi os.replace. os.replace la atomic tren
# cung filesystem, va chi chay SAU khi savetxt dong file thanh cong -> khong bao gio ton
# tai mot file dich ghi do. Khong fsync vi moi lo la process bi kill (buffer cua Python
# mat), khong phai mat dien; fsync 222k file se cham khung khiep.
def save_atomic(path, arr, fmt="%.6f"):
    tmp = path + ".tmp"
    try:
        np.savetxt(tmp, arr, fmt=fmt)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# [MOD] Dem so row/column ma khong parse float. Nhanh hon loadtxt nhung KHONG du de
# verify (bo sot hong o muc ky tu) — giu lai de chan doan nhanh thu cong.
def scan_shape(path):
    rows, ncols = 0, set()
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows += 1
                ncols.add(line.count(" ") + 1)
    return rows, ncols


# [MOD] Quet lai cac file da sinh: parse float day du, roi kiem tra so column, so row
# (phai dung bang do dai chuoi trong FASTA) va NaN/inf. Tra ve danh sach (pid, ly do).
# Doc + parse toan bo du lieu nen cham, nhung day la phep kiem tra DUY NHAT thay duoc
# hong o muc ky tu. Chay song song bang joblib.
def verify(records, set_dir, ext, dim=None, n_jobs=16):
    from joblib import Parallel, delayed

    def one(pid, seq):
        path = os.path.join(set_dir, pid + ext)
        if not os.path.exists(path):
            return pid, "thieu file"
        # [MOD] PHAI parse float that su, khong chi dem row/column. Da gap file co size
        # binh thuong, so row dung, so column dung, nhung mot gia tri bi hong o muc ky tu:
        # "-0.00161'" o row 31 column 560 cua kg_neg_clean/Negative_IEDB_40706. Kieu hong
        # nay vo hinh voi moi phep kiem tra dua tren size hay so dong — chi co parse moi thay.
        try:
            emb = np.loadtxt(path, ndmin=2)
        except ValueError as e:
            return pid, f"khong parse duoc: {e}"
        if emb.ndim != 2:
            return pid, f"ndim {emb.ndim}"
        if dim is not None and emb.shape[1] != dim:
            return pid, f"so column {emb.shape[1]} != {dim}"
        if emb.shape[0] != len(seq):
            return pid, f"so row {emb.shape[0]} != do dai chuoi {len(seq)}"
        if not np.isfinite(emb).all():
            return pid, "co NaN/inf"
        return None

    out = Parallel(n_jobs=n_jobs)(
        delayed(one)(pid, seq) for pid, seq in tqdm(records, desc="verify"))
    return [r for r in out if r is not None]


def pad_or_cut(emb, maxseq):
    """Same rule as get_dataset.py: keep the first maxseq residues, zero-pad the rest."""
    data = np.zeros((maxseq, emb.shape[1]), dtype=np.float16)
    n = min(len(emb), maxseq)
    data[:n] = emb[:n]
    return data


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained(args.model).to(device).eval()
    print(f"{args.model} on {device}")

    for name in args.sets:
        records = read_fasta(os.path.join(args.dataset_dir, f"{name}.fasta"))
        set_dir = os.path.join(args.out_dir, name)
        os.makedirs(set_dir, exist_ok=True)

        # [MOD] 0. quet lai file da co (chi khi --verify) va xoa file khong hoan chinh
        if args.verify:
            broken = verify(records, set_dir, args.ext)
            for pid, why in broken:
                path = os.path.join(set_dir, pid + args.ext)
                print(f"  XOA {pid}: {why}", flush=True)
                if os.path.exists(path):
                    os.remove(path)
            print(f"{name}: {len(broken)} file khong hoan chinh, da xoa de sinh lai",
                  flush=True)

        # 1. per-peptide text files
        # [MOD] save_atomic thay np.savetxt truc tiep: file dich chi xuat hien khi da
        # ghi xong, nen os.path.exists o day moi thuc su nghia la "da xong"
        for pid, seq in tqdm(records, desc=f"{name} embeddings"):
            path = os.path.join(set_dir, pid + args.ext)
            if not os.path.exists(path):
                save_atomic(path, embed(model, seq))

        if not args.make_npy:
            print(f"{name}: {len(records)} file per-peptide trong {set_dir} (bo qua npy)")
            continue

        # 2. stacked array in FASTA order
        # [MOD] Viet thang vao memmap thay vi dung list + np.stack. Ly do: bo
        # kg_neg_clean co 222.569 peptide -> list float16 ~12.8 GB va np.stack cap them
        # mot ban nua, dinh ~26 GB RAM cho mot thu hoan toan khong can giu trong RAM.
        # [MOD] Va kiem tra so row cua TUNG file dung bang do dai chuoi. Day la bug da
        # gap o dataset/ESM2 (34 chuoi Ind-negative bi cat am tham o 25 residue ma khong
        # ai phat hien): pad_or_cut zero-pad cho du maxseq nen file thieu row van "nhin
        # nhu binh thuong". Bao loi co ten file thay vi de loadtxt nem ValueError kho hieu.
        npy_path = os.path.join(args.out_dir, f"{name}.npy")
        data, n_cut = None, 0
        for i, (pid, seq) in enumerate(tqdm(records, desc=f"{name} -> npy")):
            path = os.path.join(set_dir, pid + args.ext)
            # [MOD] file bi cat GIUA mot dong -> loadtxt nem ValueError "number of columns
            # changed ..." khong he noi file nao. Doi thanh thong bao co ten file.
            try:
                emb = np.loadtxt(path, ndmin=2)
            except ValueError as e:
                raise SystemExit(
                    f"{path}: file khong hoan chinh ({e}). "
                    f"Chay lai voi --verify de xoa va sinh lai cac file nhu vay.") from e
            if emb.shape[0] != len(seq):
                raise SystemExit(
                    f"{path}: {emb.shape[0]} row nhung chuoi dai {len(seq)} residue. "
                    f"File khong hoan chinh — xoa no roi chay lai, hoac chay --verify.")
            if data is None:
                data = np.lib.format.open_memmap(
                    npy_path, mode="w+", dtype=np.float16,
                    shape=(len(records), 1, args.max_sequence, emb.shape[1]))
            n_cut += len(seq) > args.max_sequence
            data[i, 0] = pad_or_cut(emb, args.max_sequence)
        data.flush()
        with open(os.path.join(args.out_dir, f"{name}.order.txt"), "w") as f:
            f.write("\n".join(pid for pid, _ in records) + "\n")
        print(f"{name}: {data.shape} {data.dtype} -> {npy_path}")
        if n_cut:
            # [MOD] canh bao to: im lang cat mat residue la dung bug da gap o ESM2
            print(f"  CANH BAO: {n_cut}/{len(records)} chuoi ({n_cut/len(records)*100:.1f}%) "
                  f"dai hon maxseq={args.max_sequence} va DA BI CAT trong npy. "
                  f"File per-peptide trong {set_dir} van day du.")
        del data


if __name__ == "__main__":
    main(parser.parse_args())
