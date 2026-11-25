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
# [MOD] --verify: before stage 1, rescan the files that already exist (row count
# against sequence length, equal column counts) and DELETE any that are wrong so
# stage 1 regenerates them. Needed for data produced before save_atomic existed.
parser.add_argument("--verify", action="store_true",
                    help="[MOD] scan for and delete incomplete per-peptide files before running")
# [MOD] --no-npy: write only the per-peptide files and skip stage 2. Use it when the
# per-residue embeddings are all that is needed (for example the retrieval store) and
# the padded/truncated array is not.
parser.add_argument("--no-npy", dest="make_npy", action="store_false",
                    help="[MOD] skip stage 2 (do not write <set>.npy)")


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


# [MOD] Atomic write: np.savetxt into a .tmp file, then os.replace. os.replace is
# atomic within one filesystem and runs only AFTER savetxt has closed the file
# successfully, so a half-written destination file can never exist. No fsync: the
# failure being guarded against is the process being killed (losing Python's buffer),
# not a power cut, and fsync on 222k files would be painfully slow.
def save_atomic(path, arr, fmt="%.6f"):
    tmp = path + ".tmp"
    try:
        np.savetxt(tmp, arr, fmt=fmt)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# [MOD] Count rows and columns without parsing floats. Faster than loadtxt but NOT
# sufficient for verification (it misses character-level corruption) - kept for quick
# manual diagnosis.
def scan_shape(path):
    rows, ncols = 0, set()
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows += 1
                ncols.add(line.count(" ") + 1)
    return rows, ncols


# [MOD] Rescan the files already written: parse every float, then check the column
# count, the row count (which must equal the sequence length in the FASTA) and NaN/inf.
# Returns a list of (pid, reason). Reading and parsing all the data is slow, but this is
# the ONLY check that catches character-level corruption. Parallelised with joblib.
def verify(records, set_dir, ext, dim=None, n_jobs=16):
    from joblib import Parallel, delayed

    def one(pid, seq):
        path = os.path.join(set_dir, pid + ext)
        if not os.path.exists(path):
            return pid, "file missing"
        # [MOD] The floats MUST actually be parsed, not just the rows and columns
        # counted. A file has been seen with a normal size, the right number of rows and
        # the right number of columns, but one value corrupted at the character level:
        # "-0.00161'" at row 31, column 560 of kg_neg_clean/Negative_IEDB_40706. That
        # kind of damage is invisible to any size- or line-count-based check.
        try:
            emb = np.loadtxt(path, ndmin=2)
        except ValueError as e:
            return pid, f"cannot parse: {e}"
        if emb.ndim != 2:
            return pid, f"ndim {emb.ndim}"
        if dim is not None and emb.shape[1] != dim:
            return pid, f"{emb.shape[1]} columns != {dim}"
        if emb.shape[0] != len(seq):
            return pid, f"{emb.shape[0]} rows != sequence length {len(seq)}"
        if not np.isfinite(emb).all():
            return pid, "contains NaN/inf"
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

        # [MOD] 0. rescan existing files (only with --verify) and delete incomplete ones
        if args.verify:
            broken = verify(records, set_dir, args.ext)
            for pid, why in broken:
                path = os.path.join(set_dir, pid + args.ext)
                print(f"  DELETE {pid}: {why}", flush=True)
                if os.path.exists(path):
                    os.remove(path)
            print(f"{name}: {len(broken)} incomplete files deleted for regeneration",
                  flush=True)

        # 1. per-peptide text files
        # [MOD] save_atomic replaces a direct np.savetxt: the destination file appears
        # only once it is fully written, so os.path.exists here really does mean "done"
        for pid, seq in tqdm(records, desc=f"{name} embeddings"):
            path = os.path.join(set_dir, pid + args.ext)
            if not os.path.exists(path):
                save_atomic(path, embed(model, seq))

        if not args.make_npy:
            print(f"{name}: {len(records)} per-peptide files in {set_dir} (npy skipped)")
            continue

        # 2. stacked array in FASTA order
        # [MOD] Write straight into a memmap instead of building a list and calling
        # np.stack. The kg_neg_clean set holds 222,569 peptides, so a float16 list is
        # about 12.8 GB and np.stack allocates a second copy, peaking near 26 GB of RAM
        # for something that never needs to be held in memory at all.
        # [MOD] The row count of EVERY file is also checked against the sequence length.
        # This is the bug that hit dataset/ESM2 (34 Ind-negative sequences silently cut
        # at 25 residues and never noticed): pad_or_cut zero-pads up to maxseq, so a file
        # with missing rows still "looks normal". Raising an error that names the file
        # beats letting loadtxt throw an opaque ValueError.
        npy_path = os.path.join(args.out_dir, f"{name}.npy")
        data, n_cut = None, 0
        for i, (pid, seq) in enumerate(tqdm(records, desc=f"{name} -> npy")):
            path = os.path.join(set_dir, pid + args.ext)
            # [MOD] a file cut off IN THE MIDDLE of a line makes loadtxt raise
            # ValueError "number of columns changed ..." without naming the file.
            # Replaced with a message that does name it.
            try:
                emb = np.loadtxt(path, ndmin=2)
            except ValueError as e:
                raise SystemExit(
                    f"{path}: incomplete file ({e}). "
                    f"Re-run with --verify to delete and regenerate files like this.") from e
            if emb.shape[0] != len(seq):
                raise SystemExit(
                    f"{path}: {emb.shape[0]} rows but the sequence is {len(seq)} residues. "
                    f"Incomplete file - delete it and re-run, or run with --verify.")
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
            # [MOD] loud warning: silently losing residues is exactly the ESM2 bug
            print(f"  WARNING: {n_cut}/{len(records)} sequences ({n_cut/len(records)*100:.1f}%) "
                  f"are longer than maxseq={args.max_sequence} and WERE TRUNCATED in the npy. "
                  f"The per-peptide files in {set_dir} are still complete.")
        del data


if __name__ == "__main__":
    main(parser.parse_args())
