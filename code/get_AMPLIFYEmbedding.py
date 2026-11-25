#!/usr/bin/env python3

import os
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SETS = ["B-positive", "B-negative", "Ind-positive", "Ind-negative"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="./dataset")
parser.add_argument("--out_dir", default="./dataset/AMPLIFY")
parser.add_argument("--model", default="chandar-lab/AMPLIFY_350M")
parser.add_argument("--revision", default="223e35e4e2074fadfdc1617f3451b14fa5ace1c8",
                    help="pinned Hugging Face commit (the model runs remote code)")
parser.add_argument("--sets", nargs="+", default=SETS)
parser.add_argument("-maxseq", "--max_sequence", type=int, default=25)
parser.add_argument("--ext", default=".amplify")


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
def embed(model, tokenizer, sequence, device):
    """(L, 960) float32 per-residue embedding, <bos>/<eos> removed."""
    ids = tokenizer(sequence, return_tensors="pt")["input_ids"].to(device)
    out = model(ids, output_hidden_states=True)
    emb = model.layer_norm_2(out.hidden_states[-1])[0, 1:-1].float().cpu().numpy()
    assert emb.shape[0] == len(sequence), f"{emb.shape} vs length {len(sequence)}"
    return emb


def pad_or_cut(emb, maxseq):
    """Same rule as get_dataset.py: keep the first maxseq residues, zero-pad the rest."""
    data = np.zeros((maxseq, emb.shape[1]), dtype=np.float16)
    n = min(len(emb), maxseq)
    data[:n] = emb[:n]
    return data


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, revision=args.revision, trust_remote_code=True).to(device).eval()
    print(f"{args.model}@{args.revision[:8]} on {device}")

    for name in args.sets:
        records = read_fasta(os.path.join(args.dataset_dir, f"{name}.txt"))
        set_dir = os.path.join(args.out_dir, name)
        os.makedirs(set_dir, exist_ok=True)

        # 1. per-peptide text files
        for pid, seq in tqdm(records, desc=f"{name} embeddings"):
            path = os.path.join(set_dir, pid + args.ext)
            if not os.path.exists(path):
                np.savetxt(path, embed(model, tokenizer, seq, device), fmt="%.6f")

        # 2. stacked array in FASTA order
        arrays = [pad_or_cut(np.loadtxt(os.path.join(set_dir, pid + args.ext), ndmin=2), args.max_sequence)
                  for pid, _ in tqdm(records, desc=f"{name} -> npy")]
        data = np.stack(arrays)[:, np.newaxis]   # (N, 1, maxseq, D)
        np.save(os.path.join(args.out_dir, f"{name}.npy"), data)
        with open(os.path.join(args.out_dir, f"{name}.order.txt"), "w") as f:
            f.write("\n".join(pid for pid, _ in records) + "\n")
        print(f"{name}: {data.shape} {data.dtype} -> {args.out_dir}/{name}.npy")


if __name__ == "__main__":
    main(parser.parse_args())
