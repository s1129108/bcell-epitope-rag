#!/usr/bin/env python3

import os
import re
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

SETS = ["B-positive", "B-negative", "Ind-positive", "Ind-negative"]

# checkpoint, pinned revision, dtype, output folder, file extension, input formatter, first residue token
MODELS = {
    "prott5": dict(ckpt="Rostlab/prot_t5_xl_half_uniref50-enc", revision="94a6abc029ae13029317b140b7424e012bf8dfbf",
                   dtype=torch.float16, out="ProtT5", ext=".prott5",
                   fmt=lambda s: " ".join(re.sub(r"[UZOB]", "X", s)), start=0),
    "ankh3": dict(ckpt="ElnaggarLab/ankh3-large", revision="2be091622e8a393f0ef21735070084123c874b6e",
                  dtype=torch.float32, out="Ankh3", ext=".ankh3", fmt=lambda s: "[NLU]" + s, start=1),
    "ankh3-xl": dict(ckpt="ElnaggarLab/ankh3-xl", revision="e00113df5c95ef71df7ea3f5a73d56bd00e473a4",
                     dtype=torch.float32, out="Ankh3_XL", ext=".ankh3xl", fmt=lambda s: "[NLU]" + s, start=1),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODELS))
parser.add_argument("--dataset_dir", default="./dataset")
parser.add_argument("--out_dir", default=None, help="default ./dataset/<ProtT5|Ankh3|Ankh3_XL>")
parser.add_argument("--sets", nargs="+", default=SETS)
parser.add_argument("-maxseq", "--max_sequence", type=int, default=25)


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
def embed(model, tokenizer, spec, sequence, device):
    """(L, D) float32 per-residue embedding, special tokens removed."""
    enc = tokenizer(spec["fmt"](sequence), add_special_tokens=True, return_tensors="pt").to(device)
    hidden = model(**enc).last_hidden_state[0]
    emb = hidden[spec["start"]:spec["start"] + len(sequence)].float().cpu().numpy()
    # one token per residue + the special tokens, otherwise the slice above would be wrong
    assert hidden.shape[0] == len(sequence) + spec["start"] + 1, f"{hidden.shape} vs length {len(sequence)}"
    return emb


def pad_or_cut(emb, maxseq):
    """Same rule as get_dataset.py: keep the first maxseq residues, zero-pad the rest."""
    data = np.zeros((maxseq, emb.shape[1]), dtype=np.float16)
    n = min(len(emb), maxseq)
    data[:n] = emb[:n]
    return data


def main(args):
    spec = MODELS[args.model]
    out_dir = args.out_dir or os.path.join(args.dataset_dir, spec["out"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(spec["ckpt"], revision=spec["revision"], do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(spec["ckpt"], revision=spec["revision"], torch_dtype=spec["dtype"]).to(device).eval()
    print(f"{spec['ckpt']}@{spec['revision'][:8]} ({spec['dtype']}) on {device} -> {out_dir}")

    for name in args.sets:
        records = read_fasta(os.path.join(args.dataset_dir, f"{name}.txt"))
        set_dir = os.path.join(out_dir, name)
        os.makedirs(set_dir, exist_ok=True)

        # 1. per-peptide text files
        for pid, seq in tqdm(records, desc=f"{name} embeddings"):
            path = os.path.join(set_dir, pid + spec["ext"])
            if not os.path.exists(path):
                np.savetxt(path, embed(model, tokenizer, spec, seq, device), fmt="%.6f")

        # 2. stacked array in FASTA order
        arrays = [pad_or_cut(np.loadtxt(os.path.join(set_dir, pid + spec["ext"]), ndmin=2), args.max_sequence)
                  for pid, _ in tqdm(records, desc=f"{name} -> npy")]
        data = np.stack(arrays)[:, np.newaxis]   # (N, 1, maxseq, D)
        np.save(os.path.join(out_dir, f"{name}.npy"), data)
        with open(os.path.join(out_dir, f"{name}.order.txt"), "w") as f:
            f.write("\n".join(pid for pid, _ in records) + "\n")
        print(f"{name}: {data.shape} {data.dtype} -> {out_dir}/{name}.npy")


if __name__ == "__main__":
    main(parser.parse_args())
