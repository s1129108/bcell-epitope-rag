#!/usr/bin/env python3
"""
Concatenate all per-sequence embedding .npy files into one tensor.

Input:
    embeds/*.npy   each of shape [1, seq_len, dim]  (or [seq_len, dim])

Output:
    a single .npy file of shape (batch, maxlength, 1, dim)

Usage:
    python concatenate_embeddings.py --in_dir embeds --out_file all_embeds.npy --maxlength 64
"""

import os
import argparse
import numpy as np
from pathlib import Path


def load_and_prepare(npy_path: str, maxlength: int):
    """
    Load one embedding file and return array of shape [maxlength, 1, dim]

    Supported input shapes:
        (L, 1, D)  → squeeze middle dim → (L, D)
        (1, L, D)  → take first axis    → (L, D)
        (L, D)     → already correct
    """
    arr = np.load(npy_path)

    if arr.ndim == 3:
        # Most common case: (L, 1, D)
        if arr.shape[1] == 1:
            arr = arr.squeeze(1)          # (L, 1, D) → (L, D)
        # Alternative case: (1, L, D)
        elif arr.shape[0] == 1:
            arr = arr[0]                  # (1, L, D) → (L, D)
        else:
            raise ValueError(f"Unexpected 3-D shape {arr.shape} in {npy_path}")
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected shape {arr.shape} in {npy_path}")

    seq_len, dim = arr.shape

    # Truncate
    if seq_len > maxlength:
        arr = arr[:maxlength]

    # Pad with zeros if shorter
    if seq_len < maxlength:
        pad = np.zeros((maxlength - seq_len, dim), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)

    # Restore the singleton dimension → [1, maxlength,dim]
    arr = arr[np.newaxis, :, :]

    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate embedding .npy files into (batch, maxlength, 1, dim)"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="embeds",
        help="Directory containing individual .npy embedding files"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="all_embeds.npy",
        help="Output concatenated numpy file"
    )
    parser.add_argument(
        "--maxlength",
        type=int,
        required=True,
        help="Maximum sequence length (truncate / pad to this length)"
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        help="Sort files alphabetically before concatenation (default: filesystem order)"
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    npy_files = list(in_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {in_dir}")

    if args.sorted:
        npy_files = sorted(npy_files, key=lambda p: p.name)

    print(f"[INFO] Found {len(npy_files)} embedding files")
    print(f"[INFO] Target shape: (batch, {args.maxlength}, 1, dim)")

    batch = []
    names = []

    for path in npy_files:
        emb = load_and_prepare(str(path), args.maxlength)
        batch.append(emb)
        names.append(path.stem)
        print(f"  + {path.name}  →  {emb.shape}")

    # Stack → (batch, maxlength, 1, dim)
    all_embeds = np.stack(batch, axis=0)
    print(f"\n[INFO] Final shape: {all_embeds.shape}")

    np.save(args.out_file, all_embeds)
    print(f"[DONE] Saved to {args.out_file}")

    # Optionally save the order of sequences for later reference
    order_file = Path(args.out_file).with_suffix(".order.txt")
    with open(order_file, "w") as f:
        for name in names:
            f.write(name + "\n")
    print(f"[INFO] Sequence order saved to {order_file}")


if __name__ == "__main__":
    main()