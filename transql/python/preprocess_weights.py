"""
preprocess_weights.py — Convert .npy weight files to chunked CSV format.

Usage:
    python preprocess_weights.py --npy-dir /path/to/weights_npy \
                                 --csv-dir /path/to/weights_csv \
                                 [--chunk-size 32]

CSV format (row_id, chunk_id, v):
    row_id    INTEGER  — output dimension index (for 2D weights)
    chunk_id  INTEGER  — which chunk along the input dimension
    v         FLOAT[]  — DuckDB list literal "[f1, f2, ..., f32]"

For 2D weight [out_dim, in_dim]: chunked over in_dim (columns).
For 1D weight [dim]: row_id=0, chunked over dim.
For RoPE tables: special format — see below.
"""

import argparse
import csv
import glob
import os

import numpy as np


CHUNK_SIZE = 32
HEAD_DIM   = 128   # Llama3-8B: hidden_size / num_attention_heads = 4096 / 32


def format_list(arr):
    """Format a numpy 1D array as a DuckDB list literal: [1.0, 2.0, ...]"""
    return "[" + ", ".join(f"{x:.8g}" for x in arr) + "]"


def chunk_2d(weight, chunk_size):
    """
    Yield (row_id, chunk_id, chunk_array) for a 2D weight [out_dim, in_dim].
    Chunks are taken along in_dim (axis=1).
    """
    out_dim, in_dim = weight.shape
    assert in_dim % chunk_size == 0, \
        f"in_dim {in_dim} not divisible by chunk_size {chunk_size}"
    n_chunks = in_dim // chunk_size
    for row_id in range(out_dim):
        for chunk_id in range(n_chunks):
            chunk = weight[row_id, chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
            yield row_id, chunk_id, chunk


def chunk_1d(weight, chunk_size):
    """
    Yield (row_id=0, chunk_id, chunk_array) for a 1D weight [dim].
    Used for RMSNorm gamma weights.
    """
    dim = weight.shape[0]
    assert dim % chunk_size == 0, \
        f"dim {dim} not divisible by chunk_size {chunk_size}"
    n_chunks = dim // chunk_size
    for chunk_id in range(n_chunks):
        chunk = weight[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        yield 0, chunk_id, chunk


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["row_id", "chunk_id", "v"])
        for row_id, chunk_id, chunk in rows:
            writer.writerow([row_id, chunk_id, format_list(chunk)])


def process_rope(npy_dir, csv_dir, chunk_size):
    """
    RoPE tables: shape [max_seq_len, num_chunks, chunk_size//2].
    Combines rope_cos.npy and rope_sin.npy into a single rope.csv.
    Schema: (row_id INTEGER, chunk_id INTEGER, cos FLOAT[], sin FLOAT[])
    where row_id = position index.
    """
    half = chunk_size // 2
    cos_arr = np.load(os.path.join(npy_dir, "rope_cos.npy")).astype(np.float32)
    sin_arr = np.load(os.path.join(npy_dir, "rope_sin.npy")).astype(np.float32)
    # shape: [max_seq_len, num_chunks, half]
    max_seq, num_chunks, h = cos_arr.shape
    assert h == half, f"Expected half={half}, got {h}"
    assert cos_arr.shape == sin_arr.shape, "rope_cos and rope_sin shape mismatch"

    out_path = os.path.join(csv_dir, "rope.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["row_id", "chunk_id", "cos", "sin"])
        for pos in range(max_seq):
            for c in range(num_chunks):
                writer.writerow([pos, c,
                                  format_list(cos_arr[pos, c]),
                                  format_list(sin_arr[pos, c])])
    print(f"  rope: {max_seq} positions x {num_chunks} chunks → {out_path}")


def process_all(npy_dir, csv_dir, chunk_size):
    os.makedirs(csv_dir, exist_ok=True)

    # 1D weights (RMSNorm gamma): final_norm, layer_*_norm1, layer_*_norm2
    norm_names = (
        ["final_norm"] +
        [f"layer_{l}_norm1" for l in range(32)] +
        [f"layer_{l}_norm2" for l in range(32)]
    )
    for name in norm_names:
        npy_path = os.path.join(npy_dir, name + ".npy")
        if not os.path.exists(npy_path):
            print(f"  SKIP (not found): {name}")
            continue
        arr = np.load(npy_path).astype(np.float32)
        assert arr.ndim == 1, f"{name}: expected 1D, got shape {arr.shape}"
        write_csv(os.path.join(csv_dir, name + ".csv"), chunk_1d(arr, chunk_size))
        print(f"  {name}: {arr.shape} → {arr.shape[0] // chunk_size} chunks")

    # 2D weights: everything else except rope
    two_d_names = (
        ["embed_tokens", "lm_head"] +
        [f"layer_{l}_{w}"
         for l in range(32)
         for w in ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")]
    )
    for name in two_d_names:
        npy_path = os.path.join(npy_dir, name + ".npy")
        if not os.path.exists(npy_path):
            print(f"  SKIP (not found): {name}")
            continue
        arr = np.load(npy_path).astype(np.float32)
        assert arr.ndim == 2, f"{name}: expected 2D, got shape {arr.shape}"

        # Constant folding: absorb 1/sqrt(head_dim) into W_Q (Section 3)
        if name.endswith("_q_proj"):
            scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
            arr = (arr * scale).astype(np.float32)
            print(f"  {name}: constant folding applied (scale={scale:.6f})")

        write_csv(os.path.join(csv_dir, name + ".csv"), chunk_2d(arr, chunk_size))
        print(f"  {name}: {arr.shape} → {arr.shape[0]} rows x "
              f"{arr.shape[1] // chunk_size} chunks/row")

    # RoPE tables
    process_rope(npy_dir, csv_dir, chunk_size)

    print(f"\nDone. CSVs written to {csv_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-dir",    required=True)
    parser.add_argument("--csv-dir",    required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    process_all(args.npy_dir, args.csv_dir, args.chunk_size)
