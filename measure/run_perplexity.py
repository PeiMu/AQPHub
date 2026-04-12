"""
run_perplexity.py -- Measure TranSQL+ perplexity on WikiText-2.

Usage:
    python measure/run_perplexity.py \
        --db-path /path/to/weights.duckdb \
        --output results/transql_ppl.json \
        [--num-layers 32] [--max-chunks 64]

Computes perplexity by running prefill on WikiText-2 chunks, extracting
logits, and computing cross-entropy loss over the next-token predictions.
"""

import argparse
import json
import math
import os
import sys
import time

import duckdb
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "transql", "python"))

from run_prefill import (
    build_full_pipeline, run_steps, CHUNK_SIZE, NUM_LAYERS,
    HIDDEN_DIM,
)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
CONTEXT_LEN = 512  # tokens per chunk for perplexity computation


def read_logits(conn, seq_len, vocab_size):
    """Read logits from the dot-product intermediate table.
    logits_dp has columns (act_row, out_col, val)."""
    rows = conn.execute(
        "SELECT act_row, out_col, val FROM logits_dp "
        "ORDER BY act_row, out_col"
    ).fetchall()

    logits = np.zeros((seq_len, vocab_size), dtype=np.float32)
    for act_row, out_col, val in rows:
        logits[act_row, out_col] = val
    return logits


def compute_perplexity(db_path, num_layers, max_chunks):
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab_size = len(tokenizer)  # includes special tokens (e.g. 128256 for Llama-3)

    print("Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in ds)
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Total tokens: {len(all_tokens)}")

    # Split into chunks
    n_chunks = min(max_chunks, len(all_tokens) // CONTEXT_LEN)
    total_loss = 0.0
    total_tokens = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CONTEXT_LEN
        token_ids = all_tokens[start:start + CONTEXT_LEN]
        seq_len = len(token_ids)

        # Fresh connection per chunk
        conn = duckdb.connect(db_path, read_only=True)
        conn.execute("CREATE TEMP TABLE input_tokens "
                     "(pos INTEGER, token_id INTEGER)")
        conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                         [(i, tid) for i, tid in enumerate(token_ids)])

        pipeline = build_full_pipeline(num_layers)

        t0 = time.perf_counter()
        run_steps(conn, pipeline)
        dt = time.perf_counter() - t0

        # Read logits and compute cross-entropy
        logits = read_logits(conn, seq_len, vocab_size)
        conn.close()

        # Compute loss: -log P(token[t+1] | token[0:t+1]) for t=0..seq-2
        for t in range(seq_len - 1):
            target = token_ids[t + 1]
            # Numerically stable log-softmax
            max_logit = np.max(logits[t])
            log_sum_exp = max_logit + np.log(
                np.sum(np.exp(logits[t] - max_logit)))
            log_prob = logits[t, target] - log_sum_exp
            total_loss -= log_prob
            total_tokens += 1

        ppl_so_far = math.exp(total_loss / total_tokens)
        print(f"  Chunk {chunk_idx+1}/{n_chunks}: "
              f"{dt:.1f}s, running PPL = {ppl_so_far:.4f}")

    final_ppl = math.exp(total_loss / total_tokens)
    print(f"\nFinal perplexity: {final_ppl:.4f} "
          f"(over {total_tokens} tokens, {n_chunks} chunks)")
    return {
        "perplexity": final_ppl,
        "total_tokens": total_tokens,
        "num_chunks": n_chunks,
        "context_length": CONTEXT_LEN,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output",
                        default="measure/results/transql_ppl.json")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--max-chunks", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = compute_perplexity(args.db_path, args.num_layers,
                                args.max_chunks)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
