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
import multiprocessing
import os
import sys
import time

import duckdb
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "transql", "python"))

from run_prefill import build_full_pipeline, run_steps, CHUNK_SIZE, NUM_LAYERS

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
CONTEXT_LEN = 512  # tokens per chunk for perplexity computation



def _run_chunk(args):
    """Worker function for one chunk. Runs in a separate process."""
    db_path, token_ids, num_layers, chunk_idx, duckdb_threads = args

    conn = duckdb.connect(db_path, read_only=True,
                          config={"threads": duckdb_threads})
    conn.execute("CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)")
    conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                     [(i, tid) for i, tid in enumerate(token_ids)])

    pipeline = build_full_pipeline(num_layers)

    t0 = time.perf_counter()
    run_steps(conn, pipeline)

    # Compute cross-entropy entirely in SQL — avoids fetching seq_len×vocab_size
    # rows (e.g. 512×128256 = 65.7M) back to Python.
    # logits_dp has (act_row, out_col, val); input_tokens has (pos, token_id).
    # Position t predicts token at pos t+1.
    # l.val is FLOAT; MAX(FLOAT)=FLOAT; SUM(EXP(FLOAT)) is DOUBLE via SUM aggregate.
    # No explicit CASTs needed — DuckDB's type promotion handles precision.
    row = conn.execute("""
        WITH
          targets AS (
            SELECT pos - 1 AS t, token_id AS target_tok
            FROM input_tokens WHERE pos > 0
          ),
          mx AS (
            SELECT act_row, MAX(val) AS mv
            FROM logits_dp GROUP BY act_row
          ),
          lse AS (
            SELECT l.act_row,
                   m.mv + LOG(SUM(EXP(l.val - m.mv))) AS lse_val
            FROM logits_dp l JOIN mx m ON l.act_row = m.act_row
            GROUP BY l.act_row, m.mv
          ),
          tgt AS (
            SELECT l.act_row, l.val AS tgt_logit
            FROM logits_dp l
            JOIN targets t ON l.act_row = t.t AND l.out_col = t.target_tok
          )
        SELECT -SUM(tgt.tgt_logit - lse.lse_val) AS total_loss,
               COUNT(*) AS n_tokens
        FROM tgt JOIN lse ON tgt.act_row = lse.act_row
    """).fetchone()
    conn.close()
    dt = time.perf_counter() - t0

    chunk_loss = float(row[0])
    chunk_tokens = int(row[1])

    return chunk_idx, chunk_loss, chunk_tokens, dt


def compute_perplexity(db_path, num_layers, max_chunks, num_workers=None):
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in ds)
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Total tokens: {len(all_tokens)}")

    n_chunks = min(max_chunks, len(all_tokens) // CONTEXT_LEN)
    total_cpus = os.cpu_count() or 1
    if num_workers is None:
        # Default to 1 worker: each 32-layer worker creates ~15GB weight pivot
        # TEMP tables (not shared across workers) on top of the shared 40GB mmap.
        # On a 64GB machine that leaves room for at most 1-2 workers safely.
        # Pass --num-workers N explicitly to trade memory for parallelism.
        num_workers = 1
    # Give each worker at least 1 DuckDB thread; spread remaining cores evenly
    duckdb_threads = max(1, total_cpus // num_workers)
    print(f"  Chunks: {n_chunks}, workers: {num_workers}, "
          f"DuckDB threads/worker: {duckdb_threads}")

    chunk_args = [
        (db_path,
         all_tokens[i * CONTEXT_LEN:(i + 1) * CONTEXT_LEN],
         num_layers, i, duckdb_threads)
        for i in range(n_chunks)
    ]

    results = [None] * n_chunks
    total_loss = 0.0
    total_tokens = 0

    with multiprocessing.Pool(processes=num_workers) as pool:
        for chunk_idx, chunk_loss, chunk_tokens, dt in pool.imap_unordered(
                _run_chunk, chunk_args):
            results[chunk_idx] = (chunk_loss, chunk_tokens)
            total_loss += chunk_loss
            total_tokens += chunk_tokens
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
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Parallel chunk workers (default: num CPUs)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = compute_perplexity(args.db_path, args.num_layers,
                                args.max_chunks, args.num_workers)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
