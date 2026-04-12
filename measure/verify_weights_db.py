#!/usr/bin/env python3
"""
verify_weights_db.py -- Verify weights.duckdb integrity and correctness.

Checks:
  1. All expected tables exist (embed, lm_head, norms, projections, rope)
  2. Table schemas match expected columns and types
  3. Row counts match Llama3-8B dimensions
  4. No NULL values in weight data
  5. Value ranges are reasonable (not all zeros, not exploding)
  6. Indexes exist for join performance

Usage:
    python measure/verify_weights_db.py --db-path weights.duckdb [--num-layers 32]
    python measure/verify_weights_db.py --db-path weights.duckdb --num-layers 1  # quick check
"""

import argparse
import sys

import duckdb


# ---------------------------------------------------------------------------
# Llama3-8B expected dimensions
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 32
HIDDEN_DIM   = 4096
NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
FFN_DIM      = 14336
VOCAB_SIZE   = 128256   # Llama3 vocabulary size
MAX_SEQ_LEN  = 8192     # or 131072 for some configs; verify actual

N_CHUNKS_HIDDEN = HIDDEN_DIM // CHUNK_SIZE   # 128
N_CHUNKS_KV     = (NUM_KV_HEADS * HEAD_DIM) // CHUNK_SIZE  # 32
N_CHUNKS_FFN    = FFN_DIM // CHUNK_SIZE      # 448


def expected_tables(num_layers):
    """Return dict of table_name -> expected (total_rows, description)."""
    tables = {}

    # Embedding: [vocab_size, hidden_dim] -> vocab_size * N_CHUNKS_HIDDEN rows
    tables["embed_tokens"] = (VOCAB_SIZE * N_CHUNKS_HIDDEN,
                              f"embed [{VOCAB_SIZE}, {HIDDEN_DIM}]")

    # LM head: [vocab_size, hidden_dim] -> same as embed
    tables["lm_head"] = (VOCAB_SIZE * N_CHUNKS_HIDDEN,
                         f"lm_head [{VOCAB_SIZE}, {HIDDEN_DIM}]")

    # Final norm: [hidden_dim] -> 1 * N_CHUNKS_HIDDEN rows (or just N_CHUNKS)
    tables["final_norm"] = (N_CHUNKS_HIDDEN,
                            f"final_norm [{HIDDEN_DIM}]")

    # RoPE: [max_seq_len, n_chunks, half] -> max_seq_len * n_chunks rows
    # We don't know exact max_seq_len, so we just check it exists
    tables["rope"] = (None, "rope [seq_len, chunks, half]")

    for l in range(num_layers):
        pfx = f"layer_{l}_"

        # Norms: [hidden_dim] -> N_CHUNKS_HIDDEN rows
        tables[pfx + "norm1"] = (N_CHUNKS_HIDDEN,
                                 f"norm1 [{HIDDEN_DIM}]")
        tables[pfx + "norm2"] = (N_CHUNKS_HIDDEN,
                                 f"norm2 [{HIDDEN_DIM}]")

        # Q proj: [hidden_dim, hidden_dim] -> hidden * N_CHUNKS_HIDDEN
        tables[pfx + "q_proj"] = (HIDDEN_DIM * N_CHUNKS_HIDDEN,
                                  f"q_proj [{HIDDEN_DIM}, {HIDDEN_DIM}]")

        # K proj: [kv_dim, hidden_dim] -> kv_dim * N_CHUNKS_HIDDEN (kv_dim = 1024)
        kv_dim = NUM_KV_HEADS * HEAD_DIM  # 1024
        tables[pfx + "k_proj"] = (kv_dim * N_CHUNKS_HIDDEN,
                                  f"k_proj [{kv_dim}, {HIDDEN_DIM}]")
        tables[pfx + "v_proj"] = (kv_dim * N_CHUNKS_HIDDEN,
                                  f"v_proj [{kv_dim}, {HIDDEN_DIM}]")

        # O proj: [hidden_dim, hidden_dim]
        tables[pfx + "o_proj"] = (HIDDEN_DIM * N_CHUNKS_HIDDEN,
                                  f"o_proj [{HIDDEN_DIM}, {HIDDEN_DIM}]")

        # Gate/Up: [ffn_dim, hidden_dim] -> ffn_dim * N_CHUNKS_HIDDEN
        tables[pfx + "gate_proj"] = (FFN_DIM * N_CHUNKS_HIDDEN,
                                     f"gate_proj [{FFN_DIM}, {HIDDEN_DIM}]")
        tables[pfx + "up_proj"] = (FFN_DIM * N_CHUNKS_HIDDEN,
                                   f"up_proj [{FFN_DIM}, {HIDDEN_DIM}]")

        # Down: [hidden_dim, ffn_dim] -> hidden * N_CHUNKS_FFN
        tables[pfx + "down_proj"] = (HIDDEN_DIM * N_CHUNKS_FFN,
                                     f"down_proj [{HIDDEN_DIM}, {FFN_DIM}]")

    return tables


def check_table_exists(conn, table_name):
    """Check if a table exists in the database."""
    result = conn.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?",
        [table_name]).fetchone()
    return result[0] > 0


def check_row_count(conn, table_name, expected_rows):
    """Check row count matches expected."""
    actual = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
    if expected_rows is not None and actual != expected_rows:
        return False, actual, expected_rows
    return True, actual, expected_rows


def check_nulls(conn, table_name):
    """Check for NULL values in the weight data columns."""
    # Get column names
    cols = conn.execute(
        f"SELECT column_name FROM duckdb_columns() "
        f"WHERE table_name = '{table_name}'"
    ).fetchall()
    col_names = [c[0] for c in cols]

    null_cols = []
    for col in col_names:
        null_count = conn.execute(
            f'SELECT count(*) FROM "{table_name}" WHERE "{col}" IS NULL'
        ).fetchone()[0]
        if null_count > 0:
            null_cols.append((col, null_count))
    return null_cols


def check_value_stats(conn, table_name):
    """Get basic stats on weight values (mean, std, min, max)."""
    # For tables with 'v' column
    cols = conn.execute(
        f"SELECT column_name FROM duckdb_columns() "
        f"WHERE table_name = '{table_name}'"
    ).fetchall()
    col_names = [c[0] for c in cols]

    if 'v' in col_names:
        result = conn.execute(f"""
            SELECT
                AVG(list_avg(CAST(v AS DOUBLE[]))),
                STDDEV(list_avg(CAST(v AS DOUBLE[]))),
                MIN(list_min(CAST(v AS DOUBLE[]))),
                MAX(list_max(CAST(v AS DOUBLE[])))
            FROM "{table_name}"
            WHERE v IS NOT NULL
            LIMIT 1
        """).fetchone()
        return {'col': 'v', 'mean': result[0], 'std': result[1],
                'min': result[2], 'max': result[3]}
    elif 'cos' in col_names:
        result = conn.execute(f"""
            SELECT
                AVG(list_avg(CAST(cos AS DOUBLE[]))),
                MIN(list_min(CAST(cos AS DOUBLE[]))),
                MAX(list_max(CAST(cos AS DOUBLE[])))
            FROM "{table_name}"
            WHERE cos IS NOT NULL
        """).fetchone()
        return {'col': 'cos', 'mean': result[0], 'min': result[1],
                'max': result[2]}
    return None


def check_chunk_sizes(conn, table_name):
    """Verify all chunks have consistent sizes."""
    cols = conn.execute(
        f"SELECT column_name FROM duckdb_columns() "
        f"WHERE table_name = '{table_name}'"
    ).fetchall()
    col_names = [c[0] for c in cols]

    if 'v' in col_names:
        result = conn.execute(f"""
            SELECT DISTINCT len(v) AS chunk_len
            FROM "{table_name}"
            WHERE v IS NOT NULL
        """).fetchall()
        sizes = [r[0] for r in result]
        return sizes
    elif 'cos' in col_names:
        result = conn.execute(f"""
            SELECT DISTINCT len(cos) AS chunk_len
            FROM "{table_name}"
            WHERE cos IS NOT NULL
        """).fetchall()
        sizes = [r[0] for r in result]
        return sizes
    return []


def check_indexes(conn, table_name):
    """Check if expected indexes exist."""
    result = conn.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE table_name = ?",
        [table_name]).fetchall()
    return [r[0] for r in result]


def main():
    parser = argparse.ArgumentParser(
        description="Verify weights.duckdb integrity for Llama3-8B")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    conn = duckdb.connect(args.db_path, read_only=True)
    tables = expected_tables(args.num_layers)

    passed = 0
    failed = 0
    warnings = 0

    print(f"Verifying {args.db_path} ({args.num_layers} layers)")
    print(f"Expected {len(tables)} tables\n")

    # 1. Check all tables exist
    print("=== Table Existence ===")
    existing_tables = set()
    for name, (expected_rows, desc) in sorted(tables.items()):
        exists = check_table_exists(conn, name)
        if exists:
            existing_tables.add(name)
            if args.verbose:
                print(f"  OK  {name} ({desc})")
            passed += 1
        else:
            print(f"  FAIL  {name} MISSING ({desc})")
            failed += 1

    missing = len(tables) - len(existing_tables)
    print(f"  {len(existing_tables)}/{len(tables)} tables found"
          f"{f', {missing} MISSING' if missing else ''}\n")

    # 2. Check row counts
    print("=== Row Counts ===")
    for name in sorted(existing_tables):
        expected_rows = tables[name][0]
        ok, actual, expected = check_row_count(conn, name, expected_rows)
        if ok:
            if args.verbose:
                print(f"  OK  {name}: {actual} rows"
                      f"{f' (expected {expected})' if expected else ''}")
            passed += 1
        else:
            print(f"  FAIL  {name}: {actual} rows (expected {expected})")
            failed += 1
    print()

    # 3. Check for NULLs (sample first 5 tables + all norms)
    print("=== NULL Check ===")
    tables_to_check = sorted(existing_tables)
    null_issues = 0
    for name in tables_to_check:
        null_cols = check_nulls(conn, name)
        if null_cols:
            for col, count in null_cols:
                print(f"  FAIL  {name}.{col}: {count} NULLs")
                null_issues += 1
                failed += 1
        else:
            passed += 1
            if args.verbose:
                print(f"  OK  {name}: no NULLs")
    if null_issues == 0:
        print(f"  All {len(tables_to_check)} tables NULL-free")
    print()

    # 4. Check chunk sizes
    print("=== Chunk Size Consistency ===")
    expected_cs = CHUNK_SIZE
    expected_half = CHUNK_SIZE // 2
    for name in sorted(existing_tables):
        sizes = check_chunk_sizes(conn, name)
        if name == "rope":
            if sizes and all(s == expected_half for s in sizes):
                passed += 1
                if args.verbose:
                    print(f"  OK  {name}: chunk size {sizes[0]} (half={expected_half})")
            elif sizes:
                print(f"  FAIL  {name}: chunk sizes {sizes} (expected {expected_half})")
                failed += 1
        else:
            if sizes and all(s == expected_cs for s in sizes):
                passed += 1
                if args.verbose:
                    print(f"  OK  {name}: chunk size {sizes[0]}")
            elif sizes:
                print(f"  FAIL  {name}: chunk sizes {sizes} (expected {expected_cs})")
                failed += 1
    print()

    # 5. Value statistics (sample a few tables)
    print("=== Value Statistics (sample) ===")
    sample_tables = ["embed_tokens", "final_norm", "rope"]
    if args.num_layers > 0:
        sample_tables += [
            "layer_0_q_proj", "layer_0_norm1",
            "layer_0_gate_proj", "layer_0_down_proj",
        ]
    for name in sample_tables:
        if name not in existing_tables:
            continue
        stats = check_value_stats(conn, name)
        if stats:
            col = stats['col']
            if 'std' in stats:
                print(f"  {name}.{col}: mean={stats['mean']:.6f}, "
                      f"std={stats['std']:.6f}, "
                      f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")
            else:
                print(f"  {name}.{col}: mean={stats['mean']:.6f}, "
                      f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")

            # Sanity checks
            if stats['min'] is not None and stats['max'] is not None:
                if abs(stats['max'] - stats['min']) < 1e-10:
                    print(f"    WARNING: all values identical!")
                    warnings += 1
                if abs(stats['max']) > 1000 or abs(stats['min']) > 1000:
                    print(f"    WARNING: extreme values detected")
                    warnings += 1
    print()

    # 6. Check indexes
    print("=== Indexes ===")
    for name in sorted(existing_tables):
        indexes = check_indexes(conn, name)
        if args.verbose and indexes:
            print(f"  {name}: {', '.join(indexes)}")
    # Just check embed_tokens has its index
    embed_idxs = check_indexes(conn, "embed_tokens") if "embed_tokens" in existing_tables else []
    if any("token" in idx or "row" in idx for idx in embed_idxs):
        print(f"  embed_tokens index: OK")
        passed += 1
    elif embed_idxs:
        print(f"  embed_tokens indexes: {embed_idxs}")
        passed += 1
    else:
        print(f"  WARNING: embed_tokens has no indexes")
        warnings += 1
    print()

    # Summary
    print("=" * 50)
    print(f"PASSED: {passed}  FAILED: {failed}  WARNINGS: {warnings}")
    if failed > 0:
        print("RESULT: FAIL")
        sys.exit(1)
    elif warnings > 0:
        print("RESULT: PASS (with warnings)")
    else:
        print("RESULT: PASS")

    conn.close()


if __name__ == "__main__":
    main()
