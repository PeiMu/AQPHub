"""
load_weights_duckdb.py — Load chunked CSV weight files into a DuckDB database.

Usage:
    python load_weights_duckdb.py --csv-dir /path/to/weights_csv \
                                  --db-path  /path/to/weights.duckdb \
                                  [--chunk-size 32]

The resulting .duckdb file can be opened directly by the C++ DuckDBAdapter:
    DuckDBAdapter adapter("/path/to/weights.duckdb");
"""

import argparse
import os
import glob

import duckdb
import numpy as np
import pandas as pd


CHUNK_SIZE = 32
HEAD_DIM   = 128
HIDDEN_DIM = 4096
MAX_SEQ_LEN = 2048


def _is_moe_expert_table(name):
    """Check if table is a MOE expert weight (has expert_id column)."""
    return ("_moe_gate_proj" in name or "_moe_up_proj" in name
            or "_moe_down_proj" in name)


def table_schema(name, chunk_size):
    """Return the CREATE TABLE DDL for a given weight table."""
    half = chunk_size // 2

    if name == "rope":
        # (row_id INTEGER, chunk_id INTEGER, cos FLOAT[half], sin FLOAT[half])
        return (f"CREATE TABLE {name} "
                f"(row_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL, "
                f"cos FLOAT[{half}], sin FLOAT[{half}], "
                f"PRIMARY KEY (row_id, chunk_id))")

    if name.endswith("_norm1") or name.endswith("_norm2") or name == "final_norm":
        # 1D norm weight: (row_id INTEGER, chunk_id INTEGER, v FLOAT[chunk_size])
        return (f"CREATE TABLE {name} "
                f"(row_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL, "
                f"v FLOAT[{chunk_size}], PRIMARY KEY (chunk_id))")

    if _is_moe_expert_table(name):
        # MOE expert weight: (expert_id, row_id, chunk_id, v FLOAT[chunk_size])
        return (f"CREATE TABLE {name} "
                f"(expert_id INTEGER NOT NULL, row_id INTEGER NOT NULL, "
                f"chunk_id INTEGER NOT NULL, v FLOAT[{chunk_size}], "
                f"PRIMARY KEY (expert_id, row_id, chunk_id))")

    # 2D weight or embedding: standard chunked layout
    return (f"CREATE TABLE {name} "
            f"(row_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL, "
            f"v FLOAT[{chunk_size}], PRIMARY KEY (row_id, chunk_id))")


def load_rope_table(con, rope_base, chunk_size=CHUNK_SIZE,
                    head_dim=HEAD_DIM, hidden_dim=HIDDEN_DIM,
                    max_seq_len=MAX_SEQ_LEN):
    """Compute and load the rope table directly from formula.

    Bypasses rope.csv entirely so the correct base is always used regardless
    of how the CSV was generated.  Llama-3 uses rope_base=500000; Llama-2
    uses 10000.
    """
    num_chunks = hidden_dim // chunk_size   # 128
    half       = chunk_size // 2            # 16

    pos   = np.repeat(np.arange(max_seq_len, dtype=np.int32), num_chunks * half)
    chunk = np.tile(np.repeat(np.arange(num_chunks, dtype=np.int32), half), max_seq_len)
    pair  = np.tile(np.arange(half, dtype=np.int32), max_seq_len * num_chunks)

    global_dim = chunk.astype(np.int32) * chunk_size + 2 * pair
    pair_idx   = (global_dim % head_dim) // 2
    theta      = (1.0 / (rope_base ** (2.0 * pair_idx / head_dim))).astype(np.float64)
    angle      = pos.astype(np.float64) * theta
    cos_vals   = np.cos(angle).astype(np.float32)
    sin_vals   = np.sin(angle).astype(np.float32)

    flat_df = pd.DataFrame({'pos': pos, 'chunk': chunk, 'pair': pair,
                             'cos_val': cos_vals, 'sin_val': sin_vals})
    con.register('_rope_flat', flat_df)
    con.execute("DROP TABLE IF EXISTS rope")
    con.execute("""
        CREATE TABLE rope AS
        SELECT
            pos   AS row_id,
            chunk AS chunk_id,
            array_agg(cos_val ORDER BY pair) AS cos,
            array_agg(sin_val ORDER BY pair) AS sin
        FROM _rope_flat
        GROUP BY pos, chunk
        ORDER BY row_id, chunk_id
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_rope_chunk ON rope(chunk_id)")
    cnt = con.execute("SELECT COUNT(*) FROM rope").fetchone()[0]
    print(f"  rope: {cnt} rows (computed from formula, base={rope_base})")


def load_table(con, csv_path, table_name, chunk_size):
    schema = table_schema(table_name, chunk_size)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(schema)

    # DuckDB can read list literals [f1, f2, ...] from CSV via read_csv.
    if table_name == "rope":
        col_types = "{'row_id': 'INTEGER', 'chunk_id': 'INTEGER', 'cos': 'FLOAT[]', 'sin': 'FLOAT[]'}"
    elif _is_moe_expert_table(table_name):
        col_types = ("{'expert_id': 'INTEGER', 'row_id': 'INTEGER', "
                     "'chunk_id': 'INTEGER', 'v': 'FLOAT[]'}")
    elif (table_name.endswith("_norm1") or table_name.endswith("_norm2")
          or table_name == "final_norm"):
        col_types = "{'row_id': 'INTEGER', 'chunk_id': 'INTEGER', 'v': 'FLOAT[]'}"
    else:
        col_types = "{'row_id': 'INTEGER', 'chunk_id': 'INTEGER', 'v': 'FLOAT[]'}"

    con.execute(
        f"INSERT INTO {table_name} "
        f"SELECT * FROM read_csv('{csv_path}', columns={col_types})"
    )

    count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    print(f"  {table_name}: {count} rows")

    # Index on chunk_id for MatMul join performance.
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_chunk "
                f"ON {table_name}(chunk_id)")

    # Index on expert_id for MOE expert weight tables.
    if _is_moe_expert_table(table_name):
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_eid "
                    f"ON {table_name}(expert_id)")


def load_all(csv_dir, db_path, chunk_size, rope_base):
    print(f"Opening DuckDB at {db_path} ...")
    con = duckdb.connect(db_path)

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    # Compute rope table directly from formula — skip rope.csv to guarantee
    # the correct base regardless of how the CSV was generated.
    print(f"Computing rope table (base={rope_base})...")
    load_rope_table(con, rope_base, chunk_size)

    non_rope = [f for f in csv_files
                if os.path.splitext(os.path.basename(f))[0] != "rope"]
    print(f"Loading {len(non_rope)} weight tables from CSV...")
    for csv_path in non_rope:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            load_table(con, csv_path, table_name, chunk_size)
        except Exception as e:
            print(f"  ERROR loading {table_name}: {e}")
            raise

    # Special index for embedding lookup.
    con.execute("CREATE INDEX IF NOT EXISTS idx_embed_token_id "
                "ON embed_tokens(row_id)")

    con.close()
    print(f"\nDone. Database written to {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir",    required=True)
    parser.add_argument("--db-path",    required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--rope-base",  type=float, default=500000.0,
                        help="RoPE theta base (default: 500000 for Llama-3; "
                             "use 10000 for Llama-2)")
    args = parser.parse_args()
    load_all(args.csv_dir, args.db_path, args.chunk_size, args.rope_base)
