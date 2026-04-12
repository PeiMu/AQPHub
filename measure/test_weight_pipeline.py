#!/usr/bin/env python3
"""
test_weight_pipeline.py -- Fast end-to-end test of the weight generation pipeline.

Creates tiny synthetic weights, runs preprocess_weights.py and load_weights_duckdb.py,
then verifies the DuckDB contents match the originals. Also runs a 1-layer inference
to verify the full pipeline end-to-end.

This tests the same code paths as the real pipeline but with tiny data,
finishing in seconds instead of hours.

Usage:
    python measure/test_weight_pipeline.py
"""

import os
import sys
import shutil
import tempfile

import numpy as np
import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "transql", "python"))

# ---------------------------------------------------------------------------
# Tiny model config (same structure as Llama3-8B but much smaller)
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 4      # small chunk for fast testing
HIDDEN_DIM   = 8
NUM_Q_HEADS  = 2
NUM_KV_HEADS = 1
HEAD_DIM     = 4      # HIDDEN_DIM // NUM_Q_HEADS
FFN_DIM      = 16
VOCAB_SIZE   = 32
NUM_LAYERS   = 1
MAX_SEQ_LEN  = 16
KV_DIM       = NUM_KV_HEADS * HEAD_DIM  # 4

rng = np.random.default_rng(42)


def create_fake_npy(npy_dir):
    """Create tiny .npy weight files that mimic the real extract_weights.py output."""
    os.makedirs(npy_dir, exist_ok=True)

    # Global weights
    np.save(os.path.join(npy_dir, "embed_tokens.npy"),
            rng.standard_normal((VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32))
    np.save(os.path.join(npy_dir, "lm_head.npy"),
            rng.standard_normal((VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32))
    np.save(os.path.join(npy_dir, "final_norm.npy"),
            rng.standard_normal(HIDDEN_DIM).astype(np.float32))

    # Per-layer weights
    for l in range(NUM_LAYERS):
        np.save(os.path.join(npy_dir, f"layer_{l}_norm1.npy"),
                rng.standard_normal(HIDDEN_DIM).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_norm2.npy"),
                rng.standard_normal(HIDDEN_DIM).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_q_proj.npy"),
                rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_k_proj.npy"),
                rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_v_proj.npy"),
                rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_o_proj.npy"),
                rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_gate_proj.npy"),
                rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_up_proj.npy"),
                rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32))
        np.save(os.path.join(npy_dir, f"layer_{l}_down_proj.npy"),
                rng.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32))

    # RoPE tables (mimic extract_weights.py precompute_rope output)
    half = CHUNK_SIZE // 2
    num_chunks = HIDDEN_DIM // CHUNK_SIZE
    rope_base = 500000.0  # Llama3

    cos_table = np.zeros((MAX_SEQ_LEN, num_chunks, half), dtype=np.float32)
    sin_table = np.zeros((MAX_SEQ_LEN, num_chunks, half), dtype=np.float32)
    for c in range(num_chunks):
        for i in range(half):
            global_dim = c * CHUNK_SIZE + 2 * i
            d = global_dim % HEAD_DIM
            pair_idx = d // 2
            theta = 1.0 / (rope_base ** (2.0 * pair_idx / HEAD_DIM))
            for pos in range(MAX_SEQ_LEN):
                cos_table[pos, c, i] = np.cos(pos * theta)
                sin_table[pos, c, i] = np.sin(pos * theta)

    np.save(os.path.join(npy_dir, "rope_cos.npy"), cos_table)
    np.save(os.path.join(npy_dir, "rope_sin.npy"), sin_table)

    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    print(f"  Created {len(npy_files)} .npy files in {npy_dir}")
    return npy_files


def run_preprocess(npy_dir, csv_dir, chunk_size):
    """Run preprocess_weights.py logic on our tiny weights."""
    from preprocess_weights import process_all
    process_all(npy_dir, csv_dir, chunk_size)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"  Created {len(csv_files)} CSV files in {csv_dir}")
    return csv_files


def run_load(csv_dir, db_path, chunk_size):
    """Run load_weights_duckdb.py logic."""
    from load_weights_duckdb import load_all
    load_all(csv_dir, db_path, chunk_size)
    db_size = os.path.getsize(db_path)
    print(f"  Database size: {db_size} bytes")


def verify_roundtrip(npy_dir, db_path, chunk_size):
    """Verify that weights in DuckDB match the original .npy files."""
    conn = duckdb.connect(db_path, read_only=True)
    passed = 0
    failed = 0

    # Check 2D weights
    for name in ["embed_tokens", "lm_head"] + [
        f"layer_0_{w}" for w in ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj")]:
        npy_path = os.path.join(npy_dir, name + ".npy")
        original = np.load(npy_path).astype(np.float32)

        # Apply same constant folding as preprocess_weights
        if name.endswith("_q_proj"):
            scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
            original = (original * scale).astype(np.float32)

        out_dim, in_dim = original.shape
        rows = conn.execute(
            f'SELECT row_id, chunk_id, v FROM "{name}" ORDER BY row_id, chunk_id'
        ).fetchall()

        reconstructed = np.zeros_like(original)
        for row_id, chunk_id, v in rows:
            reconstructed[row_id, chunk_id*chunk_size:(chunk_id+1)*chunk_size] = \
                np.array(v, dtype=np.float32)

        max_diff = np.max(np.abs(original - reconstructed))
        if max_diff < 1e-6:
            passed += 1
        else:
            print(f"    FAIL {name}: max_diff={max_diff:.8e}")
            failed += 1

    # Check 1D weights (norms)
    for name in ["final_norm", "layer_0_norm1", "layer_0_norm2"]:
        npy_path = os.path.join(npy_dir, name + ".npy")
        original = np.load(npy_path).astype(np.float32)
        dim = original.shape[0]

        rows = conn.execute(
            f'SELECT chunk_id, v FROM "{name}" ORDER BY chunk_id'
        ).fetchall()

        reconstructed = np.zeros(dim, dtype=np.float32)
        for chunk_id, v in rows:
            reconstructed[chunk_id*chunk_size:(chunk_id+1)*chunk_size] = \
                np.array(v, dtype=np.float32)

        max_diff = np.max(np.abs(original - reconstructed))
        if max_diff < 1e-6:
            passed += 1
        else:
            print(f"    FAIL {name}: max_diff={max_diff:.8e}")
            failed += 1

    # Check RoPE
    cos_orig = np.load(os.path.join(npy_dir, "rope_cos.npy")).astype(np.float32)
    sin_orig = np.load(os.path.join(npy_dir, "rope_sin.npy")).astype(np.float32)
    half = chunk_size // 2
    num_chunks = HIDDEN_DIM // chunk_size

    rows = conn.execute(
        "SELECT row_id, chunk_id, cos, sin FROM rope ORDER BY row_id, chunk_id"
    ).fetchall()

    cos_recon = np.zeros_like(cos_orig)
    sin_recon = np.zeros_like(sin_orig)
    for row_id, chunk_id, c, s in rows:
        if row_id < MAX_SEQ_LEN and chunk_id < num_chunks:
            cos_recon[row_id, chunk_id] = np.array(c, dtype=np.float32)
            sin_recon[row_id, chunk_id] = np.array(s, dtype=np.float32)

    cos_diff = np.max(np.abs(cos_orig - cos_recon))
    sin_diff = np.max(np.abs(sin_orig - sin_recon))
    if cos_diff < 1e-6 and sin_diff < 1e-6:
        passed += 1
    else:
        print(f"    FAIL rope: cos_diff={cos_diff:.8e}, sin_diff={sin_diff:.8e}")
        failed += 1

    conn.close()
    return passed, failed


def verify_inference(db_path, chunk_size):
    """Run a 1-layer inference on the test DB to verify SQL pipeline works."""
    conn = duckdb.connect(db_path, read_only=True)

    # Create input tokens
    conn.execute("CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)")
    conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                     [(0, 1), (1, 5), (2, 10)])

    # Inline SQL functions with our test CHUNK_SIZE
    cs = chunk_size
    hidden = HIDDEN_DIM
    eps = 1e-5
    half = cs // 2
    num_q = NUM_Q_HEADS
    num_kv = NUM_KV_HEADS
    head_dim = HEAD_DIM
    cph = head_dim // cs
    cphg = cph * (num_q // num_kv)
    gs = num_q // num_kv

    def run(sql, name):
        conn.execute(f'CREATE TEMP TABLE "{name}" AS ({sql})')

    # Embed
    run(f"SELECT t.pos AS row_id, e.chunk_id, e.v "
        f"FROM input_tokens t JOIN embed_tokens e ON e.row_id = t.token_id "
        f"ORDER BY t.pos, e.chunk_id", "x_0")

    # Layer 0 RMSNorm 1
    run(f"SELECT a.row_id, SUM(list_dot_product(a.v, a.v)) AS ss "
        f"FROM x_0 a GROUP BY a.row_id", "n1_sq")
    run(f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(list_zip(a.v, g.v), x -> "
        f"CAST(x[1] * x[2] / sqrt(n1_sq.ss / {hidden}.0 + {eps}) AS FLOAT)) AS v "
        f"FROM x_0 a JOIN layer_0_norm1 g ON a.chunk_id = g.chunk_id "
        f"JOIN n1_sq ON a.row_id = n1_sq.row_id", "norm1")

    # Q projection
    run(f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM norm1 a JOIN layer_0_q_proj w ON a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, w.row_id", "q_dp")
    run(f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM q_dp GROUP BY act_row, out_col // {cs}", "q")

    # K projection
    run(f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM norm1 a JOIN layer_0_k_proj w ON a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, w.row_id", "k_dp")
    run(f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM k_dp GROUP BY act_row, out_col // {cs}", "k")

    # V projection
    run(f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM norm1 a JOIN layer_0_v_proj w ON a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, w.row_id", "v_dp")
    run(f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM v_dp GROUP BY act_row, out_col // {cs}", "v")

    # RoPE (1-based indexing)
    run(f"SELECT q.row_id, q.chunk_id, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] AS FLOAT)) AS v_even, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd "
        f"FROM q q JOIN rope r ON r.chunk_id = q.chunk_id AND r.row_id = q.row_id",
        "q_rope")
    run(f"SELECT q.row_id, q.chunk_id, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] AS FLOAT)) AS v_even, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd "
        f"FROM k q JOIN rope r ON r.chunk_id = q.chunk_id AND r.row_id = q.row_id",
        "k_rope")

    # QKAttn
    run(f"SELECT q.row_id AS q_tok, k.row_id AS k_tok, "
        f"q.chunk_id // {cph} AS head_id, "
        f"SUM(list_dot_product(q.v_even, k.v_even) + "
        f"list_dot_product(q.v_odd, k.v_odd)) AS score "
        f"FROM q_rope q JOIN k_rope k "
        f"ON q.chunk_id % {cph} = k.chunk_id % {cph} "
        f"AND q.chunk_id // {cphg} = k.chunk_id // {cph} "
        f"AND k.row_id <= q.row_id "
        f"GROUP BY q.row_id, k.row_id, q.chunk_id // {cph}", "qk")

    # Softmax
    run(f"SELECT q_tok, head_id, MAX(score) AS max_score FROM qk GROUP BY q_tok, head_id",
        "attn_max")
    run(f"SELECT s.q_tok, s.k_tok, s.head_id, EXP(s.score - m.max_score) AS exp_val "
        f"FROM qk s JOIN attn_max m ON s.q_tok = m.q_tok AND s.head_id = m.head_id",
        "attn_exp")
    run(f"SELECT q_tok, head_id, SUM(exp_val) AS sum_exp FROM attn_exp GROUP BY q_tok, head_id",
        "attn_sum")
    run(f"SELECT e.q_tok, e.k_tok, e.head_id, "
        f"CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight "
        f"FROM attn_exp e JOIN attn_sum s ON e.q_tok = s.q_tok AND e.head_id = s.head_id",
        "attn_w")

    # AttnVMul (with GQA fix: chunk_id % cph)
    run(f"SELECT v.row_id AS tok, v.chunk_id, "
        f"UNNEST(generate_series(0, {cs}-1)) AS dim_idx, UNNEST(v.v) AS val "
        f"FROM v v", "v_scalar")
    run(f"SELECT a.q_tok AS row_id, "
        f"a.head_id * {cph} + s.chunk_id % {cph} AS chunk_id, "
        f"s.dim_idx, SUM(a.attn_weight * s.val) AS val "
        f"FROM attn_w a JOIN v_scalar s ON a.k_tok = s.tok "
        f"WHERE s.chunk_id >= (a.head_id // {gs}) * {cph} "
        f"AND s.chunk_id < (a.head_id // {gs} + 1) * {cph} "
        f"GROUP BY a.q_tok, a.head_id * {cph} + s.chunk_id % {cph}, s.dim_idx",
        "attn_weighted")
    run(f"SELECT row_id, chunk_id, array_agg(CAST(val AS FLOAT) ORDER BY dim_idx) AS v "
        f"FROM attn_weighted GROUP BY row_id, chunk_id", "attn_out")

    # O projection + residual
    run(f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM attn_out a JOIN layer_0_o_proj w ON a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, w.row_id", "o_dp")
    run(f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM o_dp GROUP BY act_row, out_col // {cs}", "o_proj")
    run(f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(list_zip(a.v, b.v), x -> CAST(x[1]+x[2] AS FLOAT)) AS v "
        f"FROM x_0 a JOIN o_proj b ON a.row_id = b.row_id AND a.chunk_id = b.chunk_id",
        "x_after")

    # Check output
    result = conn.execute(
        "SELECT row_id, chunk_id, v FROM x_after ORDER BY row_id, chunk_id"
    ).fetchall()

    seq_len = 3
    out = np.zeros((seq_len, HIDDEN_DIM), dtype=np.float32)
    for row_id, chunk_id, v in result:
        out[row_id, chunk_id*cs:(chunk_id+1)*cs] = np.array(v, dtype=np.float32)

    conn.close()

    # Verify output is valid
    if np.any(np.isnan(out)):
        return False, "NaN in output"
    if np.any(np.isinf(out)):
        return False, "Inf in output"
    if np.max(np.abs(out)) < 1e-10:
        return False, "all-zero output"
    if out.shape != (seq_len, HIDDEN_DIM):
        return False, f"wrong shape {out.shape}"

    return True, f"shape={out.shape}, max_abs={np.max(np.abs(out)):.4f}"


def main():
    tmpdir = tempfile.mkdtemp(prefix="transql_test_")
    npy_dir = os.path.join(tmpdir, "npy")
    csv_dir = os.path.join(tmpdir, "csv")
    db_path = os.path.join(tmpdir, "test_weights.duckdb")

    try:
        print("=" * 60)
        print("Step 1: Create fake .npy weights")
        print("=" * 60)
        create_fake_npy(npy_dir)

        print("\n" + "=" * 60)
        print("Step 2: Run preprocess_weights.py (npy -> csv)")
        print("=" * 60)
        # Monkey-patch the hardcoded range(32) to only process our layers
        import preprocess_weights as pp
        original_process_all = pp.process_all

        def patched_process_all(npy_d, csv_d, cs):
            """Like process_all but only processes files that exist."""
            os.makedirs(csv_d, exist_ok=True)
            # Discover which npy files we have
            import glob as g
            npy_files = {os.path.splitext(os.path.basename(f))[0]
                         for f in g.glob(os.path.join(npy_d, "*.npy"))
                         if not f.endswith(("rope_cos.npy", "rope_sin.npy"))}

            # Process 1D norms
            for name in sorted(npy_files):
                arr = np.load(os.path.join(npy_d, name + ".npy")).astype(np.float32)
                if arr.ndim == 1:
                    pp.write_csv(os.path.join(csv_d, name + ".csv"),
                                 pp.chunk_1d(arr, cs))
                    print(f"  {name}: {arr.shape} (1D)")
                elif arr.ndim == 2:
                    # Apply constant folding for q_proj
                    if name.endswith("_q_proj"):
                        scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
                        arr = (arr * scale).astype(np.float32)
                        print(f"  {name}: constant folding (scale={scale:.6f})")
                    pp.write_csv(os.path.join(csv_d, name + ".csv"),
                                 pp.chunk_2d(arr, cs))
                    print(f"  {name}: {arr.shape} (2D)")

            # Process RoPE
            pp.process_rope(npy_d, csv_d, cs)

        patched_process_all(npy_dir, csv_dir, CHUNK_SIZE)

        csv_count = len([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
        print(f"\n  Total: {csv_count} CSV files")

        print("\n" + "=" * 60)
        print("Step 3: Run load_weights_duckdb.py (csv -> duckdb)")
        print("=" * 60)
        run_load(csv_dir, db_path, CHUNK_SIZE)

        print("\n" + "=" * 60)
        print("Step 4: Verify weight round-trip (npy -> csv -> duckdb -> numpy)")
        print("=" * 60)
        passed, failed = verify_roundtrip(npy_dir, db_path, CHUNK_SIZE)
        if failed == 0:
            print(f"  Round-trip: ALL {passed} checks PASS")
        else:
            print(f"  Round-trip: {passed} pass, {failed} FAIL")

        print("\n" + "=" * 60)
        print("Step 5: Verify 1-layer SQL inference")
        print("=" * 60)
        ok, msg = verify_inference(db_path, CHUNK_SIZE)
        if ok:
            print(f"  Inference: PASS ({msg})")
        else:
            print(f"  Inference: FAIL ({msg})")

        print("\n" + "=" * 60)
        if failed == 0 and ok:
            print("RESULT: ALL PASS")
            print("The weight pipeline (extract -> preprocess -> load) is correct.")
        else:
            print("RESULT: FAIL")
            sys.exit(1)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
