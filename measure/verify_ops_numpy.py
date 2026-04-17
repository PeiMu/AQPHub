#!/usr/bin/env python3
"""
verify_ops_numpy.py -- Verify SQL operators against NumPy reference.

Tests each SQL operator using small synthetic data in an in-memory DuckDB.
Uses the SAME SQL functions as run_prefill.py to verify they produce correct
results matching the C++ sql_template.cpp reference.

Tests:
  1. EmbedLookup
  2. MatMul (basic + pivoted)
  3. RMSNorm
  4. RoPE
  5. QKAttn
  6. Softmax
  7. AttnVMul
  8. SwiGLU
  9. ResidualAdd
  10. Full single-layer forward pass

Usage:
    python measure/verify_ops_numpy.py
    python measure/verify_ops_numpy.py -v         # verbose
    python measure/verify_ops_numpy.py --test rope # run single test
"""

import argparse
import sys
import traceback

import numpy as np
import duckdb


# ---------------------------------------------------------------------------
# Test parameters (small, fast)
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 2
SEQ_LEN      = 3
HIDDEN_DIM   = 4
KV_DIM       = 2     # num_kv_heads=1, head_dim=2
FFN_DIM      = 8
VOCAB_SIZE   = 16
NUM_Q_HEADS  = 2
NUM_KV_HEADS = 1
HEAD_DIM     = 2
EPS          = 1e-5
ATOL         = 1e-4
RTOL         = 1e-4

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def new_conn():
    return duckdb.connect(":memory:")


def load_2d(conn, name, matrix):
    """Load [n, dim] float32 matrix as chunked table (row_id, chunk_id, v FLOAT[])."""
    n, dim = matrix.shape
    conn.execute(f"CREATE TEMP TABLE {name} (row_id INTEGER, chunk_id INTEGER, v FLOAT[])")
    rows = [
        (r, c, matrix[r, c*CHUNK_SIZE:(c+1)*CHUNK_SIZE].astype(np.float32).tolist())
        for r in range(n)
        for c in range(dim // CHUNK_SIZE)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?)", rows)


def load_1d(conn, name, vector):
    """Load 1D norm weight as (chunk_id INTEGER, v FLOAT[])."""
    dim = len(vector)
    conn.execute(f"CREATE TEMP TABLE {name} (chunk_id INTEGER, v FLOAT[])")
    rows = [
        (c, vector[c*CHUNK_SIZE:(c+1)*CHUNK_SIZE].astype(np.float32).tolist())
        for c in range(dim // CHUNK_SIZE)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?)", rows)


def load_rope_table(conn, name, cos, sin):
    """Load rope table as (row_id, chunk_id, cos FLOAT[], sin FLOAT[])."""
    seq_len, num_chunks, _ = cos.shape
    conn.execute(f"CREATE TEMP TABLE {name} "
                 f"(row_id INTEGER, chunk_id INTEGER, cos FLOAT[], sin FLOAT[])")
    rows = [
        (pos, c,
         cos[pos, c].astype(np.float32).tolist(),
         sin[pos, c].astype(np.float32).tolist())
        for pos in range(seq_len)
        for c in range(num_chunks)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def read_2d(conn, name, n_rows, dim):
    """Read chunked table back into [n_rows, dim] numpy array."""
    rows = conn.execute(
        f"SELECT row_id, chunk_id, v FROM {name} ORDER BY row_id, chunk_id"
    ).fetchall()
    out = np.zeros((n_rows, dim), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        out[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = np.array(v, dtype=np.float32)
    return out


def run_steps(conn, steps):
    """Execute SQL steps, creating temp tables."""
    for sql, table_name in steps:
        conn.execute(f"CREATE TEMP TABLE {table_name} AS ({sql})")


# ---------------------------------------------------------------------------
# SQL functions (must match run_prefill.py exactly)
# ---------------------------------------------------------------------------

def embed_lookup_sql(tok_table, embed_table, out):
    return [(
        f"SELECT t.pos AS row_id, e.chunk_id, e.v "
        f"FROM {tok_table} t "
        f"JOIN {embed_table} e ON e.row_id = t.token_id "
        f"ORDER BY t.pos, e.chunk_id", out)]


def matmul_sql(act, weight, out, cs=CHUNK_SIZE):
    dp = out + "_dp"
    return [
        (f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
         f"SUM(list_dot_product(a.v, w.v)) AS val "
         f"FROM {act} a JOIN {weight} w ON a.chunk_id = w.chunk_id "
         f"GROUP BY a.row_id, w.row_id", dp),
        (f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
         f"array_agg(val ORDER BY out_col) AS v "
         f"FROM {dp} GROUP BY act_row, out_col // {cs}", out),
    ]


def pivot_sql(table_name, n_chunks):
    in_cols = ", ".join(f"{i} AS chunk{i}" for i in range(n_chunks))
    return (f"SELECT * FROM (PIVOT {table_name} "
            f"ON chunk_id IN ({in_cols}) USING FIRST(v) GROUP BY row_id)")


def pivoted_matmul_sql(act, weight, out, n_chunks, cs=CHUNK_SIZE):
    act_piv = out + "_act_piv"
    wt_piv = out + "_wt_piv"
    dp = out + "_dp"

    dot_expr = " + ".join(
        f"list_dot_product(a.chunk{c}, w.chunk{c})"
        for c in range(n_chunks))

    return [
        (pivot_sql(act, n_chunks), act_piv),
        (pivot_sql(weight, n_chunks), wt_piv),
        (f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
         f"{dot_expr} AS val "
         f"FROM {act_piv} a CROSS JOIN {wt_piv} w", dp),
        (f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
         f"array_agg(val ORDER BY out_col) AS v "
         f"FROM {dp} GROUP BY act_row, out_col // {cs}", out),
    ]


def rmsnorm_sql(inp, gamma, out, hidden_dim=HIDDEN_DIM, eps=EPS):
    sq = out + "_sq"
    return [
        (f"SELECT a.row_id, SUM(list_dot_product(a.v, a.v)) AS ss "
         f"FROM {inp} a GROUP BY a.row_id", sq),
        (f"SELECT a.row_id, a.chunk_id, "
         f"list_transform(list_zip(a.v, g.v), x -> "
         f"CAST(x[1] * x[2] / sqrt({sq}.ss / {hidden_dim}.0 + {eps}) AS FLOAT)) AS v "
         f"FROM {inp} a "
         f"JOIN {gamma} g ON a.chunk_id = g.chunk_id "
         f"JOIN {sq} ON a.row_id = {sq}.row_id", out),
    ]


def rope_sql(q_table, rope_table, out, cs=CHUNK_SIZE):
    half = cs // 2
    # DuckDB lists are 1-based
    return [(
        f"SELECT q.row_id, q.chunk_id, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] AS FLOAT)) AS v_even, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd "
        f"FROM {q_table} q "
        f"JOIN {rope_table} r ON r.chunk_id = q.chunk_id AND r.row_id = q.row_id",
        out)]


def qk_attn_sql(q_rope, k_rope, out, num_q=NUM_Q_HEADS,
                num_kv=NUM_KV_HEADS, head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    cph = head_dim // cs
    cphg = cph * (num_q // num_kv)
    return [(
        f"SELECT q.row_id AS q_tok, k.row_id AS k_tok, "
        f"q.chunk_id // {cph} AS head_id, "
        f"SUM(list_dot_product(q.v_even, k.v_even) + "
        f"list_dot_product(q.v_odd, k.v_odd)) AS score "
        f"FROM {q_rope} q JOIN {k_rope} k "
        f"ON q.chunk_id % {cph} = k.chunk_id % {cph} "
        f"AND q.chunk_id // {cphg} = k.chunk_id // {cph} "
        f"AND k.row_id <= q.row_id "
        f"GROUP BY q.row_id, k.row_id, q.chunk_id // {cph}", out)]


def softmax_sql(inp, out):
    max_t, exp_t, sum_t = out+"_max", out+"_exp", out+"_sum"
    return [
        (f"SELECT q_tok, head_id, MAX(score) AS max_score "
         f"FROM {inp} GROUP BY q_tok, head_id", max_t),
        (f"SELECT s.q_tok, s.k_tok, s.head_id, "
         f"EXP(s.score - m.max_score) AS exp_val "
         f"FROM {inp} s "
         f"JOIN {max_t} m ON s.q_tok = m.q_tok AND s.head_id = m.head_id", exp_t),
        (f"SELECT q_tok, head_id, SUM(exp_val) AS sum_exp "
         f"FROM {exp_t} GROUP BY q_tok, head_id", sum_t),
        (f"SELECT e.q_tok, e.k_tok, e.head_id, "
         f"CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight "
         f"FROM {exp_t} e "
         f"JOIN {sum_t} s ON e.q_tok = s.q_tok AND e.head_id = s.head_id", out),
    ]


def attn_vmul_sql(attn, v_table, out, num_q=NUM_Q_HEADS,
                  num_kv=NUM_KV_HEADS, head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    gs = num_q // num_kv
    cph = head_dim // cs
    scalar_t = out + "_scalar"
    weighted_t = out + "_weighted"
    return [
        (f"SELECT v.row_id AS tok, v.chunk_id, "
         f"UNNEST(generate_series(0, {cs}-1)) AS dim_idx, "
         f"UNNEST(v.v) AS val "
         f"FROM {v_table} v", scalar_t),
        (f"SELECT a.q_tok AS row_id, "
         f"a.head_id * {cph} + s.chunk_id % {cph} AS chunk_id, "
         f"s.dim_idx, SUM(a.attn_weight * s.val) AS val "
         f"FROM {attn} a JOIN {scalar_t} s ON a.k_tok = s.tok "
         f"WHERE s.chunk_id >= (a.head_id // {gs}) * {cph} "
         f"AND s.chunk_id < (a.head_id // {gs} + 1) * {cph} "
         f"GROUP BY a.q_tok, a.head_id * {cph} + s.chunk_id % {cph}, s.dim_idx",
         weighted_t),
        (f"SELECT row_id, chunk_id, "
         f"array_agg(CAST(val AS FLOAT) ORDER BY dim_idx) AS v "
         f"FROM {weighted_t} GROUP BY row_id, chunk_id", out),
    ]


def swiglu_sql(gate, up, out):
    return [(
        f"SELECT g.row_id, g.chunk_id, "
        f"list_transform(list_zip(g.v, u.v), x -> "
        f"CAST((x[1] / (1.0 + EXP(-CAST(x[1] AS DOUBLE)))) * x[2] AS FLOAT)) AS v "
        f"FROM {gate} g JOIN {up} u "
        f"ON g.row_id = u.row_id AND g.chunk_id = u.chunk_id", out)]


def residual_add_sql(a, b, out):
    return [(
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(list_zip(a.v, b.v), x -> "
        f"CAST(x[1] + x[2] AS FLOAT)) AS v "
        f"FROM {a} a JOIN {b} b "
        f"ON a.row_id = b.row_id AND a.chunk_id = b.chunk_id", out)]


# ---------------------------------------------------------------------------
# NumPy reference implementations
# ---------------------------------------------------------------------------

def ref_rmsnorm(x, gamma, eps=EPS):
    ss = np.sum(x ** 2, axis=-1, keepdims=True)
    rms = np.sqrt(ss / x.shape[-1] + eps)
    return (x / rms * gamma).astype(np.float32)


def ref_rope(q, cos, sin):
    """q: [seq, dim], cos/sin: [seq, dim//2]. Returns (even, odd) arrays."""
    half = q.shape[-1] // 2
    q_even = q[:, 0::2]  # positions 0,2,4,...
    q_odd  = q[:, 1::2]  # positions 1,3,5,...
    v_even = (q_even * cos - q_odd * sin).astype(np.float32)
    v_odd  = (q_odd * cos + q_even * sin).astype(np.float32)
    return v_even, v_odd


def ref_qk_attn(q_even, q_odd, k_even, k_odd, num_q, num_kv, head_dim):
    """Returns [seq, seq, num_q_heads] attention scores."""
    gs = num_q // num_kv
    seq = q_even.shape[0]
    half_hd = head_dim // 2
    scores = np.zeros((seq, seq, num_q), dtype=np.float64)
    for h in range(num_q):
        kv_h = h // gs
        qe = q_even[:, h*half_hd:(h+1)*half_hd]
        qo = q_odd[:, h*half_hd:(h+1)*half_hd]
        ke = k_even[:, kv_h*half_hd:(kv_h+1)*half_hd]
        ko = k_odd[:, kv_h*half_hd:(kv_h+1)*half_hd]
        scores[:, :, h] = qe @ ke.T + qo @ ko.T
    # Apply causal mask
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    scores[mask] = -1e9
    return scores.astype(np.float32)


def ref_softmax(scores):
    """scores: [seq, seq, heads]. Returns attention weights."""
    max_s = np.max(scores, axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    # Zero out future positions
    mask = np.triu(np.ones((scores.shape[0], scores.shape[1]), dtype=bool), k=1)
    exp_s[mask] = 0
    sum_s = np.sum(exp_s, axis=1, keepdims=True)
    return (exp_s / sum_s).astype(np.float32)


def ref_attn_vmul(attn, v, num_q, num_kv, head_dim):
    """attn: [seq, seq, heads], v: [seq, kv_dim]. Returns [seq, hidden]."""
    gs = num_q // num_kv
    seq = attn.shape[0]
    hidden = num_q * head_dim
    out = np.zeros((seq, hidden), dtype=np.float32)
    for h in range(num_q):
        kv_h = h // gs
        v_h = v[:, kv_h*head_dim:(kv_h+1)*head_dim]
        out[:, h*head_dim:(h+1)*head_dim] = (attn[:, :, h] @ v_h).astype(np.float32)
    return out


def ref_swiglu(gate, up):
    # Llama3 HuggingFace convention: SiLU(gate_proj) * up_proj
    silu_gate = gate / (1.0 + np.exp(-gate.astype(np.float64)))
    return (silu_gate * up).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, result, expected, atol=ATOL, rtol=RTOL):
        try:
            np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)
            self.passed += 1
            return True
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            return False


def test_embed(tr, verbose=False):
    """Test EmbedLookup."""
    embed_np = rng.standard_normal((VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32)
    token_ids = [3, 7, 11]

    conn = new_conn()
    load_2d(conn, "embed_tokens", embed_np)
    conn.execute("CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)")
    conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                     [(i, t) for i, t in enumerate(token_ids)])
    run_steps(conn, embed_lookup_sql("input_tokens", "embed_tokens", "out"))

    result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
    expected = embed_np[token_ids]

    ok = tr.check("EmbedLookup", result, expected)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} EmbedLookup")


def test_matmul(tr, verbose=False):
    """Test MatMul (basic join-based)."""
    act_np = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
    wt_np = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32)

    conn = new_conn()
    load_2d(conn, "act", act_np)
    load_2d(conn, "wt", wt_np)
    run_steps(conn, matmul_sql("act", "wt", "out", cs=CHUNK_SIZE))

    result = read_2d(conn, "out", SEQ_LEN, FFN_DIM)
    expected = (act_np @ wt_np.T).astype(np.float32)

    ok = tr.check("MatMul(basic)", result, expected, atol=1e-3)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} MatMul(basic)")


def test_pivoted_matmul(tr, verbose=False):
    """Test pivoted MatMul (ROW2COL)."""
    n_chunks = HIDDEN_DIM // CHUNK_SIZE
    act_np = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
    wt_np = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32)

    conn = new_conn()
    load_2d(conn, "act", act_np)
    load_2d(conn, "wt", wt_np)
    run_steps(conn, pivoted_matmul_sql("act", "wt", "out", n_chunks, cs=CHUNK_SIZE))

    result = read_2d(conn, "out", SEQ_LEN, FFN_DIM)
    expected = (act_np @ wt_np.T).astype(np.float32)

    ok = tr.check("MatMul(pivoted)", result, expected, atol=1e-3)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} MatMul(pivoted)")


def test_rmsnorm(tr, verbose=False):
    """Test RMSNorm."""
    x_np = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
    gamma_np = rng.standard_normal(HIDDEN_DIM).astype(np.float32)

    conn = new_conn()
    load_2d(conn, "inp", x_np)
    load_1d(conn, "gamma", gamma_np)
    run_steps(conn, rmsnorm_sql("inp", "gamma", "out",
                                hidden_dim=HIDDEN_DIM, eps=EPS))

    result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
    expected = ref_rmsnorm(x_np, gamma_np)

    ok = tr.check("RMSNorm", result, expected)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} RMSNorm")


def test_rope(tr, verbose=False):
    """Test RoPE."""
    q_np = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)

    half = CHUNK_SIZE // 2
    num_chunks = HIDDEN_DIM // CHUNK_SIZE
    cos_np = rng.standard_normal((SEQ_LEN, num_chunks, half)).astype(np.float32)
    sin_np = rng.standard_normal((SEQ_LEN, num_chunks, half)).astype(np.float32)

    conn = new_conn()
    load_2d(conn, "q", q_np)
    load_rope_table(conn, "rope", cos_np, sin_np)
    run_steps(conn, rope_sql("q", "rope", "out", cs=CHUNK_SIZE))

    # Read split output
    rows = conn.execute(
        "SELECT row_id, chunk_id, v_even, v_odd FROM out ORDER BY row_id, chunk_id"
    ).fetchall()
    result_even = np.zeros((SEQ_LEN, HIDDEN_DIM // 2), dtype=np.float32)
    result_odd = np.zeros((SEQ_LEN, HIDDEN_DIM // 2), dtype=np.float32)
    for row_id, chunk_id, v_even, v_odd in rows:
        result_even[row_id, chunk_id*half:(chunk_id+1)*half] = np.array(v_even, dtype=np.float32)
        result_odd[row_id, chunk_id*half:(chunk_id+1)*half] = np.array(v_odd, dtype=np.float32)

    # NumPy reference
    cos_flat = cos_np.reshape(SEQ_LEN, -1)  # [seq, dim//2]
    sin_flat = sin_np.reshape(SEQ_LEN, -1)
    exp_even, exp_odd = ref_rope(q_np, cos_flat, sin_flat)

    ok1 = tr.check("RoPE(even)", result_even, exp_even)
    ok2 = tr.check("RoPE(odd)", result_odd, exp_odd)
    if verbose:
        print(f"  {'PASS' if ok1 else 'FAIL'} RoPE(even)")
        print(f"  {'PASS' if ok2 else 'FAIL'} RoPE(odd)")


def test_softmax(tr, verbose=False):
    """Test Softmax on QK attention scores."""
    # Create synthetic attention scores (causal)
    scores_np = rng.standard_normal((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)

    conn = new_conn()
    # Load as scalar table (q_tok, k_tok, head_id, score)
    conn.execute("CREATE TEMP TABLE scores "
                 "(q_tok INTEGER, k_tok INTEGER, head_id INTEGER, score FLOAT)")
    rows = []
    for q in range(SEQ_LEN):
        for k in range(q + 1):  # causal: k <= q
            for h in range(NUM_Q_HEADS):
                rows.append((q, k, h, float(scores_np[q, k, h])))
    conn.executemany("INSERT INTO scores VALUES (?, ?, ?, ?)", rows)

    run_steps(conn, softmax_sql("scores", "attn_w"))

    result_rows = conn.execute(
        "SELECT q_tok, k_tok, head_id, attn_weight FROM attn_w "
        "ORDER BY q_tok, k_tok, head_id"
    ).fetchall()

    result = np.zeros((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS), dtype=np.float32)
    for q, k, h, w in result_rows:
        result[q, k, h] = w

    # NumPy reference (only causal positions)
    mask = np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool), k=1)
    scores_masked = scores_np.copy()
    scores_masked[mask] = -1e9
    expected = ref_softmax(scores_masked)

    ok = tr.check("Softmax", result, expected, atol=1e-3)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} Softmax")


def test_swiglu(tr, verbose=False):
    """Test SwiGLU: SiLU(gate) * up  (Llama3/HuggingFace convention)."""
    gate_np = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)
    up_np = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)

    conn = new_conn()
    load_2d(conn, "gate", gate_np)
    load_2d(conn, "up", up_np)
    run_steps(conn, swiglu_sql("gate", "up", "out"))

    result = read_2d(conn, "out", SEQ_LEN, FFN_DIM)
    expected = ref_swiglu(gate_np, up_np)

    ok = tr.check("SwiGLU", result, expected)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} SwiGLU")


def test_residual_add(tr, verbose=False):
    """Test ResidualAdd."""
    a_np = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
    b_np = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)

    conn = new_conn()
    load_2d(conn, "a", a_np)
    load_2d(conn, "b", b_np)
    run_steps(conn, residual_add_sql("a", "b", "out"))

    result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
    expected = (a_np + b_np).astype(np.float32)

    ok = tr.check("ResidualAdd", result, expected)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} ResidualAdd")


def test_attn_vmul(tr, verbose=False):
    """Test AttnVMul with identity attention."""
    # Use identity attention (each token attends only to itself)
    attn_np = np.zeros((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS), dtype=np.float32)
    for i in range(SEQ_LEN):
        attn_np[i, i, :] = 1.0

    v_np = rng.standard_normal((SEQ_LEN, KV_DIM)).astype(np.float32)

    conn = new_conn()
    load_2d(conn, "v", v_np)

    # Load attention as scalar table
    conn.execute("CREATE TEMP TABLE attn "
                 "(q_tok INTEGER, k_tok INTEGER, head_id INTEGER, attn_weight FLOAT)")
    rows = []
    for q in range(SEQ_LEN):
        for k in range(q + 1):
            for h in range(NUM_Q_HEADS):
                rows.append((q, k, h, float(attn_np[q, k, h])))
    conn.executemany("INSERT INTO attn VALUES (?, ?, ?, ?)", rows)

    run_steps(conn, attn_vmul_sql("attn", "v", "out",
                                   num_q=NUM_Q_HEADS, num_kv=NUM_KV_HEADS,
                                   head_dim=HEAD_DIM, cs=CHUNK_SIZE))

    result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
    expected = ref_attn_vmul(attn_np, v_np, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    ok = tr.check("AttnVMul(identity)", result, expected)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} AttnVMul(identity)")


def test_full_layer(tr, verbose=False):
    """Test a complete single-layer forward pass."""
    x_np = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
    norm1_w = rng.standard_normal(HIDDEN_DIM).astype(np.float32)
    norm2_w = rng.standard_normal(HIDDEN_DIM).astype(np.float32)
    wq = rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)
    wk = rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32)
    wv = rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32)
    wo = rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)
    wg = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32)
    wu = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32)
    wd = rng.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32)

    half = CHUNK_SIZE // 2
    num_chunks = HIDDEN_DIM // CHUNK_SIZE
    cos_np = rng.standard_normal((SEQ_LEN, num_chunks, half)).astype(np.float32)
    sin_np = rng.standard_normal((SEQ_LEN, num_chunks, half)).astype(np.float32)

    # --- NumPy reference ---
    x_norm1 = ref_rmsnorm(x_np, norm1_w)
    q = (x_norm1 @ wq.T).astype(np.float32)
    k = (x_norm1 @ wk.T).astype(np.float32)
    v = (x_norm1 @ wv.T).astype(np.float32)

    cos_flat = cos_np.reshape(SEQ_LEN, -1)
    sin_flat = sin_np.reshape(SEQ_LEN, -1)

    q_even, q_odd = ref_rope(q, cos_flat, sin_flat)
    kv_chunks = KV_DIM // CHUNK_SIZE
    k_cos = cos_np[:, :kv_chunks, :].reshape(SEQ_LEN, -1)
    k_sin = sin_np[:, :kv_chunks, :].reshape(SEQ_LEN, -1)
    k_even, k_odd = ref_rope(k, k_cos, k_sin)

    scores = ref_qk_attn(q_even, q_odd, k_even, k_odd,
                          NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
    attn = ref_softmax(scores)
    attn_out = ref_attn_vmul(attn, v, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
    o = (attn_out @ wo.T).astype(np.float32)
    x_after_attn = (x_np + o).astype(np.float32)

    x_norm2 = ref_rmsnorm(x_after_attn, norm2_w)
    gate = (x_norm2 @ wg.T).astype(np.float32)
    up = (x_norm2 @ wu.T).astype(np.float32)
    ffn_act = ref_swiglu(gate, up)
    down = (ffn_act @ wd.T).astype(np.float32)
    expected = (x_after_attn + down).astype(np.float32)

    # --- DuckDB SQL ---
    conn = new_conn()
    load_2d(conn, "x_in", x_np)
    load_1d(conn, "layer_0_norm1", norm1_w)
    load_1d(conn, "layer_0_norm2", norm2_w)
    load_2d(conn, "layer_0_q_proj", wq)
    load_2d(conn, "layer_0_k_proj", wk)
    load_2d(conn, "layer_0_v_proj", wv)
    load_2d(conn, "layer_0_o_proj", wo)
    load_2d(conn, "layer_0_gate_proj", wg)
    load_2d(conn, "layer_0_up_proj", wu)
    load_2d(conn, "layer_0_down_proj", wd)
    load_rope_table(conn, "rope", cos_np, sin_np)

    steps = []
    steps += rmsnorm_sql("x_in", "layer_0_norm1", "norm1_out")
    steps += matmul_sql("norm1_out", "layer_0_q_proj", "q")
    steps += matmul_sql("norm1_out", "layer_0_k_proj", "k")
    steps += matmul_sql("norm1_out", "layer_0_v_proj", "v")
    steps += rope_sql("q", "rope", "q_rope")
    steps += rope_sql("k", "rope", "k_rope")
    steps += qk_attn_sql("q_rope", "k_rope", "qk")
    steps += softmax_sql("qk", "attn_w")
    steps += attn_vmul_sql("attn_w", "v", "attn_out")
    steps += matmul_sql("attn_out", "layer_0_o_proj", "o_proj")
    steps += residual_add_sql("x_in", "o_proj", "x_after_attn")
    steps += rmsnorm_sql("x_after_attn", "layer_0_norm2", "norm2_out")
    steps += matmul_sql("norm2_out", "layer_0_gate_proj", "gate")
    steps += matmul_sql("norm2_out", "layer_0_up_proj", "up")
    steps += swiglu_sql("gate", "up", "ffn_act")
    steps += matmul_sql("ffn_act", "layer_0_down_proj", "down")
    steps += residual_add_sql("x_after_attn", "down", "x_out")

    run_steps(conn, steps)
    result = read_2d(conn, "x_out", SEQ_LEN, HIDDEN_DIM)

    ok = tr.check("FullLayer", result, expected, atol=5e-3, rtol=1e-3)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'} FullLayer "
              f"(max diff={np.max(np.abs(result - expected)):.6f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "embed": test_embed,
    "matmul": test_matmul,
    "pivoted_matmul": test_pivoted_matmul,
    "rmsnorm": test_rmsnorm,
    "rope": test_rope,
    "softmax": test_softmax,
    "swiglu": test_swiglu,
    "residual_add": test_residual_add,
    "attn_vmul": test_attn_vmul,
    "full_layer": test_full_layer,
}


def main():
    parser = argparse.ArgumentParser(
        description="Verify SQL operators against NumPy reference")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--test", type=str, default=None,
                        help="Run a single test (e.g., rope, matmul)")
    args = parser.parse_args()

    tr = TestResult()
    tests = {args.test: ALL_TESTS[args.test]} if args.test else ALL_TESTS

    print(f"Running {len(tests)} operator verification tests...\n")

    for name, test_fn in tests.items():
        try:
            test_fn(tr, verbose=args.verbose)
        except Exception as e:
            tr.failed += 1
            tr.errors.append((name, traceback.format_exc()))
            print(f"  ERROR {name}: {e}")

    print(f"\n{'='*50}")
    print(f"PASSED: {tr.passed}  FAILED: {tr.failed}")

    if tr.errors:
        print("\nFailures:")
        for name, msg in tr.errors:
            print(f"  {name}: {msg[:200]}")

    if tr.failed > 0:
        print("\nRESULT: FAIL")
        sys.exit(1)
    else:
        print("\nRESULT: ALL PASS")


if __name__ == "__main__":
    main()
