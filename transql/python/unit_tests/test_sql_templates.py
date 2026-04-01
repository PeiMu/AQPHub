"""
Unit tests for TranSQL+ SQL template operators.

Each test creates tiny synthetic data, loads it into an in-memory DuckDB,
executes the SQL that matches sql_template.cpp, and compares results to a
NumPy reference.

Test parameters (must stay consistent with each other):
    CHUNK_SIZE=2, HIDDEN_DIM=4, KV_DIM=2, FFN_DIM=8
    NUM_Q_HEADS=2, NUM_KV_HEADS=1, HEAD_DIM=2, SEQ_LEN=3

Run:
    cd /path/to/AQP_middleware
    python -m pytest transql/python/unittest/ -v
    # or
    python transql/python/unittest/test_sql_templates.py
"""

import unittest
import numpy as np
import duckdb

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 2
SEQ_LEN      = 3
HIDDEN_DIM   = 4      # 2 chunks per token row
KV_DIM       = 2      # 1 chunk; num_kv_heads=1, head_dim=2
FFN_DIM      = 8      # 4 chunks; used for SwiGLU gate/up
VOCAB_SIZE   = 16
NUM_Q_HEADS  = 2
NUM_KV_HEADS = 1
HEAD_DIM     = 2      # hidden_dim / num_q_heads = 4/2
EPS          = 1e-5
ATOL         = 1e-4
RTOL         = 1e-4

rng = np.random.default_rng(42)


# ===========================================================================
# DuckDB helpers
# ===========================================================================

def new_conn():
    return duckdb.connect(":memory:")


def load_2d(conn, name, matrix):
    """Load [n, dim] float32 matrix as chunked table (row_id, chunk_id, v FLOAT[])."""
    n, dim = matrix.shape
    conn.execute(
        f"CREATE TEMP TABLE {name} (row_id INTEGER, chunk_id INTEGER, v FLOAT[])"
    )
    rows = [
        (r, c, matrix[r, c*CHUNK_SIZE:(c+1)*CHUNK_SIZE].astype(np.float32).tolist())
        for r in range(n)
        for c in range(dim // CHUNK_SIZE)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?)", rows)


def load_norm_weight(conn, name, vector):
    """Load 1D norm weight as (chunk_id INTEGER, v FLOAT[])."""
    dim = len(vector)
    conn.execute(f"CREATE TEMP TABLE {name} (chunk_id INTEGER, v FLOAT[])")
    rows = [
        (c, vector[c*CHUNK_SIZE:(c+1)*CHUNK_SIZE].astype(np.float32).tolist())
        for c in range(dim // CHUNK_SIZE)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?)", rows)


def load_rope_table(conn, name, cos, sin):
    """Load rope table as (row_id INTEGER, chunk_id INTEGER, cos FLOAT[], sin FLOAT[]).
    cos, sin: [seq_len, num_chunks, half_chunk_size]
    """
    seq_len, num_chunks, _ = cos.shape
    conn.execute(
        f"CREATE TEMP TABLE {name} "
        f"(row_id INTEGER, chunk_id INTEGER, cos FLOAT[], sin FLOAT[])"
    )
    rows = [
        (pos, c,
         cos[pos, c].astype(np.float32).tolist(),
         sin[pos, c].astype(np.float32).tolist())
        for pos in range(seq_len)
        for c in range(num_chunks)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def load_rope_out(conn, name, matrix):
    """Load [seq, dim] float32 matrix directly in RoPE split output format:
    (row_id INTEGER, chunk_id INTEGER, v_even FLOAT[], v_odd FLOAT[]).
    Used to inject pre-computed RoPE results into QKAttn/AttnVMul tests.
    """
    half = CHUNK_SIZE // 2
    n, dim = matrix.shape
    conn.execute(
        f"CREATE TEMP TABLE {name} "
        f"(row_id INTEGER, chunk_id INTEGER, v_even FLOAT[], v_odd FLOAT[])"
    )
    rows = [
        (
            r, c,
            [float(matrix[r, c*CHUNK_SIZE + 2*i])     for i in range(half)],
            [float(matrix[r, c*CHUNK_SIZE + 2*i + 1]) for i in range(half)],
        )
        for r in range(n)
        for c in range(dim // CHUNK_SIZE)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def load_scores(conn, name, scores_np):
    """Load [q_tok, k_tok, heads] score array as scalar rows."""
    conn.execute(
        f"CREATE TEMP TABLE {name} "
        f"(q_tok INTEGER, k_tok INTEGER, head_id INTEGER, score FLOAT)"
    )
    q_len, k_len, n_heads = scores_np.shape
    rows = [
        (q, k, h, float(scores_np[q, k, h]))
        for q in range(q_len)
        for k in range(k_len)
        for h in range(n_heads)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def load_attn_weights(conn, name, attn_np):
    """Load [q_tok, k_tok, heads] attention-weight array as scalar rows."""
    conn.execute(
        f"CREATE TEMP TABLE {name} "
        f"(q_tok INTEGER, k_tok INTEGER, head_id INTEGER, attn_weight FLOAT)"
    )
    q_len, k_len, n_heads = attn_np.shape
    rows = [
        (q, k, h, float(attn_np[q, k, h]))
        for q in range(q_len)
        for k in range(k_len)
        for h in range(n_heads)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def read_2d(conn, name, n_rows, dim):
    """Read chunked table back to [n_rows, dim] float32 array."""
    rows = conn.execute(
        f"SELECT row_id, chunk_id, v FROM {name} ORDER BY row_id, chunk_id"
    ).fetchall()
    out = np.zeros((n_rows, dim), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        r, c = int(row_id), int(chunk_id)
        out[r, c*CHUNK_SIZE:(c+1)*CHUNK_SIZE] = v
    return out


def read_rope_out(conn, name, n_rows, dim):
    """Read RoPE output (v_even, v_odd) back to interleaved [n_rows, dim]."""
    half = CHUNK_SIZE // 2
    rows = conn.execute(
        f"SELECT row_id, chunk_id, v_even, v_odd FROM {name} "
        f"ORDER BY row_id, chunk_id"
    ).fetchall()
    out = np.zeros((n_rows, dim), dtype=np.float32)
    for row_id, chunk_id, v_even, v_odd in rows:
        r, c = int(row_id), int(chunk_id)
        for i in range(half):
            out[r, c*CHUNK_SIZE + 2*i]     = v_even[i]
            out[r, c*CHUNK_SIZE + 2*i + 1] = v_odd[i]
    return out


def read_scores(conn, name, seq_len, num_heads, val_col="score"):
    """Read (q_tok, k_tok, head_id, <val_col>) into [q, k, heads] float32 array."""
    rows = conn.execute(
        f"SELECT q_tok, k_tok, head_id, {val_col} FROM {name} "
        f"ORDER BY head_id, q_tok, k_tok"
    ).fetchall()
    out = np.zeros((seq_len, seq_len, num_heads), dtype=np.float32)
    for q_tok, k_tok, head_id, val in rows:
        out[int(q_tok), int(k_tok), int(head_id)] = val
    return out


def run_steps(conn, steps):
    """Execute [(select_sql, table_name)] as CREATE TEMP TABLE AS (...)."""
    for sql, tname in steps:
        conn.execute(f"CREATE TEMP TABLE {tname} AS ({sql})")


# ===========================================================================
# SQL template mirrors — must stay in sync with sql_template.cpp
# ===========================================================================

def embed_lookup_sql(tokens_table, embed_table, out_table):
    sql = (
        f"SELECT t.pos AS row_id, e.chunk_id, e.v "
        f"FROM {tokens_table} t "
        f"JOIN {embed_table} e ON t.token_id = e.row_id "
        f"ORDER BY t.pos, e.chunk_id"
    )
    return [(sql, out_table)]


def matmul_sql(act, weight, out, chunk_size=CHUNK_SIZE):
    dp = out + "_dp"
    cs = str(chunk_size)
    return [
        (
            f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
            f"SUM(list_dot_product(a.v, w.v)) AS val "
            f"FROM {act} a JOIN {weight} w ON a.chunk_id = w.chunk_id "
            f"GROUP BY a.row_id, w.row_id",
            dp,
        ),
        (
            f"SELECT act_row AS row_id, "
            f"out_col // {cs} AS chunk_id, "
            f"array_agg(val ORDER BY out_col) AS v "
            f"FROM {dp} GROUP BY act_row, out_col // {cs}",
            out,
        ),
    ]


def rmsnorm_sql(inp, gamma, out, hidden_dim=HIDDEN_DIM, eps=EPS):
    sq = out + "_sq"
    eps_str = f"{eps:.10f}"
    return [
        (
            f"SELECT row_id, "
            f"SUM(list_sum(list_transform(v, x -> x * x))) AS sq_sum "
            f"FROM {inp} GROUP BY row_id",
            sq,
        ),
        (
            f"SELECT n.row_id, n.chunk_id, "
            f"list_transform(generate_series(1, len(n.v)), "
            f"i -> CAST(n.v[i] / sqrt(s.sq_sum / {hidden_dim}.0 + {eps_str}) "
            f"* w.v[i] AS FLOAT)) AS v "
            f"FROM {inp} n "
            f"JOIN {sq} s ON n.row_id = s.row_id "
            f"JOIN {gamma} w ON n.chunk_id = w.chunk_id",
            out,
        ),
    ]


def rope_sql(q_table, rope_table, out_table, chunk_size=CHUNK_SIZE):
    half = chunk_size // 2
    sql = (
        f"SELECT q.row_id, q.chunk_id, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] AS FLOAT)) AS v_even, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd "
        f"FROM {q_table} q "
        f"JOIN {rope_table} r ON r.chunk_id = q.chunk_id AND r.row_id = q.row_id"
    )
    return [(sql, out_table)]


def qk_attn_sql(q_rope, k_rope, out,
                num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM, chunk_size=CHUNK_SIZE):
    cph  = head_dim // chunk_size
    cphg = cph * (num_q_heads // num_kv_heads)
    sql = (
        f"SELECT q.row_id AS q_tok, k.row_id AS k_tok, "
        f"q.chunk_id // {cph} AS head_id, "
        f"SUM(list_dot_product(q.v_even, k.v_even) + "
        f"list_dot_product(q.v_odd, k.v_odd)) AS score "
        f"FROM {q_rope} q JOIN {k_rope} k "
        f"ON q.chunk_id % {cph} = k.chunk_id % {cph} "
        f"AND q.chunk_id // {cphg} = k.chunk_id // {cph} "
        f"AND k.row_id <= q.row_id "
        f"GROUP BY q.row_id, k.row_id, q.chunk_id // {cph}"
    )
    return [(sql, out)]


def softmax_sql(inp, out):
    max_t = out + "_max"
    exp_t = out + "_exp"
    sum_t = out + "_sum"
    return [
        (
            f"SELECT q_tok, head_id, MAX(score) AS max_score "
            f"FROM {inp} GROUP BY q_tok, head_id",
            max_t,
        ),
        (
            f"SELECT s.q_tok, s.k_tok, s.head_id, "
            f"EXP(s.score - m.max_score) AS exp_val "
            f"FROM {inp} s "
            f"JOIN {max_t} m ON s.q_tok = m.q_tok AND s.head_id = m.head_id",
            exp_t,
        ),
        (
            f"SELECT q_tok, head_id, SUM(exp_val) AS sum_exp "
            f"FROM {exp_t} GROUP BY q_tok, head_id",
            sum_t,
        ),
        (
            f"SELECT e.q_tok, e.k_tok, e.head_id, "
            f"CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight "
            f"FROM {exp_t} e "
            f"JOIN {sum_t} s ON e.q_tok = s.q_tok AND e.head_id = s.head_id",
            out,
        ),
    ]


def attn_vmul_sql(attn_table, v_table, out,
                  num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
                  head_dim=HEAD_DIM, chunk_size=CHUNK_SIZE):
    cph = head_dim // chunk_size
    gs  = num_q_heads // num_kv_heads
    vs  = out + "_vs"
    wt  = out + "_w"
    return [
        (
            f"SELECT row_id AS tok, chunk_id, "
            f"unnest(generate_series(0, {chunk_size - 1})) AS elem_pos, "
            f"CAST(unnest(v) AS FLOAT) AS val "
            f"FROM {v_table}",
            vs,
        ),
        (
            f"SELECT s.q_tok, "
            f"s.head_id * {cph} + v.chunk_id % {cph} AS out_chunk_id, "
            f"v.elem_pos, "
            f"CAST(SUM(s.attn_weight * v.val) AS FLOAT) AS val "
            f"FROM {attn_table} s "
            f"JOIN {vs} v ON s.k_tok = v.tok "
            f"AND s.head_id // {gs} = v.chunk_id // {cph} "
            f"GROUP BY s.q_tok, s.head_id, v.chunk_id, v.elem_pos",
            wt,
        ),
        (
            f"SELECT q_tok AS row_id, out_chunk_id AS chunk_id, "
            f"array_agg(val ORDER BY elem_pos) AS v "
            f"FROM {wt} GROUP BY q_tok, out_chunk_id",
            out,
        ),
    ]


def swiglu_sql(gate, up, out):
    sql = (
        f"SELECT g.row_id, g.chunk_id, "
        f"list_transform(generate_series(1, len(g.v)), "
        f"i -> CAST(g.v[i] * (u.v[i] / (1.0 + exp(-u.v[i]))) AS FLOAT)) AS v "
        f"FROM {gate} g "
        f"JOIN {up} u ON g.row_id = u.row_id AND g.chunk_id = u.chunk_id"
    )
    return [(sql, out)]


def residual_add_sql(a, b, out):
    sql = (
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(generate_series(1, len(a.v)), "
        f"i -> CAST(a.v[i] + b.v[i] AS FLOAT)) AS v "
        f"FROM {a} a "
        f"JOIN {b} b ON a.row_id = b.row_id AND a.chunk_id = b.chunk_id"
    )
    return [(sql, out)]


# ===========================================================================
# NumPy reference implementations
# ===========================================================================

def ref_embed_lookup(embed, token_ids):
    return embed[token_ids].astype(np.float32)


def ref_matmul(act, weight):
    return (act @ weight.T).astype(np.float32)


def ref_rmsnorm(x, gamma, hidden_dim, eps):
    sq_sum = np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(sq_sum / hidden_dim + eps) * gamma).astype(np.float32)


def ref_rope(q, cos, sin):
    """cos, sin: [seq_len, num_chunks, half]. Position-dependent rotation."""
    half = CHUNK_SIZE // 2
    seq_len, num_chunks = q.shape[0], q.shape[1] // CHUNK_SIZE
    out = q.copy().astype(np.float32)
    for pos in range(seq_len):
        for c in range(num_chunks):
            for i in range(half):
                q_e = np.float32(q[pos, c*CHUNK_SIZE + 2*i])
                q_o = np.float32(q[pos, c*CHUNK_SIZE + 2*i + 1])
                c_i = np.float32(cos[pos, c, i])
                s_i = np.float32(sin[pos, c, i])
                out[pos, c*CHUNK_SIZE + 2*i]     = q_e * c_i - q_o * s_i
                out[pos, c*CHUNK_SIZE + 2*i + 1] = q_o * c_i + q_e * s_i
    return out


def ref_qk_attn(q_rot, k_rot, num_q_heads, num_kv_heads, head_dim):
    """q_rot, k_rot: [seq, dim] (full rotated) -> [q_tok, k_tok, heads]
    Note: 1/sqrt(d_k) scaling is absorbed into W_Q (constant folding)."""
    seq = q_rot.shape[0]
    gs  = num_q_heads // num_kv_heads
    out = np.zeros((seq, seq, num_q_heads), dtype=np.float32)
    for h in range(num_q_heads):
        kv_h = h // gs
        q_h  = q_rot[:, h*head_dim:(h+1)*head_dim]
        k_h  = k_rot[:, kv_h*head_dim:(kv_h+1)*head_dim]
        out[:, :, h] = (q_h @ k_h.T).astype(np.float32)
    return out


def ref_softmax(scores):
    """scores: [q_tok, k_tok, heads] -> row-normalised attention weights."""
    s_max  = np.max(scores, axis=1, keepdims=True)
    exp_s  = np.exp(scores - s_max).astype(np.float32)
    return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)


def ref_attn_vmul(attn, v, num_q_heads, num_kv_heads, head_dim):
    """attn: [q_tok, k_tok, heads], v: [seq, kv_dim] -> [seq, q_dim]"""
    seq = attn.shape[0]
    gs  = num_q_heads // num_kv_heads
    out = np.zeros((seq, num_q_heads * head_dim), dtype=np.float32)
    for h in range(num_q_heads):
        kv_h = h // gs
        v_h  = v[:, kv_h*head_dim:(kv_h+1)*head_dim]
        out[:, h*head_dim:(h+1)*head_dim] = (attn[:, :, h] @ v_h).astype(np.float32)
    return out


def ref_swiglu(gate, up):
    silu_up = up / (1.0 + np.exp(-up.astype(np.float64)))
    return (gate * silu_up).astype(np.float32)


def ref_residual_add(a, b):
    return (a + b).astype(np.float32)


# ===========================================================================
# Test cases
# ===========================================================================

class TestEmbedLookup(unittest.TestCase):
    """EmbedLookupSQL: tokens JOIN embed ON token_id = row_id."""

    def test_basic(self):
        embed     = rng.standard_normal((VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32)
        token_ids = rng.integers(0, VOCAB_SIZE, size=SEQ_LEN)

        conn = new_conn()
        conn.execute("CREATE TEMP TABLE tok (pos INTEGER, token_id INTEGER)")
        conn.executemany("INSERT INTO tok VALUES (?, ?)",
                         [(i, int(token_ids[i])) for i in range(SEQ_LEN)])
        load_2d(conn, "embed", embed)
        run_steps(conn, embed_lookup_sql("tok", "embed", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_embed_lookup(embed, token_ids)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="EmbedLookup mismatch")


class TestMatMul(unittest.TestCase):
    """MatMulSQL: two-step chunked dot-product + re-chunk."""

    def test_square(self):
        act    = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        weight = rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "act", act)
        load_2d(conn, "w", weight)
        run_steps(conn, matmul_sql("act", "w", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_matmul(act, weight)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="MatMul (square) mismatch")

    def test_rect_expand(self):
        """[seq × hidden] × [ffn × hidden]^T → [seq × ffn]  (gate/up projection)."""
        act    = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        weight = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "act", act)
        load_2d(conn, "w", weight)
        run_steps(conn, matmul_sql("act", "w", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, FFN_DIM)
        expected = ref_matmul(act, weight)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="MatMul (expand) mismatch")

    def test_rect_contract(self):
        """[seq × ffn] × [hidden × ffn]^T → [seq × hidden]  (down projection)."""
        act    = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)
        weight = rng.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "act", act)
        load_2d(conn, "w", weight)
        run_steps(conn, matmul_sql("act", "w", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_matmul(act, weight)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="MatMul (contract) mismatch")


class TestRMSNorm(unittest.TestCase):
    """RMSNormSQL: two-step sum-of-squares + normalise × gamma."""

    def test_basic(self):
        x     = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        gamma = (rng.standard_normal(HIDDEN_DIM) + 1.0).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "inp", x)
        load_norm_weight(conn, "gamma", gamma)
        run_steps(conn, rmsnorm_sql("inp", "gamma", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_rmsnorm(x, gamma, HIDDEN_DIM, EPS)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="RMSNorm mismatch")

    def test_unit_gamma(self):
        """With gamma=1 the output should be unit-RMS vectors."""
        x     = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        gamma = np.ones(HIDDEN_DIM, dtype=np.float32)

        conn = new_conn()
        load_2d(conn, "inp", x)
        load_norm_weight(conn, "gamma", gamma)
        run_steps(conn, rmsnorm_sql("inp", "gamma", "out"))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
        rms    = np.sqrt(np.mean(result.astype(np.float64) ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones(SEQ_LEN), atol=2e-4,
                                   err_msg="RMSNorm unit-gamma: RMS != 1")


class TestRoPE(unittest.TestCase):
    """RoPESQL: split even/odd via list_select + apply cos/sin rotation."""

    def setUp(self):
        num_chunks = HIDDEN_DIM // CHUNK_SIZE
        half       = CHUNK_SIZE // 2
        # Position-dependent: [seq_len, num_chunks, half]
        self.cos = rng.uniform(0.5, 1.0, (SEQ_LEN, num_chunks, half)).astype(np.float32)
        self.sin = rng.uniform(0.0, 0.5, (SEQ_LEN, num_chunks, half)).astype(np.float32)

    def test_basic(self):
        q = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "q", q)
        load_rope_table(conn, "rope", self.cos, self.sin)
        run_steps(conn, rope_sql("q", "rope", "out"))

        result   = read_rope_out(conn, "out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_rope(q, self.cos, self.sin)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="RoPE mismatch")

    def test_identity_at_zero_angle(self):
        """With sin=0, cos=1: rotation is identity."""
        q   = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        num_chunks = HIDDEN_DIM // CHUNK_SIZE
        half       = CHUNK_SIZE // 2
        cos = np.ones((SEQ_LEN, num_chunks, half), dtype=np.float32)
        sin = np.zeros((SEQ_LEN, num_chunks, half), dtype=np.float32)

        conn = new_conn()
        load_2d(conn, "q", q)
        load_rope_table(conn, "rope", cos, sin)
        run_steps(conn, rope_sql("q", "rope", "out"))

        result = read_rope_out(conn, "out", SEQ_LEN, HIDDEN_DIM)
        np.testing.assert_allclose(result, q, atol=ATOL, rtol=RTOL,
                                   err_msg="RoPE identity mismatch")


class TestQKAttn(unittest.TestCase):
    """QKAttnSQL: GQA-aware Q @ K^T attention scores."""

    def test_basic(self):
        # Inject pre-rotated Q/K directly in split format
        q_rot = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        k_rot = rng.standard_normal((SEQ_LEN, KV_DIM)).astype(np.float32)

        conn = new_conn()
        load_rope_out(conn, "q_rope", q_rot)
        load_rope_out(conn, "k_rope", k_rot)
        run_steps(conn, qk_attn_sql("q_rope", "k_rope", "scores"))

        result   = read_scores(conn, "scores", SEQ_LEN, NUM_Q_HEADS, "score")
        expected = ref_qk_attn(q_rot, k_rot, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        # SQL only returns lower-triangular (k_tok <= q_tok); compare only those.
        lower = np.tril(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool))
        np.testing.assert_allclose(result[lower], expected[lower], atol=ATOL, rtol=RTOL,
                                   err_msg="QKAttn mismatch")

    def test_scale(self):
        """Score of identical unit vectors: raw dot product (no 1/sqrt scaling).
        Constant folding absorbs 1/sqrt(d_k) into W_Q during preprocessing."""
        # q = k = e_0 (unit vector along first dim) for every token and head
        q = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
        k = np.zeros((1, KV_DIM),    dtype=np.float32)
        q[0, 0] = 1.0
        k[0, 0] = 1.0

        conn = new_conn()
        load_rope_out(conn, "q_rope", q)
        load_rope_out(conn, "k_rope", k)
        run_steps(conn, qk_attn_sql("q_rope", "k_rope", "scores"))

        result = read_scores(conn, "scores", 1, NUM_Q_HEADS, "score")
        # head 0 uses q[0:2] and k[0:2]: dot = 1*1 + 0*0 = 1 (no scale)
        # head 1 uses q[2:4] and k[0:2]: dot = 0
        expected_h0 = 1.0
        expected_h1 = 0.0
        self.assertAlmostEqual(float(result[0, 0, 0]), expected_h0, places=5)
        self.assertAlmostEqual(float(result[0, 0, 1]), expected_h1, places=5)


class TestSoftmax(unittest.TestCase):
    """SoftmaxSQL: numerically stable 4-step softmax over key tokens."""

    def test_basic(self):
        scores_np = rng.standard_normal((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)

        conn = new_conn()
        load_scores(conn, "scores_in", scores_np)
        run_steps(conn, softmax_sql("scores_in", "attn_w"))

        result   = read_scores(conn, "attn_w", SEQ_LEN, NUM_Q_HEADS, "attn_weight")
        expected = ref_softmax(scores_np)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="Softmax value mismatch")

    def test_rows_sum_to_one(self):
        scores_np = rng.standard_normal((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)

        conn = new_conn()
        load_scores(conn, "scores_in", scores_np)
        run_steps(conn, softmax_sql("scores_in", "attn_w"))

        result   = read_scores(conn, "attn_w", SEQ_LEN, NUM_Q_HEADS, "attn_weight")
        row_sums = result.sum(axis=1)  # sum over k_tok
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5,
                                   err_msg="Softmax rows do not sum to 1")

    def test_uniform_scores_give_uniform_weights(self):
        """All-zero scores → uniform attention (1/seq_len per key)."""
        scores_np = np.zeros((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS), dtype=np.float32)

        conn = new_conn()
        load_scores(conn, "scores_in", scores_np)
        run_steps(conn, softmax_sql("scores_in", "attn_w"))

        result   = read_scores(conn, "attn_w", SEQ_LEN, NUM_Q_HEADS, "attn_weight")
        expected = np.full_like(result, 1.0 / SEQ_LEN)
        np.testing.assert_allclose(result, expected, atol=1e-5,
                                   err_msg="Uniform softmax mismatch")


class TestAttnVMul(unittest.TestCase):
    """AttnVMulSQL: expand V → weighted sum → re-chunk."""

    def test_basic(self):
        attn_np = rng.random((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)
        attn_np /= attn_np.sum(axis=1, keepdims=True)   # normalise rows
        v_np = rng.standard_normal((SEQ_LEN, KV_DIM)).astype(np.float32)

        conn = new_conn()
        load_attn_weights(conn, "attn_w", attn_np)
        load_2d(conn, "v", v_np)
        run_steps(conn, attn_vmul_sql("attn_w", "v", "attn_out"))

        result   = read_2d(conn, "attn_out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_attn_vmul(attn_np, v_np, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="AttnVMul mismatch")

    def test_identity_attn(self):
        """Identity attention (diag weights=1) → output equals V at same position."""
        attn_np = np.eye(SEQ_LEN, dtype=np.float32)                   # [seq, seq]
        attn_np = attn_np[:, :, np.newaxis].repeat(NUM_Q_HEADS, axis=2)  # [seq, seq, heads]
        v_np = rng.standard_normal((SEQ_LEN, KV_DIM)).astype(np.float32)

        conn = new_conn()
        load_attn_weights(conn, "attn_w", attn_np)
        load_2d(conn, "v", v_np)
        run_steps(conn, attn_vmul_sql("attn_w", "v", "attn_out"))

        result   = read_2d(conn, "attn_out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_attn_vmul(attn_np, v_np, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="AttnVMul identity-attention mismatch")


class TestSwiGLU(unittest.TestCase):
    """SwiGLUSQL: gate * SiLU(up) element-wise via list_transform + list_zip."""

    def test_basic(self):
        gate_np = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)
        up_np   = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "gate", gate_np)
        load_2d(conn, "up", up_np)
        run_steps(conn, swiglu_sql("gate", "up", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, FFN_DIM)
        expected = ref_swiglu(gate_np, up_np)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="SwiGLU mismatch")

    def test_silu_positive_gate(self):
        """With gate=1 everywhere, output = SiLU(up) = up * sigmoid(up)."""
        gate_np = np.ones((SEQ_LEN, FFN_DIM), dtype=np.float32)
        up_np   = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "gate", gate_np)
        load_2d(conn, "up", up_np)
        run_steps(conn, swiglu_sql("gate", "up", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, FFN_DIM)
        expected = (up_np / (1.0 + np.exp(-up_np.astype(np.float64)))).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="SwiGLU SiLU(up) mismatch")


class TestResidualAdd(unittest.TestCase):
    """ResidualAddSQL: element-wise add via list_transform + list_zip."""

    def test_basic(self):
        a = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        b = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)

        conn = new_conn()
        load_2d(conn, "ta", a)
        load_2d(conn, "tb", b)
        run_steps(conn, residual_add_sql("ta", "tb", "out"))

        result   = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
        expected = ref_residual_add(a, b)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="ResidualAdd mismatch")

    def test_zero_addend(self):
        a = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        b = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=np.float32)

        conn = new_conn()
        load_2d(conn, "ta", a)
        load_2d(conn, "tb", b)
        run_steps(conn, residual_add_sql("ta", "tb", "out"))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM)
        np.testing.assert_allclose(result, a, atol=ATOL, rtol=RTOL,
                                   err_msg="ResidualAdd zero-addend mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
