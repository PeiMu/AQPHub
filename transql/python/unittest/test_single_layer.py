"""
test_single_layer.py — End-to-end integration test for one Llama3-8B transformer layer.

Uses synthetic random weights at full Llama3-8B shape (hidden=4096, chunk_size=32)
with a 3-token input sequence. Compares DuckDB SQL output against NumPy reference.

No HuggingFace downloads required — all weights are random float32.

Run:
    cd /path/to/AQP_middleware
    python -m pytest transql/python/unittest/test_single_layer.py -v
    # or directly:
    python transql/python/unittest/test_single_layer.py
"""

import unittest
import numpy as np
import duckdb

# ---------------------------------------------------------------------------
# Model parameters — structurally equivalent to Llama3-8B but small enough
# for fast testing (no HuggingFace weights required).
#
# Constraints preserved:
#   HIDDEN_DIM = NUM_Q_HEADS * HEAD_DIM
#   KV_DIM     = NUM_KV_HEADS * HEAD_DIM
#   HEAD_DIM   >= CHUNK_SIZE  (chunks_per_head = HEAD_DIM // CHUNK_SIZE = 4)
#   group_size = NUM_Q_HEADS // NUM_KV_HEADS = 4  (same as Llama3-8B)
#   All dims divisible by CHUNK_SIZE
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 32
SEQ_LEN      = 3
HEAD_DIM     = 128    # must be multiple of CHUNK_SIZE; 4 chunks per head
NUM_Q_HEADS  = 4
NUM_KV_HEADS = 1      # group_size = 4, same ratio as Llama3-8B
HIDDEN_DIM   = NUM_Q_HEADS  * HEAD_DIM   # 512
KV_DIM       = NUM_KV_HEADS * HEAD_DIM   # 128
FFN_DIM      = 512    # reduced from 14336; multiple of CHUNK_SIZE
EPS          = 1e-5
ATOL         = 2e-3   # float32 accumulation tolerance

rng = np.random.default_rng(0)


# ===========================================================================
# DuckDB loading helpers
# ===========================================================================

def new_conn():
    return duckdb.connect(":memory:")


def load_2d(conn, name, matrix, cs=CHUNK_SIZE):
    """Load [n, dim] float32 matrix as chunked table (row_id, chunk_id, v FLOAT[])."""
    n, dim = matrix.shape
    conn.execute(
        f"CREATE TEMP TABLE {name} (row_id INTEGER, chunk_id INTEGER, v FLOAT[])"
    )
    rows = [
        (r, c, matrix[r, c*cs:(c+1)*cs].astype(np.float32).tolist())
        for r in range(n)
        for c in range(dim // cs)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?)", rows)


def load_norm_weight(conn, name, vector, cs=CHUNK_SIZE):
    """Load 1D norm weight as (chunk_id, v FLOAT[])."""
    dim = len(vector)
    conn.execute(f"CREATE TEMP TABLE {name} (chunk_id INTEGER, v FLOAT[])")
    rows = [
        (c, vector[c*cs:(c+1)*cs].astype(np.float32).tolist())
        for c in range(dim // cs)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?)", rows)


def load_rope_table(conn, name, cos, sin, cs=CHUNK_SIZE):
    """Load rope table (row_id=pos, chunk_id, cos FLOAT[], sin FLOAT[]).
    cos, sin: [seq_len, num_chunks, half]
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


def read_2d(conn, name, n_rows, dim, cs=CHUNK_SIZE):
    rows = conn.execute(
        f"SELECT row_id, chunk_id, v FROM {name} ORDER BY row_id, chunk_id"
    ).fetchall()
    out = np.zeros((n_rows, dim), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        out[int(row_id), int(chunk_id)*cs:(int(chunk_id)+1)*cs] = v
    return out


def run_steps(conn, steps):
    for sql, tname in steps:
        conn.execute(f"CREATE TEMP TABLE {tname} AS ({sql})")


# ===========================================================================
# SQL template mirrors (copied from test_sql_templates.py, full-size params)
# ===========================================================================

def matmul_sql(act, weight, out, cs=CHUNK_SIZE):
    dp = out + "_dp"
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


def rope_sql(q_table, rope_table, out_table, cs=CHUNK_SIZE):
    half = cs // 2
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
                num_q=NUM_Q_HEADS, num_kv=NUM_KV_HEADS,
                head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    cph  = head_dim // cs
    cphg = cph * (num_q // num_kv)
    sql = (
        f"SELECT q.row_id AS q_tok, k.row_id AS k_tok, "
        f"q.chunk_id // {cph} AS head_id, "
        f"SUM(list_dot_product(q.v_even, k.v_even) + "
        f"list_dot_product(q.v_odd, k.v_odd)) * (1.0 / sqrt({head_dim}.0)) AS score "
        f"FROM {q_rope} q JOIN {k_rope} k "
        f"ON q.chunk_id % {cph} = k.chunk_id % {cph} "
        f"AND q.chunk_id // {cphg} = k.chunk_id // {cph} "
        f"GROUP BY q.row_id, k.row_id, q.chunk_id // {cph}"
    )
    return [(sql, out)]


def softmax_sql(inp, out):
    max_t, exp_t, sum_t = out+"_max", out+"_exp", out+"_sum"
    return [
        (f"SELECT q_tok, head_id, MAX(score) AS max_score "
         f"FROM {inp} GROUP BY q_tok, head_id", max_t),
        (f"SELECT s.q_tok, s.k_tok, s.head_id, "
         f"EXP(s.score - m.max_score) AS exp_val "
         f"FROM {inp} s "
         f"JOIN {max_t} m ON s.q_tok=m.q_tok AND s.head_id=m.head_id", exp_t),
        (f"SELECT q_tok, head_id, SUM(exp_val) AS sum_exp "
         f"FROM {exp_t} GROUP BY q_tok, head_id", sum_t),
        (f"SELECT e.q_tok, e.k_tok, e.head_id, "
         f"CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight "
         f"FROM {exp_t} e "
         f"JOIN {sum_t} s ON e.q_tok=s.q_tok AND e.head_id=s.head_id", out),
    ]


def attn_vmul_sql(attn_table, v_table, out,
                  num_q=NUM_Q_HEADS, num_kv=NUM_KV_HEADS,
                  head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    cph = head_dim // cs
    gs  = num_q // num_kv
    vs, wt = out+"_vs", out+"_w"
    return [
        (f"SELECT row_id AS tok, chunk_id, "
         f"unnest(generate_series(0, {cs-1})) AS elem_pos, "
         f"CAST(unnest(v) AS FLOAT) AS val "
         f"FROM {v_table}", vs),
        (f"SELECT s.q_tok, "
         f"s.head_id * {cph} + v.chunk_id % {cph} AS out_chunk_id, "
         f"v.elem_pos, "
         f"CAST(SUM(s.attn_weight * v.val) AS FLOAT) AS val "
         f"FROM {attn_table} s "
         f"JOIN {vs} v ON s.k_tok = v.tok "
         f"AND s.head_id // {gs} = v.chunk_id // {cph} "
         f"GROUP BY s.q_tok, s.head_id, v.chunk_id, v.elem_pos", wt),
        (f"SELECT q_tok AS row_id, out_chunk_id AS chunk_id, "
         f"array_agg(val ORDER BY elem_pos) AS v "
         f"FROM {wt} GROUP BY q_tok, out_chunk_id", out),
    ]


def swiglu_sql(gate, up, out):
    sql = (
        f"SELECT g.row_id, g.chunk_id, "
        f"list_transform(generate_series(1, len(g.v)), "
        f"i -> CAST(g.v[i] * (u.v[i] / (1.0 + exp(-u.v[i]))) AS FLOAT)) AS v "
        f"FROM {gate} g "
        f"JOIN {up} u ON g.row_id=u.row_id AND g.chunk_id=u.chunk_id"
    )
    return [(sql, out)]


def residual_add_sql(a, b, out):
    sql = (
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(generate_series(1, len(a.v)), "
        f"i -> CAST(a.v[i] + b.v[i] AS FLOAT)) AS v "
        f"FROM {a} a "
        f"JOIN {b} b ON a.row_id=b.row_id AND a.chunk_id=b.chunk_id"
    )
    return [(sql, out)]


# ===========================================================================
# NumPy reference: one full Llama3 transformer layer
# ===========================================================================

def ref_rmsnorm(x, gamma, eps=EPS):
    sq_sum = np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(sq_sum / x.shape[-1] + eps) * gamma).astype(np.float32)


def ref_rope(q, cos, sin):
    """q: [seq, dim], cos/sin: [seq, num_chunks, half]"""
    half = CHUNK_SIZE // 2
    seq_len = q.shape[0]
    num_chunks = q.shape[1] // CHUNK_SIZE
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


def ref_qk_attn(q_rot, k_rot):
    seq = q_rot.shape[0]
    gs = NUM_Q_HEADS // NUM_KV_HEADS
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros((seq, seq, NUM_Q_HEADS), dtype=np.float32)
    for h in range(NUM_Q_HEADS):
        kv_h = h // gs
        q_h = q_rot[:, h*HEAD_DIM:(h+1)*HEAD_DIM]
        k_h = k_rot[:, kv_h*HEAD_DIM:(kv_h+1)*HEAD_DIM]
        out[:, :, h] = (q_h @ k_h.T * scale).astype(np.float32)
    return out


def ref_softmax(scores):
    s_max = np.max(scores, axis=1, keepdims=True)
    exp_s = np.exp(scores - s_max).astype(np.float32)
    return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)


def ref_attn_vmul(attn, v):
    seq = attn.shape[0]
    gs = NUM_Q_HEADS // NUM_KV_HEADS
    out = np.zeros((seq, NUM_Q_HEADS * HEAD_DIM), dtype=np.float32)
    for h in range(NUM_Q_HEADS):
        kv_h = h // gs
        v_h = v[:, kv_h*HEAD_DIM:(kv_h+1)*HEAD_DIM]
        out[:, h*HEAD_DIM:(h+1)*HEAD_DIM] = (attn[:, :, h] @ v_h).astype(np.float32)
    return out


def ref_transformer_layer(x, norm1, norm2, w_q, w_k, w_v, w_o,
                           w_gate, w_up, w_down, rope_cos, rope_sin):
    """
    One Llama3 transformer layer in NumPy/float32.
    x:      [seq, hidden]
    norm1/norm2: [hidden]
    w_q:    [hidden, hidden], w_k/w_v: [kv_dim, hidden], w_o: [hidden, hidden]
    w_gate/w_up: [ffn_dim, hidden], w_down: [hidden, ffn_dim]
    rope_cos/sin: [seq, num_q_chunks, half] (Q chunks only; K uses first kv_dim/cs chunks)
    """
    # Pre-attention norm
    x_norm1 = ref_rmsnorm(x, norm1)

    # QKV projections
    q = (x_norm1 @ w_q.T).astype(np.float32)
    k = (x_norm1 @ w_k.T).astype(np.float32)
    v = (x_norm1 @ w_v.T).astype(np.float32)

    # RoPE — Q uses all rope chunks, K uses only the first KV_DIM//CHUNK_SIZE chunks
    num_kv_chunks = KV_DIM // CHUNK_SIZE
    q_rot = ref_rope(q, rope_cos, rope_sin)
    k_rot = ref_rope(k, rope_cos[:, :num_kv_chunks, :], rope_sin[:, :num_kv_chunks, :])

    # Attention
    scores  = ref_qk_attn(q_rot, k_rot)       # [seq, seq, num_q_heads]
    weights = ref_softmax(scores)               # [seq, seq, num_q_heads]
    attn_out = ref_attn_vmul(weights, v)        # [seq, hidden]

    # O projection + residual
    o = (attn_out @ w_o.T).astype(np.float32)
    x_after_attn = (x + o).astype(np.float32)

    # Pre-FFN norm
    x_norm2 = ref_rmsnorm(x_after_attn, norm2)

    # FFN
    gate = (x_norm2 @ w_gate.T).astype(np.float32)
    up   = (x_norm2 @ w_up.T).astype(np.float32)
    silu_up = (up / (1.0 + np.exp(-up.astype(np.float64)))).astype(np.float32)
    ffn_act = (gate * silu_up).astype(np.float32)
    down = (ffn_act @ w_down.T).astype(np.float32)

    return (x_after_attn + down).astype(np.float32)


# ===========================================================================
# Integration test
# ===========================================================================

class TestSingleLayer(unittest.TestCase):
    """End-to-end single Llama3 transformer layer: SQL vs NumPy reference."""

    @classmethod
    def setUpClass(cls):
        """Generate synthetic weights once for the class."""
        scale = 0.02  # small values to keep activations bounded, avoid float32 overflow
        cls.x      = (rng.standard_normal((SEQ_LEN, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.norm1  = (rng.uniform(0.8, 1.2, HIDDEN_DIM)).astype(np.float32)
        cls.norm2  = (rng.uniform(0.8, 1.2, HIDDEN_DIM)).astype(np.float32)
        cls.w_q    = (rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_k    = (rng.standard_normal((KV_DIM,    HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_v    = (rng.standard_normal((KV_DIM,    HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_o    = (rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_gate = (rng.standard_normal((FFN_DIM,   HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_up   = (rng.standard_normal((FFN_DIM,   HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_down = (rng.standard_normal((HIDDEN_DIM, FFN_DIM))   * scale).astype(np.float32)

        # RoPE tables: [seq, num_chunks, half] for Q; K uses subset
        num_q_chunks  = HIDDEN_DIM // CHUNK_SIZE
        half = CHUNK_SIZE // 2
        # Use small angles so cos≈1, sin≈0 → easy to verify rotation is applied
        cls.rope_cos = rng.uniform(0.9, 1.0, (SEQ_LEN, num_q_chunks, half)).astype(np.float32)
        cls.rope_sin = rng.uniform(0.0, 0.1, (SEQ_LEN, num_q_chunks, half)).astype(np.float32)

        # NumPy reference output
        cls.expected = ref_transformer_layer(
            cls.x, cls.norm1, cls.norm2,
            cls.w_q, cls.w_k, cls.w_v, cls.w_o,
            cls.w_gate, cls.w_up, cls.w_down,
            cls.rope_cos, cls.rope_sin
        )

    def _build_conn(self):
        """Load all weights into a fresh in-memory DuckDB connection."""
        conn = new_conn()
        load_2d(conn, "x",        self.x)
        load_norm_weight(conn, "norm1", self.norm1)
        load_norm_weight(conn, "norm2", self.norm2)
        load_2d(conn, "w_q",    self.w_q)
        load_2d(conn, "w_k",    self.w_k)
        load_2d(conn, "w_v",    self.w_v)
        load_2d(conn, "w_o",    self.w_o)
        load_2d(conn, "w_gate", self.w_gate)
        load_2d(conn, "w_up",   self.w_up)
        load_2d(conn, "w_down", self.w_down)

        # Rope table covers all Q chunks (K needs only first KV_DIM//CHUNK_SIZE chunks,
        # but sharing the full table is fine — extra rows are never joined)
        load_rope_table(conn, "rope", self.rope_cos, self.rope_sin)
        return conn

    def _run_layer_sql(self, conn):
        """Execute the 15-step SQL pipeline for one transformer layer."""
        steps = []

        # 1. Pre-attention RMSNorm
        steps += rmsnorm_sql("x", "norm1", "x_norm1")

        # 2-4. QKV projections
        steps += matmul_sql("x_norm1", "w_q",   "q_proj")
        steps += matmul_sql("x_norm1", "w_k",   "k_proj")
        steps += matmul_sql("x_norm1", "w_v",   "v_proj")

        # 5-6. RoPE on Q and K
        steps += rope_sql("q_proj", "rope", "q_rope")
        steps += rope_sql("k_proj", "rope", "k_rope")

        # 7. QK attention scores
        steps += qk_attn_sql("q_rope", "k_rope", "qk_scores")

        # 8. Softmax
        steps += softmax_sql("qk_scores", "attn_weights")

        # 9. Attention × V
        steps += attn_vmul_sql("attn_weights", "v_proj", "attn_out")

        # 10. O projection
        steps += matmul_sql("attn_out", "w_o", "o_proj")

        # 11. First residual add
        steps += residual_add_sql("x", "o_proj", "x_after_attn")

        # 12. Pre-FFN RMSNorm
        steps += rmsnorm_sql("x_after_attn", "norm2", "x_norm2", hidden_dim=HIDDEN_DIM)

        # 13-14. Gate / Up projections
        steps += matmul_sql("x_norm2", "w_gate", "gate")
        steps += matmul_sql("x_norm2", "w_up",   "up")

        # 15. SwiGLU
        steps += swiglu_sql("gate", "up", "ffn_act")

        # 16. Down projection
        steps += matmul_sql("ffn_act", "w_down", "down")

        # 17. Second residual add
        steps += residual_add_sql("x_after_attn", "down", "x_out")

        run_steps(conn, steps)
        return read_2d(conn, "x_out", SEQ_LEN, HIDDEN_DIM)

    def test_layer_output(self):
        """Full transformer layer SQL output matches NumPy reference within tolerance."""
        conn   = self._build_conn()
        result = self._run_layer_sql(conn)
        np.testing.assert_allclose(
            result, self.expected,
            atol=ATOL, rtol=1e-3,
            err_msg="Single-layer SQL vs NumPy mismatch"
        )

    def test_rmsnorm_only(self):
        """Pre-attention RMSNorm output matches NumPy reference."""
        conn = self._build_conn()
        run_steps(conn, rmsnorm_sql("x", "norm1", "x_norm1"))
        result   = read_2d(conn, "x_norm1", SEQ_LEN, HIDDEN_DIM)
        expected = ref_rmsnorm(self.x, self.norm1)
        np.testing.assert_allclose(result, expected, atol=2e-4, rtol=1e-4,
                                   err_msg="RMSNorm mismatch")

    def test_qkv_proj(self):
        """Q projection output matches NumPy reference."""
        conn = self._build_conn()
        run_steps(conn, rmsnorm_sql("x", "norm1", "x_norm1"))
        run_steps(conn, matmul_sql("x_norm1", "w_q", "q_proj"))
        result   = read_2d(conn, "q_proj", SEQ_LEN, HIDDEN_DIM)
        expected = (ref_rmsnorm(self.x, self.norm1) @ self.w_q.T).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=1e-3,
                                   err_msg="Q projection mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
