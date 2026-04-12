"""
test_postopt.py — Verify Section 4 post-optimizations produce identical results.

Tests each optimization (CTE merging, table fusion, ROW2COL pivoting) individually
and combined against the non-optimized SQL pipeline from test_single_layer.py.

Uses the same reduced model dimensions as test_single_layer.py.

Run:
    cd /path/to/AQP_middleware
    python -m pytest transql/python/unittest/test_postopt.py -v
"""

import unittest
import numpy as np
import duckdb

# ---------------------------------------------------------------------------
# Model parameters — same as test_single_layer.py
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 32
SEQ_LEN      = 3
HEAD_DIM     = 128
NUM_Q_HEADS  = 4
NUM_KV_HEADS = 1
HIDDEN_DIM   = NUM_Q_HEADS  * HEAD_DIM   # 512
KV_DIM       = NUM_KV_HEADS * HEAD_DIM   # 128
FFN_DIM      = 512
EPS          = 1e-5
ATOL         = 2e-3

rng = np.random.default_rng(42)

N_CHUNKS_HIDDEN = HIDDEN_DIM // CHUNK_SIZE   # 16
N_CHUNKS_FFN    = FFN_DIM // CHUNK_SIZE      # 16


# ===========================================================================
# DuckDB helpers (shared with test_single_layer.py)
# ===========================================================================

def new_conn():
    return duckdb.connect(":memory:")


def load_2d(conn, name, matrix, cs=CHUNK_SIZE):
    n, dim = matrix.shape
    conn.execute(
        f"CREATE TEMP TABLE {name} (row_id INTEGER, chunk_id INTEGER, v FLOAT[])")
    rows = [
        (r, c, matrix[r, c*cs:(c+1)*cs].astype(np.float32).tolist())
        for r in range(n) for c in range(dim // cs)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?)", rows)


def load_norm_weight(conn, name, vector, cs=CHUNK_SIZE):
    dim = len(vector)
    conn.execute(f"CREATE TEMP TABLE {name} (chunk_id INTEGER, v FLOAT[])")
    rows = [(c, vector[c*cs:(c+1)*cs].astype(np.float32).tolist())
            for c in range(dim // cs)]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?)", rows)


def load_rope_table(conn, name, cos, sin):
    seq_len, num_chunks, _ = cos.shape
    conn.execute(
        f"CREATE TEMP TABLE {name} "
        f"(row_id INTEGER, chunk_id INTEGER, cos FLOAT[], sin FLOAT[])")
    rows = [
        (pos, c,
         cos[pos, c].astype(np.float32).tolist(),
         sin[pos, c].astype(np.float32).tolist())
        for pos in range(seq_len) for c in range(num_chunks)
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
# SQL template mirrors — baseline (non-optimized)
# ===========================================================================

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


def rmsnorm_sql(inp, gamma, out, hidden_dim=HIDDEN_DIM, eps=EPS):
    sq = out + "_sq"
    eps_str = f"{eps:.10f}"
    return [
        (f"SELECT row_id, "
         f"SUM(list_sum(list_transform(v, x -> x * x))) AS sq_sum "
         f"FROM {inp} GROUP BY row_id", sq),
        (f"SELECT n.row_id, n.chunk_id, "
         f"list_transform(generate_series(1, len(n.v)), "
         f"i -> CAST(n.v[i] / sqrt(s.sq_sum / {hidden_dim}.0 + {eps_str}) "
         f"* w.v[i] AS FLOAT)) AS v "
         f"FROM {inp} n "
         f"JOIN {sq} s ON n.row_id = s.row_id "
         f"JOIN {gamma} w ON n.chunk_id = w.chunk_id", out),
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
        f"JOIN {rope_table} r ON r.chunk_id = q.chunk_id AND r.row_id = q.row_id")
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
        f"list_dot_product(q.v_odd, k.v_odd)) AS score "
        f"FROM {q_rope} q JOIN {k_rope} k "
        f"ON q.chunk_id % {cph} = k.chunk_id % {cph} "
        f"AND q.chunk_id // {cphg} = k.chunk_id // {cph} "
        f"AND k.row_id <= q.row_id "
        f"GROUP BY q.row_id, k.row_id, q.chunk_id // {cph}")
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
    # SwiGLU: SiLU(gate) * up (Llama's formula)
    return [(
        f"SELECT g.row_id, g.chunk_id, "
        f"list_transform(generate_series(1, len(g.v)), "
        f"i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] AS FLOAT)) AS v "
        f"FROM {gate} g "
        f"JOIN {up} u ON g.row_id=u.row_id AND g.chunk_id=u.chunk_id", out)]


def residual_add_sql(a, b, out):
    return [(
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(generate_series(1, len(a.v)), "
        f"i -> CAST(a.v[i] + b.v[i] AS FLOAT)) AS v "
        f"FROM {a} a "
        f"JOIN {b} b ON a.row_id=b.row_id AND a.chunk_id=b.chunk_id", out)]


# ===========================================================================
# Optimized SQL generators
# ===========================================================================

def _pivot_sql(table_name, chunk_count, chunk_start=0):
    """Generate pivot CTE body: (row_id, chunk_id, v) -> (row_id, chunk0, ..., chunkN).
    chunk_start: first chunk_id to include; column names are always chunk0..chunk{count-1}."""
    cols = ", ".join(
        f"MAX(CASE WHEN chunk_id = {chunk_start + i} THEN v END) AS chunk{i}"
        for i in range(chunk_count))
    return f"SELECT row_id, {cols} FROM {table_name} GROUP BY row_id"


def _pivoted_matmul_dp(act_pivot, weight_pivot, dp_out, n_cols,
                       subquery_width=1):
    """Subquery CTEs with CROSS JOIN + POSITIONAL JOIN reduction.
    subquery_width: columns per CROSS JOIN CTE (1 = one per column)."""
    if subquery_width <= 0:
        subquery_width = 1
    n_sq = (n_cols + subquery_width - 1) // subquery_width
    steps = []

    for sq in range(n_sq):
        ci = f"{dp_out}_sq{sq}"
        col_start = sq * subquery_width
        col_end = min(col_start + subquery_width, n_cols)
        dot_expr = " + ".join(
            f"list_dot_product(a.chunk{c}, w.chunk{c})"
            for c in range(col_start, col_end))
        sql = (f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
               f"{dot_expr} AS v{sq} "
               f"FROM {act_pivot} a CROSS JOIN {weight_pivot} w "
               f"ORDER BY a.row_id, w.row_id")
        steps.append((sql, ci))

    first = f"{dp_out}_sq0"
    sum_expr = " + ".join(f"{dp_out}_sq{i}.v{i}" for i in range(n_sq))
    from_clause = " POSITIONAL JOIN ".join(f"{dp_out}_sq{i}" for i in range(n_sq))
    steps.append((
        f"SELECT {first}.act_row, {first}.out_col, {sum_expr} AS val "
        f"FROM {from_clause}", dp_out))
    return steps


def pivoted_matmul_sql(act, weight, out, n_chunks, cs=CHUNK_SIZE,
                       pivot_width=0, subquery_width=0):
    """Full pivoted MatMul with configurable grouping.
    pivot_width: columns per pivoted sub-table (0 = all at once).
    subquery_width: columns per CROSS JOIN CTE (0 = 1 per column)."""
    if pivot_width <= 0:
        pivot_width = n_chunks
    if subquery_width <= 0:
        subquery_width = 1

    dp = out + "_dp"
    n_groups = (n_chunks + pivot_width - 1) // pivot_width

    steps = []
    group_dp_names = []

    for g in range(n_groups):
        chunk_start = g * pivot_width
        chunk_count = min(pivot_width, n_chunks - chunk_start)
        g_sfx = f"_g{g}" if n_groups > 1 else ""
        act_piv = f"{out}_act_piv{g_sfx}"
        wt_piv = f"{weight}_piv{g_sfx}"
        g_dp = f"{dp}_g{g}" if n_groups > 1 else dp

        steps.append((_pivot_sql(act, chunk_count, chunk_start), act_piv))
        steps.append((_pivot_sql(weight, chunk_count, chunk_start), wt_piv))
        steps += _pivoted_matmul_dp(act_piv, wt_piv, g_dp,
                                    chunk_count, subquery_width)
        group_dp_names.append(g_dp)

    # Sum across pivot groups via POSITIONAL JOIN
    if n_groups > 1:
        sum_expr = " + ".join(f"{n}.val" for n in group_dp_names)
        from_clause = " POSITIONAL JOIN ".join(group_dp_names)
        steps.append((
            f"SELECT {group_dp_names[0]}.act_row, {group_dp_names[0]}.out_col, "
            f"{sum_expr} AS val FROM {from_clause}", dp))

    steps.append((
        f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {dp} GROUP BY act_row, out_col // {cs}", out))
    return steps


def fused_qkv_sql(norm_out, q_wt, k_wt, v_wt, q_out, k_out, v_out,
                  q_dim=HIDDEN_DIM, kv_dim=KV_DIM, cs=CHUNK_SIZE):
    """Fused QKV: UNION ALL weight + single MatMul + 3 split re-chunks."""
    w_name = q_out + "_w_qkv"
    dp     = q_out + "_qkv_dp"
    kv_off = q_dim
    v_off  = q_dim + kv_dim

    w_sql = (f"SELECT row_id, chunk_id, v FROM {q_wt} "
             f"UNION ALL SELECT row_id + {kv_off}, chunk_id, v FROM {k_wt} "
             f"UNION ALL SELECT row_id + {v_off}, chunk_id, v FROM {v_wt}")

    dp_sql = (f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
              f"SUM(list_dot_product(a.v, w.v)) AS val "
              f"FROM {norm_out} a JOIN {w_name} w ON a.chunk_id = w.chunk_id "
              f"GROUP BY a.row_id, w.row_id")

    q_sql = (f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
             f"array_agg(val ORDER BY out_col) AS v "
             f"FROM {dp} WHERE out_col < {q_dim} "
             f"GROUP BY act_row, out_col // {cs}")

    k_sql = (f"SELECT act_row AS row_id, (out_col - {kv_off}) // {cs} AS chunk_id, "
             f"array_agg(val ORDER BY out_col) AS v "
             f"FROM {dp} WHERE out_col >= {kv_off} AND out_col < {v_off} "
             f"GROUP BY act_row, (out_col - {kv_off}) // {cs}")

    v_sql = (f"SELECT act_row AS row_id, (out_col - {v_off}) // {cs} AS chunk_id, "
             f"array_agg(val ORDER BY out_col) AS v "
             f"FROM {dp} WHERE out_col >= {v_off} "
             f"GROUP BY act_row, (out_col - {v_off}) // {cs}")

    return [(w_sql, w_name), (dp_sql, dp), (q_sql, q_out),
            (k_sql, k_out), (v_sql, v_out)]


def fused_gateup_sql(norm_out, gate_wt, up_wt, gate_out, up_out,
                     ffn_dim=FFN_DIM, cs=CHUNK_SIZE):
    """Fused gate+up: UNION ALL weight + single MatMul + 2 split re-chunks."""
    w_name = gate_out + "_w_gateup"
    dp     = gate_out + "_gateup_dp"

    w_sql = (f"SELECT row_id, chunk_id, v FROM {gate_wt} "
             f"UNION ALL SELECT row_id + {ffn_dim}, chunk_id, v FROM {up_wt}")

    dp_sql = (f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
              f"SUM(list_dot_product(a.v, w.v)) AS val "
              f"FROM {norm_out} a JOIN {w_name} w ON a.chunk_id = w.chunk_id "
              f"GROUP BY a.row_id, w.row_id")

    gate_sql = (f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
                f"array_agg(val ORDER BY out_col) AS v "
                f"FROM {dp} WHERE out_col < {ffn_dim} "
                f"GROUP BY act_row, out_col // {cs}")

    up_sql = (f"SELECT act_row AS row_id, (out_col - {ffn_dim}) // {cs} AS chunk_id, "
              f"array_agg(val ORDER BY out_col) AS v "
              f"FROM {dp} WHERE out_col >= {ffn_dim} "
              f"GROUP BY act_row, (out_col - {ffn_dim}) // {cs}")

    return [(w_sql, w_name), (dp_sql, dp), (gate_sql, gate_out),
            (up_sql, up_out)]


def _merge_to_cte(steps):
    """Merge a list of (sql, name) into a single (WITH ... <final>, final_name)."""
    if len(steps) == 1:
        return steps[0]
    cte_defs = "WITH " + ", ".join(
        f"{name} AS ({sql})" for sql, name in steps[:-1])
    return (cte_defs + " " + steps[-1][0], steps[-1][1])


# ===========================================================================
# Layer pipeline builders
# ===========================================================================

def _build_baseline_steps():
    """Non-optimized pipeline: one CREATE TEMP TABLE per SQL step."""
    steps = []
    steps += rmsnorm_sql("x", "norm1", "x_norm1")
    steps += matmul_sql("x_norm1", "w_q", "q_proj")
    steps += matmul_sql("x_norm1", "w_k", "k_proj")
    steps += matmul_sql("x_norm1", "w_v", "v_proj")
    steps += rope_sql("q_proj", "rope", "q_rope")
    steps += rope_sql("k_proj", "rope", "k_rope")
    steps += qk_attn_sql("q_rope", "k_rope", "qk_scores")
    steps += softmax_sql("qk_scores", "attn_weights")
    steps += attn_vmul_sql("attn_weights", "v_proj", "attn_out")
    steps += matmul_sql("attn_out", "w_o", "o_proj")
    steps += residual_add_sql("x", "o_proj", "x_after_attn")
    steps += rmsnorm_sql("x_after_attn", "norm2", "x_norm2")
    steps += matmul_sql("x_norm2", "w_gate", "gate")
    steps += matmul_sql("x_norm2", "w_up", "up")
    steps += swiglu_sql("gate", "up", "ffn_act")
    steps += matmul_sql("ffn_act", "w_down", "down")
    steps += residual_add_sql("x_after_attn", "down", "x_out")
    return steps


def _build_cte_merged_steps():
    """CTE-merged pipeline: 4 materialized steps per layer."""
    # G1: norm1 (shared)
    g1 = rmsnorm_sql("x", "norm1", "x_norm1")

    # G2: Q/K/V → RoPE → Attn → O_proj → ResidualAdd → x_after_attn (shared)
    g2 = []
    g2 += matmul_sql("x_norm1", "w_q", "q_proj")
    g2 += matmul_sql("x_norm1", "w_k", "k_proj")
    g2 += matmul_sql("x_norm1", "w_v", "v_proj")
    g2 += rope_sql("q_proj", "rope", "q_rope")
    g2 += rope_sql("k_proj", "rope", "k_rope")
    g2 += qk_attn_sql("q_rope", "k_rope", "qk_scores")
    g2 += softmax_sql("qk_scores", "attn_weights")
    g2 += attn_vmul_sql("attn_weights", "v_proj", "attn_out")
    g2 += matmul_sql("attn_out", "w_o", "o_proj")
    g2 += residual_add_sql("x", "o_proj", "x_after_attn")

    # G3: norm2 (shared)
    g3 = rmsnorm_sql("x_after_attn", "norm2", "x_norm2")

    # G4: gate/up → SwiGLU → down → ResidualAdd → x_out
    g4 = []
    g4 += matmul_sql("x_norm2", "w_gate", "gate")
    g4 += matmul_sql("x_norm2", "w_up", "up")
    g4 += swiglu_sql("gate", "up", "ffn_act")
    g4 += matmul_sql("ffn_act", "w_down", "down")
    g4 += residual_add_sql("x_after_attn", "down", "x_out")

    return [_merge_to_cte(g1), _merge_to_cte(g2),
            _merge_to_cte(g3), _merge_to_cte(g4)]


def _build_fusion_steps():
    """Table fusion (no CTE merge): fused QKV and gate+up."""
    steps = []
    steps += rmsnorm_sql("x", "norm1", "x_norm1")
    steps += fused_qkv_sql("x_norm1", "w_q", "w_k", "w_v",
                           "q_proj", "k_proj", "v_proj")
    steps += rope_sql("q_proj", "rope", "q_rope")
    steps += rope_sql("k_proj", "rope", "k_rope")
    steps += qk_attn_sql("q_rope", "k_rope", "qk_scores")
    steps += softmax_sql("qk_scores", "attn_weights")
    steps += attn_vmul_sql("attn_weights", "v_proj", "attn_out")
    steps += matmul_sql("attn_out", "w_o", "o_proj")
    steps += residual_add_sql("x", "o_proj", "x_after_attn")
    steps += rmsnorm_sql("x_after_attn", "norm2", "x_norm2")
    steps += fused_gateup_sql("x_norm2", "w_gate", "w_up", "gate", "up")
    steps += swiglu_sql("gate", "up", "ffn_act")
    steps += matmul_sql("ffn_act", "w_down", "down")
    steps += residual_add_sql("x_after_attn", "down", "x_out")
    return steps


def _build_row2col_steps():
    """ROW2COL pivoting for MatMul (no CTE merge, no fusion)."""
    nc = N_CHUNKS_HIDDEN
    nc_ffn = N_CHUNKS_FFN
    steps = []
    steps += rmsnorm_sql("x", "norm1", "x_norm1")
    steps += pivoted_matmul_sql("x_norm1", "w_q", "q_proj", nc)
    steps += pivoted_matmul_sql("x_norm1", "w_k", "k_proj", nc)
    steps += pivoted_matmul_sql("x_norm1", "w_v", "v_proj", nc)
    steps += rope_sql("q_proj", "rope", "q_rope")
    steps += rope_sql("k_proj", "rope", "k_rope")
    steps += qk_attn_sql("q_rope", "k_rope", "qk_scores")
    steps += softmax_sql("qk_scores", "attn_weights")
    steps += attn_vmul_sql("attn_weights", "v_proj", "attn_out")
    steps += pivoted_matmul_sql("attn_out", "w_o", "o_proj", nc)
    steps += residual_add_sql("x", "o_proj", "x_after_attn")
    steps += rmsnorm_sql("x_after_attn", "norm2", "x_norm2")
    steps += pivoted_matmul_sql("x_norm2", "w_gate", "gate", nc)
    steps += pivoted_matmul_sql("x_norm2", "w_up", "up", nc)
    steps += swiglu_sql("gate", "up", "ffn_act")
    steps += pivoted_matmul_sql("ffn_act", "w_down", "down", nc_ffn)
    steps += residual_add_sql("x_after_attn", "down", "x_out")
    return steps


# ===========================================================================
# Test class
# ===========================================================================

class TestPostOpt(unittest.TestCase):
    """Post-optimization correctness: optimized SQL == baseline SQL output."""

    @classmethod
    def setUpClass(cls):
        scale = 0.02
        cls.x      = (rng.standard_normal((SEQ_LEN, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.norm1  = rng.uniform(0.8, 1.2, HIDDEN_DIM).astype(np.float32)
        cls.norm2  = rng.uniform(0.8, 1.2, HIDDEN_DIM).astype(np.float32)
        # Constant folding: absorb 1/sqrt(head_dim) into W_Q
        cls.w_q    = (rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)) * scale
                      * np.float32(1.0 / np.sqrt(HEAD_DIM))).astype(np.float32)
        cls.w_k    = (rng.standard_normal((KV_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_v    = (rng.standard_normal((KV_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_o    = (rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_gate = (rng.standard_normal((FFN_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_up   = (rng.standard_normal((FFN_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.w_down = (rng.standard_normal((HIDDEN_DIM, FFN_DIM)) * scale).astype(np.float32)

        num_q_chunks = HIDDEN_DIM // CHUNK_SIZE
        half = CHUNK_SIZE // 2
        cls.rope_cos = rng.uniform(0.9, 1.0, (SEQ_LEN, num_q_chunks, half)).astype(np.float32)
        cls.rope_sin = rng.uniform(0.0, 0.1, (SEQ_LEN, num_q_chunks, half)).astype(np.float32)

        # Compute baseline result
        cls.baseline = cls._run_pipeline(_build_baseline_steps())

    @classmethod
    def _load_weights(cls, conn):
        load_2d(conn, "x", cls.x)
        load_norm_weight(conn, "norm1", cls.norm1)
        load_norm_weight(conn, "norm2", cls.norm2)
        load_2d(conn, "w_q", cls.w_q)
        load_2d(conn, "w_k", cls.w_k)
        load_2d(conn, "w_v", cls.w_v)
        load_2d(conn, "w_o", cls.w_o)
        load_2d(conn, "w_gate", cls.w_gate)
        load_2d(conn, "w_up", cls.w_up)
        load_2d(conn, "w_down", cls.w_down)
        load_rope_table(conn, "rope", cls.rope_cos, cls.rope_sin)

    @classmethod
    def _run_pipeline(cls, steps):
        conn = new_conn()
        cls._load_weights(conn)
        run_steps(conn, steps)
        return read_2d(conn, "x_out", SEQ_LEN, HIDDEN_DIM)

    def test_cte_merge(self):
        """CTE merging produces identical output to baseline."""
        result = self._run_pipeline(_build_cte_merged_steps())
        np.testing.assert_allclose(result, self.baseline, atol=ATOL, rtol=1e-3,
                                   err_msg="CTE merge mismatch")

    def test_cte_merge_step_count(self):
        """CTE merging reduces to 4 steps per layer."""
        steps = _build_cte_merged_steps()
        self.assertEqual(len(steps), 4, f"Expected 4 steps, got {len(steps)}")

    def test_table_fusion(self):
        """Table fusion (QKV + gate+up) produces identical output to baseline."""
        result = self._run_pipeline(_build_fusion_steps())
        np.testing.assert_allclose(result, self.baseline, atol=ATOL, rtol=1e-3,
                                   err_msg="Table fusion mismatch")

    def test_row2col_pivot(self):
        """ROW2COL pivoting produces identical output to baseline."""
        result = self._run_pipeline(_build_row2col_steps())
        np.testing.assert_allclose(result, self.baseline, atol=ATOL, rtol=1e-3,
                                   err_msg="ROW2COL pivot mismatch")

    def test_row2col_grouped_pivot(self):
        """ROW2COL with grouped pivoting (pivot_width=8, subquery_width=4)."""
        pw, sq = 8, 4
        nc = N_CHUNKS_HIDDEN
        nc_ffn = N_CHUNKS_FFN
        steps = []
        steps += rmsnorm_sql("x", "norm1", "x_norm1")
        steps += pivoted_matmul_sql("x_norm1", "w_q", "q_proj", nc, CHUNK_SIZE, pw, sq)
        steps += pivoted_matmul_sql("x_norm1", "w_k", "k_proj", nc, CHUNK_SIZE, pw, sq)
        steps += pivoted_matmul_sql("x_norm1", "w_v", "v_proj", nc, CHUNK_SIZE, pw, sq)
        steps += rope_sql("q_proj", "rope", "q_rope")
        steps += rope_sql("k_proj", "rope", "k_rope")
        steps += qk_attn_sql("q_rope", "k_rope", "qk_scores")
        steps += softmax_sql("qk_scores", "attn_weights")
        steps += attn_vmul_sql("attn_weights", "v_proj", "attn_out")
        steps += pivoted_matmul_sql("attn_out", "w_o", "o_proj", nc, CHUNK_SIZE, pw, sq)
        steps += residual_add_sql("x", "o_proj", "x_after_attn")
        steps += rmsnorm_sql("x_after_attn", "norm2", "x_norm2")
        steps += pivoted_matmul_sql("x_norm2", "w_gate", "gate", nc, CHUNK_SIZE, pw, sq)
        steps += pivoted_matmul_sql("x_norm2", "w_up", "up", nc, CHUNK_SIZE, pw, sq)
        steps += swiglu_sql("gate", "up", "ffn_act")
        steps += pivoted_matmul_sql("ffn_act", "w_down", "down", nc_ffn, CHUNK_SIZE, pw, sq)
        steps += residual_add_sql("x_after_attn", "down", "x_out")
        result = self._run_pipeline(steps)
        np.testing.assert_allclose(result, self.baseline, atol=ATOL, rtol=1e-3,
                                   err_msg="Grouped pivot (8/4) mismatch")

    def test_all_combined(self):
        """All three optimizations combined produce identical output."""
        # CTE merge + fusion + ROW2COL (grouped: pw=8, sq=4)
        pw, sq = 8, 4
        nc = N_CHUNKS_HIDDEN
        nc_ffn = N_CHUNKS_FFN

        # G1: norm1 (shared)
        g1 = rmsnorm_sql("x", "norm1", "x_norm1")

        # G2: fused QKV (no ROW2COL for fused) + attention pipeline
        g2 = []
        g2 += fused_qkv_sql("x_norm1", "w_q", "w_k", "w_v",
                            "q_proj", "k_proj", "v_proj")
        g2 += rope_sql("q_proj", "rope", "q_rope")
        g2 += rope_sql("k_proj", "rope", "k_rope")
        g2 += qk_attn_sql("q_rope", "k_rope", "qk_scores")
        g2 += softmax_sql("qk_scores", "attn_weights")
        g2 += attn_vmul_sql("attn_weights", "v_proj", "attn_out")
        g2 += pivoted_matmul_sql("attn_out", "w_o", "o_proj", nc,
                                 CHUNK_SIZE, pw, sq)
        g2 += residual_add_sql("x", "o_proj", "x_after_attn")

        # G3: norm2 (shared)
        g3 = rmsnorm_sql("x_after_attn", "norm2", "x_norm2")

        # G4: fused gate+up + SwiGLU + pivoted down + residual
        g4 = []
        g4 += fused_gateup_sql("x_norm2", "w_gate", "w_up", "gate", "up")
        g4 += swiglu_sql("gate", "up", "ffn_act")
        g4 += pivoted_matmul_sql("ffn_act", "w_down", "down", nc_ffn,
                                 CHUNK_SIZE, pw, sq)
        g4 += residual_add_sql("x_after_attn", "down", "x_out")

        steps = [_merge_to_cte(g1), _merge_to_cte(g2),
                 _merge_to_cte(g3), _merge_to_cte(g4)]

        result = self._run_pipeline(steps)
        np.testing.assert_allclose(result, self.baseline, atol=ATOL, rtol=1e-3,
                                   err_msg="Combined optimization mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
