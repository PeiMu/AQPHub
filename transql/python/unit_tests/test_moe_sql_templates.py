"""
test_moe_sql_templates.py -- Unit tests for MOE SQL templates.

Tests TopKRouting, ExpertFFN, and MoeAggregate individually and combined
against NumPy reference implementations.

Run:
    cd /path/to/AQP_middleware
    python -m pytest transql/python/unittest/test_moe_sql_templates.py -v
"""

import unittest
import numpy as np
import duckdb

# ---------------------------------------------------------------------------
# Model parameters (small MOE config for fast testing)
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 32
SEQ_LEN       = 3
HIDDEN_DIM    = 256    # small for testing
NUM_EXPERTS   = 8
TOP_K         = 2
EXPERT_FFN_DIM = 128   # per-expert FFN intermediate dim
EPS           = 1e-5
ATOL          = 2e-3

rng = np.random.default_rng(77)


# ===========================================================================
# DuckDB helpers
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


def load_expert_weight(conn, name, weight_3d, cs=CHUNK_SIZE):
    """Load [num_experts, out_dim, in_dim] as (expert_id, row_id, chunk_id, v)."""
    num_experts, out_dim, in_dim = weight_3d.shape
    conn.execute(
        f"CREATE TEMP TABLE {name} "
        f"(expert_id INTEGER, row_id INTEGER, chunk_id INTEGER, v FLOAT[])")
    rows = [
        (e, r, c, weight_3d[e, r, c*cs:(c+1)*cs].astype(np.float32).tolist())
        for e in range(num_experts) for r in range(out_dim)
        for c in range(in_dim // cs)
    ]
    conn.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)
    conn.execute(f"CREATE INDEX idx_{name}_eid ON {name}(expert_id)")


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
# SQL template mirrors
# ===========================================================================

def topk_routing_sql(act, gate_weight, out, num_experts=NUM_EXPERTS,
                     top_k=TOP_K, cs=CHUNK_SIZE):
    dp = out + "_dp"
    step1 = (
        f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {act} a JOIN {gate_weight} w ON a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, w.row_id")
    step2 = (
        f"WITH ranked AS ("
        f"SELECT act_row AS row_id, out_col AS expert_id, val AS logit, "
        f"ROW_NUMBER() OVER (PARTITION BY act_row ORDER BY val DESC) AS rk "
        f"FROM {dp}"
        f"), topk AS ("
        f"SELECT row_id, expert_id, logit FROM ranked WHERE rk <= {top_k}"
        f"), topk_max AS ("
        f"SELECT row_id, MAX(logit) AS max_l FROM topk GROUP BY row_id"
        f"), topk_exp AS ("
        f"SELECT t.row_id, t.expert_id, EXP(t.logit - m.max_l) AS exp_val "
        f"FROM topk t JOIN topk_max m ON t.row_id = m.row_id"
        f"), topk_sum AS ("
        f"SELECT row_id, SUM(exp_val) AS sum_exp FROM topk_exp GROUP BY row_id"
        f") SELECT e.row_id, e.expert_id, "
        f"CAST(e.exp_val / s.sum_exp AS FLOAT) AS gate_score "
        f"FROM topk_exp e JOIN topk_sum s ON e.row_id = s.row_id")
    return [(step1, dp), (step2, out)]


def expert_ffn_sql(act, routing, gate_proj, up_proj, down_proj, out,
                   expert_ffn_dim=EXPERT_FFN_DIM, cs=CHUNK_SIZE):
    gate_dp      = out + "_gate_dp"
    gate_rechunk = out + "_gate"
    up_dp        = out + "_up_dp"
    up_rechunk   = out + "_up"
    swiglu       = out + "_swiglu"
    down_dp      = out + "_down_dp"

    step1 = (
        f"SELECT a.row_id AS act_row, r.expert_id, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {act} a "
        f"JOIN {routing} r ON a.row_id = r.row_id "
        f"JOIN {gate_proj} w ON w.expert_id = r.expert_id "
        f"AND a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, r.expert_id, w.row_id")
    step2 = (
        f"SELECT act_row AS row_id, expert_id, "
        f"out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {gate_dp} "
        f"GROUP BY act_row, expert_id, out_col // {cs}")
    step3 = (
        f"SELECT a.row_id AS act_row, r.expert_id, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {act} a "
        f"JOIN {routing} r ON a.row_id = r.row_id "
        f"JOIN {up_proj} w ON w.expert_id = r.expert_id "
        f"AND a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, r.expert_id, w.row_id")
    step4 = (
        f"SELECT act_row AS row_id, expert_id, "
        f"out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {up_dp} "
        f"GROUP BY act_row, expert_id, out_col // {cs}")
    # SwiGLU: SiLU(gate) * up
    step5 = (
        f"SELECT g.row_id, g.expert_id, g.chunk_id, "
        f"list_transform(generate_series(1, len(g.v)), "
        f"i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] AS FLOAT)) AS v "
        f"FROM {gate_rechunk} g "
        f"JOIN {up_rechunk} u "
        f"ON g.row_id = u.row_id AND g.expert_id = u.expert_id "
        f"AND g.chunk_id = u.chunk_id")
    step6 = (
        f"SELECT a.row_id AS act_row, a.expert_id, w.row_id AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {swiglu} a "
        f"JOIN {down_proj} w ON w.expert_id = a.expert_id "
        f"AND a.chunk_id = w.chunk_id "
        f"GROUP BY a.row_id, a.expert_id, w.row_id")
    step7 = (
        f"SELECT act_row AS row_id, expert_id, "
        f"out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {down_dp} "
        f"GROUP BY act_row, expert_id, out_col // {cs}")
    return [(step1, gate_dp), (step2, gate_rechunk),
            (step3, up_dp), (step4, up_rechunk),
            (step5, swiglu), (step6, down_dp), (step7, out)]


def moe_aggregate_sql(expert_out, routing, out, cs=CHUNK_SIZE):
    scalar_t   = out + "_sc"
    weighted_t = out + "_wt"
    step1 = (
        f"SELECT row_id, expert_id, chunk_id, "
        f"unnest(generate_series(0, {cs-1})) AS elem_pos, "
        f"CAST(unnest(v) AS FLOAT) AS val "
        f"FROM {expert_out}")
    step2 = (
        f"SELECT s.row_id, s.chunk_id, s.elem_pos, "
        f"CAST(SUM(r.gate_score * s.val) AS FLOAT) AS val "
        f"FROM {scalar_t} s "
        f"JOIN {routing} r "
        f"ON s.row_id = r.row_id AND s.expert_id = r.expert_id "
        f"GROUP BY s.row_id, s.chunk_id, s.elem_pos")
    step3 = (
        f"SELECT row_id, chunk_id, "
        f"array_agg(val ORDER BY elem_pos) AS v "
        f"FROM {weighted_t} "
        f"GROUP BY row_id, chunk_id")
    return [(step1, scalar_t), (step2, weighted_t), (step3, out)]


def residual_add_sql(a, b, out):
    return [(
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(generate_series(1, len(a.v)), "
        f"i -> CAST(a.v[i] + b.v[i] AS FLOAT)) AS v "
        f"FROM {a} a "
        f"JOIN {b} b ON a.row_id=b.row_id AND a.chunk_id=b.chunk_id", out)]


# ===========================================================================
# NumPy reference
# ===========================================================================

def ref_topk_routing(x, gate_w, top_k=TOP_K):
    """Return (expert_ids [seq, top_k], gate_scores [seq, top_k])."""
    logits = (x @ gate_w.T).astype(np.float32)  # [seq, num_experts]
    topk_idx = np.argsort(-logits, axis=-1)[:, :top_k]  # [seq, top_k]
    topk_logits = np.take_along_axis(logits, topk_idx, axis=-1)
    # Softmax over top-k
    max_l = topk_logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(topk_logits - max_l).astype(np.float32)
    scores = (exp_l / exp_l.sum(axis=-1, keepdims=True)).astype(np.float32)
    return topk_idx, scores


def ref_expert_ffn(x, expert_ids, gate_proj_3d, up_proj_3d, down_proj_3d):
    """Per-expert FFN. Returns [seq, top_k, hidden_dim]."""
    seq, top_k = expert_ids.shape
    hidden = x.shape[1]
    out = np.zeros((seq, top_k, hidden), dtype=np.float32)
    for t in range(seq):
        for ki in range(top_k):
            e = expert_ids[t, ki]
            gate = (x[t] @ gate_proj_3d[e].T).astype(np.float32)
            up   = (x[t] @ up_proj_3d[e].T).astype(np.float32)
            # SwiGLU: SiLU(gate) * up
            silu_gate = (gate / (1.0 + np.exp(-gate.astype(np.float64)))).astype(np.float32)
            ffn_act = (silu_gate * up).astype(np.float32)
            out[t, ki] = (ffn_act @ down_proj_3d[e].T).astype(np.float32)
    return out


def ref_moe_aggregate(expert_out, gate_scores):
    """Weighted sum: [seq, top_k, hidden] x [seq, top_k] -> [seq, hidden]."""
    return np.einsum("stk,st->sk", expert_out, gate_scores).astype(np.float32)


# ===========================================================================
# Test class
# ===========================================================================

class TestMoeTemplates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        scale = 0.02
        cls.x = (rng.standard_normal((SEQ_LEN, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.gate_w = (rng.standard_normal((NUM_EXPERTS, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.expert_gate_proj = (rng.standard_normal(
            (NUM_EXPERTS, EXPERT_FFN_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.expert_up_proj = (rng.standard_normal(
            (NUM_EXPERTS, EXPERT_FFN_DIM, HIDDEN_DIM)) * scale).astype(np.float32)
        cls.expert_down_proj = (rng.standard_normal(
            (NUM_EXPERTS, HIDDEN_DIM, EXPERT_FFN_DIM)) * scale).astype(np.float32)

        # NumPy reference
        cls.ref_expert_ids, cls.ref_gate_scores = ref_topk_routing(cls.x, cls.gate_w)
        cls.ref_expert_out = ref_expert_ffn(
            cls.x, cls.ref_expert_ids,
            cls.expert_gate_proj, cls.expert_up_proj, cls.expert_down_proj)
        cls.ref_moe_out = ref_moe_aggregate(cls.ref_expert_out, cls.ref_gate_scores)

    def _build_conn(self):
        conn = new_conn()
        load_2d(conn, "x", self.x)
        load_2d(conn, "gate_w", self.gate_w)
        load_expert_weight(conn, "e_gate_proj", self.expert_gate_proj)
        load_expert_weight(conn, "e_up_proj", self.expert_up_proj)
        load_expert_weight(conn, "e_down_proj", self.expert_down_proj)
        return conn

    def test_topk_routing(self):
        """TopKRouting selects correct experts with correct softmax scores."""
        conn = self._build_conn()
        run_steps(conn, topk_routing_sql("x", "gate_w", "routing"))

        rows = conn.execute(
            "SELECT row_id, expert_id, gate_score "
            "FROM routing ORDER BY row_id, gate_score DESC"
        ).fetchall()

        # Check shape: TOP_K rows per token
        self.assertEqual(len(rows), SEQ_LEN * TOP_K)

        # Extract and compare
        sql_ids = np.zeros((SEQ_LEN, TOP_K), dtype=int)
        sql_scores = np.zeros((SEQ_LEN, TOP_K), dtype=np.float32)
        for i, (rid, eid, gs) in enumerate(rows):
            t = int(rid)
            k = i % TOP_K
            sql_ids[t, k] = int(eid)
            sql_scores[t, k] = float(gs)

        # Expert IDs should match (same top-k selection)
        for t in range(SEQ_LEN):
            self.assertEqual(set(sql_ids[t]), set(self.ref_expert_ids[t]),
                             f"Token {t}: expert IDs mismatch")

        # Scores should match (reorder to match)
        for t in range(SEQ_LEN):
            for k in range(TOP_K):
                ref_eid = self.ref_expert_ids[t, k]
                # Find this expert in SQL results for token t
                sql_k = np.where(sql_ids[t] == ref_eid)[0][0]
                np.testing.assert_allclose(
                    sql_scores[t, sql_k], self.ref_gate_scores[t, k],
                    atol=1e-5, rtol=1e-4,
                    err_msg=f"Token {t}, expert {ref_eid}: score mismatch")

    def test_expert_ffn(self):
        """ExpertFFN produces correct per-expert output with index filtering."""
        conn = self._build_conn()
        run_steps(conn, topk_routing_sql("x", "gate_w", "routing"))
        run_steps(conn, expert_ffn_sql("x", "routing",
                                       "e_gate_proj", "e_up_proj", "e_down_proj",
                                       "expert_out"))

        # Read expert_out: (row_id, expert_id, chunk_id, v)
        rows = conn.execute(
            "SELECT row_id, expert_id, chunk_id, v "
            "FROM expert_out ORDER BY row_id, expert_id, chunk_id"
        ).fetchall()

        # Build result array
        cs = CHUNK_SIZE
        n_chunks = HIDDEN_DIM // cs
        sql_out = {}
        for rid, eid, cid, v in rows:
            key = (int(rid), int(eid))
            if key not in sql_out:
                sql_out[key] = np.zeros(HIDDEN_DIM, dtype=np.float32)
            sql_out[key][int(cid)*cs:(int(cid)+1)*cs] = v

        # Compare against NumPy reference for each (token, expert)
        for t in range(SEQ_LEN):
            for ki in range(TOP_K):
                eid = self.ref_expert_ids[t, ki]
                key = (t, int(eid))
                self.assertIn(key, sql_out,
                              f"Missing expert output for token={t}, expert={eid}")
                np.testing.assert_allclose(
                    sql_out[key], self.ref_expert_out[t, ki],
                    atol=ATOL, rtol=1e-3,
                    err_msg=f"Expert FFN mismatch: token={t}, expert={eid}")

    def test_moe_aggregate(self):
        """MoeAggregate produces correct weighted sum of expert outputs."""
        conn = self._build_conn()
        run_steps(conn, topk_routing_sql("x", "gate_w", "routing"))
        run_steps(conn, expert_ffn_sql("x", "routing",
                                       "e_gate_proj", "e_up_proj", "e_down_proj",
                                       "expert_out"))
        run_steps(conn, moe_aggregate_sql("expert_out", "routing", "moe_out"))

        result = read_2d(conn, "moe_out", SEQ_LEN, HIDDEN_DIM)
        np.testing.assert_allclose(
            result, self.ref_moe_out, atol=ATOL, rtol=1e-3,
            err_msg="MoeAggregate mismatch")

    def test_full_moe_ffn_with_shared(self):
        """Full MOE FFN: routed experts + shared expert + residual."""
        # Create shared expert weights (single dense FFN)
        shared_gate = (rng.standard_normal(
            (EXPERT_FFN_DIM, HIDDEN_DIM)) * 0.02).astype(np.float32)
        shared_up = (rng.standard_normal(
            (EXPERT_FFN_DIM, HIDDEN_DIM)) * 0.02).astype(np.float32)
        shared_down = (rng.standard_normal(
            (HIDDEN_DIM, EXPERT_FFN_DIM)) * 0.02).astype(np.float32)

        conn = self._build_conn()
        load_2d(conn, "shared_gate_w", shared_gate)
        load_2d(conn, "shared_up_w", shared_up)
        load_2d(conn, "shared_down_w", shared_down)

        # SQL pipeline
        steps = []
        steps += topk_routing_sql("x", "gate_w", "routing")
        steps += expert_ffn_sql("x", "routing",
                                "e_gate_proj", "e_up_proj", "e_down_proj",
                                "expert_out")
        steps += moe_aggregate_sql("expert_out", "routing", "moe_routed")

        # Shared expert: standard matmul pattern
        steps.append((
            f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
            f"SUM(list_dot_product(a.v, w.v)) AS val "
            f"FROM x a JOIN shared_gate_w w ON a.chunk_id = w.chunk_id "
            f"GROUP BY a.row_id, w.row_id", "s_gate_dp"))
        steps.append((
            f"SELECT act_row AS row_id, out_col // {CHUNK_SIZE} AS chunk_id, "
            f"array_agg(val ORDER BY out_col) AS v "
            f"FROM s_gate_dp GROUP BY act_row, out_col // {CHUNK_SIZE}", "s_gate"))
        steps.append((
            f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
            f"SUM(list_dot_product(a.v, w.v)) AS val "
            f"FROM x a JOIN shared_up_w w ON a.chunk_id = w.chunk_id "
            f"GROUP BY a.row_id, w.row_id", "s_up_dp"))
        steps.append((
            f"SELECT act_row AS row_id, out_col // {CHUNK_SIZE} AS chunk_id, "
            f"array_agg(val ORDER BY out_col) AS v "
            f"FROM s_up_dp GROUP BY act_row, out_col // {CHUNK_SIZE}", "s_up"))
        # SwiGLU: SiLU(gate) * up
        steps.append((
            f"SELECT g.row_id, g.chunk_id, "
            f"list_transform(generate_series(1, len(g.v)), "
            f"i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] AS FLOAT)) AS v "
            f"FROM s_gate g JOIN s_up u "
            f"ON g.row_id=u.row_id AND g.chunk_id=u.chunk_id", "s_ffn_act"))
        steps.append((
            f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
            f"SUM(list_dot_product(a.v, w.v)) AS val "
            f"FROM s_ffn_act a JOIN shared_down_w w ON a.chunk_id = w.chunk_id "
            f"GROUP BY a.row_id, w.row_id", "s_down_dp"))
        steps.append((
            f"SELECT act_row AS row_id, out_col // {CHUNK_SIZE} AS chunk_id, "
            f"array_agg(val ORDER BY out_col) AS v "
            f"FROM s_down_dp GROUP BY act_row, out_col // {CHUNK_SIZE}", "s_down"))

        # Combine: routed + shared
        steps += residual_add_sql("moe_routed", "s_down", "moe_combined")
        # Final residual: x + moe_combined
        steps += residual_add_sql("x", "moe_combined", "x_out")

        run_steps(conn, steps)
        result = read_2d(conn, "x_out", SEQ_LEN, HIDDEN_DIM)

        # NumPy reference
        # Shared expert
        sg = (self.x @ shared_gate.T).astype(np.float32)
        su = (self.x @ shared_up.T).astype(np.float32)
        silu_su = (su / (1.0 + np.exp(-su.astype(np.float64)))).astype(np.float32)
        s_act = (sg * silu_su).astype(np.float32)
        s_down = (s_act @ shared_down.T).astype(np.float32)

        expected = (self.x + self.ref_moe_out + s_down).astype(np.float32)
        np.testing.assert_allclose(
            result, expected, atol=ATOL, rtol=1e-3,
            err_msg="Full MOE FFN (routed + shared + residual) mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
