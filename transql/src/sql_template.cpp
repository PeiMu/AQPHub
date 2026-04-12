#include "transql/sql_template.h"

#include <sstream>
#include <stdexcept>

namespace transql {

// ---------------------------------------------------------------------------
// EmbedLookup
// ---------------------------------------------------------------------------
// tokens_table(pos INTEGER, token_id INTEGER)
// embed_table(token_id INTEGER, chunk_id INTEGER, v FLOAT[])
// Output: (row_id INTEGER, chunk_id INTEGER, v FLOAT[])
SqlSteps EmbedLookupSQL(const std::string& tokens_table,
                        const std::string& embed_table,
                        const std::string& out_table) {
    std::string sql =
        "SELECT t.pos AS row_id, e.chunk_id, e.v "
        "FROM " + tokens_table + " t "
        "JOIN " + embed_table + " e ON t.token_id = e.row_id "
        "ORDER BY t.pos, e.chunk_id";
    return {{sql, out_table}};
}

// ---------------------------------------------------------------------------
// MatMul
// ---------------------------------------------------------------------------
// act(row_id, chunk_id, v FLOAT[]) x weight(row_id, chunk_id, v FLOAT[])
// Step 1 intermediate: {out}_dp (act_row, out_col, val FLOAT)
// Step 2 final:        out      (row_id, chunk_id, v FLOAT[])
SqlSteps MatMulSQL(const std::string& act_table,
                   const std::string& weight_table,
                   const std::string& out_table,
                   int chunk_size) {
    std::string dp = out_table + "_dp";
    std::string cs = std::to_string(chunk_size);

    std::string step1 =
        "SELECT a.row_id AS act_row, w.row_id AS out_col, "
        "SUM(list_dot_product(a.v, w.v)) AS val "
        "FROM " + act_table + " a "
        "JOIN " + weight_table + " w ON a.chunk_id = w.chunk_id "
        "GROUP BY a.row_id, w.row_id";

    // ORDER BY out_col directly (not out_col % cs): out_col is unique within each
    // chunk group, so the ordering is equivalent but avoids modulo sort issues.
    // Use // (integer division) to avoid DuckDB v1.3+ returning DOUBLE for / and
    // then applying banker's rounding in CAST, which mis-assigns chunk boundaries.
    std::string step2 =
        "SELECT act_row AS row_id, "
        "out_col // " + cs + " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + dp + " "
        "GROUP BY act_row, out_col // " + cs;

    return {{step1, dp}, {step2, out_table}};
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
// Step 1: {out}_sq (row_id, sq_sum FLOAT)
// Step 2: out       (row_id, chunk_id, v FLOAT[])
//
// Uses generate_series + indexed access instead of list_zip to avoid
// named-struct extraction, which is unsupported in some DuckDB versions.
SqlSteps RMSNormSQL(const std::string& input_table,
                    const std::string& gamma_table,
                    const std::string& out_table,
                    int hidden_dim,
                    float eps,
                    int /*chunk_size*/) {
    std::string sq = out_table + "_sq";

    // Use a fixed-precision string for eps to avoid exponential notation issues.
    std::ostringstream eps_ss;
    eps_ss << std::fixed;
    eps_ss.precision(10);
    eps_ss << eps;
    std::string eps_str = eps_ss.str();

    std::string step1 =
        "SELECT row_id, "
        "SUM(list_sum(list_transform(v, x -> x * x))) AS sq_sum "
        "FROM " + input_table + " "
        "GROUP BY row_id";

    // len(n.v) gives the chunk size dynamically; avoids needing it as a literal.
    std::string step2 =
        "SELECT n.row_id, n.chunk_id, "
        "list_transform(generate_series(1, len(n.v)), "
        "i -> CAST(n.v[i] / sqrt(s.sq_sum / " +
        std::to_string(hidden_dim) + ".0 + " + eps_str + ") * w.v[i] AS FLOAT)) AS v "
        "FROM " + input_table + " n "
        "JOIN " + sq + " s ON n.row_id = s.row_id "
        "JOIN " + gamma_table + " w ON n.chunk_id = w.chunk_id";

    return {{step1, sq}, {step2, out_table}};
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------
// q_table: (row_id, chunk_id, v FLOAT[chunk_size]) — standard chunked layout
// rope_table: (chunk_id, cos FLOAT[chunk_size/2], sin FLOAT[chunk_size/2])
// Output: (row_id, chunk_id, v_even FLOAT[], v_odd FLOAT[]) — split layout
//
// For pair index i (1-based, i = 1..half):
//   v_even[i] = q[2i-1]*cos[i] - q[2i]*sin[i]   (q at 1-based positions 1,3,5,...)
//   v_odd[i]  = q[2i]*cos[i]   + q[2i-1]*sin[i]  (q at 1-based positions 2,4,6,...)
//
// Uses generate_series + direct array subscripting to avoid list_zip named-struct
// extraction issues in older DuckDB versions.
SqlSteps RoPESQL(const std::string& q_table,
                 const std::string& rope_table,
                 const std::string& out_table,
                 int chunk_size) {
    std::string half = std::to_string(chunk_size / 2);

    std::string sql =
        "SELECT q.row_id, q.chunk_id, "
        "list_transform(generate_series(1, " + half + "), "
          "i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] AS FLOAT)) AS v_even, "
        "list_transform(generate_series(1, " + half + "), "
          "i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd "
        "FROM " + q_table + " q "
        "JOIN " + rope_table + " r ON r.chunk_id = q.chunk_id AND r.row_id = q.row_id";

    return {{sql, out_table}};
}

// ---------------------------------------------------------------------------
// QKAttn
// ---------------------------------------------------------------------------
// Inputs use RoPE split layout (v_even, v_odd).
// Output: (q_tok, k_tok, head_id, score) scalar.
// GQA join: q.chunk_id / (chunks_per_head * group_size) = k.chunk_id / chunks_per_head
//           q.chunk_id % chunks_per_head = k.chunk_id % chunks_per_head
SqlSteps QKAttnSQL(const std::string& q_rope_table,
                   const std::string& k_rope_table,
                   const std::string& out_table,
                   int num_q_heads,
                   int num_kv_heads,
                   int head_dim,
                   int chunk_size) {
    int group_size      = num_q_heads / num_kv_heads;   // = 4
    int chunks_per_head = head_dim / chunk_size;         // = 4 (128/32)

    std::string cph  = std::to_string(chunks_per_head);
    std::string cphg = std::to_string(chunks_per_head * group_size);

    // Note: 1/sqrt(d_k) scaling is absorbed into W_Q during preprocessing
    // (constant folding, Section 3). No runtime scaling needed.
    std::string sql =
        "SELECT q.row_id AS q_tok, k.row_id AS k_tok, "
        "q.chunk_id // " + cph + " AS head_id, "
        "SUM(list_dot_product(q.v_even, k.v_even) + "
             "list_dot_product(q.v_odd, k.v_odd)) AS score "
        "FROM " + q_rope_table + " q "
        "JOIN " + k_rope_table + " k "
        "ON q.chunk_id % " + cph + " = k.chunk_id % " + cph + " "
        "AND q.chunk_id // " + cphg + " = k.chunk_id // " + cph + " "
        "AND k.row_id <= q.row_id "
        "GROUP BY q.row_id, k.row_id, q.chunk_id // " + cph;

    return {{sql, out_table}};
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------
// Input/output: (q_tok, k_tok, head_id, score/attn_weight) scalar
SqlSteps SoftmaxSQL(const std::string& input_table,
                    const std::string& out_table) {
    std::string max_t = out_table + "_max";
    std::string exp_t = out_table + "_exp";
    std::string sum_t = out_table + "_sum";

    std::string step1 =
        "SELECT q_tok, head_id, MAX(score) AS max_score "
        "FROM " + input_table + " "
        "GROUP BY q_tok, head_id";

    std::string step2 =
        "SELECT s.q_tok, s.k_tok, s.head_id, "
        "EXP(s.score - m.max_score) AS exp_val "
        "FROM " + input_table + " s "
        "JOIN " + max_t + " m ON s.q_tok = m.q_tok AND s.head_id = m.head_id";

    std::string step3 =
        "SELECT q_tok, head_id, SUM(exp_val) AS sum_exp "
        "FROM " + exp_t + " "
        "GROUP BY q_tok, head_id";

    std::string step4 =
        "SELECT e.q_tok, e.k_tok, e.head_id, "
        "CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight "
        "FROM " + exp_t + " e "
        "JOIN " + sum_t + " s ON e.q_tok = s.q_tok AND e.head_id = s.head_id";

    return {{step1, max_t}, {step2, exp_t}, {step3, sum_t}, {step4, out_table}};
}

// ---------------------------------------------------------------------------
// AttnVMul
// ---------------------------------------------------------------------------
// attn_table: (q_tok, k_tok, head_id, attn_weight)
// v_table:    (tok, chunk_id, v FLOAT[]) standard chunked
// Output:     (row_id, chunk_id, v FLOAT[]) standard chunked [seq, hidden_dim]
// Steps: expand V → weighted sum → re-chunk
SqlSteps AttnVMulSQL(const std::string& attn_table,
                     const std::string& v_table,
                     const std::string& out_table,
                     int num_q_heads,
                     int num_kv_heads,
                     int head_dim,
                     int chunk_size) {
    int chunks_per_head = head_dim / chunk_size;          // = 4
    int group_size      = num_q_heads / num_kv_heads;     // = 4
    std::string cph = std::to_string(chunks_per_head);
    std::string gs  = std::to_string(group_size);

    std::string vs = out_table + "_vs";   // V scalar
    std::string wt = out_table + "_w";    // weighted (still scalar)

    // Expand V to scalar rows: (tok, chunk_id, elem_pos, val)
    // Use parallel UNNEST in SELECT (DuckDB-specific zipping) instead of
    // WITH ORDINALITY which is not implemented in DuckDB.
    std::string step1 =
        "SELECT row_id AS tok, chunk_id, "
        "unnest(generate_series(0, " + std::to_string(chunk_size - 1) + ")) AS elem_pos, "
        "CAST(unnest(v) AS FLOAT) AS val "
        "FROM " + v_table;

    // Weighted sum over k_tok per (q_tok, out_chunk_id, elem_pos)
    std::string step2 =
        "SELECT s.q_tok, "
        "s.head_id * " + cph + " + v.chunk_id % " + cph + " AS out_chunk_id, "
        "v.elem_pos, "
        "CAST(SUM(s.attn_weight * v.val) AS FLOAT) AS val "
        "FROM " + attn_table + " s "
        "JOIN " + vs + " v ON s.k_tok = v.tok "
        "AND s.head_id // " + gs + " = v.chunk_id // " + cph + " "
        "GROUP BY s.q_tok, s.head_id, v.chunk_id, v.elem_pos";

    // Re-chunk scalars into FLOAT[] arrays
    std::string step3 =
        "SELECT q_tok AS row_id, out_chunk_id AS chunk_id, "
        "array_agg(val ORDER BY elem_pos) AS v "
        "FROM " + wt + " "
        "GROUP BY q_tok, out_chunk_id";

    return {{step1, vs}, {step2, wt}, {step3, out_table}};
}

// ---------------------------------------------------------------------------
// SwiGLU
// ---------------------------------------------------------------------------
// SiLU(x) = x / (1 + exp(-x)), output = SiLU(gate) * up (Llama's SwiGLU)
// Uses generate_series + array subscripts to avoid list_zip struct extraction.
SqlSteps SwiGLUSQL(const std::string& gate_table,
                   const std::string& up_table,
                   const std::string& out_table) {
    std::string sql =
        "SELECT g.row_id, g.chunk_id, "
        "list_transform(generate_series(1, len(g.v)), "
        "i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] AS FLOAT)) AS v "
        "FROM " + gate_table + " g "
        "JOIN " + up_table + " u "
        "ON g.row_id = u.row_id AND g.chunk_id = u.chunk_id";
    return {{sql, out_table}};
}

// ---------------------------------------------------------------------------
// ResidualAdd
// ---------------------------------------------------------------------------
// Uses generate_series + array subscripts to avoid list_zip struct extraction.
SqlSteps ResidualAddSQL(const std::string& table_a,
                        const std::string& table_b,
                        const std::string& out_table) {
    std::string sql =
        "SELECT a.row_id, a.chunk_id, "
        "list_transform(generate_series(1, len(a.v)), "
        "i -> CAST(a.v[i] + b.v[i] AS FLOAT)) AS v "
        "FROM " + table_a + " a "
        "JOIN " + table_b + " b "
        "ON a.row_id = b.row_id AND a.chunk_id = b.chunk_id";
    return {{sql, out_table}};
}

// ---------------------------------------------------------------------------
// TopKRouting
// ---------------------------------------------------------------------------
// Step 1: dot-product gating logits (same pattern as MatMul step 1, no re-chunk)
// Step 2: top-k selection + softmax via CTE chain with window function
SqlSteps TopKRoutingSQL(const std::string& act_table,
                        const std::string& gate_weight,
                        const std::string& out_table,
                        int /*num_experts*/,
                        int top_k,
                        int /*chunk_size*/) {
    std::string dp = out_table + "_dp";
    std::string k_str = std::to_string(top_k);

    // Step 1: gating logits dot-product → (act_row, out_col, val)
    std::string step1 =
        "SELECT a.row_id AS act_row, w.row_id AS out_col, "
        "SUM(list_dot_product(a.v, w.v)) AS val "
        "FROM " + act_table + " a "
        "JOIN " + gate_weight + " w ON a.chunk_id = w.chunk_id "
        "GROUP BY a.row_id, w.row_id";

    // Step 2: top-k + softmax in one CTE chain
    std::string step2 =
        "WITH ranked AS ("
        "SELECT act_row AS row_id, out_col AS expert_id, val AS logit, "
        "ROW_NUMBER() OVER (PARTITION BY act_row ORDER BY val DESC) AS rk "
        "FROM " + dp +
        "), topk AS ("
        "SELECT row_id, expert_id, logit FROM ranked WHERE rk <= " + k_str +
        "), topk_max AS ("
        "SELECT row_id, MAX(logit) AS max_l FROM topk GROUP BY row_id"
        "), topk_exp AS ("
        "SELECT t.row_id, t.expert_id, EXP(t.logit - m.max_l) AS exp_val "
        "FROM topk t JOIN topk_max m ON t.row_id = m.row_id"
        "), topk_sum AS ("
        "SELECT row_id, SUM(exp_val) AS sum_exp FROM topk_exp GROUP BY row_id"
        ") SELECT e.row_id, e.expert_id, "
        "CAST(e.exp_val / s.sum_exp AS FLOAT) AS gate_score "
        "FROM topk_exp e JOIN topk_sum s ON e.row_id = s.row_id";

    return {{step1, dp}, {step2, out_table}};
}

// ---------------------------------------------------------------------------
// ExpertFFN
// ---------------------------------------------------------------------------
// 7 steps: gate_dp, gate_rechunk, up_dp, up_rechunk, swiglu, down_dp, down_rechunk
// All intermediates carry expert_id.
// The JOIN with routing on expert_id enables index-filtered weight retrieval.
SqlSteps ExpertFFNSQL(const std::string& act_table,
                      const std::string& routing_table,
                      const std::string& gate_proj,
                      const std::string& up_proj,
                      const std::string& down_proj,
                      const std::string& out_table,
                      int expert_ffn_dim,
                      int chunk_size) {
    std::string cs = std::to_string(chunk_size);
    std::string gate_dp      = out_table + "_gate_dp";
    std::string gate_rechunk = out_table + "_gate";
    std::string up_dp        = out_table + "_up_dp";
    std::string up_rechunk   = out_table + "_up";
    std::string swiglu       = out_table + "_swiglu";
    std::string down_dp      = out_table + "_down_dp";

    // Step 1: gate_proj dot-product (index-filtered by expert_id)
    std::string step1 =
        "SELECT a.row_id AS act_row, r.expert_id, w.row_id AS out_col, "
        "SUM(list_dot_product(a.v, w.v)) AS val "
        "FROM " + act_table + " a "
        "JOIN " + routing_table + " r ON a.row_id = r.row_id "
        "JOIN " + gate_proj + " w ON w.expert_id = r.expert_id "
        "AND a.chunk_id = w.chunk_id "
        "GROUP BY a.row_id, r.expert_id, w.row_id";

    // Step 2: re-chunk gate (carries expert_id)
    std::string step2 =
        "SELECT act_row AS row_id, expert_id, "
        "out_col // " + cs + " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + gate_dp + " "
        "GROUP BY act_row, expert_id, out_col // " + cs;

    // Step 3: up_proj dot-product (same index-filtered pattern)
    std::string step3 =
        "SELECT a.row_id AS act_row, r.expert_id, w.row_id AS out_col, "
        "SUM(list_dot_product(a.v, w.v)) AS val "
        "FROM " + act_table + " a "
        "JOIN " + routing_table + " r ON a.row_id = r.row_id "
        "JOIN " + up_proj + " w ON w.expert_id = r.expert_id "
        "AND a.chunk_id = w.chunk_id "
        "GROUP BY a.row_id, r.expert_id, w.row_id";

    // Step 4: re-chunk up (carries expert_id)
    std::string step4 =
        "SELECT act_row AS row_id, expert_id, "
        "out_col // " + cs + " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + up_dp + " "
        "GROUP BY act_row, expert_id, out_col // " + cs;

    // Step 5: SwiGLU per expert: SiLU(gate) * up
    std::string step5 =
        "SELECT g.row_id, g.expert_id, g.chunk_id, "
        "list_transform(generate_series(1, len(g.v)), "
        "i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] AS FLOAT)) AS v "
        "FROM " + gate_rechunk + " g "
        "JOIN " + up_rechunk + " u "
        "ON g.row_id = u.row_id AND g.expert_id = u.expert_id "
        "AND g.chunk_id = u.chunk_id";

    // Step 6: down_proj dot-product (expert_id from swiglu intermediate)
    std::string step6 =
        "SELECT a.row_id AS act_row, a.expert_id, w.row_id AS out_col, "
        "SUM(list_dot_product(a.v, w.v)) AS val "
        "FROM " + swiglu + " a "
        "JOIN " + down_proj + " w ON w.expert_id = a.expert_id "
        "AND a.chunk_id = w.chunk_id "
        "GROUP BY a.row_id, a.expert_id, w.row_id";

    // Step 7: re-chunk down → out (row_id, expert_id, chunk_id, v)
    std::string step7 =
        "SELECT act_row AS row_id, expert_id, "
        "out_col // " + cs + " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + down_dp + " "
        "GROUP BY act_row, expert_id, out_col // " + cs;

    return {{step1, gate_dp}, {step2, gate_rechunk},
            {step3, up_dp}, {step4, up_rechunk},
            {step5, swiglu}, {step6, down_dp}, {step7, out_table}};
}

// ---------------------------------------------------------------------------
// MoeAggregate
// ---------------------------------------------------------------------------
// Weighted sum of expert outputs, then re-chunk.
// Step 1: scalar-expand expert outputs
// Step 2: weighted sum across experts per (token, chunk, elem)
// Step 3: re-chunk
SqlSteps MoeAggregateSQL(const std::string& expert_out,
                         const std::string& routing_table,
                         const std::string& out_table,
                         int chunk_size) {
    std::string scalar_t  = out_table + "_sc";
    std::string weighted_t = out_table + "_wt";
    std::string cs = std::to_string(chunk_size);

    // Step 1: scalar-expand
    std::string step1 =
        "SELECT row_id, expert_id, chunk_id, "
        "unnest(generate_series(0, " + std::to_string(chunk_size - 1) + ")) AS elem_pos, "
        "CAST(unnest(v) AS FLOAT) AS val "
        "FROM " + expert_out;

    // Step 2: weighted sum across experts
    std::string step2 =
        "SELECT s.row_id, s.chunk_id, s.elem_pos, "
        "CAST(SUM(r.gate_score * s.val) AS FLOAT) AS val "
        "FROM " + scalar_t + " s "
        "JOIN " + routing_table + " r "
        "ON s.row_id = r.row_id AND s.expert_id = r.expert_id "
        "GROUP BY s.row_id, s.chunk_id, s.elem_pos";

    // Step 3: re-chunk
    std::string step3 =
        "SELECT row_id, chunk_id, "
        "array_agg(val ORDER BY elem_pos) AS v "
        "FROM " + weighted_t + " "
        "GROUP BY row_id, chunk_id";

    return {{step1, scalar_t}, {step2, weighted_t}, {step3, out_table}};
}

} // namespace transql
