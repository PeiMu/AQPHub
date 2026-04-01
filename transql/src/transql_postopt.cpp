#include "transql/transql_postopt.h"

#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "transql/sql_template.h"

namespace transql {

// =====================================================================
// Helpers
// =====================================================================

static int GetIntParam(const TensorDagNode& node, const std::string& key) {
    auto it = node.params.find(key);
    if (it == node.params.end())
        throw std::runtime_error("TensorDagNode missing param: " + key);
    return std::stoi(it->second);
}

static float GetFloatParam(const TensorDagNode& node, const std::string& key) {
    auto it = node.params.find(key);
    if (it == node.params.end())
        throw std::runtime_error("TensorDagNode missing param: " + key);
    return std::stof(it->second);
}

// Expand a single DAG node into raw SqlSteps using the existing templates.
static SqlSteps ExpandNode(const TensorDagNode& node) {
    const auto& in  = node.input_tables;
    const auto& out = node.output_table;

    switch (node.op_type) {
    case TensorOpType::EmbedLookup:
        return EmbedLookupSQL(in[0], in[1], out);
    case TensorOpType::MatMul:
        return MatMulSQL(in[0], in[1], out);
    case TensorOpType::RMSNorm:
        return RMSNormSQL(in[0], in[1], out,
                          GetIntParam(node, "hidden_dim"),
                          GetFloatParam(node, "eps"));
    case TensorOpType::RoPE:
        return RoPESQL(in[0], in[1], out,
                       GetIntParam(node, "chunk_size"));
    case TensorOpType::QKAttn:
        return QKAttnSQL(in[0], in[1], out,
                         GetIntParam(node, "num_q_heads"),
                         GetIntParam(node, "num_kv_heads"),
                         GetIntParam(node, "head_dim"),
                         GetIntParam(node, "chunk_size"));
    case TensorOpType::Softmax:
        return SoftmaxSQL(in[0], out);
    case TensorOpType::AttnVMul:
        return AttnVMulSQL(in[0], in[1], out,
                           GetIntParam(node, "num_q_heads"),
                           GetIntParam(node, "num_kv_heads"),
                           GetIntParam(node, "head_dim"),
                           GetIntParam(node, "chunk_size"));
    case TensorOpType::SwiGLU:
        return SwiGLUSQL(in[0], in[1], out);
    case TensorOpType::ResidualAdd:
        return ResidualAddSQL(in[0], in[1], out);
    case TensorOpType::TopKRouting:
        return TopKRoutingSQL(in[0], in[1], out,
                              GetIntParam(node, "num_experts"),
                              GetIntParam(node, "top_k"),
                              GetIntParam(node, "chunk_size"));
    case TensorOpType::ExpertFFN:
        return ExpertFFNSQL(in[0], in[1], in[2], in[3], in[4], out,
                            GetIntParam(node, "expert_ffn_dim"),
                            GetIntParam(node, "chunk_size"));
    case TensorOpType::MoeAggregate:
        return MoeAggregateSQL(in[0], in[1], out,
                               GetIntParam(node, "chunk_size"));
    default:
        throw std::runtime_error("TransqlPostOpt: unknown TensorOpType");
    }
}

// =====================================================================
// §4.1  CTE Merging
// =====================================================================

TensorDagSplitter::Step TransqlPostOpt::EmitCTEBlock(
        const std::vector<SqlStep>& steps) {
    if (steps.size() == 1) {
        return {std::unique_ptr<ir_sql_converter::AQPStmt>(
                    new ir_sql_converter::SimplestRawSQL(steps[0].first)),
                steps[0].second};
    }

    // Build: WITH cte0 AS (...), cte1 AS (...), ... <final_body>
    std::string sql = "WITH ";
    for (size_t i = 0; i + 1 < steps.size(); ++i) {
        if (i > 0) sql += ", ";
        sql += steps[i].second + " AS (" + steps[i].first + ")";
    }
    sql += " " + steps.back().first;

    return {std::unique_ptr<ir_sql_converter::AQPStmt>(
                new ir_sql_converter::SimplestRawSQL(sql)),
            steps.back().second};
}

// =====================================================================
// §4.2  Table Fusion
// =====================================================================

SqlSteps TransqlPostOpt::FusedQKVSQL(
        const std::string& norm_out,
        const std::string& q_weight, const std::string& k_weight,
        const std::string& v_weight,
        const std::string& q_out, const std::string& k_out,
        const std::string& v_out,
        int q_dim, int kv_dim, int chunk_size) {
    std::string cs = std::to_string(chunk_size);
    std::string q_dim_s  = std::to_string(q_dim);
    std::string kv_off_s = std::to_string(q_dim);
    std::string v_off_s  = std::to_string(q_dim + kv_dim);

    // Fused weight CTE: UNION ALL with row_id offsets
    std::string dp_name = q_out + "_qkv_dp";
    std::string w_cte_name = q_out + "_w_qkv";

    std::string w_cte =
        "SELECT row_id, chunk_id, v FROM " + q_weight +
        " UNION ALL "
        "SELECT row_id + " + kv_off_s + ", chunk_id, v FROM " + k_weight +
        " UNION ALL "
        "SELECT row_id + " + v_off_s + ", chunk_id, v FROM " + v_weight;

    // Single fused dot-product
    std::string dp_sql =
        "SELECT a.row_id AS act_row, w.row_id AS out_col, "
        "SUM(list_dot_product(a.v, w.v)) AS val "
        "FROM " + norm_out + " a "
        "JOIN " + w_cte_name + " w ON a.chunk_id = w.chunk_id "
        "GROUP BY a.row_id, w.row_id";

    // Split + re-chunk: Q
    std::string q_sql =
        "SELECT act_row AS row_id, out_col // " + cs + " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + dp_name + " WHERE out_col < " + q_dim_s + " "
        "GROUP BY act_row, out_col // " + cs;

    // Split + re-chunk: K
    std::string k_sql =
        "SELECT act_row AS row_id, (out_col - " + kv_off_s + ") // " + cs +
        " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + dp_name + " WHERE out_col >= " + kv_off_s +
        " AND out_col < " + v_off_s + " "
        "GROUP BY act_row, (out_col - " + kv_off_s + ") // " + cs;

    // Split + re-chunk: V
    std::string v_sql =
        "SELECT act_row AS row_id, (out_col - " + v_off_s + ") // " + cs +
        " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + dp_name + " WHERE out_col >= " + v_off_s + " "
        "GROUP BY act_row, (out_col - " + v_off_s + ") // " + cs;

    return {
        {w_cte, w_cte_name},
        {dp_sql, dp_name},
        {q_sql, q_out},
        {k_sql, k_out},
        {v_sql, v_out}
    };
}

SqlSteps TransqlPostOpt::FusedGateUpSQL(
        const std::string& norm_out,
        const std::string& gate_weight, const std::string& up_weight,
        const std::string& gate_out, const std::string& up_out,
        int ffn_dim, int chunk_size) {
    std::string cs = std::to_string(chunk_size);
    std::string ffn_s  = std::to_string(ffn_dim);

    std::string dp_name  = gate_out + "_gateup_dp";
    std::string w_cte_name = gate_out + "_w_gateup";

    std::string w_cte =
        "SELECT row_id, chunk_id, v FROM " + gate_weight +
        " UNION ALL "
        "SELECT row_id + " + ffn_s + ", chunk_id, v FROM " + up_weight;

    std::string dp_sql =
        "SELECT a.row_id AS act_row, w.row_id AS out_col, "
        "SUM(list_dot_product(a.v, w.v)) AS val "
        "FROM " + norm_out + " a "
        "JOIN " + w_cte_name + " w ON a.chunk_id = w.chunk_id "
        "GROUP BY a.row_id, w.row_id";

    std::string gate_sql =
        "SELECT act_row AS row_id, out_col // " + cs + " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + dp_name + " WHERE out_col < " + ffn_s + " "
        "GROUP BY act_row, out_col // " + cs;

    std::string up_sql =
        "SELECT act_row AS row_id, (out_col - " + ffn_s + ") // " + cs +
        " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + dp_name + " WHERE out_col >= " + ffn_s + " "
        "GROUP BY act_row, (out_col - " + ffn_s + ") // " + cs;

    return {
        {w_cte, w_cte_name},
        {dp_sql, dp_name},
        {gate_sql, gate_out},
        {up_sql, up_out}
    };
}

// =====================================================================
// §4.3  ROW2COL Pivoting
// =====================================================================

std::string TransqlPostOpt::PivotSQL(const std::string& table_name,
                                      int chunk_count, int chunk_start) {
    // SELECT row_id,
    //   MAX(CASE WHEN chunk_id=start THEN v END) AS chunk0,
    //   MAX(CASE WHEN chunk_id=start+1 THEN v END) AS chunk1, ...
    // FROM table_name GROUP BY row_id
    std::string sql = "SELECT row_id";
    for (int i = 0; i < chunk_count; ++i) {
        sql += ", MAX(CASE WHEN chunk_id = " + std::to_string(chunk_start + i) +
               " THEN v END) AS chunk" + std::to_string(i);
    }
    sql += " FROM " + table_name + " GROUP BY row_id";
    return sql;
}

SqlSteps TransqlPostOpt::PivotedMatMulDotProduct(
        const std::string& act_pivot,
        const std::string& weight_pivot,
        const std::string& dp_out,
        int n_cols, int subquery_width) {
    if (subquery_width <= 0) subquery_width = 1;

    SqlSteps steps;
    int n_sq = (n_cols + subquery_width - 1) / subquery_width;

    for (int sq = 0; sq < n_sq; ++sq) {
        std::string ci = dp_out + "_sq" + std::to_string(sq);
        int col_start = sq * subquery_width;
        int col_end   = std::min(col_start + subquery_width, n_cols);

        // Build sum of dot products for this subquery's columns
        std::string dot_expr;
        for (int c = col_start; c < col_end; ++c) {
            std::string col = "chunk" + std::to_string(c);
            if (c > col_start) dot_expr += " + ";
            dot_expr += "list_dot_product(a." + col + ", w." + col + ")";
        }

        std::string sql =
            "SELECT a.row_id AS act_row, w.row_id AS out_col, "
            + dot_expr + " AS v" + std::to_string(sq) + " "
            "FROM " + act_pivot + " a CROSS JOIN " + weight_pivot + " w "
            "ORDER BY a.row_id, w.row_id";
        steps.push_back({sql, ci});
    }

    // POSITIONAL JOIN reduction
    std::string first = dp_out + "_sq0";
    std::string sum_expr = first + ".v0";
    for (int i = 1; i < n_sq; ++i)
        sum_expr += " + " + dp_out + "_sq" + std::to_string(i) +
                    ".v" + std::to_string(i);

    std::string from_clause = first;
    for (int i = 1; i < n_sq; ++i)
        from_clause += " POSITIONAL JOIN " + dp_out + "_sq" + std::to_string(i);

    std::string final_sql =
        "SELECT " + first + ".act_row, " + first + ".out_col, "
        + sum_expr + " AS val "
        "FROM " + from_clause;

    steps.push_back({final_sql, dp_out});
    return steps;
}

SqlSteps TransqlPostOpt::PivotedMatMulSQL(
        const std::string& act_table,
        const std::string& weight_table,
        const std::string& out_table,
        int n_chunks, int chunk_size,
        int pivot_width, int subquery_width) {
    std::string cs = std::to_string(chunk_size);
    std::string dp_name = out_table + "_dp";

    // Defaults: all chunks at once, one column per CTE
    if (pivot_width <= 0)    pivot_width = n_chunks;
    if (subquery_width <= 0) subquery_width = 1;

    int n_groups = (n_chunks + pivot_width - 1) / pivot_width;

    SqlSteps steps;
    std::vector<std::string> group_dp_names;

    for (int g = 0; g < n_groups; ++g) {
        int chunk_start = g * pivot_width;
        int chunk_count = std::min(pivot_width, n_chunks - chunk_start);

        std::string g_sfx = (n_groups > 1)
            ? "_g" + std::to_string(g) : "";
        std::string act_piv = out_table + "_act_piv" + g_sfx;
        std::string wt_piv  = weight_table + "_piv" + g_sfx;
        std::string g_dp    = (n_groups > 1)
            ? dp_name + "_g" + std::to_string(g) : dp_name;

        // Pivot activation and weight for this group's chunk range
        steps.push_back({PivotSQL(act_table,    chunk_count, chunk_start), act_piv});
        steps.push_back({PivotSQL(weight_table, chunk_count, chunk_start), wt_piv});

        // Subquery CROSS JOINs + POSITIONAL JOIN within group
        auto dp_steps = PivotedMatMulDotProduct(
                act_piv, wt_piv, g_dp, chunk_count, subquery_width);
        steps.insert(steps.end(), dp_steps.begin(), dp_steps.end());

        group_dp_names.push_back(g_dp);
    }

    // Sum across pivot groups via POSITIONAL JOIN
    if (n_groups > 1) {
        std::string sum_expr = group_dp_names[0] + ".val";
        for (int i = 1; i < n_groups; ++i)
            sum_expr += " + " + group_dp_names[i] + ".val";

        std::string from_clause = group_dp_names[0];
        for (int i = 1; i < n_groups; ++i)
            from_clause += " POSITIONAL JOIN " + group_dp_names[i];

        steps.push_back({
            "SELECT " + group_dp_names[0] + ".act_row, " +
            group_dp_names[0] + ".out_col, " + sum_expr + " AS val "
            "FROM " + from_clause,
            dp_name});
    }

    // Re-chunk: same as MatMul step2
    std::string rechunk =
        "SELECT act_row AS row_id, "
        "out_col // " + cs + " AS chunk_id, "
        "array_agg(val ORDER BY out_col) AS v "
        "FROM " + dp_name + " "
        "GROUP BY act_row, out_col // " + cs;
    steps.push_back({rechunk, out_table});

    return steps;
}

// =====================================================================
// Convert — main entry point
// =====================================================================

std::vector<TensorDagSplitter::Step>
TransqlPostOpt::Convert(const TensorComputeDAG& dag) {
    const auto& nodes = dag.Nodes();
    int output_id = dag.OutputNodeId();

    // ── Phase 1: Generate raw SqlSteps per node, with fusion if enabled ──

    // Build a list of (SqlSteps, is_shared) groups per node.
    // Nodes may be consumed by fusion (skipped individually).
    std::unordered_set<int> fused_node_ids;  // nodes consumed by fusion
    std::vector<std::pair<SqlSteps, bool>> node_groups;  // (steps, is_shared)
    node_groups.reserve(nodes.size());

    for (size_t ni = 0; ni < nodes.size(); ++ni) {
        if (fused_node_ids.count(static_cast<int>(ni)))
            continue;

        const auto& node = nodes[ni];
        bool at_output = (node.id == output_id);

        // ── Table fusion detection ──
        if (opts_.table_fusion && node.op_type == TensorOpType::MatMul &&
            ni + 2 < nodes.size()) {
            const auto& n1 = nodes[ni + 1];
            const auto& n2 = nodes[ni + 2];

            // QKV: 3 consecutive MatMuls with same activation input
            if (n1.op_type == TensorOpType::MatMul &&
                n2.op_type == TensorOpType::MatMul &&
                node.input_tables[0] == n1.input_tables[0] &&
                node.input_tables[0] == n2.input_tables[0]) {

                // Infer dimensions from weight table naming convention:
                // Q weight has hidden_dim output rows, K/V have kv_dim output rows.
                // For Llama3-8B: q_dim=4096, kv_dim=1024
                // The dims are encoded in the weight table shapes, but we need
                // them as parameters. Derive from the DAG's model params.
                // For now, find the layer's QKAttn node to get the params.
                int q_dim = -1, kv_dim = -1, cs = 32;
                for (size_t j = ni + 3; j < nodes.size(); ++j) {
                    if (nodes[j].op_type == TensorOpType::QKAttn) {
                        int num_q  = GetIntParam(nodes[j], "num_q_heads");
                        int num_kv = GetIntParam(nodes[j], "num_kv_heads");
                        int hd     = GetIntParam(nodes[j], "head_dim");
                        cs         = GetIntParam(nodes[j], "chunk_size");
                        q_dim  = num_q * hd;
                        kv_dim = num_kv * hd;
                        break;
                    }
                }

                if (q_dim > 0) {
                    auto fused = FusedQKVSQL(
                        node.input_tables[0],
                        node.input_tables[1], n1.input_tables[1],
                        n2.input_tables[1],
                        node.output_table, n1.output_table, n2.output_table,
                        q_dim, kv_dim, cs);

                    node_groups.push_back({std::move(fused), false});
                    fused_node_ids.insert(static_cast<int>(ni + 1));
                    fused_node_ids.insert(static_cast<int>(ni + 2));
                    continue;
                }
            }

            // Gate+up: 2 consecutive MatMuls with same activation input
            if (n1.op_type == TensorOpType::MatMul &&
                node.input_tables[0] == n1.input_tables[0]) {

                // For Llama3-8B, gate_proj and up_proj both have ffn_dim=14336
                // output rows. We can infer ffn_dim from the DAG structure.
                // The SwiGLU node following gate+up confirms this pattern.
                bool has_swiglu = false;
                int ffn_dim = -1;
                for (size_t j = ni + 2; j < nodes.size(); ++j) {
                    if (nodes[j].op_type == TensorOpType::SwiGLU &&
                        nodes[j].input_tables[0] == node.output_table &&
                        nodes[j].input_tables[1] == n1.output_table) {
                        has_swiglu = true;
                        break;
                    }
                }

                if (has_swiglu) {
                    // Infer FFN dim: find the down_proj MatMul that follows SwiGLU.
                    // The down_proj weight has shape [hidden_dim, ffn_dim],
                    // meaning its row_id count = hidden_dim and chunks = ffn_dim/cs.
                    // We need the output dim of gate_proj = ffn_dim.
                    // For Llama3-8B: ffn_dim = 14336.
                    // Heuristic: the gate_proj weight table's row count = ffn_dim.
                    // Since we can't query the DB here, use the Llama3 constant
                    // or derive from the down_proj weight table dimensions.
                    // For generality, store ffn_dim in node params.
                    // Fallback: Llama3-8B hardcoded.
                    ffn_dim = 14336;  // Llama3-8B default
                    int cs = 32;

                    auto fused = FusedGateUpSQL(
                        node.input_tables[0],
                        node.input_tables[1], n1.input_tables[1],
                        node.output_table, n1.output_table,
                        ffn_dim, cs);

                    node_groups.push_back({std::move(fused), false});
                    fused_node_ids.insert(static_cast<int>(ni + 1));
                    continue;
                }
            }
        }

        // ── ROW2COL pivoting for non-fused MatMul ──
        if (opts_.row2col_pivot && node.op_type == TensorOpType::MatMul) {
            // Determine n_chunks from the contracted dimension.
            // The activation and weight share the same chunk_id range.
            // For Llama3-8B: hidden_dim/cs=128, ffn_dim/cs=448, etc.
            // We need to know the chunk count. Since the weight table is a
            // permanent table with known structure, we derive from model params.
            // Default: hidden_dim=4096, cs=32 → 128 chunks.
            // For down_proj: input is ffn_dim, chunks=ffn_dim/cs=448.
            //
            // Heuristic: if the weight table name contains "down_proj",
            // the contracted dim is ffn_dim; otherwise hidden_dim.
            int cs = 32;
            int n_chunks;
            const std::string& wt = node.input_tables[1];
            if (wt.find("down_proj") != std::string::npos) {
                n_chunks = 14336 / cs;  // ffn_dim / cs
            } else {
                n_chunks = 4096 / cs;   // hidden_dim / cs
            }

            auto pivoted = PivotedMatMulSQL(
                    node.input_tables[0], node.input_tables[1],
                    node.output_table, n_chunks, cs,
                    opts_.pivot_width, opts_.subquery_width);
            node_groups.push_back({std::move(pivoted),
                                   node.is_shared || at_output});
            continue;
        }

        // ── Default: use existing template expansion ──
        auto steps = ExpandNode(node);
        node_groups.push_back({std::move(steps),
                               node.is_shared || at_output});
    }

    // ── Phase 2: CTE merge (§4.1) ──

    if (!opts_.cte_merge) {
        // No merging — emit every step as a separate CREATE TEMP TABLE
        std::vector<TensorDagSplitter::Step> result;
        for (auto& group : node_groups) {
            for (auto& s : group.first) {
                result.push_back(
                    {std::unique_ptr<ir_sql_converter::AQPStmt>(
                         new ir_sql_converter::SimplestRawSQL(s.first)),
                     s.second});
            }
        }
        return result;
    }

    // Merge consecutive non-shared groups into CTE blocks.
    std::vector<TensorDagSplitter::Step> result;
    std::vector<SqlStep> current_ctes;

    for (auto& group : node_groups) {
        current_ctes.insert(current_ctes.end(),
                            group.first.begin(), group.first.end());

        if (group.second) {  // is_shared
            result.push_back(EmitCTEBlock(current_ctes));
            current_ctes.clear();
        }
    }

    // Flush any remaining steps (shouldn't happen if output node is shared)
    if (!current_ctes.empty()) {
        result.push_back(EmitCTEBlock(current_ctes));
    }

    return result;
}

} // namespace transql
