#pragma once

#include <string>
#include <utility>
#include <vector>

#include "dag_to_tree.h"
#include "sql_template.h"
#include "tensor_dag.h"

namespace transql {

// Options controlling which Section 4 post-optimizations are applied.
struct PostOptOptions {
    bool cte_merge     = true;   // §4.1: merge non-shared steps into CTEs
    bool table_fusion  = true;   // §4.2: fuse QKV / gate+up weight tables via UNION ALL
    bool row2col_pivot = true;   // §4.3: ROW2COL pivoting for MatMul (CROSS JOIN + POSITIONAL JOIN)
};

// Section 4 post-optimizer: produces optimized SQL steps from a TensorComputeDAG.
//
// Drop-in replacement for TensorDagSplitter::Convert() — produces the same
// Step type (SimplestRawSQL + table_name) but with fewer, larger SQL statements.
//
// Usage:
//   TransqlPostOpt opt({.cte_merge=true, .table_fusion=true, .row2col_pivot=true});
//   auto steps = opt.Convert(dag);
class TransqlPostOpt {
public:
    explicit TransqlPostOpt(PostOptOptions opts = {}) : opts_(opts) {}

    // Convert DAG to optimized SQL steps.
    // Returns (SimplestRawSQL, table_name) pairs — same type as TensorDagSplitter.
    std::vector<TensorDagSplitter::Step> Convert(const TensorComputeDAG& dag);

private:
    PostOptOptions opts_;

    // ── CTE Merging (§4.1) ──────────────────────────────────────────────

    // Collect raw (sql, table_name) pairs for a group of DAG nodes,
    // then merge all but the last into a WITH ... AS (...) CTE block.
    // The last step becomes the body, materialized as a temp table.
    static TensorDagSplitter::Step EmitCTEBlock(
            const std::vector<SqlStep>& steps);

    // ── Table Fusion (§4.2) ─────────────────────────────────────────────

    // Fused QKV: single MatMul with UNION ALL weight + 3 split re-chunks.
    // norm_out: activation table name (shared norm output).
    // q/k/v_weight: weight table names.
    // q/k/v_out: output table names for the 3 splits.
    // Returns SqlSteps that produce q_out, k_out, v_out.
    static SqlSteps FusedQKVSQL(
            const std::string& norm_out,
            const std::string& q_weight, const std::string& k_weight,
            const std::string& v_weight,
            const std::string& q_out, const std::string& k_out,
            const std::string& v_out,
            int q_dim, int kv_dim, int chunk_size);

    // Fused gate+up: single MatMul with UNION ALL weight + 2 split re-chunks.
    static SqlSteps FusedGateUpSQL(
            const std::string& norm_out,
            const std::string& gate_weight, const std::string& up_weight,
            const std::string& gate_out, const std::string& up_out,
            int ffn_dim, int chunk_size);

    // ── ROW2COL Pivoting (§4.3) ─────────────────────────────────────────

    // Generate SQL to pivot (row_id, chunk_id, v) → (row_id, chunk0, ..., chunkN).
    // Returns a single SELECT statement (used as CTE body).
    static std::string PivotSQL(const std::string& table_name, int n_chunks);

    // Generate pivoted MatMul: per-chunk CTEs with CROSS JOIN + POSITIONAL JOIN.
    // act_pivot, weight_pivot: names of pivoted tables/CTEs.
    // Returns SqlSteps: [c0, c1, ..., cN-1, matmul_dp].
    static SqlSteps PivotedMatMulDotProduct(
            const std::string& act_pivot,
            const std::string& weight_pivot,
            const std::string& dp_out,
            int n_chunks);

    // Full pivoted MatMul: pivot activation + dot product + re-chunk.
    // weight_table: original (row_id, chunk_id, v) weight table —
    //               if a *_pivot table exists, it is used directly.
    static SqlSteps PivotedMatMulSQL(
            const std::string& act_table,
            const std::string& weight_table,
            const std::string& out_table,
            int n_chunks, int chunk_size);
};

} // namespace transql
