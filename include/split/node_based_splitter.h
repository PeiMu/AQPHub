/*
 * NodeBasedSplitter: DuckDB MiddleOptimize-driven split strategy.
 *
 * Mirrors DuckDB's client_context.cpp split loop.  All DuckDB-specific state
 * (plan, QuerySplit, SubqueryPreparer, …) lives here; ExecuteSplitLoop drives
 * it through the standard SplitAlgorithm interface.
 *
 * Loop structure per iteration:
 *   ExtractNextSubquery():
 *     BLOCK 1 – if pending subqueries: MergeSubquery + UpdateProjHead
 *               MiddleOptimize + Clear + Split
 *     BLOCK 2 – ReorderTables + MergeSubquery + UpdateProjHead + Clear + Split
 *     Early terminal (size==1): MergeToSubquery, return is_final extraction
 *     Normal: GenerateProjHead → ConvertPlanToIR, return sub-IR extraction
 *   UpdateRemainingIR():
 *     SetNewTableIndex + MergeDataChunk + MergeSibling
 *     UpdateSubqueriesIndex + UpdateTableExpr
 *     Late terminal (size==1): MergeToSubquery + UpdateProjHead,
 *                              return final plan IR as remaining_ir
 */

#pragma once

#ifdef HAVE_DUCKDB

#include "adapters/duckdb_adapter.h"
#include "duckdb/optimizer/reorder_get.h"
#include "split/split_algorithm.h"
#include <memory>
#include <vector>

namespace middleware {

class NodeBasedSplitter : public SplitAlgorithm {
public:
  NodeBasedSplitter(DBAdapter *exec_adapter, DuckDBAdapter *plan_adapter,
                    bool enable_debug_print = false);

  // Takes the DuckDB plan from plan_adapter_ and initialises split machinery.
  // The ir parameter is unused (DuckDB plan drives everything).
  void Preprocess(std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) override;

  // Runs BLOCK 1 + BLOCK 2 of the DuckDB split loop.
  // – Early terminal: returns is_final extraction with final plan IR.
  // – Normal: returns extraction whose sub_ir is the next sub-plan IR.
  std::unique_ptr<SubqueryExtraction>
  ExtractNextSubquery(ir_sql_converter::SimplestStmt *remaining_ir) override;

  // Returns true once the plan has been fully merged and handed off.
  bool IsComplete(const ir_sql_converter::SimplestStmt *remaining_ir) override;

  // DuckDB's UpdateSubqueriesIndex / UpdateTableExpr keep all bindings
  // consistent; the generic IR-level UpdateRemainingIRIndices must be skipped.
  bool SkipUpdateIndices() const override { return true; }

  // Inserts a CHUNK_GET node for the just-executed temp table and advances
  // the DuckDB split state.  Returns the final plan IR when late-terminal,
  // nullptr otherwise (remaining_ir is unused by NodeBased logic).
  std::unique_ptr<ir_sql_converter::SimplestStmt> UpdateRemainingIR(
      std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
      const std::set<unsigned int> &executed_table_indices,
      unsigned int temp_table_index, const std::string &temp_table_name,
      uint64_t temp_table_cardinality,
      const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
      const std::vector<std::string> &column_names) override;

  std::string GetStrategyName() const override { return "NodeBased"; }

private:
  // ── DuckDB adapter that owns the plan and the binder ─────────────────────
  DuckDBAdapter *plan_adapter_;

  // ── Per-query split state (initialised in Preprocess) ────────────────────
  duckdb::ClientContext *ctx_ = nullptr;
  duckdb::unique_ptr<duckdb::LogicalOperator> plan_;
  std::unique_ptr<duckdb::QuerySplit> qs_;
  std::unique_ptr<duckdb::SubqueryPreparer> sp_;
  std::unique_ptr<duckdb::ReorderGet> reorder_get_;

  // Subquery groups produced by the most recent Split call
  duckdb::subquery_queue subqueries_;
  duckdb::table_expr_info table_expr_queue_;
  std::vector<duckdb::TableExpr> proj_expr_;

  // Sibling node saved from the previous ExtractNextSubquery (parallel path)
  duckdb::unique_ptr<duckdb::LogicalOperator> last_sibling_node_;

  // Output types of the sub-plan produced in ExtractNextSubquery;
  // needed in UpdateRemainingIR to build the ColumnDataCollection.
  duckdb::vector<duckdb::LogicalType> sub_plan_types_;

  bool merge_sibling_expr_ = false;
  bool terminal_ = false;
  bool external_execution_ = false; // true when exec adapter != DuckDB

  // ── Helpers ──────────────────────────────────────────────────────────────
  // Run MiddleOptimize inside a transaction (matches client_context.cpp).
  void RunMiddleOptimize();

  // Convert the current plan_ to an IR and return it.  plan_ is moved into
  // the adapter and consumed by ConvertPlanToIR; caller must not use plan_
  // afterwards (terminal paths only).
  std::unique_ptr<ir_sql_converter::SimplestStmt> TakePlanAsIR();

  bool enable_debug_print_ = false;
};

} // namespace middleware

#endif // HAVE_DUCKDB
