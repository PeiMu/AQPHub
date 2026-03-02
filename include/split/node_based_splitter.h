/*
 * NodeBasedSplitter: DuckDB MiddleOptimize-driven split strategy.
 *
 * This class is a minimal SplitAlgorithm stub. The actual loop logic lives in
 * IRQuerySplitter::ExecuteSplitLoopNodeBased, which detects NODE_BASED early
 * in ExecuteWithSplit and bypasses the generic SplitAlgorithm loop entirely.
 * This stub exists only so splitter_ is non-null and NeedsSplit() returns true.
 */

#pragma once

#ifdef HAVE_DUCKDB

#include "adapters/duckdb_adapter.h"
#include "split/split_algorithm.h"

namespace middleware {

class NodeBasedSplitter : public SplitAlgorithm {
public:
  NodeBasedSplitter(DBAdapter *exec_adapter, DuckDBAdapter *plan_adapter);

  void
  Preprocess(std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) override;

  std::unique_ptr<SubqueryExtraction>
  ExtractNextSubquery(ir_sql_converter::SimplestStmt *remaining_ir) override;

  bool IsComplete(const ir_sql_converter::SimplestStmt *remaining_ir) override;

  std::unique_ptr<ir_sql_converter::SimplestStmt> UpdateRemainingIR(
      std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
      const std::set<unsigned int> &executed_table_indices,
      unsigned int temp_table_index, const std::string &temp_table_name,
      uint64_t temp_table_cardinality,
      const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
      const std::vector<std::string> &column_names) override;

  std::string GetStrategyName() const override { return "NodeBased"; }
};

} // namespace middleware

#endif // HAVE_DUCKDB
