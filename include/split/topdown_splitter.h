/*
 * TopDown split strategy implementation
 * Splits at FILTER nodes or JOIN nodes (build side)
 */

#pragma once

#include "split/ir_reorder_get.h"
#include "split/split_algorithm.h"
#include <queue>
#include <set>

namespace middleware {

class TopDownSplitter : public SplitAlgorithm {
public:
  explicit TopDownSplitter(DBAdapter *adapter, bool enable_reorder = true)
      : SplitAlgorithm(adapter), reorder_get_(adapter),
        enable_reorder_(enable_reorder) {}

  // Preprocess: Run IR-level ReorderGet if enabled
  void Preprocess(std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) override;

  std::unique_ptr<SubqueryExtraction>
  ExtractNextSubquery(ir_sql_converter::SimplestStmt *remaining_ir) override;

  bool IsComplete(const ir_sql_converter::SimplestStmt *remaining_ir) override;

  std::string GetStrategyName() const override { return "TopDown"; }

  // Update remaining IR by replacing subtree (DuckDB style)
  // Because split points align with subtree boundaries
  // Takes ownership but modifies in-place and returns the same IR
  std::unique_ptr<ir_sql_converter::SimplestStmt> UpdateRemainingIR(
      std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
      const std::set<unsigned int> &executed_table_indices,
      unsigned int temp_table_index, const std::string &temp_table_name,
      uint64_t temp_table_cardinality,
      const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
      const std::vector<std::string> &column_names) override;

private:
  // Visit operator tree and identify split points
  void VisitOperator(ir_sql_converter::SimplestStmt *node, bool is_top_most);

  // Collect all table indices in a subtree
  std::set<unsigned int>
  CollectTableIndices(const ir_sql_converter::SimplestStmt *node) const;

  // Count number of base tables (scans) in the IR
  int CountBaseTables(const ir_sql_converter::SimplestStmt *node) const;

  // Helper to get node type name for debugging
  std::string GetNodeTypeName(ir_sql_converter::SimplestNodeType type) const;

  bool
  CheckSameTableInSubtree(ir_sql_converter::SimplestStmt *node,
                          std::unordered_set<std::string> &seen_tables) const;

  IRReorderGet reorder_get_;
  bool enable_reorder_;

  // Pointer to the found split point during VisitOperator traversal
  // This is set during tree traversal and used in ExtractNextSubquery
  ir_sql_converter::SimplestStmt *found_split_node_ = nullptr;

  // Current subquery index
  int query_split_index_ = 0;

  // table_index -> table_name
  std::unordered_map<unsigned int, std::string> table_names_;
};

} // namespace middleware
