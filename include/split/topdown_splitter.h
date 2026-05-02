/*
 * TopDown split strategy implementation
 * Splits at FILTER nodes or JOIN nodes (build side)
 */

#pragma once

#include "split/ir_reorder_get.h"
#include "split/split_algorithm.h"
#include <map>
#include <queue>
#include <set>

namespace middleware {

class TopDownSplitter : public AQPSplitter {
public:
  explicit TopDownSplitter(EngineAdapter *adapter, bool enable_reorder = true)
      : AQPSplitter(adapter), reorder_get_(adapter),
        enable_reorder_(enable_reorder) {}

  // Preprocess: Run IR-level ReorderGet if enabled
  void Preprocess(std::unique_ptr<ir_sql_converter::AQPStmt> &ir) override;

  std::unique_ptr<SubqueryExtraction>
  SplitIR(ir_sql_converter::AQPStmt *remaining_ir) override;

  bool IsComplete(const ir_sql_converter::AQPStmt *remaining_ir) override;

  std::string GetStrategyName() const override { return "TopDown"; }

  // Re-run IR-level reorder before each split iteration so that temp tables
  // (with actual cardinalities) influence join ordering.
  void ReorderBeforeSplit(std::unique_ptr<ir_sql_converter::AQPStmt> &ir) override;

  // Update remaining IR by replacing subtree (DuckDB style)
  // Because split points align with subtree boundaries
  // Takes ownership but modifies in-place and returns the same IR
  std::unique_ptr<ir_sql_converter::AQPStmt> UpdateRemainingIR(
      std::unique_ptr<ir_sql_converter::AQPStmt> remaining_ir,
      const std::set<unsigned int> &executed_table_indices,
      unsigned int temp_table_index, const std::string &temp_table_name,
      uint64_t temp_table_cardinality,
      const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
      const std::vector<std::string> &column_names) override;

private:
  // Visit operator tree and identify the next split point.
  // No parameter: top_most_ is a member variable (mirrors DuckDB's top_most
  // member in TopDownSplit), reset to true before each call from
  // SplitIR.
  void Visit(ir_sql_converter::AQPStmt *node);

  // Collect all table indices in a subtree
  std::set<unsigned int>
  CollectTableIndices(const ir_sql_converter::AQPStmt *node) const;

  // Count number of base tables (scans) in the IR
  int CountBaseTables(const ir_sql_converter::AQPStmt *node) const;

  // Helper to get node type name for debugging
  std::string GetNodeTypeName(ir_sql_converter::SimplestNodeType type) const;

  bool
  CheckSameTableInSubtree(ir_sql_converter::AQPStmt *node,
                          std::unordered_set<std::string> &seen_tables) const;

  // Fallback for when Visit() finds no split point: pick the deepest
  // right child (build side) in the tree.
  ir_sql_converter::AQPStmt *
  FindDeepestRightChild(ir_sql_converter::AQPStmt *node) const;

  // Find the deepest JoinNode in a left-deep chain.  Used when Visit()
  // picks a split that covers the entire tree — we drill down to the
  // smallest join (2-3 tables) as the first subquery instead.
  ir_sql_converter::AQPStmt *
  FindDeepestJoin(ir_sql_converter::AQPStmt *node) const;

  // Collect the minimum set of attributes that the remaining IR needs from
  // found_split_node_'s subtree.  Mirrors FK-splitter's required_attrs logic:
  // (a) top-level target_list attrs from subquery tables,
  // (b) AGGR/ORDER node attrs from subquery tables,
  // (c) cross-boundary join condition attrs (one side in subquery, other not).
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
  CollectRequiredAttrs(const ir_sql_converter::AQPStmt *full_ir,
                       const std::set<unsigned int> &subquery_tables) const;

  // Wrap found_split_node_ in a SimplestProjection node in-place.
  // Finds found_split_node_'s parent, extracts it, wraps in a projection with
  // required_attrs as target_list, puts the projection back.
  // Updates found_split_node_ to point to the new projection.
  // Returns pointer to the new projection, or nullptr on failure.
  ir_sql_converter::AQPStmt *
  WrapInProjection(ir_sql_converter::AQPStmt *remaining_ir,
                   std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
                       required_attrs);

  // Build column name map from the IR (before ReOptimizeIR corrupts names).
  // Maps (table_name, col_idx) → correct column name.
  void BuildColumnNameMap(const ir_sql_converter::AQPStmt *ir);

  // Look up the correct column name for a given table and col_idx.
  std::string LookupColumnName(const std::string &table_name,
                               unsigned int col_idx) const;

  // Fix corrupted attr names in the IR after ReOptimizeIR.
  void FixAttrName(ir_sql_converter::SimplestAttr *attr) const;
  void FixExprNames(ir_sql_converter::AQPExpr *expr) const;
  void FixAllAttrNames(ir_sql_converter::AQPStmt *node) const;

  IRReorderGet reorder_get_;
  bool enable_reorder_;

  // Pointer to the found split point during SplitIR traversal
  // This is set during tree traversal and used in SplitIR
  ir_sql_converter::AQPStmt *found_split_node_ = nullptr;

  // Mirrors DuckDB's TopDownSplit::top_most member variable.
  // Starts true at the beginning of each SplitIR traversal and is set to
  // false the first time a split-worthy JOIN or top-level FILTER is seen.
  // Using a member variable (not a parameter) so the flag propagates across
  // recursive calls through non-split nodes (AGGR, ORDER, LIMIT, etc.).
  bool top_most_ = true;

  // Current subquery index
  int query_split_index_ = 0;

  // Correct column names from the original IR (before ReOptimizeIR).
  // Key: (table_name, col_idx), Value: correct column name.
  std::map<std::pair<std::string, unsigned int>, std::string> col_name_map_;

};

} // namespace middleware
