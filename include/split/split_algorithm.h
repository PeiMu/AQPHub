/*
 * Base interface for query split algorithms
 */

#pragma once

#include "adapters/db_adapter.h"
#include "simplest_ir.h"
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace middleware {

// Result of extracting a subquery
struct SubqueryExtraction {
  explicit SubqueryExtraction(std::set<unsigned int> table_indices,
                              std::string temp_name = "")
      : executed_table_indices(std::move(table_indices)),
        temp_table_name(std::move(temp_name)) {}

  // Get the IR to execute (prefers sub_ir over pipeline_breaker_ptr)
  ir_sql_converter::SimplestStmt *GetExecutableIR() const {
    if (sub_ir) {
      return sub_ir.get();
    }
    return pipeline_breaker_ptr;
  }

  // Set of table indices that will be executed in this subquery
  std::set<unsigned int> executed_table_indices;

  // Optional: name for the temporary table to create
  std::string temp_table_name;

  // The built sub-IR for this subquery (owns the IR)
  // For FK-based strategies: NEW sub-IR built from cluster tables (used for
  // execution) For TopDown: typically null (uses pipeline_breaker_ptr instead)
  std::unique_ptr<ir_sql_converter::SimplestStmt> sub_ir;

  // Optimizer's estimated rows for this subquery (from
  // EXPLAIN/GetEstimatedCost) Used when update_temp_card is disabled to
  // simulate inaccurate cardinality
  double estimated_rows = 0;

  // Pointer to node in the ORIGINAL tree that should be replaced with temp
  // table For FK-based strategies: the LCA node containing cluster tables (used
  // for UpdateRemainingIR) For TopDown: points to the subtree for both
  // execution AND replacement
  ir_sql_converter::SimplestStmt *pipeline_breaker_ptr = nullptr;

  // When true: sub_ir holds the final plan IR ready for direct execution.
  // ExecuteOneIteration sets remaining_ir = sub_ir and stops without running
  // SQL or calling UpdateRemainingIR. Used by NodeBasedSplitter to hand the
  // merged final plan back to ExecuteSplitLoop (both early and late terminal).
  bool is_final = false;
};

class SplitAlgorithm {
public:
  explicit SplitAlgorithm(DBAdapter *adapter) : adapter_(adapter) {}
  virtual ~SplitAlgorithm() = default;

  // Strategy-specific preprocessing (called once before splitting loop)
  // For TopDown: runs IR-level ReorderGet
  // For FK-based: extracts foreign keys
  virtual void Preprocess(std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) {
  }

  // Extract next subquery to execute from the remaining IR
  // Returns: SubqueryExtraction with sub-IR and table indices
  // Returns nullptr when no more splits needed
  virtual std::unique_ptr<SubqueryExtraction>
  ExtractNextSubquery(ir_sql_converter::SimplestStmt *remaining_ir) = 0;

  // Check if splitting is complete (typically when only 1 table left)
  virtual bool
  IsComplete(const ir_sql_converter::SimplestStmt *remaining_ir) = 0;

  // Get strategy name for logging
  virtual std::string GetStrategyName() const = 0;

  // Update remaining IR after executing a subquery
  // Different strategies have different implementations:
  // - TopDown: Replace subtree directly (DuckDB style) - returns same IR
  // modified in-place
  // - FK-based: Rebuild the IR (PostgreSQL style) - consumes old IR, returns
  // new one column_mappings: (old_table_idx, old_col_idx) for each column
  // column_names: computed column names matching SQL generator's convention
  // Takes ownership of remaining_ir to allow moving expressions instead of
  // cloning Returns the updated remaining IR
  virtual std::unique_ptr<ir_sql_converter::SimplestStmt> UpdateRemainingIR(
      std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
      const std::set<unsigned int> &executed_table_indices,
      unsigned int temp_table_index, const std::string &temp_table_name,
      uint64_t temp_table_cardinality,
      const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
      const std::vector<std::string> &column_names) = 0;

  // When true, ExecuteOneIteration skips the generic UpdateRemainingIRIndices
  // step. NodeBasedSplitter returns true because DuckDB's UpdateSubqueriesIndex
  // / UpdateTableExpr handle all index updates internally.
  virtual bool SkipUpdateIndices() const { return false; }

  // Get the maximum table index in the original IR
  // Used to generate new table indices for temp tables
  unsigned int GetMaxTableIndex() const { return max_table_index_; }

  // Table name lookup (used by IRQuerySplitter::ComputeColumnAlias)
  std::string GetTableName(unsigned int table_idx) const {
    auto it = table_index_to_name_.find(table_idx);
    if (it != table_index_to_name_.end()) {
      return it->second;
    }
    return "";
  }

  // Add a table mapping (used when creating temp tables)
  void AddTableMapping(unsigned int idx, const std::string &name) {
    table_index_to_name_[idx] = name;
  }

protected:
  // Collect table names from IR scan nodes (shared by all strategies)
  void CollectTableNames(const ir_sql_converter::SimplestStmt *ir) {
    if (!ir)
      return;
    if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
      auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(ir);
      if (scan) {
        table_index_to_name_[scan->GetTableIndex()] = scan->GetTableName();
      }
    }
    for (const auto &child : ir->children) {
      CollectTableNames(child.get());
    }
  }

  DBAdapter *adapter_;

  // Iteration counter
  int split_iteration_ = 0;

  // Track which tables have been executed (now part of a temp table)
  std::set<unsigned int> executed_tables_;

  // Track the maximum table index (for generating new temp table indices)
  unsigned int max_table_index_ = 0;

  // Table index to name mapping (collected in Preprocess, updated each
  // iteration)
  std::map<unsigned int, std::string> table_index_to_name_;
};

} // namespace middleware
