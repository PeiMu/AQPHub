/*
 * IR-level query splitter main pipeline
 */

#pragma once

#include "adapters/db_adapter.h"
#include "simplest_ir.h"
#include "split/fk_based_splitter.h"
#include "split/foreign_key_extractor.h"
#include "split/split_algorithm.h"
#include "split/topdown_splitter.h"
#ifdef HAVE_DUCKDB
#include "split/node_based_splitter.h"
#endif
#include "util/param_config.h"
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace middleware {

// Mapping entry for column index updates
struct ColumnMapping {
  unsigned int old_table_idx;
  unsigned int old_col_idx;
  std::string column_name;

  ColumnMapping(unsigned int table_idx, unsigned int col_idx, std::string name)
      : old_table_idx(table_idx), old_col_idx(col_idx),
        column_name(std::move(name)) {}
};

// Temp table information after executing a subquery
struct TempTableInfo {
  std::string table_name;
  unsigned int table_index;
  uint64_t cardinality;
  std::vector<std::string> column_names;

  // Mapping from old (table_idx, col_idx) to position in this temp table
  // column_mappings[i] contains the original (table_idx, col_idx) for column i
  std::vector<ColumnMapping> column_mappings;

  TempTableInfo(std::string name, unsigned int idx, uint64_t card)
      : table_name(std::move(name)), table_index(idx), cardinality(card) {}

  // Find the new column index for a given old (table_idx, col_idx)
  // Returns -1 if not found
  int FindNewColumnIndex(unsigned int old_table_idx,
                         unsigned int old_col_idx) const {
    for (size_t i = 0; i < column_mappings.size(); i++) {
      if (column_mappings[i].old_table_idx == old_table_idx &&
          column_mappings[i].old_col_idx == old_col_idx) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
};

class IRQuerySplitter {
public:
  IRQuerySplitter(DBAdapter *adapter, const ParamConfig &config);
  ~IRQuerySplitter() = default;

  // Main entry: Execute query with optional splitting
  QueryResult ExecuteWithSplit(const std::string &sql);

  // Statistics
  int GetIterationCount() const { return iteration_count_; }
  std::vector<double> GetIterationTimes() const { return iteration_times_; }

private:
  // === IR-based Iterative Split-Execute Loop (all strategies) ===
  QueryResult
  ExecuteSplitLoop(std::unique_ptr<ir_sql_converter::SimplestStmt> whole_ir);

  // Single iteration: extract → execute → update remaining IR
  bool ExecuteOneIteration(
      std::unique_ptr<ir_sql_converter::SimplestStmt> &remaining_ir);

  // Execute a sub-IR and create temp table
  TempTableInfo
  ExecuteSubIR(std::unique_ptr<ir_sql_converter::SimplestStmt> sub_ir,
               const std::set<unsigned int> &executed_table_indices);

  // === Shared Index Update Functions ===
  // Update table/column indices in remaining IR after creating temp table
  // Called after strategy-specific UpdateRemainingIR
  void
  UpdateRemainingIRIndices(ir_sql_converter::SimplestStmt *remaining_ir,
                           const TempTableInfo &temp_table,
                           const std::set<unsigned int> &old_table_indices);

  // Helper: Update a single SimplestAttr if it references an executed table
  std::unique_ptr<ir_sql_converter::SimplestAttr>
  UpdateAttrIndices(const ir_sql_converter::SimplestAttr *attr,
                    const TempTableInfo &temp_table,
                    const std::set<unsigned int> &old_table_indices);

  // Helper: Recursively update all attributes in an IR node
  void UpdateNodeIndices(ir_sql_converter::SimplestStmt *node,
                         const TempTableInfo &temp_table,
                         const std::set<unsigned int> &old_table_indices);

  // Helper: Recursively update attributes in an expression tree
  void UpdateExprIndices(ir_sql_converter::SimplestExpr *expr,
                         const TempTableInfo &temp_table,
                         const std::set<unsigned int> &old_table_indices);

  // === Helper Functions ===
  std::string GenerateTempTableName();

  // Check if remaining IR is trivial (just a temp table reference)
  std::string GetTrivialTempTable(ir_sql_converter::SimplestStmt *ir) const;

  DBAdapter *adapter_;
#ifdef HAVE_DUCKDB
  // Owned helper DuckDB adapter; non-null only when engine != DUCKDB and
  // strategy == NODE_BASED. Null when engine == DUCKDB (adapter_ is the DuckDB
  // adapter in that case).
  std::unique_ptr<DuckDBAdapter> owned_duckdb_adapter_;
  // Non-owning pointer to the DuckDB adapter used for planning.
  // Valid whenever strategy == NODE_BASED.
  DuckDBAdapter *duckdb_adapter_ = nullptr;
#endif
  ParamConfig config_;
  std::unique_ptr<SplitAlgorithm> splitter_;

  // Iteration tracking
  int iteration_count_ = 0;
  std::vector<TempTableInfo> temp_tables_;
  std::vector<double> iteration_times_;

  // Sub-plan combiner: collected (temp_name, sql) pairs
  std::vector<std::pair<std::string, std::string>> sub_plan_sqls_;

  // Helper: Compute column alias using SQL generator's convention
  std::string ComputeColumnAlias(unsigned int table_idx,
                                 const std::string &col_name) const;

  // Helper: Build a combined CTE SQL from collected sub-plans + final SQL
  std::string BuildCombinedSQL(
      const std::vector<std::pair<std::string, std::string>> &sub_plans,
      const std::string &final_sql) const;
};

} // namespace middleware
