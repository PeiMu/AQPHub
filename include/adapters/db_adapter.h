/*
 * EngineAdapter as an interface
 * */

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interface.h"
#include "util/util.h"

namespace middleware {
struct QueryResult {
  std::vector<std::string> column_names;
  std::vector<std::vector<std::string>> rows;
  int num_rows;
  int num_columns;

  QueryResult() : num_rows(0), num_columns(0) {}
};

// DuckDB's optimizer adds prefix-range predicates for LIKE patterns
// (e.g. LIKE 'USA:%' → col >= 'USA:' AND col < 'USA;').  These rely on
// C-locale byte ordering and produce wrong results on engines whose
// default collation is not C (PostgreSQL en_GB.UTF-8, MariaDB utf8mb4,
// etc.).  Strip them before generating SQL for external engines; the
// LIKE clause already provides the correct filter.
//
// Pattern detected per qual_vec entry:
//   LogicalAnd(LogicalAnd(col >= prefix, col < prefix'), IS NOT NULL(col))
// where prefix' is prefix with the last byte incremented by 1.
inline void StripCollationDependentRangeQuals(ir_sql_converter::AQPStmt &ir) {
  namespace irc = ir_sql_converter;

  auto is_prefix_range = [](const irc::AQPExpr *expr) -> bool {
    if (!expr || expr->GetNodeType() != irc::LogicalExprNode)
      return false;
    auto *outer = static_cast<const irc::SimplestLogicalExpr *>(expr);
    if (outer->GetLogicalOp() != irc::LogicalAnd)
      return false;
    // right must be IS NOT NULL
    if (!outer->right_expr ||
        outer->right_expr->GetNodeType() != irc::IsNullExprNode)
      return false;
    if (outer->right_expr->GetSimplestExprType() != irc::NonNullType)
      return false;
    // left must be AND(col >= prefix, col < prefix')
    if (!outer->left_expr ||
        outer->left_expr->GetNodeType() != irc::LogicalExprNode)
      return false;
    auto *inner = static_cast<const irc::SimplestLogicalExpr *>(
        outer->left_expr.get());
    if (inner->GetLogicalOp() != irc::LogicalAnd)
      return false;
    if (!inner->left_expr || !inner->right_expr)
      return false;
    // left child: col >= prefix (GreaterEqual, VarConstComparison)
    if (inner->left_expr->GetNodeType() != irc::VarConstComparisonNode ||
        inner->left_expr->GetSimplestExprType() != irc::GreaterEqual)
      return false;
    // right child: col < prefix' (LessThan, VarConstComparison)
    if (inner->right_expr->GetNodeType() != irc::VarConstComparisonNode ||
        inner->right_expr->GetSimplestExprType() != irc::LessThan)
      return false;
    auto *ge = static_cast<const irc::SimplestVarConstComparison *>(
        inner->left_expr.get());
    auto *lt = static_cast<const irc::SimplestVarConstComparison *>(
        inner->right_expr.get());
    if (!ge->const_var || !lt->const_var)
      return false;
    if (ge->const_var->GetType() != irc::StringVar ||
        lt->const_var->GetType() != irc::StringVar)
      return false;
    std::string lo = ge->const_var->GetStringValue();
    std::string hi = lt->const_var->GetStringValue();
    if (lo.empty() || lo.size() != hi.size())
      return false;
    // hi must be lo with the last byte incremented by 1
    if (hi.substr(0, hi.size() - 1) != lo.substr(0, lo.size() - 1))
      return false;
    return static_cast<unsigned char>(hi.back()) ==
           static_cast<unsigned char>(lo.back()) + 1;
  };

  // Remove matching quals from this node's qual_vec.
  auto &qv = ir.qual_vec;
  qv.erase(std::remove_if(qv.begin(), qv.end(),
               [&](const std::unique_ptr<irc::AQPExpr> &q) {
                 return is_prefix_range(q.get());
               }),
           qv.end());

  // Recurse into children.
  for (auto &child : ir.children)
    if (child)
      StripCollationDependentRangeQuals(*child);
}

// Strip trailing ';' and whitespace so an "EXPLAIN ..." prefix produces a
// single statement. Used by ExplainAnalyze overrides across adapters.
inline std::string StripSqlTerminator(const std::string &sql) {
  size_t end = sql.size();
  while (end > 0 && (sql[end - 1] == ';' || sql[end - 1] == ' ' ||
                     sql[end - 1] == '\n' || sql[end - 1] == '\r' ||
                     sql[end - 1] == '\t')) {
    --end;
  }
  return sql.substr(0, end);
}

class EngineAdapter {
public:
  EngineAdapter() = default;

  virtual ~EngineAdapter() = default;

  // Parse SQL and return logical plan
  virtual void ParseSQL(const std::string &sql) = 0;

  // Convert logical plan to IR
  virtual std::unique_ptr<ir_sql_converter::AQPStmt> ConvertPlanToIR() = 0;

  // Convert IR to SQL.
  // Strips DuckDB's collation-dependent prefix-range quals before
  // generating SQL so external engines with non-C collation get
  // correct results.  Harmless for DuckDB (re-optimization re-adds).
  std::string GenerateSQL(ir_sql_converter::AQPStmt &simplest_stmt,
                          int query_id, bool save_file = false,
                          const std::string &sql_path = "") {
    StripCollationDependentRangeQuals(simplest_stmt);
    auto sql = ir_sql_converter::ConvertIRToSQL(simplest_stmt, query_id,
                                                save_file, sql_path);
    return sql;
  }

  // Apply an engine configuration statement (e.g. "SET ...") outside the
  // timed query path — must not emit timing columns. Default: no-op for
  // engines without such settings.
  virtual void ApplyEngineSetting(const std::string &sql) {}

  // Execute SQL query
  virtual QueryResult ExecuteSQL(const std::string &sql) = 0;
  virtual void ExecuteSQLandCreateTempTable(const std::string &sql,
                                            const std::string &temp_table_name,
                                            bool update_temp_card) = 0;

  // Execute IR directly (bypassing SQL generation) and create temp table.
  // Default: GenerateSQL + ExecuteSQLandCreateTempTable.
  virtual void ExecuteIRandCreateTempTable(
      ir_sql_converter::AQPStmt &ir,
      const std::string &temp_table_name,
      bool update_temp_card) {
    std::string sql = GenerateSQL(ir, subquery_index++);
    ExecuteSQLandCreateTempTable(sql, temp_table_name, update_temp_card);
  }

  // Execute IR directly and return results (for final query).
  // Default: GenerateSQL + ExecuteSQL.
  virtual QueryResult ExecuteIRQuery(ir_sql_converter::AQPStmt &ir) {
    std::string sql = GenerateSQL(ir, subquery_index++);
    return ExecuteSQL(sql);
  }

  // Temp table management
  virtual void CreateTempTable(const std::string &table_name,
                               const QueryResult &result) = 0;

  virtual void DropTempTable(const std::string &table_name) = 0;

  virtual bool TempTableExists(const std::string &table_name) = 0;

  // Get cardinality of temp table after execution
  virtual uint64_t
  GetTempTableCardinality(const std::string &temp_table_name) = 0;

  // Override the engine's internal cardinality for a temp table
  // Used for A/B testing: sets the engine's stats to an estimated value
  // so that subsequent EXPLAIN queries use the overridden cardinality
  virtual void SetTempTableCardinality(const std::string &temp_table_name,
                                       uint64_t cardinality) = 0;

  // Get estimated cost and rows for a query using EXPLAIN
  // Returns {estimated_cost, estimated_rows}
  virtual std::pair<double, double>
  GetEstimatedCost(const std::string &sql) = 0;

  // Return the EXPLAIN ANALYZE plan of a sub-SQL as formatted text (used by
  // --explain mode to display per-sub-query plans). Default returns "" for
  // engines that don't implement plan display; overridden per adapter.
  virtual std::string ExplainAnalyze(const std::string &sql) { return ""; }

  // Batch version: evaluate multiple EXPLAIN queries in one round-trip
  // Default implementation calls GetEstimatedCost sequentially (fine for
  // in-process engines like DuckDB; overridden for network-based engines)
  virtual std::vector<std::pair<double, double>>
  BatchGetEstimatedCosts(const std::vector<std::string> &sqls) {
    std::vector<std::pair<double, double>> results;
    results.reserve(sqls.size());
    for (const auto &sql : sqls) {
      results.push_back(GetEstimatedCost(sql));
    }
    return results;
  }

  virtual std::string GetEngineName() const = 0;

  // Pipeline detection: is this IR node a pipeline breaker?
  // Default: infer from IR node type (AggregateNode, HashNode, SortNode).
  // Engines with physical plan access (DuckDB) override with IsSink() check.
  virtual bool IsPipelineBreaker(
      const ir_sql_converter::AQPStmt &node) const {
    auto nt = node.GetNodeType();
    return nt == ir_sql_converter::SimplestNodeType::AggregateNode ||
           nt == ir_sql_converter::SimplestNodeType::HashNode ||
           nt == ir_sql_converter::SimplestNodeType::SortNode;
  }

  // Re-optimize the remaining IR through the engine's full optimizer.
  // Default: return the IR unchanged (no re-optimization).
  // DuckDB override: GenerateSQL → ParseSQL → Optimize → ConvertPlanToIR.
  virtual std::unique_ptr<ir_sql_converter::AQPStmt>
  ReOptimizeIR(std::unique_ptr<ir_sql_converter::AQPStmt> ir) {
    return ir;
  }

  virtual void CleanUp() = 0;

  virtual void ResetQueryState() {}

  unsigned int subquery_index = 0;

  // std::string intermediate_table_name, int64_t created_table_size
  std::unordered_map<std::string, int64_t> temp_table_card_;

  std::unordered_map<std::string, int64_t> GetTempTableCardSnapshot() const {
    return temp_table_card_;
  }

  bool enable_timing_ = false;
};
} // namespace middleware