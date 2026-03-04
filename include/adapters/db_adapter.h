/*
 * DBAdapter as an interface
 * */

#pragma once

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

class DBAdapter {
public:
  DBAdapter() = default;

  virtual ~DBAdapter() = default;

  // Parse SQL and return logical plan
  virtual void ParseSQL(const std::string &sql) = 0;

  // Convert logical plan to IR
  virtual std::unique_ptr<ir_sql_converter::SimplestStmt> ConvertPlanToIR() = 0;

  // Convert IR to SQL
  std::string GenerateSQL(ir_sql_converter::SimplestStmt &simplest_stmt,
                          int query_id, bool save_file = false,
                          const std::string &sql_path = "") {
    auto sql = ir_sql_converter::ConvertIRToSQL(simplest_stmt, query_id,
                                                save_file, sql_path);
    return sql;
  }

  // Execute SQL query
  virtual QueryResult ExecuteSQL(const std::string &sql) = 0;
  virtual void ExecuteSQLandCreateTempTable(const std::string &sql,
                                            const std::string &temp_table_name,
                                            bool update_temp_card,
                                            bool enable_timing) = 0;

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

  virtual void CleanUp() = 0;

  unsigned int subquery_index = 0;

  // std::string intermediate_table_name, int64_t created_table_size
  std::unordered_map<std::string, int64_t> temp_table_card_;
};
} // namespace middleware