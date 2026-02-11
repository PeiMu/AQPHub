/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "adapters/db_adapter.h"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_constraint.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/storage/data_table.hpp"

namespace duckdb {
class DuckDB;
class Connection;
class ClientContext;
class LogicalOperator;
class Planner;
class Optimizer;
} // namespace duckdb

namespace middleware {
class DuckDBAdapter : public DBAdapter {
public:
  explicit DuckDBAdapter(const std::string &db_path = ":memory :");
  ~DuckDBAdapter() override;

  // Parse SQL and return logical plan
  void ParseSQL(const std::string &sql) override;

  // Optimizer
  void FilterOptimize();
  void PostOptimizePlan();

  void *GetLogicalPlan();

  void PrintLogicalPlan() { plan->Print(); };

  // Convert logical plan to IR
  std::unique_ptr<ir_sql_converter::SimplestStmt> ConvertPlanToIR() override;

  // Execute SQL query
  QueryResult ExecuteSQL(const std::string &sql) override;
  void ExecuteSQLandCreateTempTable(const std::string &sql,
                                    const std::string &temp_table_name,
                                    bool update_temp_card) override;

  // Temp table management
  void CreateTempTable(const std::string &table_name,
                       const QueryResult &result) override;

  void DropTempTable(const std::string &table_name) override;

  bool TempTableExists(const std::string &table_name) override;

  uint64_t GetTempTableCardinality(const std::string &temp_table_name) override;

  void SetTempTableCardinality(const std::string &temp_table_name,
                               uint64_t cardinality) override;

  // Get estimated cost and rows for a query using EXPLAIN
  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;

  std::string GetEngineName() const override { return "DuckDB"; }

  void CleanUp() override;

  // Get context and binder for IR conversion
  duckdb::ClientContext *GetClientContext();

  struct pair_hash {
    template <class T1, class T2>
    uint64_t operator()(const std::pair<T1, T2> &p) const {
      auto h1 = std::hash<T1>{}(p.first);
      auto h2 = std::hash<T2>{}(p.second);

      // Mainly for demonstration purposes, i.e. works but is overly simple
      // In the real world, use sth. like boost.hash_combine
      return h1 ^ h2;
    }
  };

private:
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> conn;
  std::unique_ptr<duckdb::Planner> planner;
  duckdb::unique_ptr<duckdb::LogicalOperator> plan;

  std::unordered_map<std::pair<uint64_t, uint64_t>, std::string, pair_hash>
      table_column_mappings;

  // <temp%, subquery_dd_index>
  std::unordered_map<unsigned int, std::string> intermediate_table_map;
};
} // namespace middleware
