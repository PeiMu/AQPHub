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
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/column/column_data_scan_states.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_constraint.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"

#define IN_MEM_TMP_TABLE false

namespace duckdb {
class DuckDB;
class Connection;
class ClientContext;
class LogicalOperator;
class Planner;
class Optimizer;
} // namespace duckdb

namespace middleware {

#if IN_MEM_TMP_TABLE
// Stored result for replacement-scan-based temp tables
struct StoredTempResult {
  duckdb::unique_ptr<duckdb::ColumnDataCollection> collection;
  std::vector<std::string> column_names;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;
};

// ReplacementScanData subclass: holds pointer to temp_collections_ map
struct TempCollectionScanData : public duckdb::ReplacementScanData {
  explicit TempCollectionScanData(
      std::unordered_map<std::string, StoredTempResult> *collections)
      : temp_collections(collections) {}
  std::unordered_map<std::string, StoredTempResult> *temp_collections;
};

// TableFunctionInfo subclass: holds pointer to temp_collections_ map
struct TempCollectionScanInfo : public duckdb::TableFunctionInfo {
  explicit TempCollectionScanInfo(
      std::unordered_map<std::string, StoredTempResult> *collections)
      : temp_collections(collections) {}
  std::unordered_map<std::string, StoredTempResult> *temp_collections;
};
#endif

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

#if IN_MEM_TMP_TABLE
private:
  // Register the temp collection table function and replacement scan
  void RegisterTempCollectionScan();

  // Table function callbacks (static)
  static duckdb::unique_ptr<duckdb::FunctionData>
  TempCollectionBind(duckdb::ClientContext &context,
                     duckdb::TableFunctionBindInput &input,
                     duckdb::vector<duckdb::LogicalType> &return_types,
                     duckdb::vector<duckdb::string> &names);

  static duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
  TempCollectionInitGlobal(duckdb::ClientContext &context,
                           duckdb::TableFunctionInitInput &input);

  static void TempCollectionScanFunc(duckdb::ClientContext &context,
                                     duckdb::TableFunctionInput &data,
                                     duckdb::DataChunk &output);

  static duckdb::unique_ptr<duckdb::NodeStatistics>
  TempCollectionCardinality(duckdb::ClientContext &context,
                            const duckdb::FunctionData *bind_data);

  // Replacement scan callback (static)
  static duckdb::unique_ptr<duckdb::TableRef>
  TempCollectionReplacementScan(duckdb::ClientContext &context,
                                const duckdb::string &table_name,
                                duckdb::ReplacementScanData *data);
  // Replacement scan: in-memory temp table storage
  std::unordered_map<std::string, StoredTempResult> temp_collections_;
#endif
};
} // namespace middleware
