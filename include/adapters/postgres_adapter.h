/*
 * PostgreSQL adapter for binding IR to the PostgreSQL engine
 * */

#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include <libpq-fe.h>

#include "adapters/db_adapter.h"
#include "pg_query.h"

using json = nlohmann::json;

namespace middleware {
class PostgreSQLAdapter : public EngineAdapter {
public:
  explicit PostgreSQLAdapter(const std::string &connection_string);
  ~PostgreSQLAdapter() override;

  // Parse SQL and return logical plan
  void ParseSQL(const std::string &sql) override;

  json GetParseTree() { return parse_tree; }

  // Convert logical plan to IR
  std::unique_ptr<ir_sql_converter::AQPStmt> ConvertPlanToIR() override;

  // Execute SQL query
  QueryResult ExecuteSQL(const std::string &sql) override;
  void ExecuteSQLandCreateTempTable(const std::string &sql,
                                    const std::string &temp_table_name,
                                    bool update_temp_card,
                                    bool enable_timing) override;

  // Temp table management
  void CreateTempTable(const std::string &table_name,
                       const QueryResult &result) override;

  void DropTempTable(const std::string &table_name) override;

  bool TempTableExists(const std::string &table_name) override;

  uint64_t GetTempTableCardinality(const std::string &temp_table_name) override;

  void SetTempTableCardinality(const std::string &temp_table_name,
                               uint64_t estimated_rows) override;

  // Get estimated cost and rows for a query using EXPLAIN
  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;

  // Batch EXPLAIN: send multiple EXPLAIN queries in one PQsendQuery round-trip
  std::vector<std::pair<double, double>>
  BatchGetEstimatedCosts(const std::vector<std::string> &sqls) override;

  std::string GetEngineName() const override { return "PostgreSQL"; }

  void CleanUp() override;

  // Get connection handle
  PGconn *GetConnection() { return conn; }

  void CheckConnection();
  // JIT profiling: run EXPLAIN ANALYZE with JIT forced on                                    
    struct JITProfileResult {                                                                   
      double planning_time_ms = 0;                                                              
      double execution_time_ms = 0;                                                             
      int    jit_functions = 0;                                                                 
      double jit_generation_ms = 0;                                                             
      double jit_inlining_ms = 0;                                                               
      double jit_optimization_ms = 0;                                                          
      double jit_emission_ms = 0;                                                               
      double jit_total_ms = 0;                                                                  
      double estimated_cost = 0;                                                               
      double estimated_rows = 0;                                                                
    };                                                                                          
    JITProfileResult GetJITProfile(const std::string &sql); 
private:
  PGconn *conn;
  json parse_tree;
};
} // namespace middleware
