/*
 * MariaDB adapter for binding IR to the MariaDB engine
 */

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

#include <mysql/mysql.h>

#include "adapters/db_adapter.h"
#include "pg_query.h"

#ifdef HAVE_POSTGRES
#include "adapters/postgres_adapter.h"
#endif

#define SUB_SQL_TIMEOUT 1000

using json = nlohmann::json;

namespace middleware {
class MariaDBAdapter : public DBAdapter {
public:
  // connection_string: MariaDB connection ("host=... dbname=... user=...")
  // estimator_connection: optional PostgreSQL connection for cost estimation;
  //                       empty string = use MariaDB's own EXPLAIN
  explicit MariaDBAdapter(const std::string &connection_string,
                          const std::string &estimator_connection = "");
  ~MariaDBAdapter() override;

  // Parse SQL and return logical plan
  void ParseSQL(const std::string &sql) override;

  json GetParseTree() { return parse_tree_; }

  // Convert logical plan to IR
  std::unique_ptr<ir_sql_converter::SimplestStmt> ConvertPlanToIR() override;

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

  // Batch EXPLAIN: send multiple EXPLAIN queries via CLIENT_MULTI_STATEMENTS
  std::vector<std::pair<double, double>>
  BatchGetEstimatedCosts(const std::vector<std::string> &sqls) override;

  std::string GetEngineName() const override { return "MariaDB"; }

  void CleanUp() override;

  // Get connection handle
  MYSQL *GetConnection() { return conn_; }

  void CheckConnection();

private:
  MYSQL *conn_;
  json parse_tree_;

#ifdef HAVE_POSTGRES
  // Optional PostgreSQL adapter used as cost estimator instead of MariaDB
  // EXPLAIN
  std::unique_ptr<PostgreSQLAdapter> pg_estimator_;
#endif

  // Parse connection string "host=X port=Y dbname=Z user=U password=P"
  // into individual parameters for mysql_real_connect()
  struct ConnParams {
    std::string host = "localhost";
    unsigned int port = 3306;
    std::string user = "root";
    std::string password;
    std::string dbname;
  };
  static ConnParams ParseConnectionString(const std::string &conn_str);

  // Probe a SELECT query's result metadata to check for TEXT/BLOB columns.
  // Runs SELECT ... LIMIT 0 (no data transfer, metadata only).
  // Returns true if any output column is TEXT/BLOB (incompatible with MEMORY
  // engine).
  bool HasTextBlobColumns(const std::string &select_sql);

  // Returns true if sql references a middleware temp table (temp1, temp2, ...).
  // Used to decide whether to delegate cost estimation to the PG estimator.
  static bool HasTempTable(const std::string &sql);

  // Run EXPLAIN FORMAT=JSON for each SQL via MariaDB CLIENT_MULTI_STATEMENTS.
  std::vector<std::pair<double, double>>
  BatchExplainWithMariaDB(const std::vector<std::string> &sqls);
};
} // namespace middleware
