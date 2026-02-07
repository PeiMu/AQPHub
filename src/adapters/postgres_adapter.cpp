/*
 * PostgreSQL adapter for binding IR to the PostgreSQL engine
 * */

#include "adapters/postgres_adapter.h"

namespace middleware {

PostgreSQLAdapter::PostgreSQLAdapter(const std::string &connection_string)
    : conn(nullptr), parse_tree() {
  // Connect to PostgreSQL
  conn = PQconnectdb(connection_string.c_str());

  if (CONNECTION_OK != PQstatus(conn)) {
    std::string error_msg = PQerrorMessage(conn);
    PQfinish(conn);
    conn = nullptr;
    throw std::runtime_error("PostgreSQL connection failed: " + error_msg);
  }
#ifndef NDEBUG
  std::cout << "[PostgreSQL] Connected to database: " << PQdb(conn) << "@"
            << PQhost(conn) << std::endl;
#endif
}

PostgreSQLAdapter::~PostgreSQLAdapter() { CleanUp(); }

void PostgreSQLAdapter::ParseSQL(const std::string &sql) {
  CheckConnection();

  // Parse SQL using libpg_query
  PgQueryParseResult result = pg_query_parse(sql.c_str());

  if (result.error) {
    std::string error_msg =
        "Parse error: " + std::string(result.error->message);
    pg_query_free_parse_result(result);
    throw std::runtime_error(error_msg);
  }

  // Parse JSON
  parse_tree = json::parse(result.parse_tree);
  pg_query_free_parse_result(result);
}

std::unique_ptr<ir_sql_converter::SimplestStmt>
PostgreSQLAdapter::ConvertPlanToIR() {
  if (parse_tree.empty()) {
    throw std::runtime_error("No parse tree available. Call ParseSQL first.");
  }

  // Use schema-aware conversion if global schema parser is initialized,
  // otherwise fall back to basic conversion (column indices will be 0)
  std::unique_ptr<ir_sql_converter::SimplestStmt> stmt =
      ir_sql_converter::ConvertParseTreeToIRWithSchema(parse_tree,
                                                       subquery_index);
  return std::move(stmt);
}

QueryResult PostgreSQLAdapter::ExecuteSQL(const std::string &sql) {
  CheckConnection();

  QueryResult result;

  // Execute query
  PGresult *pg_result = PQexec(conn, sql.c_str());

  // Check for errors
  ExecStatusType status = PQresultStatus(pg_result);
  if (PGRES_TUPLES_OK != status && PGRES_COMMAND_OK != status) {
    std::string error_msg =
        "Query execution failed: " + std::string(PQerrorMessage(conn));
    PQclear(pg_result);
    throw std::runtime_error(error_msg);
  }

  // Get column information
  result.num_columns = PQnfields(pg_result);
  for (int i = 0; i < result.num_columns; i++) {
    result.column_names.emplace_back(PQfname(pg_result, i));
  }

  // Get row data
  result.num_rows = PQntuples(pg_result);
  result.rows.reserve(result.num_rows);

  for (int row = 0; row < result.num_rows; row++) {
    std::vector<std::string> row_data;
    row_data.reserve(result.num_columns);

    for (int col = 0; col < result.num_columns; col++) {
      // Check if value is NULL
      if (PQgetisnull(pg_result, row, col)) {
        row_data.emplace_back("NULL");
      } else {
        row_data.emplace_back(PQgetvalue(pg_result, row, col));
      }
    }

    result.rows.push_back(std::move(row_data));
  }

  PQclear(pg_result);
  return result;
}

void PostgreSQLAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name) {
  CheckConnection();

  // Create temp table with query results using CREATE TEMP TABLE AS
  std::string create_sql = "CREATE TEMP TABLE " + temp_table_name + " AS (" +
                           sql.substr(0, sql.size() - 1) + ");";

#ifndef NDEBUG
  std::cout << "[PostgreSQL] Creating temp table: " << temp_table_name
            << std::endl;
#endif

  PGresult *pg_result = PQexec(conn, create_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_COMMAND_OK) {
    std::string error_msg =
        "Failed to create temp table: " + std::string(PQerrorMessage(conn));
    PQclear(pg_result);
    throw std::runtime_error(error_msg);
  }

  PQclear(pg_result);

#ifndef NDEBUG
  std::cout << "[PostgreSQL] Created temp table: " << temp_table_name
            << std::endl;
#endif
}

void PostgreSQLAdapter::CreateTempTable(const std::string &table_name,
                                        const QueryResult &result) {}

void PostgreSQLAdapter::DropTempTable(const std::string &table_name) {
  CheckConnection();

  std::string drop_sql = "DROP TABLE IF EXISTS " + table_name;
  PGresult *pg_result = PQexec(conn, drop_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_COMMAND_OK) {
    std::string error_msg =
        "Failed to drop temp table: " + std::string(PQerrorMessage(conn));
    PQclear(pg_result);
    throw std::runtime_error(error_msg);
  }

  PQclear(pg_result);

#ifndef NDEBUG
  std::cout << "[PostgreSQL] Dropped temp table: " << table_name << std::endl;
#endif
}

bool PostgreSQLAdapter::TempTableExists(const std::string &table_name) {
  CheckConnection();

  std::string check_sql = "SELECT EXISTS ("
                          "  SELECT 1 FROM pg_tables "
                          "  WHERE tablename = '" +
                          table_name +
                          "' "
                          "  AND schemaname = 'pg_temp_" +
                          std::to_string(PQbackendPID(conn)) +
                          "'"
                          ")";

  PGresult *pg_result = PQexec(conn, check_sql.c_str());

  bool exists = false;
  if (PQresultStatus(pg_result) == PGRES_TUPLES_OK &&
      PQntuples(pg_result) > 0) {
    exists = (PQgetvalue(pg_result, 0, 0)[0] == 't');
  }

  PQclear(pg_result);
  return exists;
}

uint64_t
PostgreSQLAdapter::GetTempTableCardinality(const std::string &temp_table_name) {
  CheckConnection();

  std::string count_sql = "SELECT COUNT(*) FROM " + temp_table_name;
  PGresult *pg_result = PQexec(conn, count_sql.c_str());

  uint64_t cardinality = 0;
  if (PQresultStatus(pg_result) == PGRES_TUPLES_OK &&
      PQntuples(pg_result) > 0) {
    cardinality = std::stoull(PQgetvalue(pg_result, 0, 0));
  }

  PQclear(pg_result);
  return cardinality;
}

std::pair<double, double>
PostgreSQLAdapter::GetEstimatedCost(const std::string &sql) {
  CheckConnection();

  // Use EXPLAIN (FORMAT JSON) to get structured output with cost and rows
  std::string explain_sql = "EXPLAIN (FORMAT JSON) " + sql;
  PGresult *pg_result = PQexec(conn, explain_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_TUPLES_OK) {
    std::cerr << "[PostgreSQL] EXPLAIN failed: " << PQerrorMessage(conn)
              << std::endl;
    PQclear(pg_result);
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }

  double estimated_cost = std::numeric_limits<double>::max();
  double estimated_rows = std::numeric_limits<double>::max();

  if (PQntuples(pg_result) > 0 && PQnfields(pg_result) > 0) {
    std::string json_str = PQgetvalue(pg_result, 0, 0);

    try {
      json explain_json = json::parse(json_str);

      // PostgreSQL EXPLAIN JSON format:
      // [{"Plan": {"Total Cost": ..., "Plan Rows": ..., ...}}]
      if (explain_json.is_array() && !explain_json.empty()) {
        auto &plan = explain_json[0]["Plan"];
        if (plan.contains("Total Cost")) {
          estimated_cost = plan["Total Cost"].get<double>();
        }
        if (plan.contains("Plan Rows")) {
          estimated_rows = plan["Plan Rows"].get<double>();
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "[PostgreSQL] Failed to parse EXPLAIN JSON: " << e.what()
                << std::endl;
    }
  }

  PQclear(pg_result);
  return {estimated_cost, estimated_rows};
}

void PostgreSQLAdapter::CleanUp() {
  parse_tree.clear();

  // Close connection
  if (conn) {
    PQfinish(conn);
    conn = nullptr;
#ifndef NDEBUG
    std::cout << "[PostgreSQL] Connection closed" << std::endl;
#endif
  }
}

void PostgreSQLAdapter::CheckConnection() {
  if (!conn || CONNECTION_OK != PQstatus(conn)) {
    throw std::runtime_error("PostgreSQL connection is not valid");
  }
}
} // namespace middleware