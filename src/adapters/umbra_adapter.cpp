#include "adapters/umbra_adapter.h"
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>

namespace middleware {

UmbraAdapter::UmbraAdapter(const std::string &connection_string)
    : PostgreSQLAdapter(connection_string), conn_string_(connection_string) {}

void UmbraAdapter::SetTempTableCardinality(const std::string &temp_table_name,
                                           uint64_t estimated_rows) {
  // Try the same UPDATE pg_class approach as PostgreSQL.
  // Umbra may not expose pg_class, so treat failure as non-fatal.
  std::string update_sql =
      "UPDATE pg_class SET reltuples = " + std::to_string(estimated_rows) +
      " WHERE relname = '" + temp_table_name + "'";
  PGresult *pg_result = PQexec(GetConnection(), update_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_COMMAND_OK) {
#ifndef NDEBUG
    std::cerr << "[Umbra] Warning: Failed to set reltuples (non-fatal): "
              << PQerrorMessage(GetConnection()) << std::endl;
#endif
  }

  PQclear(pg_result);

#ifndef NDEBUG
  std::cout << "[Umbra] SetTempTableCardinality: " << temp_table_name << " = "
            << estimated_rows << std::endl;
#endif
}

void UmbraAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card) {
  CheckConnection();

  std::string create_sql = "CREATE TEMP TABLE " + temp_table_name + " AS (" +
                           sql.substr(0, sql.size() - 1) + ");";
#ifndef NDEBUG
  std::cout << "[Umbra] Creating temp table: " << temp_table_name << std::endl;
#endif

  PGresult *pg_result = PQexec(GetConnection(), create_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_COMMAND_OK) {
    std::string error_msg = "Failed to create temp table: " +
                            std::string(PQerrorMessage(GetConnection()));
    PQclear(pg_result);
    throw std::runtime_error(error_msg);
  }
  PQclear(pg_result);

  // Cardinality is looked up lazily in GetTempTableCardinality (via
  // pg_class or COUNT(*)).  ResetQueryState uses reconnect to drop
  // session temp tables, so temp_table_card_ doesn't need to be
  // populated here.

#ifndef NDEBUG
  std::cout << "[Umbra] Created temp table: " << temp_table_name << std::endl;
#endif
}

uint64_t
UmbraAdapter::GetTempTableCardinality(const std::string &temp_table_name) {
  // Check cache first (populated by ExecuteSQLandCreateTempTable)
  auto it = temp_table_card_.find(temp_table_name);
  if (it != temp_table_card_.end()) {
    return it->second;
  }

  // Umbra auto-collects stats — pg_class.reltuples is cheaper than COUNT(*)
  CheckConnection();
  std::string sql = "SELECT reltuples::bigint FROM pg_class WHERE relname = '" +
                    temp_table_name + "'";
  PGresult *r = PQexec(GetConnection(), sql.c_str());
  if (PQresultStatus(r) == PGRES_TUPLES_OK && PQntuples(r) > 0) {
    int64_t reltuples = std::stoll(PQgetvalue(r, 0, 0));
    PQclear(r);
    if (reltuples >= 0) {
      temp_table_card_[temp_table_name] = static_cast<uint64_t>(reltuples);
      return static_cast<uint64_t>(reltuples);
    }
  }
  if (r)
    PQclear(r);

  // Fallback to COUNT(*) via parent, then cache
  uint64_t count = PostgreSQLAdapter::GetTempTableCardinality(temp_table_name);
  temp_table_card_[temp_table_name] = count;
  return count;
}

std::vector<std::pair<double, double>>
UmbraAdapter::BatchGetEstimatedCosts(const std::vector<std::string> &sqls) {
  std::vector<std::pair<double, double>> results;
  results.reserve(sqls.size());
  for (const auto &sql : sqls) {
    results.push_back(GetEstimatedCost(sql));
  }
  return results;
}

std::pair<double, double>
UmbraAdapter::GetEstimatedCost(const std::string &sql) {
  CheckConnection();

  std::string explain_sql = "EXPLAIN (FORMAT JSON) " + sql;
  PGresult *pg_result = PQexec(GetConnection(), explain_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_TUPLES_OK) {
    std::cerr << "[Umbra] EXPLAIN failed: " << PQerrorMessage(GetConnection())
              << std::endl;
    PQclear(pg_result);
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }

  double estimated_rows = std::numeric_limits<double>::max();

  if (PQntuples(pg_result) > 0 && PQnfields(pg_result) > 0) {
    std::string json_str = PQgetvalue(pg_result, 0, 0);

    try {
      auto explain_json = nlohmann::json::parse(json_str);

      // Umbra EXPLAIN JSON: {"plan":{"operator":"...", "cardinality":N, ...}}
      if (explain_json.contains("plan") &&
          explain_json["plan"].contains("cardinality")) {
        estimated_rows = explain_json["plan"]["cardinality"].get<double>();
      }
    } catch (const std::exception &e) {
      std::cerr << "[Umbra] Failed to parse EXPLAIN JSON: " << e.what()
                << std::endl;
    }
  }

  PQclear(pg_result);
  // Umbra has no "Total Cost" — use cardinality as cost proxy
  return {estimated_rows, estimated_rows};
}

std::string UmbraAdapter::ExplainAnalyze(const std::string &sql) {
  // Umbra accepts plain "EXPLAIN ANALYZE" but may reject Postgres-only options
  // (VERBOSE/BUFFERS). Reuse the inherited ExecuteSQL and join the plan rows.
  try {
    QueryResult r = ExecuteSQL("EXPLAIN ANALYZE " + StripSqlTerminator(sql));
    std::string plan_text;
    for (const auto &row : r.rows) {
      if (!row.empty())
        plan_text += row.front();
      plan_text += "\n";
    }
    return plan_text;
  } catch (const std::exception &e) {
    return std::string("EXPLAIN ANALYZE failed: ") + e.what();
  }
}

void UmbraAdapter::DropTempTable(const std::string &table_name) {
  CheckConnection();
  std::string drop_sql = "DROP TABLE IF EXISTS " + table_name;
  PGresult *pg_result = PQexec(GetConnection(), drop_sql.c_str());
  if (PQresultStatus(pg_result) != PGRES_COMMAND_OK) {
#ifndef NDEBUG
    std::cerr << "[Umbra] Warning: DROP TABLE " << table_name
              << " failed (non-fatal): " << PQerrorMessage(GetConnection())
              << std::endl;
#endif
  }
  PQclear(pg_result);
}

void UmbraAdapter::ResetQueryState() {
  // Umbra may hold an internal schema lock after query execution (background
  // compaction on temp tables), causing both DROP and CREATE to fail with
  // "could not lock the schema".  Reconnecting releases all session-scoped
  // temp tables and clears the lock — no need to drop them individually.
  temp_table_card_.clear();
  parse_tree.clear();
  subquery_index = 0;

  if (conn) {
    PQfinish(conn);
    conn = nullptr;
  }
  conn = PQconnectdb(conn_string_.c_str());
  if (CONNECTION_OK != PQstatus(conn)) {
    std::string err = PQerrorMessage(conn);
    PQfinish(conn);
    conn = nullptr;
    throw std::runtime_error("Umbra reconnect failed: " + err);
  }
  PGresult *r = PQexec(conn, "SET synchronous_commit = off");
  if (r)
    PQclear(r);
}

} // namespace middleware
