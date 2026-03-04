/*
 * MariaDB adapter for binding IR to the MariaDB engine
 */

#include "adapters/mariadb_adapter.h"

namespace middleware {

MariaDBAdapter::ConnParams
MariaDBAdapter::ParseConnectionString(const std::string &conn_str) {
  ConnParams params;
  std::istringstream stream(conn_str);
  std::string token;

  while (stream >> token) {
    auto eq_pos = token.find('=');
    if (eq_pos == std::string::npos)
      continue;

    std::string key = token.substr(0, eq_pos);
    std::string value = token.substr(eq_pos + 1);

    if (key == "host") {
      params.host = value;
    } else if (key == "port") {
      params.port = static_cast<unsigned int>(std::stoul(value));
    } else if (key == "user") {
      params.user = value;
    } else if (key == "password") {
      params.password = value;
    } else if (key == "dbname" || key == "database") {
      params.dbname = value;
    }
  }

  return params;
}

MariaDBAdapter::MariaDBAdapter(const std::string &connection_string,
                               const std::string &estimator_connection)
    : conn_(nullptr), parse_tree_() {
  auto params = ParseConnectionString(connection_string);

  conn_ = mysql_init(nullptr);
  if (!conn_) {
    throw std::runtime_error("MariaDB mysql_init() failed");
  }

  // Use Unix socket for localhost (faster than TCP loopback)
  const char *host = (params.host == "localhost" || params.host.empty())
                         ? nullptr
                         : params.host.c_str();

  if (!mysql_real_connect(
          conn_, host, params.user.c_str(),
          params.password.empty() ? nullptr : params.password.c_str(),
          params.dbname.empty() ? nullptr : params.dbname.c_str(),
          host ? params.port : 0, nullptr, CLIENT_MULTI_STATEMENTS)) {
    std::string error_msg = mysql_error(conn_);
    mysql_close(conn_);
    conn_ = nullptr;
    throw std::runtime_error("MariaDB connection failed: " + error_msg);
  }

#ifndef NDEBUG
  std::cout << "[MariaDB] Connected to database: " << params.dbname << "@"
            << params.host << std::endl;
#endif

  // Performance tuning for temp table workloads
  mysql_query(conn_, "SET SESSION innodb_flush_log_at_trx_commit = 0");
  mysql_query(conn_, "SET SESSION tmp_table_size = 268435456");
  mysql_query(conn_, "SET SESSION max_heap_table_size = 268435456");

#ifdef HAVE_POSTGRES
  // Initialize PostgreSQL estimator if a connection string was provided
  if (!estimator_connection.empty()) {
    pg_estimator_ = std::make_unique<PostgreSQLAdapter>(estimator_connection);
#ifndef NDEBUG
    std::cout << "[MariaDB] Using PostgreSQL estimator for cost estimation"
              << std::endl;
#endif
  }
#endif
}

MariaDBAdapter::~MariaDBAdapter() { CleanUp(); }

void MariaDBAdapter::ParseSQL(const std::string &sql) {
  CheckConnection();

  // Parse SQL using libpg_query (same as PostgreSQL — standard SQL parses
  // identically)
  PgQueryParseResult result = pg_query_parse(sql.c_str());

  if (result.error) {
    std::string error_msg =
        "Parse error: " + std::string(result.error->message);
    pg_query_free_parse_result(result);
    throw std::runtime_error(error_msg);
  }

  parse_tree_ = json::parse(result.parse_tree);
  pg_query_free_parse_result(result);
}

std::unique_ptr<ir_sql_converter::SimplestStmt>
MariaDBAdapter::ConvertPlanToIR() {
  if (parse_tree_.empty()) {
    throw std::runtime_error("No parse tree available. Call ParseSQL first.");
  }

  std::unique_ptr<ir_sql_converter::SimplestStmt> stmt =
      ir_sql_converter::ConvertParseTreeToIRWithSchema(parse_tree_,
                                                       subquery_index);
  return std::move(stmt);
}

QueryResult MariaDBAdapter::ExecuteSQL(const std::string &sql) {
  CheckConnection();

  QueryResult result;

  if (mysql_query(conn_, sql.c_str())) {
    throw std::runtime_error("Query execution failed: " +
                             std::string(mysql_error(conn_)));
  }

  MYSQL_RES *mysql_result = mysql_store_result(conn_);

  if (!mysql_result) {
    // Could be a non-SELECT statement (INSERT, CREATE, etc.)
    if (mysql_field_count(conn_) == 0) {
      // Statement that doesn't return rows
      return result;
    }
    throw std::runtime_error("Failed to store result: " +
                             std::string(mysql_error(conn_)));
  }

  // Get column information
  result.num_columns = static_cast<int>(mysql_num_fields(mysql_result));
  MYSQL_FIELD *fields = mysql_fetch_fields(mysql_result);
  for (int i = 0; i < result.num_columns; i++) {
    result.column_names.emplace_back(fields[i].name);
  }

  // Get row data
  result.num_rows = static_cast<int>(mysql_num_rows(mysql_result));
  result.rows.reserve(result.num_rows);

  MYSQL_ROW row;
  while ((row = mysql_fetch_row(mysql_result))) {
    unsigned long *lengths = mysql_fetch_lengths(mysql_result);
    std::vector<std::string> row_data;
    row_data.reserve(result.num_columns);

    for (int col = 0; col < result.num_columns; col++) {
      if (row[col] == nullptr) {
        row_data.emplace_back("NULL");
      } else {
        row_data.emplace_back(row[col], lengths[col]);
      }
    }

    result.rows.push_back(std::move(row_data));
  }

  mysql_free_result(mysql_result);
  return result;
}

void MariaDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card, bool enable_timing) {
  CheckConnection();

#ifndef NDEBUG
  std::cout << "[MariaDB] Creating temp table: " << temp_table_name
            << std::endl;
#endif

  std::string select_body = sql.substr(0, sql.size() - 1);

  // Probe output column types to decide engine (avoids try-fail-retry)
  bool use_memory = !HasTextBlobColumns(select_body);

#ifndef NDEBUG
  std::cout << "[MariaDB] Engine: "
            << (use_memory ? "MEMORY" : "InnoDB (TEXT/BLOB detected)")
            << std::endl;
#endif

  std::string create_sql =
      "SET STATEMENT max_statement_time=" + std::to_string(SUB_SQL_TIMEOUT) +
      " FOR CREATE TEMPORARY TABLE " + temp_table_name +
      (use_memory ? " ENGINE=MEMORY" : "") + " AS (" + select_body + ")";

  std::string combined_sql;
  if (update_temp_card) {
    combined_sql = create_sql + "; ANALYZE TABLE " + temp_table_name + ";";
  } else {
    combined_sql = create_sql + ";";
  }

  if (mysql_query(conn_, combined_sql.c_str())) {
    std::string error_msg =
        "Failed to create temp table: " + std::string(mysql_error(conn_));
    // Drain any partial results
    while (mysql_next_result(conn_) == 0) {
      MYSQL_RES *r = mysql_store_result(conn_);
      if (r)
        mysql_free_result(r);
    }
    throw std::runtime_error(error_msg);
  }

  // mysql_affected_rows() returns row count for CREATE TABLE ... AS SELECT
  uint64_t row_count = mysql_affected_rows(conn_);
  temp_table_card_[temp_table_name] = static_cast<int64_t>(row_count);

  // Drain remaining results (ANALYZE TABLE result if present)
  while (mysql_next_result(conn_) == 0) {
    MYSQL_RES *r = mysql_store_result(conn_);
    if (r)
      mysql_free_result(r);
  }

#ifndef NDEBUG
  std::cout << "[MariaDB] Created temp table: " << temp_table_name
            << " (rows=" << temp_table_card_[temp_table_name] << ")"
            << std::endl;
#endif
}

// void MariaDBAdapter::ExecuteSQLandCreateTempTable(
//     const std::string &sql, const std::string &temp_table_name,
//     bool update_temp_card, bool enable_timing) {
//   CheckConnection();
//
// #ifndef NDEBUG
//   std::cout << "[MariaDB] Creating temp table: " << temp_table_name
//             << std::endl;
// #endif
//
//   std::string create_sql = "CREATE TEMPORARY TABLE " + temp_table_name +
//                            " ENGINE=MEMORY AS (" +
//                            sql.substr(0, sql.size() - 1) + ")";
//
//   std::string combined_sql;
//   if (update_temp_card) {
//     combined_sql = create_sql + "; ANALYZE TABLE " + temp_table_name + ";";
//   } else {
//     combined_sql = create_sql + ";";
//   }
//
//   if (mysql_query(conn_, combined_sql.c_str())) {
//     unsigned int err = mysql_errno(conn_);
//     if (err == 1163) {
//       // MEMORY engine doesn't support BLOB/TEXT — retry without
//       ENGINE=MEMORY
// #ifndef NDEBUG
//       std::cerr << "[MariaDB] MEMORY engine failed (BLOB/TEXT), retrying with
//       "
//                    "default engine"
//                 << std::endl;
// #endif
//       // Drain any partial results from the failed multi-statement
//       while (mysql_next_result(conn_) == 0) {
//         MYSQL_RES *r = mysql_store_result(conn_);
//         if (r)
//           mysql_free_result(r);
//       }
//
//       create_sql = "CREATE TEMPORARY TABLE " + temp_table_name + " AS (" +
//                    sql.substr(0, sql.size() - 1) + ")";
//       if (update_temp_card) {
//         combined_sql =
//             create_sql + "; ANALYZE TABLE " + temp_table_name + ";";
//       } else {
//         combined_sql = create_sql + ";";
//       }
//
//       if (mysql_query(conn_, combined_sql.c_str())) {
//         std::string error_msg = "Failed to create temp table: " +
//                                 std::string(mysql_error(conn_));
//         // Drain results
//         while (mysql_next_result(conn_) == 0) {
//           MYSQL_RES *r = mysql_store_result(conn_);
//           if (r)
//             mysql_free_result(r);
//         }
//         throw std::runtime_error(error_msg);
//       }
//     } else {
//       std::string error_msg = "Failed to create temp table: " +
//                               std::string(mysql_error(conn_));
//       // Drain results
//       while (mysql_next_result(conn_) == 0) {
//         MYSQL_RES *r = mysql_store_result(conn_);
//         if (r)
//           mysql_free_result(r);
//       }
//       throw std::runtime_error(error_msg);
//     }
//   }
//
//   // mysql_affected_rows() returns row count for CREATE TABLE ... AS SELECT
//   uint64_t row_count = mysql_affected_rows(conn_);
//   temp_table_card_[temp_table_name] = static_cast<int64_t>(row_count);
//
//   // Drain remaining results (ANALYZE TABLE result if present)
//   while (mysql_next_result(conn_) == 0) {
//     MYSQL_RES *r = mysql_store_result(conn_);
//     if (r)
//       mysql_free_result(r);
//   }
//
// #ifndef NDEBUG
//   std::cout << "[MariaDB] Created temp table: " << temp_table_name
//             << " (rows=" << temp_table_card_[temp_table_name] << ")"
//             << std::endl;
// #endif
// }

void MariaDBAdapter::CreateTempTable(const std::string &table_name,
                                     const QueryResult &result) {}

void MariaDBAdapter::DropTempTable(const std::string &table_name) {
  CheckConnection();

  std::string drop_sql = "DROP TEMPORARY TABLE IF EXISTS " + table_name;
  if (mysql_query(conn_, drop_sql.c_str())) {
    throw std::runtime_error("Failed to drop temp table: " +
                             std::string(mysql_error(conn_)));
  }

#ifndef NDEBUG
  std::cout << "[MariaDB] Dropped temp table: " << table_name << std::endl;
#endif
}

bool MariaDBAdapter::TempTableExists(const std::string &table_name) {
  CheckConnection();

  // Try a lightweight query on the table — if it fails, the table doesn't exist
  std::string check_sql = "SELECT 1 FROM " + table_name + " LIMIT 0";
  if (mysql_query(conn_, check_sql.c_str())) {
    return false;
  }

  MYSQL_RES *r = mysql_store_result(conn_);
  if (r)
    mysql_free_result(r);
  return true;
}

uint64_t
MariaDBAdapter::GetTempTableCardinality(const std::string &temp_table_name) {
  // Check cache first (populated by ExecuteSQLandCreateTempTable via
  // mysql_affected_rows)
  auto it = temp_table_card_.find(temp_table_name);
  if (it != temp_table_card_.end()) {
    return it->second;
  }

  // Fallback to SELECT COUNT(*)
  CheckConnection();
  std::string count_sql = "SELECT COUNT(*) FROM " + temp_table_name;
  if (mysql_query(conn_, count_sql.c_str())) {
    return 0;
  }

  uint64_t cardinality = 0;
  MYSQL_RES *mysql_result = mysql_store_result(conn_);
  if (mysql_result) {
    MYSQL_ROW row = mysql_fetch_row(mysql_result);
    if (row && row[0]) {
      cardinality = std::stoull(row[0]);
    }
    mysql_free_result(mysql_result);
  }

  return cardinality;
}

void MariaDBAdapter::SetTempTableCardinality(const std::string &temp_table_name,
                                             uint64_t estimated_rows) {
  CheckConnection();

  // MariaDB has no pg_class equivalent for direct stats override.
  // ANALYZE TABLE refreshes stats from actual data — best-effort approach.
  std::string analyze_sql = "ANALYZE TABLE " + temp_table_name;
  if (mysql_query(conn_, analyze_sql.c_str())) {
#ifndef NDEBUG
    std::cerr << "[MariaDB] Warning: ANALYZE TABLE failed (non-fatal): "
              << mysql_error(conn_) << std::endl;
#endif
  }

  // Drain the ANALYZE TABLE result set
  MYSQL_RES *r = mysql_store_result(conn_);
  if (r)
    mysql_free_result(r);

#ifndef NDEBUG
  std::cout << "[MariaDB] SetTempTableCardinality: " << temp_table_name << " = "
            << estimated_rows << " (ANALYZE TABLE, best-effort)" << std::endl;
#endif
}

bool MariaDBAdapter::HasTempTable(const std::string &sql) {
  // Middleware temp tables are named temp1, temp2, etc.
  // Scan for the literal "temp" followed immediately by a digit.
  auto pos = sql.find("temp");
  while (pos != std::string::npos) {
    size_t after = pos + 4;
    if (after < sql.size() &&
        std::isdigit(static_cast<unsigned char>(sql[after]))) {
      return true;
    }
    pos = sql.find("temp", pos + 1);
  }
  return false;
}

std::vector<std::pair<double, double>>
MariaDBAdapter::BatchExplainWithMariaDB(const std::vector<std::string> &sqls) {
  CheckConnection();

  std::string combined;
  for (const auto &sql : sqls) {
    combined += "EXPLAIN FORMAT=JSON " + sql + ";";
  }

  if (mysql_query(conn_, combined.c_str())) {
    std::cerr << "[MariaDB] BatchExplainWithMariaDB: mysql_query failed: "
              << mysql_error(conn_) << std::endl;
    while (mysql_next_result(conn_) == 0) {
      MYSQL_RES *r = mysql_store_result(conn_);
      if (r)
        mysql_free_result(r);
    }
    return std::vector<std::pair<double, double>>(
        sqls.size(), {std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max()});
  }

  std::vector<std::pair<double, double>> results;
  results.reserve(sqls.size());

  for (size_t i = 0; i < sqls.size(); i++) {
    double estimated_cost = std::numeric_limits<double>::max();
    double estimated_rows = std::numeric_limits<double>::max();

    MYSQL_RES *mysql_result = mysql_store_result(conn_);
    if (mysql_result) {
      MYSQL_ROW row = mysql_fetch_row(mysql_result);
      if (row && row[0]) {
        try {
          json explain_json = json::parse(row[0]);
          if (explain_json.contains("query_block")) {
            auto &qb = explain_json["query_block"];
            if (qb.contains("cost")) {
              estimated_cost = qb["cost"].get<double>();
            }
            if (qb.contains("table") && qb["table"].contains("rows")) {
              estimated_rows = qb["table"]["rows"].get<double>();
            } else if (qb.contains("nested_loop")) {
              double total_rows = 0;
              for (auto &entry : qb["nested_loop"]) {
                if (entry.contains("table") &&
                    entry["table"].contains("rows")) {
                  double r = entry["table"]["rows"].get<double>();
                  if (total_rows == 0)
                    total_rows = r;
                  else
                    total_rows *= r;
                }
              }
              if (total_rows > 0)
                estimated_rows = total_rows;
            }
            if (estimated_rows == std::numeric_limits<double>::max() &&
                estimated_cost != std::numeric_limits<double>::max()) {
              estimated_rows = estimated_cost;
            }
          }
        } catch (const std::exception &e) {
#ifndef NDEBUG
          std::cerr << "[MariaDB] BatchExplainWithMariaDB: parse failed for "
                       "query "
                    << i << ": " << e.what() << std::endl;
#endif
        }
      }
      mysql_free_result(mysql_result);
    }

    results.push_back({estimated_cost, estimated_rows});

    if (i < sqls.size() - 1) {
      int status = mysql_next_result(conn_);
      if (status > 0) {
        for (size_t j = i + 1; j < sqls.size(); j++) {
          results.push_back({std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max()});
        }
        break;
      }
    }
  }

  while (mysql_next_result(conn_) == 0) {
    MYSQL_RES *r = mysql_store_result(conn_);
    if (r)
      mysql_free_result(r);
  }

  return results;
}

std::pair<double, double>
MariaDBAdapter::GetEstimatedCost(const std::string &sql) {
#ifdef HAVE_POSTGRES
  if (pg_estimator_ && !HasTempTable(sql)) {
    return pg_estimator_->GetEstimatedCost(sql);
  }
#endif
  CheckConnection();

  // Use EXPLAIN FORMAT=JSON to get structured output with cost and rows
  std::string explain_sql = "EXPLAIN FORMAT=JSON " + sql;
  if (mysql_query(conn_, explain_sql.c_str())) {
    std::cerr << "[MariaDB] EXPLAIN failed: " << mysql_error(conn_)
              << std::endl;
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }

  MYSQL_RES *mysql_result = mysql_store_result(conn_);
  if (!mysql_result) {
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }

  double estimated_cost = std::numeric_limits<double>::max();
  double estimated_rows = std::numeric_limits<double>::max();

  MYSQL_ROW row = mysql_fetch_row(mysql_result);
  if (row && row[0]) {
    try {
      json explain_json = json::parse(row[0]);

      // MariaDB EXPLAIN FORMAT=JSON:
      // {"query_block": {"select_id": 1, "cost": 123.45,
      //   "table": {"chunk_name": "t1", "rows": 100, "cost": 10.0}}}
      if (explain_json.contains("query_block")) {
        auto &qb = explain_json["query_block"];
        if (qb.contains("cost")) {
          estimated_cost = qb["cost"].get<double>();
        }
        // Walk nested structure to find rows estimate
        if (qb.contains("table") && qb["table"].contains("rows")) {
          estimated_rows = qb["table"]["rows"].get<double>();
        } else if (qb.contains("nested_loop")) {
          // For joins, sum rows from all tables in nested_loop array
          double total_rows = 0;
          for (auto &entry : qb["nested_loop"]) {
            if (entry.contains("table") && entry["table"].contains("rows")) {
              double r = entry["table"]["rows"].get<double>();
              if (total_rows == 0)
                total_rows = r;
              else
                total_rows *= r;
            }
          }
          if (total_rows > 0)
            estimated_rows = total_rows;
        }
        // If we got cost but not rows, use cost as proxy (like Umbra)
        if (estimated_rows == std::numeric_limits<double>::max() &&
            estimated_cost != std::numeric_limits<double>::max()) {
          estimated_rows = estimated_cost;
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "[MariaDB] Failed to parse EXPLAIN JSON: " << e.what()
                << std::endl;
    }
  }

  mysql_free_result(mysql_result);
  return {estimated_cost, estimated_rows};
}

std::vector<std::pair<double, double>>
MariaDBAdapter::BatchGetEstimatedCosts(const std::vector<std::string> &sqls) {
  if (sqls.empty()) {
    return {};
  }

#ifdef HAVE_POSTGRES
  if (pg_estimator_) {
    // Route per-SQL: base-table queries → PG estimator (accurate),
    // temp-table queries → MariaDB EXPLAIN (PG can't see MariaDB temp tables).
    std::vector<size_t> pg_indices, maria_indices;
    std::vector<std::string> pg_sqls, maria_sqls;

    for (size_t i = 0; i < sqls.size(); i++) {
      if (HasTempTable(sqls[i])) {
        maria_indices.push_back(i);
        maria_sqls.push_back(sqls[i]);
      } else {
        pg_indices.push_back(i);
        pg_sqls.push_back(sqls[i]);
      }
    }

    std::vector<std::pair<double, double>> out(sqls.size());

    if (!pg_sqls.empty()) {
      auto pg_results = pg_estimator_->BatchGetEstimatedCosts(pg_sqls);
      for (size_t j = 0; j < pg_indices.size(); j++) {
        out[pg_indices[j]] = pg_results[j];
      }
    }

    if (!maria_sqls.empty()) {
      auto maria_results = BatchExplainWithMariaDB(maria_sqls);
      for (size_t j = 0; j < maria_indices.size(); j++) {
        out[maria_indices[j]] = maria_results[j];
      }
    }

    return out;
  }
#endif

  return BatchExplainWithMariaDB(sqls);
}

void MariaDBAdapter::CleanUp() {
  parse_tree_.clear();

  if (conn_) {
    mysql_close(conn_);
    conn_ = nullptr;
#ifndef NDEBUG
    std::cout << "[MariaDB] Connection closed" << std::endl;
#endif
  }
}

void MariaDBAdapter::CheckConnection() {
  if (!conn_) {
    throw std::runtime_error("MariaDB connection is not valid");
  }

  // mysql_ping() reconnects automatically if the connection was dropped
  if (mysql_ping(conn_)) {
    throw std::runtime_error("MariaDB connection lost: " +
                             std::string(mysql_error(conn_)));
  }
}
bool MariaDBAdapter::HasTextBlobColumns(const std::string &select_sql) {
  // Run the SELECT with LIMIT 0 wrapped in a subquery to get column metadata
  // without transferring any data. The subquery wrapper ensures LIMIT 0
  // doesn't conflict with any existing LIMIT/ORDER BY in the original query.
  std::string probe_sql =
      "SELECT * FROM (" + select_sql + ") AS _probe LIMIT 0";

  if (mysql_query(conn_, probe_sql.c_str())) {
    // On error, assume TEXT is present (safe fallback to InnoDB)
    return true;
  }

  MYSQL_RES *result = mysql_store_result(conn_);
  if (!result) {
    return true;
  }

  bool has_text_blob = false;
  unsigned int num_fields = mysql_num_fields(result);
  MYSQL_FIELD *fields = mysql_fetch_fields(result);

  for (unsigned int i = 0; i < num_fields; i++) {
    switch (fields[i].type) {
    case MYSQL_TYPE_TINY_BLOB:
    case MYSQL_TYPE_BLOB:
    case MYSQL_TYPE_MEDIUM_BLOB:
    case MYSQL_TYPE_LONG_BLOB:
      has_text_blob = true;
      break;
    default:
      break;
    }
    if (has_text_blob)
      break;
  }

  mysql_free_result(result);
  return has_text_blob;
}

} // namespace middleware
