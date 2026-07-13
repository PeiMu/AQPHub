/*
 * PostgreSQL adapter for binding IR to the PostgreSQL engine
 * */

#include "adapters/postgres_adapter.h"

#include <cassert>
#include <cstdlib>

#ifdef HAVE_LLVM

#define QJIT_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "[AQP-QJIT] FATAL: " << (msg) << "\n";                     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#include "qjit/query_jit_abi.h"
#include "qjit/query_jit_runtime.h"
#include "qjit/query_jit_steps.h"

namespace {
static bool IsSelectStatement(const std::string &sql) {
  size_t i = sql.find_first_not_of(" \t\r\n");
  if (i == std::string::npos)
    return false;
  auto starts_with = [&](const char *kw) {
    size_t n = std::strlen(kw);
    if (sql.size() - i < n)
      return false;
    for (size_t k = 0; k < n; k++)
      if (std::toupper(static_cast<unsigned char>(sql[i + k])) != kw[k])
        return false;
    return true;
  };
  return starts_with("SELECT") || starts_with("WITH");
}
static aqp_jit::SimdISA ResolveSimdISA(uint32_t flags) {
  if (flags & AQP_JIT_SIMD)
    return aqp_jit::SimdISA::AUTO;
  switch (flags & AQP_JIT_SIMD_MASK) {
  case AQP_JIT_SIMD_OFF:
    return aqp_jit::SimdISA::OFF;
  case AQP_JIT_SIMD_SSE2:
    return aqp_jit::SimdISA::SSE2;
  case AQP_JIT_SIMD_AVX:
    return aqp_jit::SimdISA::AVX;
  case AQP_JIT_SIMD_AVX2:
    return aqp_jit::SimdISA::AVX2;
  case AQP_JIT_SIMD_AVX512:
    return aqp_jit::SimdISA::AVX512;
  case AQP_JIT_SIMD_AUTO:
    return aqp_jit::SimdISA::AUTO;
  }
  return aqp_jit::SimdISA::OFF;
}
} // namespace
#endif

#ifdef HAVE_LLVM
#include "jit/aqp_jit_abi.h"
namespace {
static uint64_t ExtractEstimatedRows(const std::string &json_str) {
  if (json_str.empty())
    return 0;
  try {
    auto j = json::parse(json_str);
    if (j.is_array() && !j.empty() && j[0].contains("Plan"))
      if (j[0]["Plan"].contains("Plan Rows"))
        return j[0]["Plan"]["Plan Rows"].get<uint64_t>();
  } catch (...) {
  }
  return 0;
}

static void IrTargetListToDtypes(const ir_sql_converter::AQPStmt &ir,
                                 std::vector<int32_t> &dtypes,
                                 std::vector<std::string> &col_names) {
  for (const auto &attr : ir.target_list) {
    if (!attr)
      continue;
    int32_t dt;
    switch (attr->GetType()) {
    case ir_sql_converter::IntVar:
    case ir_sql_converter::Date:
      dt = attr->GetBitWidth() == 64 ? AQP_DTYPE_INT64 : AQP_DTYPE_INT32;
      break;
    case ir_sql_converter::StringVar:
      dt = AQP_DTYPE_VARCHAR;
      break;
    case ir_sql_converter::FloatVar:
      dt = attr->GetBitWidth() == 64 ? AQP_DTYPE_DOUBLE : AQP_DTYPE_FLOAT;
      break;
    case ir_sql_converter::BoolVar:
      dt = AQP_DTYPE_BOOL;
      break;
    default:
      dt = AQP_DTYPE_VARCHAR;
      break;
    }
    dtypes.push_back(dt);
    col_names.push_back(attr->GetColumnName());
  }
}
} // namespace
#endif

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

  // Temp tables don't need WAL durability — skip fsync on every commit
  PGresult *sync_result = PQexec(conn, "SET synchronous_commit = off");
  if (sync_result)
    PQclear(sync_result);
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

std::unique_ptr<ir_sql_converter::AQPStmt>
PostgreSQLAdapter::ConvertPlanToIR() {
  if (parse_tree.empty()) {
    throw std::runtime_error("No parse tree available. Call ParseSQL first.");
  }

  // Use schema-aware conversion if global schema parser is initialized,
  // otherwise fall back to basic conversion (column indices will be 0)
  std::unique_ptr<ir_sql_converter::AQPStmt> stmt =
      ir_sql_converter::ConvertParseTreeToIRWithSchema(parse_tree,
                                                       subquery_index);
  return std::move(stmt);
}

QueryResult PostgreSQLAdapter::ExecuteSQL(const std::string &sql) {
  CheckConnection();

  QueryResult result;

#ifdef HAVE_LLVM
  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

  if (query_jit_ && qjit_storage_plan_ && qjit_storage_plan_->IsLoaded() &&
      IsSelectStatement(sql)) {
    ParseSQL(sql);
    auto ir = ConvertPlanToIR();
    if (ir) {
      AnnotateBuildSidesFromExplain(sql, *ir);
      auto analysis = qjit::AnalyzeQueryJit(*ir, "result");
      if (analysis.accepted) {
        auto compiled = TryCompileQueryJit(*ir, analysis, "result");
        if (compiled) {
          if (enable_timing_) {
            auto compile_us =
                chrono_toc(&timer, "qjit final compile time\n", false);
            std::ofstream log_file;
            log_file.open(g_timing_log_name, std::ios_base::app);
            log_file << std::fixed << std::setprecision(3)
                     << ((compile_us + ConsumeSpecWaitUs()) / 1000.0) << ", ";
            log_file.close();
          }
          auto qjit_result = ExecuteQueryJitFinal(*compiled);
          if (qjit_result.num_rows >= 0) {
            if (enable_timing_) {
              auto run_us =
                  chrono_toc(&timer, "qjit final execute time\n", false);
              std::ofstream log_file;
              log_file.open(g_timing_log_name, std::ios_base::app);
              log_file << std::fixed << std::setprecision(3)
                       << (run_us / 1000.0) << ", ";
              log_file.close();
            }
            return qjit_result;
          }
        }
      }
    }
    if (enable_timing_)
      timer = chrono_tic();
  }
  if (enable_timing_ && (jit_flags_ & AQP_JIT_QUERY_JIT)) {
    auto zero_us = chrono_toc(&timer, "", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (ConsumeSpecWaitUs() / 1000.0) << ", ";
    log_file.close();
    timer = chrono_tic();
  }
#endif

  // Execute query
#ifndef HAVE_LLVM
  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();
#endif
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
  if (enable_timing_) {
    auto run_us = chrono_toc(&timer, "ExecuteSQL final_exe time\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (run_us / 1000.0) << ", ";
    log_file.close();
  }
  return result;
}

void PostgreSQLAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card) {
  CheckConnection();

  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

#ifdef HAVE_LLVM
  if (query_jit_ && qjit_storage_plan_ && qjit_storage_plan_->IsLoaded()) {
    // Spec HIT path: bg thread already compiled this sub-query
    if (qjit_spec_hit_) {
      auto spec = std::move(qjit_spec_hit_);
      std::string reason;
      QJIT_ASSERT(ResolveQjitSources(spec->plan, *spec->compiled, reason),
                  "spec-hit resolve failed: " + reason);
      if (enable_timing_) {
        auto compile_us =
            chrono_toc(&timer, "qjit spec-hit resolve time\n", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << ((compile_us + ConsumeSpecWaitUs()) / 1000.0) << ", ";
        log_file.close();
      }
      int64_t rows =
          ExecuteQueryJit(*spec->compiled, temp_table_name);
      QJIT_ASSERT(rows >= 0, "spec-hit execution failed");
      temp_table_card_[temp_table_name] = rows;
      if (enable_timing_) {
        auto execute_us = chrono_toc(
            &timer, "qjit execute time\n", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (execute_us / 1000.0) << ", ";
        log_file.close();
      }
      MaterializeQjitTempToPostgreSQL(temp_table_name, update_temp_card);
      if (enable_timing_) {
        auto mat_us = chrono_toc(
            &timer, "qjit extra_materialize time\n", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (mat_us / 1000.0) << ", ";
        log_file.close();
      }
      if (post_prepare_hook_)
        post_prepare_hook_(temp_table_name, {}, {}, "", rows, true);
#ifndef NDEBUG
      std::cout << "[PostgreSQL] Created temp table (qjit spec-hit): "
                << temp_table_name << " (rows=" << rows << ")"
                << std::endl;
#endif
      return;
    }

    // Fast path: splitter passed the IR directly — skip ParseSQL+ConvertPlanToIR
    const ir_sql_converter::AQPStmt *fast_ir = qjit_pending_ir_;
    qjit_pending_ir_ = nullptr;
    ir_sql_converter::AQPStmt *ir_ptr = nullptr;
    std::unique_ptr<ir_sql_converter::AQPStmt> ir_owned;

    if (fast_ir) {
      ir_ptr = const_cast<ir_sql_converter::AQPStmt *>(fast_ir);
    } else {
      ParseSQL(sql);
      ir_owned = ConvertPlanToIR();
      ir_ptr = ir_owned.get();
    }

    QJIT_ASSERT(ir_ptr, "IR conversion failed for sub-query");

    std::string explain_json;
    AnnotateBuildSidesFromExplain(sql, *ir_ptr, &explain_json);

    if (post_prepare_hook_) {
      std::vector<int32_t> dtypes;
      std::vector<std::string> col_names;
      IrTargetListToDtypes(*ir_ptr, dtypes, col_names);
      uint64_t est = ExtractEstimatedRows(explain_json);
      post_prepare_hook_(temp_table_name, dtypes, col_names,
                         explain_json, est, false);
    }

    auto analysis = qjit::AnalyzeQueryJit(*ir_ptr, temp_table_name);
    if (!analysis.accepted) {
      // Splitter-provided IR must always be accepted — abort on rejection.
      // Re-parsed IR may contain cross products etc. — fall through to PG.
      QJIT_ASSERT(!fast_ir,
                   "analysis rejected sub-query: " + analysis.reject_reason);
    } else {
    auto compiled =
        TryCompileQueryJit(*ir_ptr, analysis, temp_table_name);
    if (!compiled) {
      // Fall through to PG: missing temp source (cascading from a prior
      // PG-executed subquery), unsupported IR node, or StoragePlan gap.
    } else {
    if (enable_timing_) {
      auto compile_us =
          chrono_toc(&timer, "qjit compile time\n", false);
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << ((compile_us + ConsumeSpecWaitUs()) / 1000.0) << ", ";
      log_file.close();
    }
    int64_t rows = ExecuteQueryJit(*compiled, temp_table_name);
    QJIT_ASSERT(rows >= 0, "execution failed for sub-query");
    temp_table_card_[temp_table_name] = rows;
    if (enable_timing_) {
      auto execute_us =
          chrono_toc(&timer, "qjit execute time\n", false);
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (execute_us / 1000.0) << ", ";
      log_file.close();
    }
    MaterializeQjitTempToPostgreSQL(temp_table_name, update_temp_card);
    if (enable_timing_) {
      auto mat_us =
          chrono_toc(&timer, "qjit extra_materialize time\n",
                     false);
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (mat_us / 1000.0) << ", ";
      log_file.close();
    }
    if (post_prepare_hook_)
      post_prepare_hook_(temp_table_name, {}, {}, "", rows,
                         true);
#ifndef NDEBUG
    std::cout << "[PostgreSQL] Created temp table (qjit): "
              << temp_table_name << " (rows=" << rows << ")"
              << std::endl;
#endif
    return;
    } // compiled
    } // analysis.accepted
    if (enable_timing_) {
      auto reject_us = chrono_toc(&timer, "", false);
      (void)reject_us;
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (ConsumeSpecWaitUs() / 1000.0) << ", ";
      log_file.close();
      timer = chrono_tic();
    }
  }
#endif

  // Build SQL: CREATE TEMP TABLE + optional ANALYZE in one round-trip
  std::string create_sql = "CREATE TEMP TABLE " + temp_table_name + " AS (" +
                           sql.substr(0, sql.size() - 1) + ")";

#ifndef NDEBUG
  std::cout << "[PostgreSQL] Creating temp table: " << temp_table_name
            << std::endl;
#endif

  std::string combined_sql;
  if (update_temp_card) {
    combined_sql = create_sql + "; ANALYZE " + temp_table_name + ";";
  } else {
    combined_sql = create_sql + ";";
  }

  // Use PQsendQuery to send all statements in one round-trip
  // and PQgetResult to retrieve each result individually
  if (!PQsendQuery(conn, combined_sql.c_str())) {
    throw std::runtime_error("Failed to send query: " +
                             std::string(PQerrorMessage(conn)));
  }
  if (enable_timing_) {
    auto execute_sub_sql_time =
        chrono_toc(&timer, "Execute sub-SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_sub_sql_time / 1000.0) << ", ";
    log_file.close();
  }

  // First result: CREATE TEMP TABLE — extract row count from command tag
  PGresult *create_result = PQgetResult(conn);
  if (!create_result || PQresultStatus(create_result) != PGRES_COMMAND_OK) {
    std::string error_msg =
        "Failed to create temp table: " + std::string(PQerrorMessage(conn));
    if (create_result)
      PQclear(create_result);
    while (PGresult *r = PQgetResult(conn))
      PQclear(r);
    throw std::runtime_error(error_msg);
  }

  // PQcmdTuples returns row count for CREATE TABLE AS (avoids SELECT COUNT(*))
  const char *cmd_tuples = PQcmdTuples(create_result);
  if (cmd_tuples && cmd_tuples[0] != '\0') {
    temp_table_card_[temp_table_name] = std::stoull(cmd_tuples);
  }
  PQclear(create_result);

  // Drain remaining results (ANALYZE result if present, then NULL terminator)
  while (PGresult *r = PQgetResult(conn)) {
    PQclear(r);
  }

  if (enable_timing_) {
    auto extra_materialize_time =
        chrono_toc(&timer, "Extra materialize time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extra_materialize_time / 1000.0) << ", ";
    log_file.close();
  }

#ifndef NDEBUG
  std::cout << "[PostgreSQL] Created temp table: " << temp_table_name
            << " (rows=" << temp_table_card_[temp_table_name] << ")"
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
  // Check cache first (populated by ExecuteSQLandCreateTempTable via
  // PQcmdTuples)
  auto it = temp_table_card_.find(temp_table_name);
  if (it != temp_table_card_.end()) {
    return it->second;
  }

  // Fallback to SELECT COUNT(*)
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

void PostgreSQLAdapter::SetTempTableCardinality(
    const std::string &temp_table_name, uint64_t estimated_rows) {
  CheckConnection();

  // Update pg_class.reltuples so PostgreSQL's planner uses this estimated_rows
  std::string update_sql =
      "UPDATE pg_class SET reltuples = " + std::to_string(estimated_rows) +
      " WHERE relname = '" + temp_table_name + "'";
  PGresult *pg_result = PQexec(conn, update_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_COMMAND_OK) {
    std::cerr << "[PostgreSQL] Failed to set reltuples: "
              << PQerrorMessage(conn) << std::endl;
  }

  PQclear(pg_result);

#ifndef NDEBUG
  std::cout << "[PostgreSQL] SetTempTableCardinality: " << temp_table_name
            << " = " << estimated_rows << std::endl;
#endif
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

std::string PostgreSQLAdapter::ExplainAnalyze(const std::string &sql) {
  // Reuse ExecuteSQL: EXPLAIN (ANALYZE, ...) returns one plan line per row in a
  // single "QUERY PLAN" column. Failures are returned as text (never thrown) so
  // a plan error can't abort the surrounding split run.
  try {
    QueryResult r =
        ExecuteSQL("EXPLAIN (ANALYZE, VERBOSE, BUFFERS) " + StripSqlTerminator(sql));
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

std::vector<std::pair<double, double>>
PostgreSQLAdapter::BatchGetEstimatedCosts(
    const std::vector<std::string> &sqls) {
  if (sqls.empty()) {
    return {};
  }

  CheckConnection();

  // Concatenate all EXPLAIN queries into one string
  std::string combined;
  for (const auto &sql : sqls) {
    combined += "EXPLAIN (FORMAT JSON) " + sql + ";";
  }

  // Send all at once via PQsendQuery
  if (!PQsendQuery(conn, combined.c_str())) {
    std::cerr << "[PostgreSQL] BatchGetEstimatedCosts: PQsendQuery failed: "
              << PQerrorMessage(conn) << std::endl;
    // Fallback: return max for all
    return std::vector<std::pair<double, double>>(
        sqls.size(), {std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max()});
  }

  std::vector<std::pair<double, double>> results;
  results.reserve(sqls.size());

  for (size_t i = 0; i < sqls.size(); i++) {
    PGresult *pg_result = PQgetResult(conn);

    double estimated_cost = std::numeric_limits<double>::max();
    double estimated_rows = std::numeric_limits<double>::max();

    if (pg_result && PQresultStatus(pg_result) == PGRES_TUPLES_OK &&
        PQntuples(pg_result) > 0 && PQnfields(pg_result) > 0) {
      std::string json_str = PQgetvalue(pg_result, 0, 0);
      try {
        json explain_json = json::parse(json_str);
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
        std::cerr << "[PostgreSQL] BatchGetEstimatedCosts: parse failed for "
                     "query "
                  << i << ": " << e.what() << std::endl;
      }
    } else {
#ifndef NDEBUG
      std::cerr << "[PostgreSQL] BatchGetEstimatedCosts: EXPLAIN failed for "
                   "query "
                << i << std::endl;
#endif
    }

    if (pg_result)
      PQclear(pg_result);
    results.push_back({estimated_cost, estimated_rows});
  }

  // Drain NULL terminator
  while (PGresult *r = PQgetResult(conn)) {
    PQclear(r);
  }

  return results;
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

void PostgreSQLAdapter::ResetQueryState() {
  for (const auto &kv : temp_table_card_)
    DropTempTable(kv.first);
  parse_tree.clear();
  temp_table_card_.clear();
  subquery_index = 0;
#ifdef HAVE_LLVM
  qjit_temps_.clear();
  qjit_pending_ir_ = nullptr;
  qjit_spec_hit_.reset();
  spec_wait_extra_us_ = 0;
  if (jit_compiler_)
    jit_compiler_->ResetModules();
#endif
}

void PostgreSQLAdapter::CheckConnection() {
  if (!conn || CONNECTION_OK != PQstatus(conn)) {
    throw std::runtime_error("PostgreSQL connection is not valid");
  }
}

// ---------------------------------------------------------------------------
// Query-jit support (Steps 2-4)
// ---------------------------------------------------------------------------
#ifdef HAVE_LLVM

namespace {

static void
CollectIRSourceIndices(const ir_sql_converter::AQPStmt *ir,
                       std::unordered_set<unsigned int> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode)
    out.insert(
        static_cast<const ir_sql_converter::SimplestScan *>(ir)->GetTableIndex());
  else if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode)
    out.insert(
        static_cast<const ir_sql_converter::SimplestChunk *>(ir)->GetTableIndex());
  for (const auto &child : ir->children)
    CollectIRSourceIndices(child.get(), out);
}

static void
CollectIRJoinNodes(ir_sql_converter::AQPStmt *ir,
                   std::vector<ir_sql_converter::SimplestJoin *> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    out.push_back(static_cast<ir_sql_converter::SimplestJoin *>(ir));
  for (auto &child : ir->children)
    CollectIRJoinNodes(child.get(), out);
}

static void
BuildTableNameToIndex(const ir_sql_converter::AQPStmt *ir,
                      std::unordered_map<std::string, unsigned int> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    out[scan->GetTableName()] = scan->GetTableIndex();
  } else if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = static_cast<const ir_sql_converter::SimplestChunk *>(ir);
    out[chunk->GetChunkName()] = chunk->GetTableIndex();
  }
  for (const auto &child : ir->children)
    BuildTableNameToIndex(child.get(), out);
}

static void
CollectRelationNames(const json &plan_node, std::vector<std::string> &out) {
  if (plan_node.contains("Relation Name"))
    out.push_back(plan_node["Relation Name"].get<std::string>());
  if (plan_node.contains("Alias") && !plan_node.contains("Relation Name"))
    out.push_back(plan_node["Alias"].get<std::string>());
  if (plan_node.contains("Plans")) {
    for (const auto &child : plan_node["Plans"])
      CollectRelationNames(child, out);
  }
}

static void
AnnotateFromExplainRec(
    const json &plan_node,
    ir_sql_converter::AQPStmt &ir,
    const std::unordered_map<std::string, unsigned int> &name_to_index) {
  if (plan_node.contains("Plans")) {
    for (const auto &child : plan_node["Plans"])
      AnnotateFromExplainRec(child, ir, name_to_index);
  }

  std::string node_type;
  if (plan_node.contains("Node Type"))
    node_type = plan_node["Node Type"].get<std::string>();
  if (node_type != "Hash Join" && node_type != "Nested Loop" &&
      node_type != "Merge Join")
    return;
  if (!plan_node.contains("Plans"))
    return;

  const json *inner_child = nullptr;
  const json *outer_child = nullptr;
  for (const auto &child : plan_node["Plans"]) {
    if (!child.contains("Parent Relationship"))
      continue;
    std::string rel = child["Parent Relationship"].get<std::string>();
    if (rel == "Inner")
      inner_child = &child;
    else if (rel == "Outer")
      outer_child = &child;
  }
  if (!inner_child || !outer_child)
    return;

  std::vector<std::string> build_names, probe_names;
  CollectRelationNames(*inner_child, build_names);
  CollectRelationNames(*outer_child, probe_names);

  std::unordered_set<unsigned int> build_set, probe_set;
  for (const auto &name : build_names) {
    auto it = name_to_index.find(name);
    if (it != name_to_index.end())
      build_set.insert(it->second);
  }
  for (const auto &name : probe_names) {
    auto it = name_to_index.find(name);
    if (it != name_to_index.end())
      probe_set.insert(it->second);
  }
  if (build_set.empty() || probe_set.empty())
    return;

  std::vector<ir_sql_converter::SimplestJoin *> joins;
  CollectIRJoinNodes(&ir, joins);
  for (auto *join : joins) {
    if (join->GetBuildChild() != -1 || join->children.size() != 2)
      continue;
    std::unordered_set<unsigned int> c0_set, c1_set;
    CollectIRSourceIndices(join->children[0].get(), c0_set);
    CollectIRSourceIndices(join->children[1].get(), c1_set);
    if (c0_set.empty() || c1_set.empty())
      continue;
    if (c0_set == probe_set && c1_set == build_set) {
      join->SetBuildChild(1);
      return;
    }
    if (c0_set == build_set && c1_set == probe_set) {
      join->SetBuildChild(0);
      return;
    }
  }
#ifndef NDEBUG
  std::cerr << "[AQP-QJIT] AnnotateFromExplainRec: no IR JoinNode matched "
            << node_type << "\n";
#endif
}

static uint64_t
EstimateSubtreeCard(const ir_sql_converter::AQPStmt *node,
                    const storage::StoragePlan *sp,
                    const std::unordered_map<std::string, int64_t> &temp_card) {
  if (!node) return 0;
  auto nt = node->GetNodeType();
  if (nt == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(node);
    if (sp) {
      const auto *ft = sp->GetTable(scan->GetTableName());
      if (ft) return ft->row_count;
    }
    return 1000000;
  }
  if (nt == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = static_cast<const ir_sql_converter::SimplestChunk *>(node);
    auto it = temp_card.find(chunk->GetChunkName());
    if (it != temp_card.end()) return it->second;
    return 1000;
  }
  uint64_t card = 0;
  for (const auto &child : node->children) {
    uint64_t c = EstimateSubtreeCard(child.get(), sp, temp_card);
    card = (card == 0) ? c : std::min(card, c);
  }
  return card;
}

static void
AnnotateUnannotatedJoinsByCard(
    ir_sql_converter::AQPStmt &ir,
    const storage::StoragePlan *sp,
    const std::unordered_map<std::string, int64_t> &temp_card) {
  std::vector<ir_sql_converter::SimplestJoin *> joins;
  CollectIRJoinNodes(&ir, joins);
  for (auto *join : joins) {
    if (join->GetBuildChild() != -1 || join->children.size() != 2)
      continue;
    uint64_t c0 = EstimateSubtreeCard(join->children[0].get(), sp, temp_card);
    uint64_t c1 = EstimateSubtreeCard(join->children[1].get(), sp, temp_card);
    join->SetBuildChild(c1 <= c0 ? 1 : 0);
#ifndef NDEBUG
    std::cerr << "[AQP-QJIT] fallback build-side annotation: c0=" << c0
              << " c1=" << c1 << " → build=" << join->GetBuildChild() << "\n";
#endif
  }
}

} // anonymous namespace

// Step 2: Build-side annotation from EXPLAIN JSON
void PostgreSQLAdapter::AnnotateBuildSidesFromExplain(
    const std::string &sql, ir_sql_converter::AQPStmt &ir,
    std::string *out_json) {
  std::string explain_sql =
      "EXPLAIN (FORMAT JSON) " + StripSqlTerminator(sql);
  PGresult *pg_result = PQexec(conn, explain_sql.c_str());

  if (PQresultStatus(pg_result) != PGRES_TUPLES_OK ||
      PQntuples(pg_result) == 0) {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] EXPLAIN failed for build-side annotation: %s\n",
            PQerrorMessage(conn));
#endif
    PQclear(pg_result);
    return;
  }

  std::string json_str = PQgetvalue(pg_result, 0, 0);
  PQclear(pg_result);
  if (out_json)
    *out_json = json_str;

  json explain_json;
  try {
    explain_json = json::parse(json_str);
  } catch (const std::exception &e) {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] EXPLAIN JSON parse failed: %s\n", e.what());
#endif
    return;
  }

  if (!explain_json.is_array() || explain_json.empty() ||
      !explain_json[0].contains("Plan"))
    return;

  std::unordered_map<std::string, unsigned int> name_to_index;
  BuildTableNameToIndex(&ir, name_to_index);

  AnnotateFromExplainRec(explain_json[0]["Plan"], ir, name_to_index);

  AnnotateUnannotatedJoinsByCard(ir, qjit_storage_plan_, temp_table_card_);
}

// Runtime symbol registration (same as DuckDB adapter)
void PostgreSQLAdapter::RegisterQjitRuntimeSymbols(
    aqp_jit::IrToLlvmCompiler *comp) {
  comp->RegisterRuntimeSymbol("qjit_parallel_for", (void *)&qjit_parallel_for);
  comp->RegisterRuntimeSymbol("qjit_buffer_grow", (void *)&qjit_buffer_grow);
  comp->RegisterRuntimeSymbol("qjit_ht_append", (void *)&qjit_ht_append);
  comp->RegisterRuntimeSymbol("qjit_ht_begin", (void *)&qjit_ht_begin);
  comp->RegisterRuntimeSymbol("qjit_ht_append_slow",
                              (void *)&qjit_ht_append_slow);
  comp->RegisterRuntimeSymbol("qjit_ht_end", (void *)&qjit_ht_end);
  comp->RegisterRuntimeSymbol("qjit_ht_finalize", (void *)&qjit_ht_finalize);
  comp->RegisterRuntimeSymbol("qjit_str_arena_copy",
                              (void *)&qjit_str_arena_copy);
  comp->RegisterRuntimeSymbol("qjit_table_append_i32",
                              (void *)&qjit_table_append_i32);
  comp->RegisterRuntimeSymbol("qjit_table_append_i64",
                              (void *)&qjit_table_append_i64);
  comp->RegisterRuntimeSymbol("qjit_table_append_str",
                              (void *)&qjit_table_append_str);
  comp->RegisterRuntimeSymbol("qjit_table_append_null",
                              (void *)&qjit_table_append_null);
  comp->RegisterRuntimeSymbol("qjit_table_finish_row",
                              (void *)&qjit_table_finish_row);
  comp->RegisterRuntimeSymbol("qjit_ht_dir", (void *)&qjit_ht_dir);
  comp->RegisterRuntimeSymbol("qjit_ht_mask", (void *)&qjit_ht_mask);
  comp->RegisterRuntimeSymbol("qjit_ht_key0_min", (void *)&qjit_ht_key0_min);
  comp->RegisterRuntimeSymbol("qjit_ht_key0_max", (void *)&qjit_ht_key0_max);
  comp->RegisterRuntimeSymbol("qjit_ht_entries", (void *)&qjit_ht_entries);
  comp->RegisterRuntimeSymbol("qjit_agg_update_i64",
                              (void *)&qjit_agg_update_i64);
  comp->RegisterRuntimeSymbol("qjit_agg_update_str",
                              (void *)&qjit_agg_update_str);
  comp->RegisterRuntimeSymbol("qjit_agg_update_count",
                              (void *)&qjit_agg_update_count);
  comp->RegisterRuntimeSymbol("qjit_table_begin", (void *)&qjit_table_begin);
  comp->RegisterRuntimeSymbol("qjit_table_col_slow",
                              (void *)&qjit_table_col_slow);
  comp->RegisterRuntimeSymbol("qjit_table_null_slow",
                              (void *)&qjit_table_null_slow);
  comp->RegisterRuntimeSymbol("qjit_table_end", (void *)&qjit_table_end);
  comp->RegisterRuntimeSymbol("qjit_table_str_copy",
                              (void *)&qjit_table_str_copy);
}

// Step 4 helpers: BuildOutputDescsFromIR, ResolveQjitSources, TryCompileQueryJit

static void CollectTableNames(
    const ir_sql_converter::AQPStmt &node,
    std::unordered_map<unsigned int, std::string> &table_names) {
  if (node.GetNodeType() == ir_sql_converter::ScanNode) {
    auto &scan = node.Cast<ir_sql_converter::SimplestScan>();
    table_names[scan.GetTableIndex()] = scan.GetTableName();
  } else if (node.GetNodeType() == ir_sql_converter::ChunkNode) {
    auto &chunk = node.Cast<ir_sql_converter::SimplestChunk>();
    table_names[chunk.GetTableIndex()] = chunk.GetChunkName();
  }
  for (const auto &child : node.children) {
    if (child)
      CollectTableNames(*child, table_names);
  }
}

static std::string IrColumnAlias(
    const ir_sql_converter::SimplestAttr &attr,
    const std::unordered_map<unsigned int, std::string> &table_names) {
  unsigned int tidx = attr.GetTableIndex();
  auto it = table_names.find(tidx);
  if (it != table_names.end()) {
    return it->second + "_" + std::to_string(tidx) + "_" +
           attr.GetColumnName();
  }
  return "col_" + std::to_string(tidx) + "_" + attr.GetColumnName();
}

bool PostgreSQLAdapter::BuildOutputDescsFromIR(
    const qjit::QjitQueryPlan &plan, QjitCompiled &compiled,
    std::string &reason) {
  const qjit::QjitStep &last = plan.steps.back();

  std::unordered_map<unsigned int, std::string> table_names;
  if (plan.ir)
    CollectTableNames(*plan.ir, table_names);

  auto out_name = [&](size_t i) -> std::string {
    if (plan.ir && i < plan.ir->target_list.size() &&
        plan.ir->target_list[i]) {
      return IrColumnAlias(*plan.ir->target_list[i], table_names);
    }
    return "col_" + std::to_string(i);
  };

  if (plan.has_agg) {
    for (size_t i = 0; i < plan.agg_output_cells.size(); i++) {
      const qjit::QjitAggCellPlan &cell =
          last.agg_cells[(size_t)plan.agg_output_cells[i]];
      int32_t dt;
      if (cell.fn == qjit::QjitAggFn::Count ||
          cell.fn == qjit::QjitAggFn::CountStar) {
        dt = AQP_DTYPE_INT64;
      } else if (cell.arg.dtype == AQP_DTYPE_INT32) {
        dt = AQP_DTYPE_INT32;
      } else if (cell.arg.dtype == AQP_DTYPE_VARCHAR) {
        dt = AQP_DTYPE_VARCHAR;
      } else {
        reason = "output:agg-type";
        return false;
      }
      compiled.out_descs.push_back({dt, out_name(i)});
    }
    compiled.agg_output_cells = plan.agg_output_cells;
    compiled.agg_descs.reserve(last.agg_cells.size());
    for (const auto &cell : last.agg_cells) {
      qjit::QjitAggDType adt =
          (cell.has_arg && cell.arg.dtype == AQP_DTYPE_VARCHAR)
              ? qjit::QjitAggDType::Str
              : qjit::QjitAggDType::I64;
      compiled.agg_descs.push_back({cell.fn, adt});
    }
  } else {
    for (size_t i = 0; i < last.outputs.size(); i++) {
      int32_t dt = last.outputs[i].dtype;
      if (dt != AQP_DTYPE_INT32 && dt != AQP_DTYPE_VARCHAR) {
        reason = "output:unsupported-dtype";
        return false;
      }
      compiled.out_descs.push_back({dt, out_name(i)});
    }
  }
  return true;
}

bool PostgreSQLAdapter::ResolveQjitSources(const qjit::QjitQueryPlan &plan,
                                           QjitCompiled &compiled,
                                           std::string &reason) {
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded()) {
    reason = "no-storage-plan";
    return false;
  }
  if (!qjit_executor_)
    qjit_executor_ = std::make_unique<qjit::QjitExecutor>(
        query_jit_threads_,
        query_jit_morsel_ > 0 ? (uint64_t)query_jit_morsel_ : 0);
  compiled.srcs.resize(plan.steps.size());
  for (size_t k = 0; k < plan.steps.size(); k++) {
    const qjit::QjitStep &st = plan.steps[k];
    if (st.source_is_temp) {
      auto it = qjit_temps_.find(st.source_table);
      if (it == qjit_temps_.end() || !it->second) {
        reason = "source:temp-missing:" + st.source_table;
        return false;
      }
      if (!qjit_executor_->ResolveTempSource(*it->second, st.cols,
                                             compiled.srcs[k], reason))
        return false;
      continue;
    }
    const auto *flat = qjit_storage_plan_->GetTable(st.source_table);
    if (!flat) {
      reason = "source:table-missing:" + st.source_table;
      return false;
    }
    if (!qjit_executor_->ResolveSource(*flat, st.cols, compiled.srcs[k],
                                       reason, st.block_skip_col))
      return false;
  }
  return true;
}

std::unique_ptr<PostgreSQLAdapter::QjitCompiled>
PostgreSQLAdapter::TryCompileQueryJit(
    const ir_sql_converter::AQPStmt &ir,
    const qjit::QjitAnalysisResult &analysis, const std::string &label) {
  auto fallback = [&](const std::string &reason)
      -> std::unique_ptr<QjitCompiled> {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] fallback:%s label=%s\n", reason.c_str(),
            label.c_str());
#else
    (void)reason;
#endif
    return nullptr;
  };

  if (!analysis.accepted)
    return nullptr;
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded())
    return fallback("no-storage-plan");

  qjit::QjitQueryPlan plan;
  std::string reason;
  if (!qjit::BuildExecutionSteps(ir, plan, reason))
    return fallback(reason);
  plan.ir = &ir;

  auto compiled = std::make_unique<QjitCompiled>();
  if (!BuildOutputDescsFromIR(plan, *compiled, reason))
    return fallback(reason);

  if (!ResolveQjitSources(plan, *compiled, reason))
    return fallback(reason);

  compiled->ht_tuple_sizes.reserve(plan.hts.size());
  compiled->ht_key0_offsets.reserve(plan.hts.size());
  for (const auto &ht : plan.hts) {
    compiled->ht_tuple_sizes.push_back(ht.tuple_size);
    compiled->ht_key0_offsets.push_back(ht.prefix_bytes);
  }

  auto want_simd = ResolveSimdISA(jit_flags_);
  auto want_fast =
      static_cast<aqp_jit::FastCompileBackend>(compile_mode_);
  if (jit_compiler_ &&
      (jit_compiler_->GetFastMode() != want_fast ||
       jit_compiler_->GetSimdISA() != want_simd)) {
    jit_compiler_.reset();
    qjit_syms_registered_ = false;
  }
  if (!jit_compiler_) {
    jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
        jit_debug_, want_simd, want_fast);
    jit_compiler_->SetPrefetch(jit_prefetch_, jit_prefetch_distance_);
  }
  if (!qjit_syms_registered_) {
    RegisterQjitRuntimeSymbols(jit_compiler_.get());
    qjit_syms_registered_ = true;
  }
  jit_compiler_->SetSkipHashCmp(skip_hash_cmp_);
  if (jit_cache_)
    jit_compiler_->SetCache(jit_cache_);
  if (!jit_cache_dir_.empty())
    jit_compiler_->SetDiskCacheDir(jit_cache_dir_);
  jit_compiler_->ResetModules();

  compiled->fn =
      jit_compiler_->CompileQuerySteps(plan, &compiled->params_buf);
  if (!compiled->fn)
    return fallback("compile-failed");

  return compiled;
}

// Execute compiled query-jit plan, store result as temp in qjit_temps_
int64_t PostgreSQLAdapter::ExecuteQueryJit(QjitCompiled &compiled,
                                           const std::string &temp_table_name) {
  auto qtable = std::make_unique<qjit::QjitTable>(
      compiled.out_descs, qjit_executor_->NumWorkers());
  int64_t rows = qjit_executor_->Run(
      reinterpret_cast<QjitQueryFn>(compiled.fn), compiled.srcs,
      compiled.ht_tuple_sizes, compiled.agg_descs, compiled.agg_output_cells,
      *qtable, compiled.ht_key0_offsets, compiled.params_buf);
  if (rows >= 0) {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] exec label=%s rows=%lld\n",
            temp_table_name.c_str(), (long long)rows);
#endif
    qjit_temps_[temp_table_name] = std::move(qtable);
  } else {
    fprintf(stderr, "[AQP-QJIT] fallback:run-error(rc=%lld) label=%s\n",
            (long long)rows, temp_table_name.c_str());
  }
  return rows;
}

// Execute compiled query-jit plan, return result as QueryResult (final query)
QueryResult
PostgreSQLAdapter::ExecuteQueryJitFinal(QjitCompiled &compiled) {
  qjit::QjitTable qtable(compiled.out_descs, qjit_executor_->NumWorkers());
  int64_t rows = qjit_executor_->Run(
      reinterpret_cast<QjitQueryFn>(compiled.fn), compiled.srcs,
      compiled.ht_tuple_sizes, compiled.agg_descs, compiled.agg_output_cells,
      qtable, compiled.ht_key0_offsets, compiled.params_buf);
  QueryResult result;
  if (rows < 0) {
    fprintf(stderr, "[AQP-QJIT] fallback:run-error(rc=%lld) label=result\n",
            (long long)rows);
    return result;
  }
#ifndef NDEBUG
  fprintf(stderr, "[AQP-QJIT] exec label=result rows=%lld\n",
          (long long)rows);
#endif
  result.num_columns = qtable.NumCols();
  for (size_t i = 0; i < qtable.NumCols(); i++)
    result.column_names.push_back(qtable.Col(i).name);
  result.num_rows = 0;
  for (uint64_t r = 0; r < (uint64_t)rows; r++) {
    std::vector<std::string> row_data;
    row_data.reserve(result.num_columns);
    for (size_t col = 0; col < result.num_columns; col++) {
      if (!qtable.ValueValid(col, r)) {
        row_data.emplace_back("NULL");
      } else if (qtable.Col(col).dtype == AQP_DTYPE_INT32) {
        row_data.push_back(std::to_string(qtable.GetI32(col, r)));
      } else if (qtable.Col(col).dtype == AQP_DTYPE_INT64) {
        row_data.push_back(std::to_string(qtable.GetI64(col, r)));
      } else {
        QjitString s = qtable.GetStr(col, r);
        row_data.emplace_back(qjit::StringData(s), qjit::StringLen(s));
      }
    }
    result.rows.push_back(std::move(row_data));
    result.num_rows++;
  }
  return result;
}

// Materialize a QjitTable temp result into PostgreSQL via COPY FROM STDIN
void PostgreSQLAdapter::MaterializeQjitTempToPostgreSQL(
    const std::string &name, bool update_temp_card) {
  auto it = qjit_temps_.find(name);
  if (it == qjit_temps_.end() || !it->second)
    return;
  const qjit::QjitTable &qtable = *it->second;
  uint64_t nrows = qtable.NumRows();
  size_t ncols = qtable.NumCols();

  // Build CREATE TEMP TABLE with correct column types
  std::string create_sql = "CREATE TEMP TABLE " + name + " (";
  for (size_t c = 0; c < ncols; c++) {
    if (c > 0)
      create_sql += ", ";
    create_sql += qtable.Col(c).name + " ";
    if (qtable.Col(c).dtype == AQP_DTYPE_INT32)
      create_sql += "integer";
    else if (qtable.Col(c).dtype == AQP_DTYPE_INT64)
      create_sql += "bigint";
    else
      create_sql += "text";
  }
  create_sql += ")";
  PGresult *cr = PQexec(conn, create_sql.c_str());
  if (PQresultStatus(cr) != PGRES_COMMAND_OK) {
    fprintf(stderr,
            "[AQP-QJIT] MaterializeQjitTemp: CREATE failed: %s\n",
            PQerrorMessage(conn));
    PQclear(cr);
    return;
  }
  PQclear(cr);

  // COPY FROM STDIN in text mode
  std::string copy_sql = "COPY " + name + " FROM STDIN";
  cr = PQexec(conn, copy_sql.c_str());
  if (PQresultStatus(cr) != PGRES_COPY_IN) {
    fprintf(stderr,
            "[AQP-QJIT] MaterializeQjitTemp: COPY start failed: %s\n",
            PQerrorMessage(conn));
    PQclear(cr);
    return;
  }
  PQclear(cr);

  for (uint64_t r = 0; r < nrows; r++) {
    std::string line;
    for (size_t c = 0; c < ncols; c++) {
      if (c > 0)
        line += '\t';
      if (!qtable.ValueValid(c, r)) {
        line += "\\N";
      } else if (qtable.Col(c).dtype == AQP_DTYPE_INT32) {
        line += std::to_string(qtable.GetI32(c, r));
      } else if (qtable.Col(c).dtype == AQP_DTYPE_INT64) {
        line += std::to_string(qtable.GetI64(c, r));
      } else {
        QjitString s = qtable.GetStr(c, r);
        int32_t slen = qjit::StringLen(s);
        const char *sdata = qjit::StringData(s);
        for (int32_t i = 0; i < slen; i++) {
          char ch = sdata[i];
          if (ch == '\\') line += "\\\\";
          else if (ch == '\t') line += "\\t";
          else if (ch == '\n') line += "\\n";
          else if (ch == '\r') line += "\\r";
          else line += ch;
        }
      }
    }
    line += '\n';
    if (PQputCopyData(conn, line.c_str(), (int)line.size()) != 1) {
      fprintf(stderr, "[AQP-QJIT] MaterializeQjitTemp: PQputCopyData failed\n");
      PQputCopyEnd(conn, "aborted");
      while (PGresult *r2 = PQgetResult(conn))
        PQclear(r2);
      return;
    }
  }

  if (PQputCopyEnd(conn, nullptr) != 1) {
    fprintf(stderr, "[AQP-QJIT] MaterializeQjitTemp: PQputCopyEnd failed\n");
  }
  PGresult *copy_result = PQgetResult(conn);
  if (copy_result) {
    if (PQresultStatus(copy_result) != PGRES_COMMAND_OK) {
      fprintf(stderr,
              "[AQP-QJIT] MaterializeQjitTemp: COPY result error: %s\n",
              PQerrorMessage(conn));
    }
    PQclear(copy_result);
  }

  if (update_temp_card) {
    std::string analyze_sql = "ANALYZE " + name;
    PGresult *ar = PQexec(conn, analyze_sql.c_str());
    if (ar) {
      if (PQresultStatus(ar) != PGRES_COMMAND_OK) {
        std::cerr << "[AQP-QJIT] MaterializeQjitTemp: ANALYZE failed: "
                  << PQerrorMessage(conn) << "\n";
      }
      PQclear(ar);
    }
  }

#ifndef NDEBUG
  fprintf(stderr,
          "[AQP-QJIT] materialized temp=%s rows=%llu cols=%zu\n",
          name.c_str(), (unsigned long long)nrows, ncols);
#endif
}

// Bg-thread speculative compile: parse → IR → annotate from JSON → compile.
// No PGconn access — all server interaction must happen on the main thread.
std::unique_ptr<PostgreSQLAdapter::QjitSpecCompiled>
PostgreSQLAdapter::SpeculativeQueryJitCompileFromJSON(
    const std::string &sql, const std::string &explain_json,
    const std::string &label, unsigned int sub_plan_id,
    aqp_jit::IrToLlvmCompiler *spec_comp) {
  auto reject = [&](const std::string &reason)
      -> std::unique_ptr<QjitSpecCompiled> {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] spec-reject:%s label=%s\n", reason.c_str(),
            label.c_str());
#else
    (void)reason;
#endif
    return nullptr;
  };
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded())
    return reject("no-storage-plan");

  try {
    // 1. Parse SQL (thread-safe: libpg_query uses __thread globals)
    PgQueryParseResult pg_result = pg_query_parse(sql.c_str());
    if (pg_result.error) {
      std::string err = pg_result.error->message;
      pg_query_free_parse_result(pg_result);
      return reject("parse-failed:" + err);
    }
    json parse_tree_local;
    try {
      parse_tree_local = json::parse(pg_result.parse_tree);
    } catch (...) {
      pg_query_free_parse_result(pg_result);
      return reject("json-parse-failed");
    }
    pg_query_free_parse_result(pg_result);

    // 2. Convert to IR (thread-safe: SchemaParser is read-only after init).
    // sub_plan_id captured on main thread — NOT adapter->subquery_index.
    auto ir = ir_sql_converter::ConvertParseTreeToIRWithSchema(
        parse_tree_local, sub_plan_id);
    if (!ir)
      return reject("ir-conversion-failed");

    // 3. Annotate build sides from cached EXPLAIN JSON (no PGconn needed)
    if (!explain_json.empty()) {
      json ej;
      try {
        ej = json::parse(explain_json);
      } catch (...) {
      }
      if (ej.is_array() && !ej.empty() && ej[0].contains("Plan")) {
        std::unordered_map<std::string, unsigned int> name_to_index;
        BuildTableNameToIndex(ir.get(), name_to_index);
        AnnotateFromExplainRec(ej[0]["Plan"], *ir, name_to_index);
      }
    }

    // 4. Analyze + build steps
    auto analysis = qjit::AnalyzeQueryJit(*ir, label);
    if (!analysis.accepted)
      return nullptr;

    auto payload = std::make_unique<QjitSpecCompiled>();
    std::string reason;
    if (!qjit::BuildExecutionSteps(*ir, payload->plan, reason))
      return reject(reason);
    payload->plan.ir = ir.get();

    payload->compiled = std::make_unique<QjitCompiled>();
    if (!BuildOutputDescsFromIR(payload->plan, *payload->compiled, reason))
      return reject(reason);

    payload->compiled->ht_tuple_sizes.reserve(payload->plan.hts.size());
    payload->compiled->ht_key0_offsets.reserve(payload->plan.hts.size());
    for (const auto &ht : payload->plan.hts) {
      payload->compiled->ht_tuple_sizes.push_back(ht.tuple_size);
      payload->compiled->ht_key0_offsets.push_back(ht.prefix_bytes);
    }

    // 5. LLVM compile
    payload->compiled->fn = spec_comp->CompileQuerySteps(
        payload->plan, &payload->compiled->params_buf);
    if (!payload->compiled->fn)
      return reject("compile-failed");

    payload->ir = std::move(ir);
    return payload;
  } catch (std::exception &e) {
    return reject(std::string("exception:") + e.what());
  }
}

#endif // HAVE_LLVM

} // namespace middleware