#include "adapters/lingodb_adapter.h"
#include "util/util.h"

#include <fstream>
#include <iomanip>
#include <sstream>

#include <arrow/api.h>
#include <arrow/table.h>

#include <lingodb/catalog/Defs.h>
#include <lingodb/catalog/TableCatalogEntry.h>
#include <lingodb/catalog/Types.h>
#include <lingodb/compiler/mlir-support/eval.h>
#include <lingodb/execution/Execution.h>
#include <lingodb/execution/ResultProcessing.h>
#include <lingodb/execution/Timing.h>
#include <lingodb/runtime/RelationHelper.h>
#include <lingodb/scheduler/Scheduler.h>

using json = nlohmann::json;

namespace {

static bool lingodb_eval_initialized = false;
static std::unique_ptr<lingodb::scheduler::SchedulerHandle> global_scheduler;

void EnsureLingoDBInit() {
  if (!lingodb_eval_initialized) {
    lingodb::compiler::support::eval::init();
    lingodb_eval_initialized = true;
  }
  if (!global_scheduler) {
    global_scheduler = lingodb::scheduler::startScheduler();
  }
}

lingodb::catalog::Type ArrowTypeToLingoDBType(
    const std::shared_ptr<arrow::DataType> &arrow_type) {
  using namespace lingodb::catalog;
  switch (arrow_type->id()) {
  case arrow::Type::BOOL:
    return Type::boolean();
  case arrow::Type::INT8:
    return Type::int8();
  case arrow::Type::INT16:
    return Type::int16();
  case arrow::Type::INT32:
    return Type::int32();
  case arrow::Type::INT64:
    return Type::int64();
  case arrow::Type::FLOAT:
    return Type::f32();
  case arrow::Type::DOUBLE:
    return Type::f64();
  case arrow::Type::DECIMAL128: {
    auto dec = std::static_pointer_cast<arrow::Decimal128Type>(arrow_type);
    return Type::decimal(dec->precision(), dec->scale());
  }
  case arrow::Type::DATE32:
    return Type::makeIntType(32, true);
  case arrow::Type::DATE64:
    return Type::makeIntType(64, true);
  case arrow::Type::TIMESTAMP:
    return Type::timestamp();
  case arrow::Type::STRING:
  case arrow::Type::LARGE_STRING:
    return Type::stringType();
  case arrow::Type::FIXED_SIZE_BINARY: {
    auto fsb = std::static_pointer_cast<arrow::FixedSizeBinaryType>(arrow_type);
    return Type::charType(fsb->byte_width());
  }
  default:
    return Type::stringType();
  }
}

} // namespace

namespace {
const std::vector<std::string> kLingoDBTimingColumns = {
    "frontend", "QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerArrow",
    "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen",
    "baselineLowering", "baselineCodeGen", "baselineEmit",
    "executionTime", "total"};
} // namespace

namespace middleware {

void LingoDBAdapter::WriteLingoDBTimingRow(
    const std::unordered_map<std::string, double> &timing) {
  if (timing.empty()) return;

  std::ofstream log_file;
  log_file.open(g_lingodb_compile_log_name, std::ios_base::app);
  for (size_t i = 0; i < kLingoDBTimingColumns.size(); i++) {
    const auto &col = kLingoDBTimingColumns[i];
    auto it = timing.find(col);
    if (it != timing.end()) {
      log_file << std::fixed << std::setprecision(3) << it->second;
    }
    if (i + 1 < kLingoDBTimingColumns.size())
      log_file << ", ";
  }
  log_file << "\n";
  log_file.close();
}

QueryResult LingoDBAdapter::ArrowTableToQueryResult(
    const std::shared_ptr<arrow::Table> &table) {
  QueryResult result;
  if (!table || table->num_columns() == 0) {
    return result;
  }

  result.num_columns = table->num_columns();
  for (int i = 0; i < table->num_columns(); i++) {
    result.column_names.push_back(table->schema()->field(i)->name());
  }

  result.num_rows = static_cast<int>(table->num_rows());
  result.rows.reserve(result.num_rows);

  for (int64_t row = 0; row < table->num_rows(); row++) {
    std::vector<std::string> row_data;
    row_data.reserve(result.num_columns);

    for (int col = 0; col < table->num_columns(); col++) {
      auto chunk_array = table->column(col);
      int64_t offset = row;
      int chunk_idx = 0;
      while (chunk_idx < chunk_array->num_chunks() &&
             offset >= chunk_array->chunk(chunk_idx)->length()) {
        offset -= chunk_array->chunk(chunk_idx)->length();
        chunk_idx++;
      }
      if (chunk_idx >= chunk_array->num_chunks()) {
        row_data.emplace_back("NULL");
        continue;
      }
      auto arr = chunk_array->chunk(chunk_idx);
      if (arr->IsNull(offset)) {
        row_data.emplace_back("NULL");
      } else {
        auto scalar = arr->GetScalar(offset);
        if (scalar.ok()) {
          row_data.push_back((*scalar)->ToString());
        } else {
          row_data.emplace_back("NULL");
        }
      }
    }
    result.rows.push_back(std::move(row_data));
  }

  return result;
}

LingoDBAdapter::LingoDBAdapter(const std::string &db_path) {
  EnsureLingoDBInit();

  if (db_path == ":memory:" || db_path.empty()) {
    session_ = lingodb::runtime::Session::createSession();
  } else {
    session_ = lingodb::runtime::Session::createSession(db_path, true);
  }
}

LingoDBAdapter::~LingoDBAdapter() { CleanUp(); }

void LingoDBAdapter::ParseSQL(const std::string &sql) {
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

std::unique_ptr<ir_sql_converter::AQPStmt> LingoDBAdapter::ConvertPlanToIR() {
  if (parse_tree_.empty()) {
    throw std::runtime_error("No parse tree available. Call ParseSQL first.");
  }

  return ir_sql_converter::ConvertParseTreeToIRWithSchema(parse_tree_,
                                                          subquery_index);
}

void LingoDBAdapter::SetExecutionMode(const std::string &mode) {
  if (mode == "SPEED") {
    exec_mode_ = lingodb::execution::ExecutionMode::SPEED;
  } else if (mode == "BASELINE_SPEED") {
    exec_mode_ = lingodb::execution::ExecutionMode::BASELINE_SPEED;
  } else {
    throw std::runtime_error("[LingoDB] Unknown execution mode: " + mode +
                             " (valid: SPEED, BASELINE_SPEED)");
  }
}

QueryResult LingoDBAdapter::ExecuteSingleSQL(const std::string &sql) {
  auto config =
      lingodb::execution::createQueryExecutionConfig(exec_mode_, true);

  std::shared_ptr<arrow::Table> result_table;
  config->resultProcessor =
      lingodb::execution::createTableRetriever(result_table);

  lingodb::execution::TimingCollector *timing_collector = nullptr;
  if (enable_timing_) {
    auto collector =
        std::make_unique<lingodb::execution::TimingCollector>();
    timing_collector = collector.get();
    config->timingProcessor = std::move(collector);
  }

  auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(
      std::move(config), *session_);
  executer->fromData(sql);
  lingodb::scheduler::awaitEntryTask(
      std::make_unique<lingodb::execution::QueryExecutionTask>(
          std::move(executer)));

  if (enable_timing_ && timing_collector) {
    WriteLingoDBTimingRow(timing_collector->getTiming());
  }

  if (result_table) {
    return ArrowTableToQueryResult(result_table);
  }
  return QueryResult();
}

QueryResult LingoDBAdapter::ExecuteSQL(const std::string &sql) {
  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

  auto result = ExecuteSingleSQL(sql);

  if (enable_timing_) {
    auto exe_time = chrono_toc(&timer, "Execute SQL time is\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (exe_time / 1000.0)
             << ", ";
    log_file.close();
  }
  return result;
}

void LingoDBAdapter::CreateTempTableFromArrow(
    const std::string &table_name, std::shared_ptr<arrow::Table> table) {
  auto catalog = session_->getCatalog();

  // Build CreateTableDef from arrow schema
  lingodb::catalog::CreateTableDef def;
  def.name = table_name;
  for (int i = 0; i < table->num_columns(); i++) {
    auto field = table->schema()->field(i);
    auto lingo_type = ArrowTypeToLingoDBType(field->type());
    def.columns.emplace_back(field->name(), lingo_type, field->nullable());
  }

  auto entry =
      lingodb::catalog::LingoDBTableCatalogEntry::createFromCreateTable(def);
  catalog->insertEntry(entry, true);

  lingodb::runtime::RelationHelper::appendToTable(*session_, table_name, table);

  temp_table_card_[table_name] =
      static_cast<int64_t>(table->num_rows());
}

void LingoDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card) {
  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

  auto config =
      lingodb::execution::createQueryExecutionConfig(exec_mode_, true);

  std::shared_ptr<arrow::Table> result_table;
  config->resultProcessor =
      lingodb::execution::createTableRetriever(result_table);

  lingodb::execution::TimingCollector *timing_collector = nullptr;
  if (enable_timing_) {
    auto collector =
        std::make_unique<lingodb::execution::TimingCollector>();
    timing_collector = collector.get();
    config->timingProcessor = std::move(collector);
  }

  auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(
      std::move(config), *session_);

  std::string select_sql = sql;
  if (!select_sql.empty() && select_sql.back() == ';') {
    select_sql.pop_back();
  }
  executer->fromData(select_sql);
  lingodb::scheduler::awaitEntryTask(
      std::make_unique<lingodb::execution::QueryExecutionTask>(
          std::move(executer)));

  if (enable_timing_) {
    auto execute_sub_sql_time =
        chrono_toc(&timer, "Execute sub-SQL time is\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_sub_sql_time / 1000.0) << ", ";
    log_file.close();
  }

  if (enable_timing_ && timing_collector) {
    WriteLingoDBTimingRow(timing_collector->getTiming());
  }

  if (!result_table) {
    throw std::runtime_error(
        "[LingoDB] ExecuteSQLandCreateTempTable: query returned no results");
  }

  CreateTempTableFromArrow(temp_table_name, result_table);

  if (enable_timing_) {
    auto extra_materialize_time =
        chrono_toc(&timer, "Extra materialize time is\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extra_materialize_time / 1000.0) << ", ";
    log_file.close();
  }

#ifndef NDEBUG
  std::cout << "[LingoDB] Created temp table: " << temp_table_name
            << " (rows=" << temp_table_card_[temp_table_name] << ")"
            << std::endl;
#endif
}

void LingoDBAdapter::CreateTempTable(const std::string &table_name,
                                     const QueryResult &result) {
  // Not used directly — ExecuteSQLandCreateTempTable handles this path
}

void LingoDBAdapter::DropTempTable(const std::string &table_name) {
  auto catalog = session_->getCatalog();
  catalog->removeEntry(table_name);
  // Also remove any associated index
  catalog->removeEntry(table_name + ".pk.hashidx");
  temp_table_card_.erase(table_name);

#ifndef NDEBUG
  std::cout << "[LingoDB] Dropped temp table: " << table_name << std::endl;
#endif
}

bool LingoDBAdapter::TempTableExists(const std::string &table_name) {
  return session_->getCatalog()->getEntry(table_name).has_value();
}

uint64_t
LingoDBAdapter::GetTempTableCardinality(const std::string &temp_table_name) {
  auto it = temp_table_card_.find(temp_table_name);
  if (it != temp_table_card_.end()) {
    return it->second;
  }

  auto catalog = session_->getCatalog();
  auto entry =
      catalog->getTypedEntry<lingodb::catalog::TableCatalogEntry>(
          temp_table_name);
  if (entry.has_value()) {
    return entry.value()->getNumRows();
  }
  return 0;
}

void LingoDBAdapter::SetTempTableCardinality(
    const std::string &temp_table_name, uint64_t cardinality) {
  temp_table_card_[temp_table_name] = static_cast<int64_t>(cardinality);
}

std::pair<double, double>
LingoDBAdapter::GetEstimatedCost(const std::string &sql) {
  double rows = lingodb::execution::estimateQueryRows(*session_, sql);
  if (rows < 0) {
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }
  return {rows, rows};
}

void LingoDBAdapter::CleanUp() {
  parse_tree_.clear();
  session_.reset();
}

void LingoDBAdapter::LoadTablesFromCSV(const std::string &schema_path,
                                       const std::string &csv_dir) {
  if (schema_path.empty()) {
    throw std::runtime_error(
        "[LingoDB] LoadTablesFromCSV: schema_path is required");
  }

  std::ifstream schema_file(schema_path);
  if (!schema_file.is_open()) {
    throw std::runtime_error("[LingoDB] Cannot open schema file: " +
                             schema_path);
  }

  std::string schema_sql((std::istreambuf_iterator<char>(schema_file)),
                          std::istreambuf_iterator<char>());
  schema_file.close();

  // Extract table names from CREATE TABLE statements
  std::vector<std::string> table_names;
  std::string::size_type pos = 0;
  while ((pos = schema_sql.find("CREATE TABLE", pos)) != std::string::npos) {
    auto start = pos + 13;
    while (start < schema_sql.size() && schema_sql[start] == ' ')
      start++;
    auto end = start;
    while (end < schema_sql.size() && schema_sql[end] != ' ' &&
           schema_sql[end] != '(')
      end++;
    if (end > start)
      table_names.push_back(schema_sql.substr(start, end - start));
    pos = end;
  }

  // Execute CREATE TABLE statements (skip COPY, SET, etc.)
  std::istringstream stream(schema_sql);
  std::string statement;
  std::string line;
  while (std::getline(stream, line)) {
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    if (trimmed.empty() || trimmed.substr(0, 2) == "--")
      continue;
    statement += line + "\n";
    if (trimmed.back() == ';') {
      std::string lower_stmt = statement;
      std::transform(lower_stmt.begin(), lower_stmt.end(), lower_stmt.begin(),
                     ::tolower);
      if (lower_stmt.find("create table") != std::string::npos) {
        ExecuteSingleSQL(statement);
      }
      statement.clear();
    }
  }

  // Load CSV data using COPY (same convention as DuckDB adapter)
  if (!csv_dir.empty()) {
    for (const auto &table_name : table_names) {
      std::string csv_path = csv_dir;
      if (!csv_path.empty() && csv_path.back() != '/')
        csv_path += '/';
      csv_path += table_name + ".csv";
      std::string copy_sql = "copy " + table_name + " from '" + csv_path +
                             "' csv escape '\\';\n";
      ExecuteSingleSQL(copy_sql);
    }
  }

  std::cout << "[LingoDB] Loaded " << table_names.size()
            << " tables from CSV: " << csv_dir << std::endl;
}

} // namespace middleware
