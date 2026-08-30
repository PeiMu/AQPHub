#include "adapters/lingodb_adapter.h"
#include "util/util.h"
#ifdef HAVE_LLVM
#include "storage/storage_plan.h"
#endif

#include <fstream>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>

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
    bool has_nulls = table->column(i)->null_count() > 0;
    bool nullable = field->nullable() && has_nulls;
    def.columns.emplace_back(field->name(), lingo_type, nullable);
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

  // Extract table names from CREATE TABLE statements (case-insensitive)
  std::string schema_lower = schema_sql;
  std::transform(schema_lower.begin(), schema_lower.end(), schema_lower.begin(),
                 ::tolower);
  std::vector<std::string> table_names;
  std::string::size_type pos = 0;
  while ((pos = schema_lower.find("create table", pos)) != std::string::npos) {
    auto start = pos + 13;
    while (start < schema_lower.size() && schema_lower[start] == ' ')
      start++;
    auto end = start;
    while (end < schema_lower.size() && schema_lower[end] != ' ' &&
           schema_lower[end] != '(' && schema_lower[end] != '\n' &&
           schema_lower[end] != '\r')
      end++;
    if (end > start)
      table_names.push_back(schema_lower.substr(start, end - start));
    pos = end;
  }

  // Execute CREATE TABLE statements (skip COPY, SET, etc.)
  // Skip tables with unsupported types (e.g. SQL TIME — lingodb hangs on it).
  std::set<std::string> skipped_tables;
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
        // Check for unsupported column types
        bool has_unsupported = false;
        std::istringstream col_stream(lower_stmt);
        std::string col_line;
        while (std::getline(col_stream, col_line)) {
          std::string ct = col_line;
          ct.erase(0, ct.find_first_not_of(" \t"));
          // Match "colname  time  ," or "colname  time  )" patterns
          if (std::regex_search(ct, std::regex("\\btime\\b")) &&
              ct.find("time_sk") == std::string::npos &&
              ct.find("time_id") == std::string::npos &&
              ct.find("time_dim") == std::string::npos &&
              ct.find("timestamp") == std::string::npos &&
              ct.find("create table") == std::string::npos) {
            has_unsupported = true;
            break;
          }
        }
        if (has_unsupported) {
          // Extract table name to skip its CSV load too
          auto tpos = lower_stmt.find("create table");
          if (tpos != std::string::npos) {
            auto ts = tpos + 13;
            while (ts < lower_stmt.size() && lower_stmt[ts] == ' ') ts++;
            auto te = ts;
            while (te < lower_stmt.size() && lower_stmt[te] != ' ' &&
                   lower_stmt[te] != '(') te++;
            if (te > ts)
              skipped_tables.insert(lower_stmt.substr(ts, te - ts));
          }
#ifndef NDEBUG
          std::cerr << "[LingoDB] Skipping table with unsupported type (time)"
                    << std::endl;
#endif
        } else {
          ExecuteSingleSQL(statement);
        }
      }
      statement.clear();
    }
  }

  // Load CSV data using COPY
  if (!csv_dir.empty()) {
    for (const auto &table_name : table_names) {
      if (skipped_tables.count(table_name))
        continue;
      std::string csv_path = csv_dir;
      if (!csv_path.empty() && csv_path.back() != '/')
        csv_path += '/';
      csv_path += table_name + ".csv";
      std::ifstream check(csv_path);
      if (!check.good())
        continue;
      std::string copy_sql = "copy " + table_name + " from '" + csv_path +
                             "' csv escape '\\';\n";
      ExecuteSingleSQL(copy_sql);
    }
  }

#ifndef NDEBUG
  std::cerr << "[LingoDB] Loaded " << table_names.size()
            << " tables from CSV: " << csv_dir << std::endl;
#endif
}

} // namespace middleware

// ===================================================================
// IR-to-MLIR execution + Query-JIT (merged from lingodb_runtime_adapter.cpp)
// ===================================================================
#include "util/util.h"
#ifdef HAVE_LLVM
#include "storage/storage_plan.h"
#endif

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#include <arrow/api.h>
#include <arrow/table.h>

#include <lingodb/catalog/Catalog.h>
#include <lingodb/catalog/Defs.h>
#include <lingodb/catalog/MLIRTypes.h>
#include <lingodb/catalog/TableCatalogEntry.h>
#include <lingodb/catalog/Types.h>
#include <lingodb/compiler/Dialect/DB/IR/DBDialect.h>
#include <lingodb/compiler/Dialect/DB/IR/DBOps.h>
#include <lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h>
#include <lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h>
#include <lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h>
#include <lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h>
#include <lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h>
#include <lingodb/compiler/Dialect/RelAlg/Passes.h>
#include <lingodb/compiler/helper.h>
#include <lingodb/execution/Execution.h>
#include <lingodb/execution/Frontend.h>
#include <lingodb/execution/ResultProcessing.h>
#include <lingodb/execution/Timing.h>
#include <lingodb/runtime/DatasourceRestrictionProperty.h>
#include <lingodb/runtime/RelationHelper.h>
#include <lingodb/runtime/Session.h>
#include <lingodb/scheduler/Scheduler.h>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h>

namespace relalg = lingodb::compiler::dialect::relalg;
namespace db = lingodb::compiler::dialect::db;
namespace tuples = lingodb::compiler::dialect::tuples;
namespace subop = lingodb::compiler::dialect::subop;

namespace middleware {
namespace {

class IRToRelAlgConverter {
public:
  IRToRelAlgConverter(mlir::MLIRContext &ctx,
                      lingodb::catalog::Catalog &catalog,
                      lingodb::runtime::Session *session = nullptr)
      : ctx_(ctx), catalog_(catalog), session_(session),
        colMgr_(ctx.getLoadedDialect<tuples::TupleStreamDialect>()
                    ->getColumnManager()),
        memberMgr_(ctx.getLoadedDialect<subop::SubOperatorDialect>()
                       ->getMemberManager()) {}

  const std::unordered_map<int64_t, ir_sql_converter::SimplestJoin *> &
  GetJoinMap() const { return irJoinMap_; }

  mlir::ModuleOp convert(ir_sql_converter::AQPStmt &ir) {
    auto loc = mlir::UnknownLoc::get(&ctx_);
    mlir::OpBuilder builder(&ctx_);

    auto moduleOp = builder.create<mlir::ModuleOp>(loc);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto *queryInnerBlock = new mlir::Block;
    subop::LocalTableType localTableType;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(queryInnerBlock);

      mlir::Value tree = convertNode(builder, ir);

      llvm::SmallVector<mlir::Attribute> colRefAttrs;
      llvm::SmallVector<mlir::Attribute> colNameAttrs;
      llvm::SmallVector<subop::Member> members;

      for (auto &attr : ir.target_list) {
        auto scope = resolveScope(attr->GetTableIndex());
        auto colName = attr->GetColumnName();
        auto colRef = resolveColRef(scope, colName);
        auto mlirType = colRef.getColumn().type;

        // If this column was remapped by a projection, use the actual
        // stream column ref instead (the projection alias doesn't exist
        // in the tuple stream)
        std::string key = scope + "::" + colName;
        auto mapIt = projToStreamMap_.find(key);
        if (mapIt != projToStreamMap_.end()) {
          colRef = mapIt->second;
          mlirType = colRef.getColumn().type;
        }

        if (!mlirType) {
          throw std::runtime_error(
              "[IRToRelAlg] Column type is null for scope=" + scope +
              " colName=" + colName +
              " tableIdx=" + std::to_string(attr->GetTableIndex()));
        }

        auto tableName = findTableName(attr->GetTableIndex(), ir);
        std::string alias;
        if (!tableName.empty())
          alias = tableName + "_" + std::to_string(attr->GetTableIndex()) +
                  "_" + colName;
        else
          alias = "t" + std::to_string(attr->GetTableIndex()) + "_" + colName;
        if (alias.size() > 63) {
          uint64_t h = 14695981039346656037ULL;
          for (unsigned char c : alias) { h ^= c; h *= 1099511628211ULL; }
          char buf[18];
          snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
          alias = std::string("c_") + buf;
        }

        colRefAttrs.push_back(colRef);
        colNameAttrs.push_back(builder.getStringAttr(alias));
        members.push_back(memberMgr_.createMember(alias, mlirType));
      }

      localTableType = subop::LocalTableType::get(
          &ctx_, subop::StateMembersAttr::get(&ctx_, members),
          builder.getArrayAttr(colNameAttrs));

      auto matOp = builder.create<relalg::MaterializeOp>(
          loc, localTableType, tree, builder.getArrayAttr(colRefAttrs),
          builder.getArrayAttr(colNameAttrs));
      builder.create<relalg::QueryReturnOp>(loc, matOp.getResult());
    }

    auto *funcBlock = new mlir::Block;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(funcBlock);

      auto queryOp = builder.create<relalg::QueryOp>(
          loc, mlir::TypeRange{localTableType}, mlir::ValueRange{});
      queryOp.getQueryOps().getBlocks().clear();
      queryOp.getQueryOps().push_back(queryInnerBlock);

      builder.create<subop::SetResultOp>(loc, 0,
                                         queryOp.getResults()[0]);
      builder.create<mlir::func::ReturnOp>(loc);
    }

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, "main", builder.getFunctionType({}, {}));
    funcOp.getBody().push_back(funcBlock);

    return moduleOp;
  }

private:
  mlir::MLIRContext &ctx_;
  lingodb::catalog::Catalog &catalog_;
  lingodb::runtime::Session *session_;
  tuples::ColumnManager &colMgr_;
  subop::MemberManager &memberMgr_;

  // table_index -> scope name
  std::unordered_map<unsigned int, std::string> scopeMap_;
  // table_index -> catalog table name
  std::unordered_map<unsigned int, std::string> tableNameMap_;
  // Output columns from the last converted aggregate (for projection mapping)
  std::vector<tuples::ColumnRefAttr> lastAggOutputRefs_;
  // Mapping: projection output column ref -> actual stream column ref
  std::unordered_map<std::string, tuples::ColumnRefAttr> projToStreamMap_;
  // table indices that are mark join outputs (always single column "col0")
  std::set<unsigned int> markIndices_;
  // IR join ID → IR SimplestJoin* mapping for MLIR-based build/probe annotation
  std::unordered_map<int64_t, ir_sql_converter::SimplestJoin *> irJoinMap_;
  int64_t nextJoinId_ = 0;

  std::string resolveScope(unsigned int tableIndex) const {
    auto it = scopeMap_.find(tableIndex);
    if (it != scopeMap_.end())
      return it->second;
    return "unknown_" + std::to_string(tableIndex);
  }

  std::string resolveColumnName(unsigned int tableIndex,
                                unsigned int colIndex,
                                const std::string &irName) {
    if (markIndices_.count(tableIndex))
      return "col0";
    if (!irName.empty())
      return irName;
    auto it = tableNameMap_.find(tableIndex);
    if (it == tableNameMap_.end())
      return "col" + std::to_string(colIndex);
    auto entry =
        catalog_.getTypedEntry<lingodb::catalog::TableCatalogEntry>(it->second);
    if (!entry.has_value())
      return "col" + std::to_string(colIndex);
    auto cols = entry.value()->getColumns();
    if (colIndex < cols.size())
      return std::string(cols[colIndex].getColumnName());
    return "col" + std::to_string(colIndex);
  }

  // Resolve column ref with fallback: if the column isn't defined in the
  // primary scope (DuckDB sometimes mis-attributes columns after
  // optimization), search all known scopes for a matching column name.
  tuples::ColumnRefAttr resolveColRef(const std::string &scope,
                                      const std::string &colName) {
    auto ref = colMgr_.createRef(scope, colName);
    if (ref.getColumn().type)
      return ref;
    // Column not in scope — check projToStreamMap_
    std::string key = scope + "::" + colName;
    auto mapIt = projToStreamMap_.find(key);
    if (mapIt != projToStreamMap_.end() && mapIt->second.getColumn().type)
      return mapIt->second;
    // Search all other scopes
    for (auto &[idx, otherScope] : scopeMap_) {
      if (otherScope == scope)
        continue;
      auto otherRef = colMgr_.createRef(otherScope, colName);
      if (otherRef.getColumn().type)
        return otherRef;
    }
    return ref; // return original (null type) if not found anywhere
  }

  mlir::Type getColumnMLIRType(const std::string &tableName,
                               const std::string &colName) {
    auto entry =
        catalog_
            .getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName);
    if (entry.has_value()) {
      for (auto &col : entry.value()->getColumns()) {
        if (col.getColumnName() == colName) {
          auto baseType =
              col.getLogicalType().getMLIRTypeCreator()->createType(&ctx_);
          if (col.getIsNullable())
            return db::NullableType::get(&ctx_, baseType);
          return baseType;
        }
      }
    }
    return db::NullableType::get(&ctx_,
                                 db::StringType::get(&ctx_));
  }

  mlir::Type irTypeToMLIR(const ir_sql_converter::SimplestAttr &attr) {
    switch (attr.GetType()) {
    case ir_sql_converter::SimplestVarType::IntVar: {
      unsigned bw = attr.GetBitWidth();
      if (bw == 0 || bw == 64)
        return db::NullableType::get(&ctx_,
                                     mlir::IntegerType::get(&ctx_, 64));
      return db::NullableType::get(
          &ctx_, mlir::IntegerType::get(&ctx_, bw));
    }
    case ir_sql_converter::SimplestVarType::FloatVar:
      return db::NullableType::get(&ctx_, mlir::Float64Type::get(&ctx_));
    case ir_sql_converter::SimplestVarType::StringVar:
      return db::NullableType::get(&ctx_, db::StringType::get(&ctx_));
    case ir_sql_converter::SimplestVarType::Date:
      return db::NullableType::get(&ctx_,
                                   mlir::IntegerType::get(&ctx_, 32));
    default:
      return db::NullableType::get(&ctx_, db::StringType::get(&ctx_));
    }
  }

  // Resolve MLIR type for a column attribute: catalog first, fallback to IR
  mlir::Type resolveColumnType(const ir_sql_converter::SimplestAttr &attr,
                               const std::string &tableName) {
    auto entry =
        catalog_
            .getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName);
    if (entry.has_value()) {
      for (auto &col : entry.value()->getColumns()) {
        if (col.getColumnName() == attr.GetColumnName()) {
          auto baseType =
              col.getLogicalType().getMLIRTypeCreator()->createType(&ctx_);
          if (col.getIsNullable())
            return db::NullableType::get(&ctx_, baseType);
          return baseType;
        }
      }
    }
    return irTypeToMLIR(attr);
  }

  // Find the table name for a given table_index
  std::string findTableName(unsigned int tableIndex,
                            const ir_sql_converter::AQPStmt &root) {
    // Walk the tree to find a Scan or Chunk with the matching index
    if (auto *scan =
            dynamic_cast<const ir_sql_converter::SimplestScan *>(&root)) {
      if (scan->GetTableIndex() == tableIndex)
        return scan->GetTableName();
    }
    if (auto *chunk =
            dynamic_cast<const ir_sql_converter::SimplestChunk *>(&root)) {
      if (chunk->GetTableIndex() == tableIndex)
        return chunk->GetChunkName();
    }
    for (auto &child : root.children) {
      auto name = findTableName(tableIndex, *child);
      if (!name.empty())
        return name;
    }
    return "";
  }

  // ============== Node Conversion ==============

  mlir::Value convertNode(mlir::OpBuilder &builder,
                          ir_sql_converter::AQPStmt &node) {
    auto loc = builder.getUnknownLoc();
    auto nt = node.GetNodeType();
    using NT = ir_sql_converter::SimplestNodeType;
    switch (nt) {
    case NT::ScanNode:
      return convertScan(builder,
                         static_cast<ir_sql_converter::SimplestScan &>(node));
    case NT::ChunkNode:
      return convertChunk(
          builder, static_cast<ir_sql_converter::SimplestChunk &>(node));
    case NT::FilterNode: {
      auto &filter =
          static_cast<ir_sql_converter::SimplestFilter &>(node);
      // Detect pattern: Filter[mark_col] -> MarkJoin => convert to SemiJoin
      if (isMarkFilterPattern(filter)) {
        return convertMarkFilterAsSemiJoin(builder, filter);
      }
      mlir::Value child = convertNode(builder, *filter.children[0]);
      return convertFilter(builder, filter, child);
    }
    case NT::JoinNode:
      return convertJoin(
          builder, static_cast<ir_sql_converter::SimplestJoin &>(node));
    case NT::CrossProductNode:
      return convertCrossProduct(
          builder,
          static_cast<ir_sql_converter::SimplestCrossProduct &>(node));
    case NT::ProjectionNode: {
      auto &proj =
          static_cast<ir_sql_converter::SimplestProjection &>(node);
      mlir::Value child = convertNode(builder, *proj.children[0]);
      return convertProjection(builder, proj, child);
    }
    case NT::AggregateNode: {
      auto &agg =
          static_cast<ir_sql_converter::SimplestAggregate &>(node);
      mlir::Value child = convertNode(builder, *agg.children[0]);
      return convertAggregate(builder, agg, child);
    }
    case NT::OrderNode: {
      auto &order =
          static_cast<ir_sql_converter::SimplestOrderBy &>(node);
      mlir::Value child = convertNode(builder, *order.children[0]);
      return convertOrderBy(builder, order, child);
    }
    case NT::LimitNode: {
      auto &limit =
          static_cast<ir_sql_converter::SimplestLimit &>(node);
      mlir::Value child = convertNode(builder, *limit.children[0]);
      return convertLimit(builder, limit, child);
    }
    case NT::SortNode: {
      // SimplestSort is also used for ORDER BY in some paths
      mlir::Value child = convertNode(builder, *node.children[0]);
      return child; // pass-through: sort info handled by OrderNode
    }
    case NT::StmtNode:
      // Generic statement wrapper — drill to child
      if (!node.children.empty())
        return convertNode(builder, *node.children[0]);
      throw std::runtime_error(
          "[IRToRelAlg] StmtNode with no children");
    default:
      throw std::runtime_error(
          "[IRToRelAlg] Unsupported IR node type: " +
          std::to_string(static_cast<int>(nt)));
    }
  }

  std::vector<mlir::NamedAttribute> buildAllCatalogColumns(
      mlir::OpBuilder &builder, const std::string &tableName,
      const std::string &scope) {
    std::vector<mlir::NamedAttribute> columns;
    auto entry =
        catalog_
            .getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName);
    if (!entry.has_value())
      throw std::runtime_error("[IRToRelAlg] Table not in catalog: " +
                               tableName);

    auto catalogCols = entry.value()->getColumns();
    for (auto &cc : catalogCols) {
      auto colName = cc.getColumnName();
      auto attrDef = colMgr_.createDef(scope, colName);
      auto baseType =
          cc.getLogicalType().getMLIRTypeCreator()->createType(&ctx_);
      mlir::Type colType =
          cc.getIsNullable()
              ? (mlir::Type)db::NullableType::get(&ctx_, baseType)
              : baseType;
      attrDef.getColumn().type = colType;
      columns.push_back(builder.getNamedAttr(colName, attrDef));
    }
    return columns;
  }

  void createInlineTable(const std::string &tableName,
                         const std::vector<std::string> &contents) {
    // Create a single-column string table in the catalog
    lingodb::catalog::CreateTableDef def;
    def.name = tableName;
    def.columns.emplace_back("col0", lingodb::catalog::Type::stringType(),
                             true);

    auto entry =
        lingodb::catalog::LingoDBTableCatalogEntry::createFromCreateTable(def);
    catalog_.insertEntry(entry, true);

    // Build Arrow table with the contents
    auto strBuilder = std::make_shared<arrow::StringBuilder>();
    for (auto &v : contents) {
      (void)strBuilder->Append(v);
    }
    std::shared_ptr<arrow::Array> arr;
    (void)strBuilder->Finish(&arr);
    auto schema = arrow::schema({arrow::field("col0", arrow::utf8(), true)});
    auto table = arrow::Table::Make(schema, {arr});

    lingodb::runtime::RelationHelper::appendToTable(
        *session_, tableName, table);
  }

  void collectTableIndices(ir_sql_converter::AQPStmt &node,
                           std::set<unsigned int> &indices) {
    using NT = ir_sql_converter::SimplestNodeType;
    auto nt = node.GetNodeType();
    if (nt == NT::ScanNode) {
      indices.insert(
          static_cast<ir_sql_converter::SimplestScan &>(node).GetTableIndex());
    } else if (nt == NT::ChunkNode) {
      indices.insert(
          static_cast<ir_sql_converter::SimplestChunk &>(node).GetTableIndex());
    }
    for (auto &child : node.children) {
      collectTableIndices(*child, indices);
    }
  }

  mlir::Value ensureI1(mlir::OpBuilder &builder, mlir::Value val) {
    if (mlir::isa<db::NullableType>(val.getType()))
      return builder.create<db::DeriveTruth>(builder.getUnknownLoc(), val);
    return val;
  }

  mlir::Value wrapWithSelection(
      mlir::OpBuilder &builder,
      std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> &quals,
      mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    auto selOp = builder.create<relalg::SelectionOp>(
        loc, tuples::TupleStreamType::get(&ctx_), input);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);

    mlir::Value tupleArg = pred->getArgument(0);
    mlir::Value predResult = buildPredicateFromQuals(predBuilder, quals, tupleArg);
    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));

    selOp.getPredicate().push_back(pred);
    return selOp.getResult();
  }

  mlir::Value convertScan(mlir::OpBuilder &builder,
                          ir_sql_converter::SimplestScan &scan) {
    auto loc = builder.getUnknownLoc();
    auto tableName = scan.GetTableName();
    auto tableIndex = scan.GetTableIndex();

    auto scope = colMgr_.getUniqueScope(tableName);
    scopeMap_[tableIndex] = scope;
    tableNameMap_[tableIndex] = tableName;

    auto columns = buildAllCatalogColumns(builder, tableName, scope);

    mlir::Value result = builder.create<relalg::BaseTableOp>(
        loc, tuples::TupleStreamType::get(&ctx_), tableName,
        builder.getDictionaryAttr(columns),
        lingodb::runtime::DatasourceRestrictionProperty{});

    if (!scan.qual_vec.empty()) {
      result = wrapWithSelection(builder, scan.qual_vec, result);
    }
    return result;
  }

  mlir::Value convertChunk(mlir::OpBuilder &builder,
                           ir_sql_converter::SimplestChunk &chunk) {
    auto loc = builder.getUnknownLoc();
    auto chunkName = chunk.GetChunkName();
    auto tableIndex = chunk.GetTableIndex();

    if (chunkName.empty() && !chunk.GetContents().empty()) {
      chunkName = "__inline_" + std::to_string(tableIndex);
      createInlineTable(chunkName, chunk.GetContents());
    }

    auto scope = colMgr_.getUniqueScope(chunkName);
    scopeMap_[tableIndex] = scope;
    tableNameMap_[tableIndex] = chunkName;

    auto columns = buildAllCatalogColumns(builder, chunkName, scope);

    mlir::Value result = builder.create<relalg::BaseTableOp>(
        loc, tuples::TupleStreamType::get(&ctx_), chunkName,
        builder.getDictionaryAttr(columns),
        lingodb::runtime::DatasourceRestrictionProperty{});

    if (!chunk.qual_vec.empty()) {
      result = wrapWithSelection(builder, chunk.qual_vec, result);
    }
    return result;
  }

  bool isMarkFilterPattern(ir_sql_converter::SimplestFilter &filter) {
    if (filter.qual_vec.size() != 1 || filter.children.empty())
      return false;
    auto &qual = filter.qual_vec[0];
    if (qual->GetNodeType() != ir_sql_converter::SimplestNodeType::SingleAttrExprNode)
      return false;
    auto &sae = static_cast<ir_sql_converter::SimplestSingleAttrExpr &>(*qual);
    auto tableIdx = sae.attr->GetTableIndex();
    auto &child = *filter.children[0];
    if (child.GetNodeType() != ir_sql_converter::SimplestNodeType::JoinNode)
      return false;
    auto &join = static_cast<ir_sql_converter::SimplestJoin &>(child);
    return join.GetSimplestJoinType() == ir_sql_converter::SimplestJoinType::Mark
        && join.GetMarkIndex() == tableIdx;
  }

  mlir::Value convertMarkFilterAsSemiJoin(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestFilter &filter) {
    auto loc = builder.getUnknownLoc();
    auto &join = static_cast<ir_sql_converter::SimplestJoin &>(*filter.children[0]);

    mlir::Value left = convertNode(builder, *join.children[0]);
    mlir::Value right = convertNode(builder, *join.children[1]);

    std::set<unsigned int> leftTableIndices, rightTableIndices;
    collectTableIndices(*join.children[0], leftTableIndices);
    collectTableIndices(*join.children[1], rightTableIndices);

    auto semiJoinOp = builder.create<relalg::SemiJoinOp>(
        loc, tuples::TupleStreamType::get(&ctx_), left, right);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);
    mlir::Value tupleArg = pred->getArgument(0);

    std::vector<mlir::Value> conditions;
    llvm::SmallVector<mlir::Attribute> leftHashKeys, rightHashKeys, nullsEqualAttrs;
    bool canUseHash = !join.join_conditions.empty();

    for (auto &jc : join.join_conditions) {
      conditions.push_back(convertVarComparison(predBuilder, *jc, tupleArg));
      if (canUseHash && jc->GetSimplestExprType() == ir_sql_converter::SimplestExprType::Equal) {
        auto &la = *jc->left_attr;
        auto &ra = *jc->right_attr;
        auto lScope = resolveScope(la.GetTableIndex());
        auto rScope = resolveScope(ra.GetTableIndex());
        auto lColName = resolveColumnName(la.GetTableIndex(), la.GetColumnIndex(), la.GetColumnName());
        auto rColName = resolveColumnName(ra.GetTableIndex(), ra.GetColumnIndex(), ra.GetColumnName());
        auto lRef = resolveColRef(lScope, lColName);
        auto rRef = resolveColRef(rScope, rColName);

        bool lIsLeft = leftTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsRight = rightTableIndices.count(ra.GetTableIndex()) > 0;
        bool lIsRight = rightTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsLeft = leftTableIndices.count(ra.GetTableIndex()) > 0;

        if (lIsLeft && rIsRight) {
          leftHashKeys.push_back(lRef);
          rightHashKeys.push_back(rRef);
        } else if (lIsRight && rIsLeft) {
          leftHashKeys.push_back(rRef);
          rightHashKeys.push_back(lRef);
        } else {
          canUseHash = false;
        }
        nullsEqualAttrs.push_back(
            mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx_, 8), 0));
      }
    }

    mlir::Value predResult;
    if (conditions.size() == 1) {
      predResult = conditions[0];
    } else if (conditions.size() > 1) {
      predResult = predBuilder.create<db::AndOp>(loc, conditions);
    } else {
      predResult = predBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }
    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));
    semiJoinOp.getPredicate().push_back(pred);

    if (canUseHash && !leftHashKeys.empty()) {
      semiJoinOp->setAttr("useHashJoin", mlir::UnitAttr::get(&ctx_));
      semiJoinOp->setAttr("leftHash", builder.getArrayAttr(leftHashKeys));
      semiJoinOp->setAttr("rightHash", builder.getArrayAttr(rightHashKeys));
      semiJoinOp->setAttr("nullsEqual", builder.getArrayAttr(nullsEqualAttrs));
    }

    return semiJoinOp.getResult();
  }

  mlir::Value convertFilter(mlir::OpBuilder &builder,
                            ir_sql_converter::SimplestFilter &filter,
                            mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    auto selOp = builder.create<relalg::SelectionOp>(
        loc, tuples::TupleStreamType::get(&ctx_), input);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);

    mlir::Value tupleArg = pred->getArgument(0);
    mlir::Value predResult = buildPredicateFromQuals(predBuilder, filter.qual_vec, tupleArg);
    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));

    selOp.getPredicate().push_back(pred);
    return selOp.getResult();
  }

  mlir::Value convertJoin(mlir::OpBuilder &builder,
                          ir_sql_converter::SimplestJoin &join) {
    auto loc = builder.getUnknownLoc();
    auto joinType = join.GetSimplestJoinType();

    // DuckDB: children[0]=probe, children[1]=build
    // LingoDB: left=build, right=probe
    // Swap to match conventions.
    mlir::Value left = convertNode(builder, *join.children[1]);
    mlir::Value right = convertNode(builder, *join.children[0]);

    std::set<unsigned int> leftTableIndices, rightTableIndices;
    collectTableIndices(*join.children[1], leftTableIndices);
    collectTableIndices(*join.children[0], rightTableIndices);

    // Handle Mark Join (used for IN clauses) — kept as fallback but
    // normally converted to SemiJoin via isMarkFilterPattern
    if (joinType == ir_sql_converter::SimplestJoinType::Mark) {
      auto markIdx = join.GetMarkIndex();
      auto markScope = colMgr_.getUniqueScope("mark");
      scopeMap_[markIdx] = markScope;
      markIndices_.insert(markIdx);
      auto markDef = colMgr_.createDef(markScope, "col0");
      markDef.getColumn().type = builder.getI1Type();

      auto markJoinOp = builder.create<relalg::MarkJoinOp>(
          loc, tuples::TupleStreamType::get(&ctx_), markDef, left, right);

      auto *pred = new mlir::Block;
      pred->addArgument(tuples::TupleType::get(&ctx_), loc);
      mlir::OpBuilder predBuilder(&ctx_);
      predBuilder.setInsertionPointToStart(pred);
      mlir::Value tupleArg = pred->getArgument(0);

      std::vector<mlir::Value> conditions;
      for (auto &jc : join.join_conditions) {
        conditions.push_back(convertVarComparison(predBuilder, *jc, tupleArg));
      }
      mlir::Value predResult;
      if (conditions.size() == 1) {
        predResult = conditions[0];
      } else if (conditions.size() > 1) {
        predResult = predBuilder.create<db::AndOp>(loc, conditions);
      } else {
        predResult = predBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));
      markJoinOp.getPredicate().push_back(pred);

      int64_t jid = nextJoinId_++;
      markJoinOp->setAttr("irJoinId", builder.getI64IntegerAttr(jid));
      irJoinMap_[jid] = &join;

      return markJoinOp.getResult();
    }

    auto joinOp = builder.create<relalg::InnerJoinOp>(
        loc, tuples::TupleStreamType::get(&ctx_), left, right);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);

    mlir::Value tupleArg = pred->getArgument(0);

    // Extract hash keys from equality join conditions
    llvm::SmallVector<mlir::Attribute> leftHashKeys, rightHashKeys, nullsEqualAttrs;
    bool canUseHash = !join.join_conditions.empty();

    // Build join predicate from join_conditions + qual_vec
    std::vector<mlir::Value> conditions;
    for (auto &jc : join.join_conditions) {
      conditions.push_back(convertVarComparison(predBuilder, *jc, tupleArg));

      if (canUseHash && jc->GetSimplestExprType() == ir_sql_converter::SimplestExprType::Equal) {
        auto &la = *jc->left_attr;
        auto &ra = *jc->right_attr;
        auto lScope = resolveScope(la.GetTableIndex());
        auto rScope = resolveScope(ra.GetTableIndex());
        auto lColName = resolveColumnName(la.GetTableIndex(), la.GetColumnIndex(), la.GetColumnName());
        auto rColName = resolveColumnName(ra.GetTableIndex(), ra.GetColumnIndex(), ra.GetColumnName());
        auto lRef = resolveColRef(lScope, lColName);
        auto rRef = resolveColRef(rScope, rColName);

        bool lIsLeft = leftTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsRight = rightTableIndices.count(ra.GetTableIndex()) > 0;
        bool lIsRight = rightTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsLeft = leftTableIndices.count(ra.GetTableIndex()) > 0;

        if (lIsLeft && rIsRight) {
          leftHashKeys.push_back(lRef);
          rightHashKeys.push_back(rRef);
        } else if (lIsRight && rIsLeft) {
          leftHashKeys.push_back(rRef);
          rightHashKeys.push_back(lRef);
        } else {
          canUseHash = false;
        }
        nullsEqualAttrs.push_back(
            mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx_, 8), 0));
      }
    }
    for (auto &q : join.qual_vec) {
      conditions.push_back(convertExpr(predBuilder, *q, tupleArg));
    }

    mlir::Value predResult;
    if (conditions.size() == 1) {
      predResult = conditions[0];
    } else if (conditions.size() > 1) {
      predResult = predBuilder.create<db::AndOp>(loc, conditions);
    } else {
      predResult =
          predBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }

    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));
    joinOp.getPredicate().push_back(pred);

    // Set hash join attributes
    if (canUseHash && !leftHashKeys.empty()) {
      joinOp->setAttr("useHashJoin", mlir::UnitAttr::get(&ctx_));
      joinOp->setAttr("leftHash", builder.getArrayAttr(leftHashKeys));
      joinOp->setAttr("rightHash", builder.getArrayAttr(rightHashKeys));
      joinOp->setAttr("nullsEqual", builder.getArrayAttr(nullsEqualAttrs));
    }

    int64_t jid = nextJoinId_++;
    joinOp->setAttr("irJoinId", builder.getI64IntegerAttr(jid));
    irJoinMap_[jid] = &join;

    return joinOp.getResult();
  }

  mlir::Value convertCrossProduct(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestCrossProduct &cp) {
    auto loc = builder.getUnknownLoc();
    mlir::Value left = convertNode(builder, *cp.children[0]);
    mlir::Value right = convertNode(builder, *cp.children[1]);
    return builder.create<relalg::CrossProductOp>(
        loc, tuples::TupleStreamType::get(&ctx_), left, right);
  }

  mlir::Value convertProjection(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestProjection &proj,
      mlir::Value input) {
    auto loc = builder.getUnknownLoc();

    // First try: resolve all target_list columns directly
    bool allResolved = true;
    llvm::SmallVector<mlir::Attribute> colRefs;
    for (auto &attr : proj.target_list) {
      auto scope = resolveScope(attr->GetTableIndex());
      auto colRef = resolveColRef(scope, attr->GetColumnName());
      if (!colRef.getColumn().type) {
        allResolved = false;
        break;
      }
      colRefs.push_back(colRef);
    }

    if (allResolved) {
      return builder.create<relalg::ProjectionOp>(
          loc, tuples::TupleStreamType::get(&ctx_), relalg::SetSemantic::all,
          input, builder.getArrayAttr(colRefs));
    }

    // Fallback: projection renames columns from child (e.g., aggregate
    // output with empty target_list). Map by column index into child refs.
    // DuckDB may deduplicate aggregates, so multiple projection entries
    // can reference the same child column (via GetColumnIndex()).
    colRefs.clear();
    auto childRefs = lastAggOutputRefs_;
    auto projScope = colMgr_.getUniqueScope("proj");

    for (auto &attr : proj.target_list) {
      scopeMap_[attr->GetTableIndex()] = projScope;
    }

    for (size_t i = 0; i < proj.target_list.size(); i++) {
      auto colIdx = proj.target_list[i]->GetColumnIndex();
      if (colIdx < childRefs.size() && childRefs[colIdx].getColumn().type) {
        auto childRef = childRefs[colIdx];
        auto projColName = proj.target_list[i]->GetColumnName();
        auto projDef = colMgr_.createDef(projScope, projColName);
        projDef.getColumn().type = childRef.getColumn().type;
        colRefs.push_back(childRef);
        std::string key = projScope + "::" + projColName;
        projToStreamMap_[key] = childRef;
      } else {
        auto scope = resolveScope(proj.target_list[i]->GetTableIndex());
        colRefs.push_back(
            resolveColRef(scope,
                          proj.target_list[i]->GetColumnName()));
      }
    }

    return builder.create<relalg::ProjectionOp>(
        loc, tuples::TupleStreamType::get(&ctx_), relalg::SetSemantic::all,
        input, builder.getArrayAttr(colRefs));
  }

  mlir::Value convertAggregate(mlir::OpBuilder &builder,
                               ir_sql_converter::SimplestAggregate &agg,
                               mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    auto tupleStreamTy = tuples::TupleStreamType::get(&ctx_);

    // Group-by column refs
    llvm::SmallVector<mlir::Attribute> groupByRefs;
    for (auto &grp : agg.groups) {
      auto scope = resolveScope(grp->GetTableIndex());
      groupByRefs.push_back(resolveColRef(scope, grp->GetColumnName()));
    }

    // Create a unique scope for aggregate output columns
    auto aggScope = colMgr_.getUniqueScope("agg");

    // Build aggregation function block
    auto *block = new mlir::Block;
    block->addArgument(tupleStreamTy, loc);
    block->addArgument(tuples::TupleType::get(&ctx_), loc);

    mlir::OpBuilder aggrBuilder(&ctx_);
    aggrBuilder.setInsertionPointToStart(block);

    mlir::Value relation = block->getArgument(0);

    std::vector<mlir::Value> createdValues;
    llvm::SmallVector<mlir::Attribute> createdCols;

    auto &aggFns = agg.agg_fns;
    size_t aggIdx = 0;
    for (auto &[aggAttr, aggFnType] : aggFns) {
      std::string outColName;
      if (aggIdx < agg.target_list.size())
        outColName = agg.target_list[aggIdx]->GetColumnName();
      else
        outColName = "agg_" + std::to_string(aggIdx);

      auto outDef = colMgr_.createDef(aggScope, outColName);
      mlir::Value expr;

      if (aggFnType == ir_sql_converter::SimplestAggFnType::CountStar) {
        auto resultType = mlir::IntegerType::get(&ctx_, 64);
        expr = aggrBuilder.create<relalg::CountRowsOp>(loc, resultType,
                                                        relation);
        outDef.getColumn().type = resultType;
      } else {
        auto scope = resolveScope(aggAttr->GetTableIndex());
        auto colRef =
            resolveColRef(scope, aggAttr->GetColumnName());

        relalg::AggrFunc fn;
        switch (aggFnType) {
        case ir_sql_converter::SimplestAggFnType::Min:
          fn = relalg::AggrFunc::min;
          break;
        case ir_sql_converter::SimplestAggFnType::Max:
          fn = relalg::AggrFunc::max;
          break;
        case ir_sql_converter::SimplestAggFnType::Sum:
          fn = relalg::AggrFunc::sum;
          break;
        case ir_sql_converter::SimplestAggFnType::Average:
          fn = relalg::AggrFunc::avg;
          break;
        case ir_sql_converter::SimplestAggFnType::Count:
          fn = relalg::AggrFunc::count;
          break;
        default:
          fn = relalg::AggrFunc::min;
          break;
        }

        // Result type: for MIN/MAX, same as input; for COUNT, i64;
        // for SUM, promote to i64 for integer inputs
        mlir::Type inputType = colRef.getColumn().type;
        mlir::Type baseInputType = inputType;
        if (auto nullTy = mlir::dyn_cast<db::NullableType>(inputType))
          baseInputType = nullTy.getType();

        mlir::Type resultType;
        if (fn == relalg::AggrFunc::count) {
          resultType = mlir::IntegerType::get(&ctx_, 64);
        } else if (fn == relalg::AggrFunc::sum) {
          if (mlir::isa<mlir::IntegerType>(baseInputType))
            resultType = mlir::IntegerType::get(&ctx_, 64);
          else
            resultType = baseInputType;
        } else if (fn == relalg::AggrFunc::avg) {
          resultType = mlir::Float64Type::get(&ctx_);
        } else {
          // min/max: same as input base type
          resultType = baseInputType;
        }

        // Aggregate results are nullable
        auto nullableResultType = db::NullableType::get(&ctx_, resultType);
        expr = aggrBuilder.create<relalg::AggrFuncOp>(
            loc, nullableResultType, fn, relation, colRef);
        outDef.getColumn().type = nullableResultType;
      }

      createdCols.push_back(outDef);
      createdValues.push_back(expr);
      aggIdx++;
    }

    aggrBuilder.create<tuples::ReturnOp>(loc, createdValues);

    auto aggOp = builder.create<relalg::AggregationOp>(
        loc, tupleStreamTy, input, builder.getArrayAttr(groupByRefs),
        builder.getArrayAttr(createdCols));
    aggOp.getAggrFunc().push_back(block);

    lastAggOutputRefs_.clear();
    for (auto &col : createdCols) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(col);
      lastAggOutputRefs_.push_back(
          colMgr_.createRef(&def.getColumn()));
    }

    for (size_t i = 0; i < agg.target_list.size(); i++) {
      scopeMap_[agg.target_list[i]->GetTableIndex()] = aggScope;
    }

    return aggOp.getResult();
  }

  mlir::Value convertOrderBy(mlir::OpBuilder &builder,
                             ir_sql_converter::SimplestOrderBy &orderBy,
                             mlir::Value input) {
    auto loc = builder.getUnknownLoc();

    llvm::SmallVector<mlir::Attribute> sortSpecs;
    for (auto &order : orderBy.orders) {
      auto scope = resolveScope(order.attr->GetTableIndex());
      auto colRef = resolveColRef(scope, order.attr->GetColumnName());

      auto spec = (order.order_type ==
                       ir_sql_converter::SimplestOrderType::Ascending ||
                   order.order_type ==
                       ir_sql_converter::SimplestOrderType::ORDER_DEFAULT)
                      ? relalg::SortSpec::asc
                      : relalg::SortSpec::desc;

      sortSpecs.push_back(
          relalg::SortSpecificationAttr::get(&ctx_, colRef, spec));
    }

    return builder.create<relalg::SortOp>(
        loc, tuples::TupleStreamType::get(&ctx_), input,
        builder.getArrayAttr(sortSpecs));
  }

  mlir::Value convertLimit(mlir::OpBuilder &builder,
                           ir_sql_converter::SimplestLimit &limit,
                           mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    int32_t maxRows = static_cast<int32_t>(limit.limit_val.val);
    return builder.create<relalg::LimitOp>(
        loc, tuples::TupleStreamType::get(&ctx_), maxRows, input);
  }

  // ============== Expression Conversion ==============

  mlir::Value buildPredicateFromQuals(
      mlir::OpBuilder &builder,
      std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> &quals,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    if (quals.empty())
      return builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);

    std::vector<mlir::Value> conditions;
    for (auto &q : quals) {
      conditions.push_back(convertExpr(builder, *q, tuple));
    }

    if (conditions.size() == 1)
      return conditions[0];
    return builder.create<db::AndOp>(loc, conditions);
  }

  mlir::Value convertExpr(mlir::OpBuilder &builder,
                          ir_sql_converter::AQPExpr &expr,
                          mlir::Value tuple) {
    using NT = ir_sql_converter::SimplestNodeType;
    auto nt = expr.GetNodeType();

    switch (nt) {
    case NT::VarComparisonNode:
      return convertVarComparison(
          builder,
          static_cast<ir_sql_converter::SimplestVarComparison &>(expr),
          tuple);
    case NT::VarConstComparisonNode:
      return convertVarConstComparison(
          builder,
          static_cast<ir_sql_converter::SimplestVarConstComparison &>(
              expr),
          tuple);
    case NT::LogicalExprNode:
      return convertLogicalExpr(
          builder,
          static_cast<ir_sql_converter::SimplestLogicalExpr &>(expr),
          tuple);
    case NT::IsNullExprNode:
      return convertIsNullExpr(
          builder,
          static_cast<ir_sql_converter::SimplestIsNullExpr &>(expr),
          tuple);
    case NT::InExprNode:
      return convertInExpr(
          builder,
          static_cast<ir_sql_converter::SimplestInExpr &>(expr), tuple);
    case NT::ArithExprNode:
      return convertArithExpr(
          builder,
          static_cast<ir_sql_converter::SimplestArithExpr &>(expr),
          tuple);
    case NT::CastExprNode:
      return convertCastExpr(
          builder,
          static_cast<ir_sql_converter::SimplestCastExpr &>(expr),
          tuple);
    case NT::SingleAttrExprNode: {
      auto &sae =
          static_cast<ir_sql_converter::SimplestSingleAttrExpr &>(expr);
      auto &attr = *sae.attr;
      auto scope = resolveScope(attr.GetTableIndex());
      auto colName = resolveColumnName(attr.GetTableIndex(),
                                       attr.GetColumnIndex(),
                                       attr.GetColumnName());
      auto colRef = resolveColRef(scope, colName);
      return builder.create<tuples::GetColumnOp>(
          builder.getUnknownLoc(), colRef.getColumn().type, colRef, tuple);
    }
    case NT::ExprNode: {
      auto &gen =
          static_cast<ir_sql_converter::SimplestGeneralComparison &>(expr);
      auto leftVal = convertExpr(builder, *gen.left_expr, tuple);
      auto rightVal = convertExpr(builder, *gen.right_expr, tuple);
      auto pred = mapCmpPredicate(gen.GetSimplestExprType());
      return builder.create<db::CmpOp>(builder.getUnknownLoc(), pred,
                                       leftVal, rightVal);
    }
    case NT::FunctionExprNodeType: {
      auto &fn =
          static_cast<ir_sql_converter::SimplestFunctionExpr &>(expr);
      auto loc = builder.getUnknownLoc();
      std::vector<mlir::Value> args;
      for (auto &arg : fn.args)
        args.push_back(convertExpr(builder, *arg, tuple));
      auto lingoName = mapFunctionName(fn.fn_name);
      auto resType = inferFunctionResultType(builder, lingoName, args);
      return builder.create<db::RuntimeCall>(loc, resType, lingoName,
                                             mlir::ValueRange(args))
          .getRes();
    }
    case NT::ConstVarNode: {
      auto &ce =
          static_cast<ir_sql_converter::SimplestConstExpr &>(expr);
      auto loc = builder.getUnknownLoc();
      return createConstantUntyped(builder, *ce.value);
    }
    default:
      throw std::runtime_error(
          "[IRToRelAlg] Unsupported expression type: " +
          std::to_string(static_cast<int>(nt)));
    }
  }

  mlir::Value convertVarComparison(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestVarComparison &cmp,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &leftAttr = *cmp.left_attr;
    auto &rightAttr = *cmp.right_attr;

    auto leftScope = resolveScope(leftAttr.GetTableIndex());
    auto rightScope = resolveScope(rightAttr.GetTableIndex());

    auto leftColName = resolveColumnName(leftAttr.GetTableIndex(),
                                         leftAttr.GetColumnIndex(),
                                         leftAttr.GetColumnName());
    auto rightColName = resolveColumnName(rightAttr.GetTableIndex(),
                                          rightAttr.GetColumnIndex(),
                                          rightAttr.GetColumnName());
    auto leftRef = resolveColRef(leftScope, leftColName);
    auto rightRef = resolveColRef(rightScope, rightColName);

    if (!leftRef.getColumn().type) {
      throw std::runtime_error(
          "[IRToRelAlg] VarComparison: left column type is null for scope=" +
          leftScope + " col=" + leftColName +
          " tableIdx=" + std::to_string(leftAttr.GetTableIndex()) +
          " colIdx=" + std::to_string(leftAttr.GetColumnIndex()));
    }
    if (!rightRef.getColumn().type) {
      throw std::runtime_error(
          "[IRToRelAlg] VarComparison: right column type is null for scope=" +
          rightScope + " col=" + rightColName +
          " tableIdx=" + std::to_string(rightAttr.GetTableIndex()) +
          " colIdx=" + std::to_string(rightAttr.GetColumnIndex()));
    }

    auto leftVal = builder.create<tuples::GetColumnOp>(
        loc, leftRef.getColumn().type, leftRef, tuple);
    auto rightVal = builder.create<tuples::GetColumnOp>(
        loc, rightRef.getColumn().type, rightRef, tuple);

    auto pred = mapCmpPredicate(cmp.GetSimplestExprType());
    return builder.create<db::CmpOp>(loc, pred, leftVal, rightVal);
  }

  mlir::Value convertVarConstComparison(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestVarConstComparison &cmp,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &attr = *cmp.attr;
    auto &constVar = *cmp.const_var;

    auto scope = resolveScope(attr.GetTableIndex());
    auto colName = resolveColumnName(attr.GetTableIndex(),
                                     attr.GetColumnIndex(),
                                     attr.GetColumnName());
    auto colRef = resolveColRef(scope, colName);
    if (!colRef.getColumn().type) {
      throw std::runtime_error(
          "[IRToRelAlg] VarConstComparison: column type is null for scope=" +
          scope + " col=" + colName +
          " tableIdx=" + std::to_string(attr.GetTableIndex()) +
          " colIdx=" + std::to_string(attr.GetColumnIndex()));
    }
    auto colVal = builder.create<tuples::GetColumnOp>(
        loc, colRef.getColumn().type, colRef, tuple);

    auto colType = colRef.getColumn().type;
    auto baseColType = colType;
    if (auto nullTy = mlir::dyn_cast<db::NullableType>(colType))
      baseColType = nullTy.getType();

    auto constVal = createConstant(builder, constVar, baseColType);

    auto exprType = cmp.GetSimplestExprType();
    if (exprType == ir_sql_converter::SimplestExprType::TextLike ||
        exprType == ir_sql_converter::SimplestExprType::Text_Not_Like) {
      bool isNullable = mlir::isa<db::NullableType>(colType) ||
                        mlir::isa<db::NullableType>(constVal.getType());
      mlir::Type resType =
          isNullable
              ? (mlir::Type)db::NullableType::get(&ctx_,
                                                  builder.getI1Type())
              : (mlir::Type)builder.getI1Type();
      auto like = builder.create<db::RuntimeCall>(
          loc, resType, "Like", mlir::ValueRange({colVal, constVal}));
      mlir::Value result = like.getRes();
      if (exprType == ir_sql_converter::SimplestExprType::Text_Not_Like)
        result = builder.create<db::NotOp>(loc, result);
      return result;
    }

    auto pred = mapCmpPredicate(exprType);
    return builder.create<db::CmpOp>(loc, pred, colVal, constVal);
  }

  mlir::Value convertLogicalExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestLogicalExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();

    auto op = expr.GetLogicalOp();
    if (op == ir_sql_converter::SimplestLogicalOp::LogicalNot) {
      auto rightVal = convertExpr(builder, *expr.right_expr, tuple);
      return builder.create<db::NotOp>(loc, rightVal);
    }

    auto leftVal = convertExpr(builder, *expr.left_expr, tuple);
    auto rightVal = convertExpr(builder, *expr.right_expr, tuple);

    std::vector<mlir::Value> operands = {leftVal, rightVal};

    if (op == ir_sql_converter::SimplestLogicalOp::LogicalAnd)
      return builder.create<db::AndOp>(loc, operands);
    else
      return builder.create<db::OrOp>(loc, operands);
  }

  mlir::Value convertIsNullExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestIsNullExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &attr = *expr.attr;

    auto scope = resolveScope(attr.GetTableIndex());
    auto colRef = resolveColRef(scope, attr.GetColumnName());
    auto colType = colRef.getColumn().type;

    bool isNonNullType = !colType.isa<db::NullableType>();
    if (isNonNullType) {
      // Column is non-nullable: IS NULL is always false, IS NOT NULL always true
      bool isNotNull = expr.GetSimplestExprType() ==
                       ir_sql_converter::SimplestExprType::NonNullType;
      return builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(), isNotNull ? 1 : 0));
    }

    auto colVal = builder.create<tuples::GetColumnOp>(
        loc, colType, colRef, tuple);
    auto isNull = builder.create<db::IsNullOp>(loc, colVal);

    if (expr.GetSimplestExprType() ==
        ir_sql_converter::SimplestExprType::NonNullType)
      return builder.create<db::NotOp>(loc, isNull);
    return isNull;
  }

  mlir::Value convertInExpr(mlir::OpBuilder &builder,
                            ir_sql_converter::SimplestInExpr &expr,
                            mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &attr = *expr.attr;

    auto scope = resolveScope(attr.GetTableIndex());
    auto colRef = resolveColRef(scope, attr.GetColumnName());
    auto colVal = builder.create<tuples::GetColumnOp>(
        loc, colRef.getColumn().type, colRef, tuple);

    auto colType = colRef.getColumn().type;
    auto baseColType = colType;
    if (auto nullTy = mlir::dyn_cast<db::NullableType>(colType))
      baseColType = nullTy.getType();

    std::vector<mlir::Value> values;
    values.push_back(colVal);
    for (auto &v : expr.values) {
      values.push_back(createConstant(builder, *v, baseColType));
    }

    mlir::Value result = builder.create<db::OneOfOp>(loc, values);
    if (expr.negated)
      result = builder.create<db::NotOp>(loc, result);
    return result;
  }

  mlir::Value convertArithExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestArithExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto leftVal = convertExpr(builder, *expr.left, tuple);
    auto rightVal = convertExpr(builder, *expr.right, tuple);

    switch (expr.arith_op) {
    case ir_sql_converter::SimplestArithOp::ArithAdd:
      return builder.create<db::AddOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithSub:
      return builder.create<db::SubOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithMul:
      return builder.create<db::MulOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithDiv:
      return builder.create<db::DivOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithMod:
      return builder.create<db::ModOp>(loc, leftVal, rightVal);
    default:
      throw std::runtime_error("[IRToRelAlg] Unsupported arith op");
    }
  }

  mlir::Value convertCastExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestCastExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto child = convertExpr(builder, *expr.child, tuple);
    return child;
  }

  // ============== Helpers ==============

  std::string mapFunctionName(const std::string &duckName) {
    static const std::unordered_map<std::string, std::string> map = {
        {"substring", "Substring"},
        {"substr", "Substring"},
        {"upper", "ToUpper"},
        {"lower", "ToLower"},
        {"length", "StringLength"},
        {"contains", "Contains"},
        {"replace", "Replace"},
        {"regexp_replace", "RegexpReplace"},
        {"concat", "Concatenate"},
        {"abs", "AbsInt"},
    };
    auto it = map.find(duckName);
    if (it != map.end())
      return it->second;
    std::string pascal;
    bool capitalize = true;
    for (char c : duckName) {
      if (c == '_') {
        capitalize = true;
      } else {
        pascal += capitalize ? (char)toupper(c) : c;
        capitalize = false;
      }
    }
    return pascal;
  }

  mlir::Type inferFunctionResultType(mlir::OpBuilder &builder,
                                     const std::string &fn,
                                     const std::vector<mlir::Value> &args) {
    if (fn == "StringLength")
      return builder.getI64Type();
    if (!args.empty())
      return args[0].getType();
    return db::StringType::get(&ctx_);
  }

  mlir::Value createConstantUntyped(mlir::OpBuilder &builder,
                                    ir_sql_converter::SimplestConstVar &cv) {
    auto loc = builder.getUnknownLoc();
    switch (cv.GetType()) {
    case ir_sql_converter::SimplestVarType::IntVar:
      return builder.create<db::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cv.GetIntValue()));
    case ir_sql_converter::SimplestVarType::FloatVar:
      return builder.create<db::ConstantOp>(
          loc, mlir::Float64Type::get(&ctx_),
          builder.getF64FloatAttr(cv.GetFloatValue()));
    case ir_sql_converter::SimplestVarType::StringVar:
      return builder.create<db::ConstantOp>(
          loc, db::StringType::get(&ctx_),
          builder.getStringAttr(cv.GetStringValue()));
    default:
      return builder.create<db::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
    }
  }

  mlir::Value createConstant(mlir::OpBuilder &builder,
                             ir_sql_converter::SimplestConstVar &constVar,
                             mlir::Type targetType) {
    auto loc = builder.getUnknownLoc();

    switch (constVar.GetType()) {
    case ir_sql_converter::SimplestVarType::IntVar: {
      auto val = constVar.GetIntValue();
      return builder.create<db::ConstantOp>(
          loc, targetType, builder.getI32IntegerAttr(val));
    }
    case ir_sql_converter::SimplestVarType::FloatVar: {
      auto val = constVar.GetFloatValue();
      return builder.create<db::ConstantOp>(
          loc, targetType, builder.getStringAttr(std::to_string(val)));
    }
    case ir_sql_converter::SimplestVarType::StringVar: {
      auto val = constVar.GetStringValue();
      return builder.create<db::ConstantOp>(
          loc, targetType, builder.getStringAttr(val));
    }
    case ir_sql_converter::SimplestVarType::StringVarArr: {
      auto vals = constVar.GetStringVecValue();
      if (!vals.empty())
        return builder.create<db::ConstantOp>(
            loc, targetType, builder.getStringAttr(vals[0]));
      return builder.create<db::NullOp>(
          loc, db::NullableType::get(&ctx_, targetType));
    }
    default:
      return builder.create<db::NullOp>(
          loc, db::NullableType::get(&ctx_, targetType));
    }
  }

  db::DBCmpPredicate
  mapCmpPredicate(ir_sql_converter::SimplestExprType exprType) {
    using ET = ir_sql_converter::SimplestExprType;
    switch (exprType) {
    case ET::Equal:
      return db::DBCmpPredicate::eq;
    case ET::NotEqual:
      return db::DBCmpPredicate::neq;
    case ET::LessThan:
      return db::DBCmpPredicate::lt;
    case ET::GreaterThan:
      return db::DBCmpPredicate::gt;
    case ET::LessEqual:
      return db::DBCmpPredicate::lte;
    case ET::GreaterEqual:
      return db::DBCmpPredicate::gte;
    default:
      return db::DBCmpPredicate::eq;
    }
  }
};

class DecomposeInnerJoinsOnly
    : public mlir::PassWrapper<DecomposeInnerJoinsOnly,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeInnerJoinsOnly)

  llvm::StringRef getArgument() const override {
    return "relalg-decompose-inner-joins-only";
  }

  void runOnOperation() override {
    using namespace lingodb::compiler::dialect;
    std::vector<relalg::InnerJoinOp> joins;
    getOperation().walk(
        [&](relalg::InnerJoinOp op) { joins.push_back(op); });
    for (auto innerJoin : joins) {
      mlir::OpBuilder builder(innerJoin);
      auto cp = builder.create<relalg::CrossProductOp>(
          innerJoin->getLoc(), innerJoin.getLeft(), innerJoin.getRight());
      auto sel = builder.create<relalg::SelectionOp>(innerJoin->getLoc(),
                                                      cp);
      sel.getPredicate().getBlocks().splice(sel.getPredicate().end(),
                                            innerJoin.getPredicate().getBlocks());
      innerJoin.replaceAllUsesWith(sel.getResult());
      innerJoin->erase();
    }
  }
};

class DecomposeSelectionsOnly
    : public mlir::PassWrapper<DecomposeSelectionsOnly,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeSelectionsOnly)

  llvm::StringRef getArgument() const override {
    return "relalg-decompose-selections-only";
  }

private:
  void getConditionVals(mlir::Value v,
                        std::vector<mlir::Value> &values) {
    using namespace lingodb::compiler::dialect;
    if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(v.getDefiningOp())) {
      for (auto operand : andOp.getVals())
        getConditionVals(operand, values);
    } else {
      values.push_back(v);
    }
  }

  void decomposeSelection(mlir::Value v, mlir::Value &tree) {
    using namespace lingodb::compiler::dialect;
    auto currentSel =
        mlir::dyn_cast_or_null<relalg::SelectionOp>(v.getDefiningOp()->getParentOp());
    mlir::OpBuilder builder(currentSel);
    mlir::IRMapping mapping;
    auto newSel = builder.create<relalg::SelectionOp>(
        currentSel->getLoc(),
        tuples::TupleStreamType::get(builder.getContext()), tree);
    tree = newSel;
    newSel.initPredicate();
    mapping.map(currentSel.getPredicateArgument(),
                newSel.getPredicateArgument());
    builder.setInsertionPointToStart(&newSel.getPredicate().front());
    relalg::detail::inlineOpIntoBlock(
        v.getDefiningOp(), v.getDefiningOp()->getParentOp(),
        &newSel.getPredicateBlock(), mapping);
    builder.create<tuples::ReturnOp>(currentSel->getLoc(),
                                     mapping.lookup(v));
    auto *terminator = newSel.getLambdaBlock().getTerminator();
    terminator->erase();
  }

  void decomposeMap(lingodb::compiler::dialect::relalg::MapOp currentMap,
                    mlir::Value &tree) {
    using namespace lingodb::compiler::dialect;
    auto *terminator = currentMap.getPredicate().front().getTerminator();
    if (auto returnOp =
            mlir::dyn_cast_or_null<tuples::ReturnOp>(terminator)) {
      assert(returnOp.getResults().size() ==
             currentMap.getComputedCols().size());
      auto computedValRange = returnOp.getResults();
      for (size_t i = 0; i < computedValRange.size(); i++) {
        mlir::OpBuilder builder(currentMap);
        mlir::IRMapping mapping;
        auto currentAttr = mlir::cast<tuples::ColumnDefAttr>(
            currentMap.getComputedCols()[i]);
        mlir::Value currentVal = computedValRange[i];
        auto newMap = builder.create<relalg::MapOp>(
            currentMap->getLoc(),
            tuples::TupleStreamType::get(builder.getContext()), tree,
            builder.getArrayAttr({currentAttr}));
        tree = newMap;
        newMap.getPredicate().push_back(new mlir::Block);
        newMap.getPredicate().addArgument(
            tuples::TupleType::get(builder.getContext()),
            currentMap->getLoc());
        builder.setInsertionPointToStart(&newMap.getPredicate().front());
        auto ret1 = builder.create<tuples::ReturnOp>(currentMap->getLoc());
        mapping.map(currentMap.getLambdaArgument(),
                    newMap.getLambdaArgument());
        relalg::detail::inlineOpIntoBlock(
            currentVal.getDefiningOp(),
            currentVal.getDefiningOp()->getParentOp(),
            &newMap.getLambdaBlock(), mapping);
        builder.create<tuples::ReturnOp>(currentMap->getLoc(),
                                         mapping.lookup(currentVal));
        ret1->erase();
      }
    }
  }

  void runOnOperation() override {
    using namespace lingodb::compiler::dialect;
    std::vector<mlir::Operation *> toErase;

    // Decompose multi-condition SelectionOps into individual ones
    getOperation().walk([&](relalg::SelectionOp op) {
      auto *terminator = op.getRegion().front().getTerminator();
      mlir::Value val = op.getRel();
      if (terminator->getNumOperands() > 0) {
        std::vector<mlir::Value> conditionValues;
        getConditionVals(terminator->getOperand(0), conditionValues);
        if (conditionValues.size() > 1) {
          for (auto condition : conditionValues)
            decomposeSelection(condition, val);
          op.replaceAllUsesWith(val);
          toErase.push_back(op.getOperation());
        }
      } else {
        op.replaceAllUsesWith(val);
        toErase.push_back(op.getOperation());
      }
    });

    getOperation().walk([&](relalg::MapOp op) {
      mlir::Value val = op.getRel();
      if (op.getComputedCols().size() == 1)
        return;
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(
              op.getRegion().front().getTerminator())) {
        bool anyRelalgOp = false;
        for (auto v : returnOp.getResults()) {
          if (auto *defOp = v.getDefiningOp()) {
            if (defOp->getDialect() ==
                op.getContext()
                    ->getLoadedDialect<relalg::RelAlgDialect>()) {
              anyRelalgOp = true;
              break;
            }
          }
        }
        if (!anyRelalgOp)
          return;
      }
      decomposeMap(op, val);
      op.replaceAllUsesWith(val);
      toErase.push_back(op.getOperation());
    });

    for (auto *op : toErase)
      op->erase();
  }
};

class PartialQueryOptimizer : public lingodb::execution::QueryOptimizer {
  lingodb::catalog::Catalog &catalog_;

public:
  explicit PartialQueryOptimizer(lingodb::catalog::Catalog &catalog)
      : catalog_(catalog) {}

  void optimize(mlir::ModuleOp &module) override {
    auto start = std::chrono::high_resolution_clock::now();

    mlir::PassManager pm(module.getContext());
    using namespace lingodb::compiler::dialect;
    // Full LingoDB optimizer pipeline with:
    // - DecomposeLambdasPass replaced by DecomposeSelectionsOnly
    //   (InnerJoin→CrossProduct causes nested-loop joins and catastrophic
    //   performance, so we keep InnerJoinOps with hash join attributes)
    // - OptimizeJoinOrderPass skipped (preserve DuckDB's join ordering)
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createSimplifyAggregationsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createExtractNestedOperatorsPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(lingodb::compiler::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createInferNotNullConditionsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<DecomposeSelectionsOnly>());
    pm.addPass(lingodb::compiler::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createImplicitToExplicitJoinsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createInferNotNullConditionsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<DecomposeSelectionsOnly>());
    pm.addNestedPass<mlir::func::FuncOp>(relalg::createPushdownPass());
    pm.addNestedPass<mlir::func::FuncOp>(relalg::createUnnestingPass());
    pm.addNestedPass<mlir::func::FuncOp>(relalg::createColumnFoldingPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<DecomposeSelectionsOnly>());
    pm.addNestedPass<mlir::func::FuncOp>(relalg::createPushdownPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createAttachMetaDataPass(catalog_));
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createReduceGroupByKeysPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createExpandTransitiveEqualities());
    // OptimizeJoinOrderPass deliberately skipped
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createCombinePredicatesPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createEliminateNullableTypesPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createOptimizeImplementationsPass());
    pm.addNestedPass<mlir::func::FuncOp>(relalg::createDetachMetaDataPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        relalg::createCommonSubtreeEliminationPass());
    pm.addPass(lingodb::compiler::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(relalg::createIntroduceTmpPass());

    if (mlir::failed(pm.run(module)))
      llvm::errs() << "PartialQueryOptimizer: pass pipeline failed\n";

    auto end = std::chrono::high_resolution_clock::now();
    timing["QOpt"] = std::chrono::duration_cast<std::chrono::microseconds>(
                         end - start)
                         .count() /
                     1000.0;
#ifndef NDEBUG
    std::cerr << "[LingoDB-Runtime] Optimized MLIR:" << std::endl;
    module.print(llvm::errs());
    std::cerr << std::endl;
#endif
  }
};

class IRFrontend : public lingodb::execution::Frontend {
  ir_sql_converter::AQPStmt *ir_;
  lingodb::runtime::Session *session_;
  mlir::MLIRContext *context_ = nullptr;
  mlir::OwningOpRef<mlir::ModuleOp> module_;

public:
  IRFrontend(ir_sql_converter::AQPStmt *ir, lingodb::runtime::Session *session)
      : ir_(ir), session_(session) {}

  void setContext(mlir::MLIRContext *context) override {
    context_ = context;
  }

  void loadFromString(std::string) override {
    IRToRelAlgConverter converter(*context_, *catalog, session_);
    auto moduleOp = converter.convert(*ir_);
#ifndef NDEBUG
    std::cerr << "[LingoDB-Runtime] Generated MLIR:" << std::endl;
    moduleOp.print(llvm::errs());
    std::cerr << std::endl;
#endif
    module_ = moduleOp;
  }

  void loadFromFile(std::string) override {}

  mlir::ModuleOp *getModule() override {
    assert(module_);
    return module_.operator->();
  }
};

} // anonymous namespace

// ============== LingoDBAdapter ==============


static bool hasCrossProduct(const ir_sql_converter::AQPStmt *node) {
  if (!node) return false;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::CrossProductNode)
    return true;
  for (auto &child : node->children)
    if (hasCrossProduct(child.get())) return true;
  return false;
}

void LingoDBAdapter::ExecuteIRandCreateTempTable(
    ir_sql_converter::AQPStmt &ir, const std::string &temp_table_name,
    bool update_temp_card) {
  if (hasCrossProduct(&ir)) {
    std::string sql = GenerateSQL(ir, subquery_index++);
    ExecuteSQLandCreateTempTable(sql, temp_table_name, update_temp_card);
    return;
  }

#ifdef HAVE_LLVM
  if (query_jit_ && qjit_storage_plan_ && qjit_storage_plan_->IsLoaded()) {
    std::chrono::high_resolution_clock::time_point jit_timer;
    if (enable_timing_)
      jit_timer = chrono_tic();

    AnnotateBuildSidesFromMLIR(ir);
    auto analysis = qjit::AnalyzeQueryJit(ir, temp_table_name);
    if (analysis.accepted) {
      auto compiled = TryCompileQueryJit(ir, analysis, temp_table_name);
      if (compiled) {
        if (enable_timing_) {
          auto compile_us = chrono_toc(&jit_timer, "qjit compile\n", false);
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << (compile_us / 1000.0) << ", ";
          log_file.close();
        }

        int64_t rows = ExecuteQueryJit(*compiled, temp_table_name);
        if (rows >= 0) {
          if (enable_timing_) {
            auto exec_us = chrono_toc(&jit_timer, "qjit exec\n", false);
            std::ofstream log_file;
            log_file.open(g_timing_log_name, std::ios_base::app);
            log_file << std::fixed << std::setprecision(3)
                     << (exec_us / 1000.0) << ", ";
            log_file.close();
          }

          MaterializeQjitTempToLingoDB(temp_table_name);
          temp_table_card_[temp_table_name] = rows;

          if (enable_timing_) {
            auto mat_us = chrono_toc(&jit_timer, "qjit materialize\n", false);
            std::ofstream log_file;
            log_file.open(g_timing_log_name, std::ios_base::app);
            log_file << std::fixed << std::setprecision(3)
                     << (mat_us / 1000.0) << ", ";
            log_file.close();
          }
#ifndef NDEBUG
          std::cout << "[LingoDB-Runtime] Created temp table (qjit): "
                    << temp_table_name << " (rows=" << rows << ")"
                    << std::endl;
#endif
          return;
        }
      }
    }
#ifndef NDEBUG
    std::cerr << "[LingoDB] QJIT rejected/failed for "
              << temp_table_name << ", falling back to SQL\n";
#endif
    // JIT rejected — fall back to SQL (not MLIR, which may hang due to
    // PartialQueryOptimizer skipping join reorder on temp tables)
    std::string sql = GenerateSQL(ir, subquery_index++);
    ExecuteSQLandCreateTempTable(sql, temp_table_name, update_temp_card);
    return;
  }
#endif // HAVE_LLVM

  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

  // MLIR fallback (only reached when query-JIT is not enabled)
  auto config =
      lingodb::execution::createQueryExecutionConfig(exec_mode_, false);
  config->queryOptimizer = std::make_unique<PartialQueryOptimizer>(*session_->getCatalog());
  config->frontend = std::make_unique<IRFrontend>(&ir, session_.get());

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
  executer->fromData("");
  lingodb::scheduler::awaitEntryTask(
      std::make_unique<lingodb::execution::QueryExecutionTask>(
          std::move(executer)));

  if (enable_timing_) {
    auto execute_time =
        chrono_toc(&timer, "Execute MLIR time\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_time / 1000.0) << ", ";
    log_file.close();
  }

  if (enable_timing_ && timing_collector) {
    WriteLingoDBTimingRow(timing_collector->getTiming());
  }

  if (!result_table) {
    std::string sql = GenerateSQL(ir, subquery_index++);
    ExecuteSQLandCreateTempTable(sql, temp_table_name, update_temp_card);
    return;
  }

  // 3. Store result as temp table
  CreateTempTableFromArrow(temp_table_name, result_table);

  if (enable_timing_) {
    auto materialize_time =
        chrono_toc(&timer, "Materialize temp table time\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (materialize_time / 1000.0) << ", ";
    log_file.close();
  }

#ifndef NDEBUG
  std::cout << "[LingoDB-Runtime] Created temp table: " << temp_table_name
            << " (rows=" << temp_table_card_[temp_table_name] << ")"
            << std::endl;
#endif
}

QueryResult LingoDBAdapter::ExecuteIRQuery(
    ir_sql_converter::AQPStmt &ir) {
  if (hasCrossProduct(&ir)) {
    std::string sql = GenerateSQL(ir, subquery_index++);
    auto result = ExecuteSQL(sql);
    if (enable_timing_) {
      WriteLingoDBTimingRow({});
    }
    return result;
  }

#ifdef HAVE_LLVM
  if (query_jit_ && qjit_storage_plan_ && qjit_storage_plan_->IsLoaded()) {
    std::chrono::high_resolution_clock::time_point jit_timer;
    if (enable_timing_)
      jit_timer = chrono_tic();

    AnnotateBuildSidesFromMLIR(ir);
    auto analysis = qjit::AnalyzeQueryJit(ir, "final");
    if (analysis.accepted) {
      auto compiled = TryCompileQueryJit(ir, analysis, "final");
      if (compiled) {
        if (enable_timing_) {
          auto compile_us = chrono_toc(&jit_timer, "qjit compile final\n", false);
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << (compile_us / 1000.0) << ", ";
          log_file.close();
        }

        auto result = ExecuteQueryJitFinal(*compiled);

        if (enable_timing_) {
          auto exec_us = chrono_toc(&jit_timer, "qjit exec final\n", false);
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << (exec_us / 1000.0) << ", ";
          log_file.close();
        }

#ifndef NDEBUG
        std::cerr << "[LingoDB-Runtime] Final query executed via QJIT ("
                  << result.num_rows << " rows)\n";
#endif
        return result;
      }
    }
#ifndef NDEBUG
    std::cerr << "[LingoDB] QJIT rejected/failed for final query, "
              << "falling back to SQL\n";
#endif
    std::string sql = GenerateSQL(ir, subquery_index++);
    return ExecuteSQL(sql);
  }
#endif // HAVE_LLVM

  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

  // MLIR fallback (only reached when query-JIT is not enabled)
  auto config =
      lingodb::execution::createQueryExecutionConfig(exec_mode_, false);
  config->queryOptimizer = std::make_unique<PartialQueryOptimizer>(*session_->getCatalog());
  config->frontend = std::make_unique<IRFrontend>(&ir, session_.get());

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
  executer->fromData("");
  lingodb::scheduler::awaitEntryTask(
      std::make_unique<lingodb::execution::QueryExecutionTask>(
          std::move(executer)));

  if (enable_timing_) {
    auto execute_time =
        chrono_toc(&timer, "Execute MLIR time\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_time / 1000.0) << ", ";
    log_file.close();
  }

  if (enable_timing_ && timing_collector) {
    WriteLingoDBTimingRow(timing_collector->getTiming());
  }

  if (result_table) {
    return ArrowTableToQueryResult(result_table);
  }
  // MLIR execution produced no result — fall back to SQL
  std::string sql = GenerateSQL(ir, subquery_index++);
  return ExecuteSQL(sql);
}

} // namespace middleware

#ifdef HAVE_LLVM

#include "jit/aqp_jit_abi.h"
#include "qjit/qjit_annotate.h"
#include "qjit/query_jit_abi.h"
#include "qjit/query_jit_runtime.h"
#include "qjit/query_jit_steps.h"
#include "storage/storage_plan.h"

namespace {
static aqp_jit::SimdISA ResolveSimdISA(uint32_t flags) {
  if (flags & AQP_JIT_SIMD)
    return aqp_jit::SimdISA::AUTO;
  switch (flags & AQP_JIT_SIMD_MASK) {
  case AQP_JIT_SIMD_OFF:   return aqp_jit::SimdISA::OFF;
  case AQP_JIT_SIMD_SSE2:  return aqp_jit::SimdISA::SSE2;
  case AQP_JIT_SIMD_AVX:   return aqp_jit::SimdISA::AVX;
  case AQP_JIT_SIMD_AVX2:  return aqp_jit::SimdISA::AVX2;
  case AQP_JIT_SIMD_AVX512:return aqp_jit::SimdISA::AVX512;
  case AQP_JIT_SIMD_AUTO:  return aqp_jit::SimdISA::AUTO;
  }
  return aqp_jit::SimdISA::OFF;
}
} // namespace

namespace middleware {

void LingoDBAdapter::AnnotateBuildSidesByCard(
    ::ir_sql_converter::AQPStmt &ir) {
  ::middleware::AnnotateUnannotatedJoinsByCard(ir, qjit_storage_plan_, temp_table_card_);
}

uint64_t LingoDBAdapter::EstimateIRCard(
    const ::ir_sql_converter::AQPStmt &ir) const {
  return ::middleware::EstimateSubtreeCard(&ir, qjit_storage_plan_, temp_table_card_);
}

void LingoDBAdapter::AnnotateBuildSidesFromMLIR(
    ::ir_sql_converter::AQPStmt &ir) {
  if (!session_ || !session_->getCatalog()) {
    AnnotateBuildSidesByCard(ir);
    return;
  }

  auto &catalog = *session_->getCatalog();
  auto *context = new mlir::MLIRContext();
  lingodb::execution::initializeContext(*context, false);

  IRToRelAlgConverter converter(*context, catalog, session_.get());
  mlir::ModuleOp moduleOp = converter.convert(ir);

  PartialQueryOptimizer optimizer(catalog);
  optimizer.optimize(moduleOp);

  auto &joinMap = converter.GetJoinMap();

  // Walk InnerJoinOp: MLIR right = IR children[0] (due to swap in convertJoin).
  // LinGo-DB convention: right operand = build side (materialized into MultiMap).
  // So IR build child = 0.
  moduleOp.walk([&](relalg::InnerJoinOp joinOp) {
    auto addrAttr = joinOp->getAttrOfType<mlir::IntegerAttr>("irJoinId");
    if (!addrAttr) return;
    int64_t jid = addrAttr.getValue().getSExtValue();
    auto it = joinMap.find(jid);
    if (it == joinMap.end()) return;
    ir_sql_converter::SimplestJoin *irJoin = it->second;
    if (irJoin->GetBuildChild() != -1) return;
    // MLIR right = build = IR children[0]
    irJoin->SetBuildChild(0);
  });

  // Walk MarkJoinOp: same convention, but check reverseSides.
  moduleOp.walk([&](relalg::MarkJoinOp joinOp) {
    auto addrAttr = joinOp->getAttrOfType<mlir::IntegerAttr>("irJoinId");
    if (!addrAttr) return;
    int64_t jid = addrAttr.getValue().getSExtValue();
    auto it = joinMap.find(jid);
    if (it == joinMap.end()) return;
    ir_sql_converter::SimplestJoin *irJoin = it->second;
    if (irJoin->GetBuildChild() != -1) return;
    bool reversed = joinOp->hasAttr("reverseSides");
    // Without reverse: right=build=IR children[0]. With reverse: left=build=IR children[1].
    irJoin->SetBuildChild(reversed ? 1 : 0);
  });

  moduleOp.erase();
  delete context;

  // Fallback: annotate any joins not covered by MLIR walk
  AnnotateBuildSidesByCard(ir);
}

void LingoDBAdapter::EnsureJITCompiler() {
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
}

void LingoDBAdapter::RegisterQjitRuntimeSymbols(
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

bool LingoDBAdapter::BuildOutputDescsFromIR(
    const qjit::QjitQueryPlan &plan, QjitCompiled &compiled,
    std::string &reason) {
  const qjit::QjitStep &last = plan.steps.back();

  std::unordered_map<unsigned int, std::string> table_names;
  if (plan.ir)
    ::middleware::CollectTableNames(*plan.ir, table_names);

  auto out_name = [&](size_t i) -> std::string {
    if (plan.ir && i < plan.ir->target_list.size() &&
        plan.ir->target_list[i]) {
      return ::middleware::IrColumnAlias(*plan.ir->target_list[i], table_names);
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
      } else if (cell.fn == qjit::QjitAggFn::Average) {
        dt = AQP_DTYPE_DOUBLE;
      } else if (cell.arg.dtype == AQP_DTYPE_INT32) {
        dt = AQP_DTYPE_INT32;
      } else if (cell.arg.dtype == AQP_DTYPE_INT64) {
        dt = AQP_DTYPE_INT64;
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
      qjit::QjitAggDType adt;
      if (cell.fn == qjit::QjitAggFn::Average)
        adt = (cell.has_arg && cell.arg.dtype == AQP_DTYPE_VARCHAR)
                  ? qjit::QjitAggDType::Str
                  : (cell.has_arg && (cell.arg.dtype == AQP_DTYPE_DOUBLE ||
                                      cell.arg.dtype == AQP_DTYPE_FLOAT))
                        ? qjit::QjitAggDType::F64
                        : qjit::QjitAggDType::I64;
      else
        adt = (cell.has_arg && cell.arg.dtype == AQP_DTYPE_VARCHAR)
                  ? qjit::QjitAggDType::Str
                  : qjit::QjitAggDType::I64;
      compiled.agg_descs.push_back({cell.fn, adt});
    }
  } else {
    for (size_t i = 0; i < last.outputs.size(); i++) {
      int32_t dt = last.outputs[i].dtype;
      if (dt != AQP_DTYPE_INT32 && dt != AQP_DTYPE_INT64 &&
          dt != AQP_DTYPE_VARCHAR && dt != AQP_DTYPE_DOUBLE &&
          dt != AQP_DTYPE_FLOAT) {
        reason = "output:unsupported-dtype";
        return false;
      }
      compiled.out_descs.push_back({dt, out_name(i)});
    }
  }
  if (plan.ir) {
    std::vector<int32_t> sem;
    std::vector<std::string> names;
    ::middleware::IrTargetListToDtypes(*plan.ir, sem, names);
    for (size_t i = 0; i < compiled.out_descs.size() && i < sem.size(); i++) {
      if (sem[i] == AQP_DTYPE_DATE &&
          compiled.out_descs[i].dtype == AQP_DTYPE_INT32)
        compiled.out_descs[i].dtype = AQP_DTYPE_DATE;
    }
  }
  return true;
}

bool LingoDBAdapter::ResolveQjitSources(
    const qjit::QjitQueryPlan &plan, QjitCompiled &compiled,
    std::string &reason) {
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded()) {
    reason = "no-storage-plan";
    return false;
  }
  if (!qjit_executor_)
    qjit_executor_ = std::make_unique<qjit::QjitExecutor>(
        query_jit_threads_,
        query_jit_morsel_ > 0 ? query_jit_morsel_ : 0);
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

std::unique_ptr<LingoDBAdapter::QjitCompiled>
LingoDBAdapter::TryCompileQueryJit(
    ::ir_sql_converter::AQPStmt &ir,
    const qjit::QjitAnalysisResult &analysis, const std::string &label) {
  auto fallback = [&](const std::string &reason)
      -> std::unique_ptr<QjitCompiled> {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT-LDB] fallback:%s label=%s\n", reason.c_str(),
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

  EnsureJITCompiler();
  jit_compiler_->ResetModules();

  compiled->fn =
      jit_compiler_->CompileQuerySteps(plan, &compiled->params_buf);
  compiled->replay_cache_key = jit_compiler_->LastCacheKey();
  compiled->replay_fn_name = jit_compiler_->LastEntryName();
  if (!compiled->fn)
    return fallback("compile-failed");

  return compiled;
}

int64_t LingoDBAdapter::ExecuteQueryJit(
    QjitCompiled &compiled, const std::string &temp_table_name) {
  auto qtable = std::make_unique<qjit::QjitTable>(
      compiled.out_descs, qjit_executor_->NumWorkers());
  int64_t rows = qjit_executor_->Run(
      reinterpret_cast<QjitQueryFn>(compiled.fn), compiled.srcs,
      compiled.ht_tuple_sizes, compiled.agg_descs, compiled.agg_output_cells,
      *qtable, compiled.ht_key0_offsets, compiled.params_buf);
  if (rows >= 0) {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT-LDB] exec label=%s rows=%lld\n",
            temp_table_name.c_str(), (long long)rows);
#endif
    qjit_temps_[temp_table_name] = std::move(qtable);
  } else {
    fprintf(stderr, "[AQP-QJIT-LDB] fallback:run-error(rc=%lld) label=%s\n",
            (long long)rows, temp_table_name.c_str());
  }
  return rows;
}

QueryResult
LingoDBAdapter::ExecuteQueryJitFinal(QjitCompiled &compiled) {
  qjit::QjitTable qtable(compiled.out_descs, qjit_executor_->NumWorkers());
  int64_t rows = qjit_executor_->Run(
      reinterpret_cast<QjitQueryFn>(compiled.fn), compiled.srcs,
      compiled.ht_tuple_sizes, compiled.agg_descs, compiled.agg_output_cells,
      qtable, compiled.ht_key0_offsets, compiled.params_buf);
  QueryResult result;
  if (rows < 0) {
    fprintf(stderr,
            "[AQP-QJIT-LDB] fallback:run-error(rc=%lld) label=result\n",
            (long long)rows);
    return result;
  }
#ifndef NDEBUG
  fprintf(stderr, "[AQP-QJIT-LDB] exec label=result rows=%lld\n",
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
      } else if (qtable.Col(col).dtype == AQP_DTYPE_DOUBLE ||
                 qtable.Col(col).dtype == AQP_DTYPE_FLOAT) {
        row_data.push_back(std::to_string(qtable.GetF64(col, r)));
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

void LingoDBAdapter::MaterializeQjitTempToLingoDB(
    const std::string &temp_table_name) {
  auto it = qjit_temps_.find(temp_table_name);
  if (it == qjit_temps_.end() || !it->second)
    return;

  const qjit::QjitTable &qtable = *it->second;
  uint64_t num_rows = qtable.NumRows();

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  fields.reserve(qtable.NumCols());
  arrays.reserve(qtable.NumCols());

  for (size_t c = 0; c < qtable.NumCols(); c++) {
    const auto &col = qtable.Col(c);
    if (col.dtype == AQP_DTYPE_INT32 || col.dtype == AQP_DTYPE_DATE) {
      arrow::Int32Builder builder;
      (void)builder.Reserve(num_rows);
      for (uint64_t r = 0; r < num_rows; r++) {
        if (!qtable.ValueValid(c, r))
          (void)builder.AppendNull();
        else
          (void)builder.Append(qtable.GetI32(c, r));
      }
      std::shared_ptr<arrow::Array> arr;
      (void)builder.Finish(&arr);
      fields.push_back(arrow::field(col.name, arrow::int32()));
      arrays.push_back(std::move(arr));
    } else if (col.dtype == AQP_DTYPE_INT64) {
      arrow::Int64Builder builder;
      (void)builder.Reserve(num_rows);
      for (uint64_t r = 0; r < num_rows; r++) {
        if (!qtable.ValueValid(c, r))
          (void)builder.AppendNull();
        else
          (void)builder.Append(qtable.GetI64(c, r));
      }
      std::shared_ptr<arrow::Array> arr;
      (void)builder.Finish(&arr);
      fields.push_back(arrow::field(col.name, arrow::int64()));
      arrays.push_back(std::move(arr));
    } else {
      arrow::StringBuilder builder;
      (void)builder.Reserve(num_rows);
      for (uint64_t r = 0; r < num_rows; r++) {
        if (!qtable.ValueValid(c, r)) {
          (void)builder.AppendNull();
        } else {
          QjitString s = qtable.GetStr(c, r);
          (void)builder.Append(qjit::StringData(s),
                               (int32_t)qjit::StringLen(s));
        }
      }
      std::shared_ptr<arrow::Array> arr;
      (void)builder.Finish(&arr);
      fields.push_back(arrow::field(col.name, arrow::utf8()));
      arrays.push_back(std::move(arr));
    }
  }

  // LinGo-DB's LingoDBTable rejects tables > 1M rows when a column's
  // validity buffer is null (no nulls). Force a validity bitmap on all
  // columns to avoid the "too many nulls" error.
  for (size_t c = 0; c < arrays.size(); c++) {
    if (arrays[c]->null_count() == 0 && num_rows > 0) {
      auto data = arrays[c]->data()->Copy();
      if (!data->buffers[0]) {
        auto buf_res = arrow::AllocateBitmap(num_rows);
        if (buf_res.ok()) {
          auto buf = std::move(*buf_res);
          memset(buf->mutable_data(), 0xFF, buf->size());
          data->buffers[0] = std::move(buf);
          data->null_count = 0;
          arrays[c] = arrow::MakeArray(data);
        }
      }
    }
  }

  auto schema = arrow::schema(fields);
  auto table = arrow::Table::Make(schema, arrays, (int64_t)num_rows);
  CreateTempTableFromArrow(temp_table_name, table);
}

#endif // HAVE_LLVM

} // namespace middleware
