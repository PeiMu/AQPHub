/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#include "adapters/duckdb_adapter.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef HAVE_LLVM
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#endif

namespace middleware {

#if IN_MEM_TMP_TABLE
// Anonymous namespace for table function internal state
namespace {

// Bind data: holds pointer to collection and override cardinality
struct TempCollectionFunctionData : public duckdb::FunctionData {
  duckdb::ColumnDataCollection *collection = nullptr;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
    auto result = duckdb::make_uniq<TempCollectionFunctionData>();
    result->collection = collection;
    result->has_override_cardinality = has_override_cardinality;
    result->override_cardinality = override_cardinality;
    return std::move(result);
  }

  bool Equals(const duckdb::FunctionData &other_p) const override {
    auto &other = other_p.Cast<TempCollectionFunctionData>();
    return collection == other.collection;
  }
};

// Global state: holds scan state for ColumnDataCollection
struct TempCollectionGlobalState : public duckdb::GlobalTableFunctionState {
  duckdb::ColumnDataScanState scan_state;
  bool initialized = false;
};

} // anonymous namespace
#endif

DuckDBAdapter::DuckDBAdapter(const std::string &db_path) {
  db = std::make_unique<duckdb::DuckDB>(db_path);
  conn = std::make_unique<duckdb::Connection>(*db);
#if IN_MEM_TMP_TABLE
  RegisterTempCollectionScan();
#endif
}

DuckDBAdapter::~DuckDBAdapter() { CleanUp(); }

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::RegisterTempCollectionScan() {
  auto context = GetClientContext();

  // Create the table function
  duckdb::TableFunction func(
      "scan_temp_collection", {duckdb::LogicalType::VARCHAR},
      TempCollectionScanFunc, TempCollectionBind, TempCollectionInitGlobal);
  func.cardinality = TempCollectionCardinality;
  func.function_info =
      duckdb::make_shared_ptr<TempCollectionScanInfo>(&temp_collections_);

  // Register the table function in the catalog
  duckdb::CreateTableFunctionInfo info(func);
  auto &catalog = duckdb::Catalog::GetSystemCatalog(*context);
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }
  catalog.CreateTableFunction(*context, info);
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }

  // Register the replacement scan
  auto &db_config = duckdb::DBConfig::GetConfig(*context);
  auto scan_data =
      duckdb::make_uniq<TempCollectionScanData>(&temp_collections_);
  db_config.replacement_scans.emplace_back(TempCollectionReplacementScan,
                                           std::move(scan_data));
}

// Table function callbacks
duckdb::unique_ptr<duckdb::FunctionData> DuckDBAdapter::TempCollectionBind(
    duckdb::ClientContext &context, duckdb::TableFunctionBindInput &input,
    duckdb::vector<duckdb::LogicalType> &return_types,
    duckdb::vector<duckdb::string> &names) {

  auto &info = input.info->Cast<TempCollectionScanInfo>();
  auto table_name = input.inputs[0].GetValue<duckdb::string>();

  auto it = info.temp_collections->find(table_name);
  if (it == info.temp_collections->end()) {
    throw duckdb::BinderException("Temp collection '%s' not found", table_name);
  }

  auto &stored = it->second;
  return_types = stored.collection->Types();
  for (auto &col_name : stored.column_names) {
    names.push_back(col_name);
  }

  auto result = duckdb::make_uniq<TempCollectionFunctionData>();
  result->collection = stored.collection.get();
  result->has_override_cardinality = stored.has_override_cardinality;
  result->override_cardinality = stored.override_cardinality;
  return std::move(result);
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
DuckDBAdapter::TempCollectionInitGlobal(duckdb::ClientContext &context,
                                        duckdb::TableFunctionInitInput &input) {
  return duckdb::make_uniq<TempCollectionGlobalState>();
}

void DuckDBAdapter::TempCollectionScanFunc(duckdb::ClientContext &context,
                                           duckdb::TableFunctionInput &data,
                                           duckdb::DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<TempCollectionFunctionData>();
  auto &state = data.global_state->Cast<TempCollectionGlobalState>();

  if (!state.initialized) {
    bind_data.collection->InitializeScan(state.scan_state);
    state.initialized = true;
  }

  bind_data.collection->Scan(state.scan_state, output);
}

duckdb::unique_ptr<duckdb::NodeStatistics>
DuckDBAdapter::TempCollectionCardinality(
    duckdb::ClientContext &context, const duckdb::FunctionData *bind_data) {
  auto &data = bind_data->Cast<TempCollectionFunctionData>();
  duckdb::idx_t cardinality;
  if (data.has_override_cardinality) {
    cardinality = data.override_cardinality;
  } else {
    cardinality = data.collection->Count();
  }
  return duckdb::make_uniq<duckdb::NodeStatistics>(cardinality, cardinality);
}

// Replacement scan callback
duckdb::unique_ptr<duckdb::TableRef>
DuckDBAdapter::TempCollectionReplacementScan(
    duckdb::ClientContext &context, duckdb::ReplacementScanInput &input,
    duckdb::optional_ptr<duckdb::ReplacementScanData> data) {

  auto &scan_data = data->Cast<TempCollectionScanData>();
  const auto &table_name = input.table_name;
  if (scan_data.temp_collections->find(table_name) ==
      scan_data.temp_collections->end()) {
    return nullptr;
  }

  auto table_ref = duckdb::make_uniq<duckdb::TableFunctionRef>();
  duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
  children.push_back(
      duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(table_name)));
  table_ref->function = duckdb::make_uniq<duckdb::FunctionExpression>(
      "scan_temp_collection", std::move(children));
  table_ref->alias = table_name;
  return std::move(table_ref);
}

#endif

void DuckDBAdapter::ParseSQL(const std::string &sql) {
  auto context = GetClientContext();

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  duckdb::Parser parser(context->GetParserOptions());
  parser.ParseQuery(sql);

  if (parser.statements.empty()) {
    throw std::runtime_error("No statements found!");
  }

  if (duckdb::StatementType::SELECT_STATEMENT != parser.statements[0]->type) {
    throw std::runtime_error("Only SELECT queries supported!");
  }

  planner = std::make_unique<duckdb::Planner>(*context);
  planner->CreatePlan(std::move(parser.statements[0]));

  if (!planner->plan) {
    throw std::runtime_error("Failed to create logical plan");
  }

  plan = std::move(planner->plan);

  // Commit transaction if we started one
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

void DuckDBAdapter::FilterOptimize() {
  auto context = GetClientContext();

  if (!plan) {
    throw std::runtime_error("Cannot optimize null plan");
  }

  // Check if optimization is enabled and required
  if (!plan->RequireOptimizer()) {
    std::cout << "[DuckDB] Plan does not require optimization" << std::endl;
    return;
  }

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  if (!planner || !planner->binder) {
    throw std::runtime_error("Binder not available. Call ParseSQL first.");
  }

  // Create optimizer and run PreOptimize
  duckdb::Optimizer optimizer(*planner->binder, *context);
  auto optimized_plan = optimizer.FilterOptimize(std::move(plan));

  // Store the optimized plan
  plan = std::move(optimized_plan);

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

void DuckDBAdapter::PostOptimizePlan() {
  auto context = GetClientContext();

  if (!plan) {
    throw std::runtime_error("Cannot optimize null plan");
  }

  // Check if optimization is enabled and required
  if (!plan->RequireOptimizer()) {
    std::cout << "[DuckDB] Plan does not require optimization" << std::endl;
    return;
  }

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  if (!planner || !planner->binder) {
    throw std::runtime_error("Binder not available. Call ParseSQL first.");
  }

  // Create optimizer and run PreOptimize
  duckdb::Optimizer optimizer(*planner->binder, *context);
  auto optimized_plan = optimizer.PostOptimize(std::move(plan));

  // Store the optimized plan
  plan = std::move(optimized_plan);

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

void *DuckDBAdapter::GetLogicalPlan() {
  return static_cast<void *>(plan.get());
}

std::unique_ptr<ir_sql_converter::AQPStmt>
DuckDBAdapter::ConvertPlanToIR() {
  auto context = GetClientContext();
  auto logical_plan = std::move(plan);
  auto ir = ir_sql_converter::ConvertDuckDBPlanToIR(
      *planner->binder, *context, logical_plan.get(), intermediate_table_map,
      false, &chunk_col_names_);

  return std::move(ir);
}

QueryResult DuckDBAdapter::ExecuteSQL(const std::string &sql) {
  QueryResult result;

#ifdef HAVE_LLVM
  std::cerr << "[AQP-JIT-TRACE] ExecuteSQL: jit_pending_ir_=" << (void*)jit_pending_ir_ << "\n";
  // If JIT pending IR is set, use Prepare path so we can walk the physical
  // plan and register compiled filters before execution.
  if (jit_pending_ir_) {
    auto prepared = conn->Prepare(sql);
    if (!prepared->HasError() && prepared->data && prepared->data->physical_plan) {
      jit_compiler_.reset(); // fresh compiler per query execution
      RegisterJITFilters(prepared->data->physical_plan->Root(), *jit_pending_ir_);
    }
    jit_pending_ir_ = nullptr;
    duckdb::vector<duckdb::Value> bound;
    auto duckdb_result = prepared->Execute(bound, false);
    if (duckdb_result->HasError())
      throw std::runtime_error("Query failed: " + duckdb_result->GetError());
    result.num_columns = duckdb_result->ColumnCount();
    for (size_t i = 0; i < result.num_columns; i++)
      result.column_names.push_back(duckdb_result->ColumnName(i));
    result.num_rows = 0;
    while (true) {
      auto chunk = duckdb_result->Fetch();
      if (!chunk || 0 == chunk->size()) break;
      for (size_t row = 0; row < chunk->size(); row++) {
        std::vector<std::string> row_data;
        for (size_t col = 0; col < result.num_columns; col++)
          row_data.push_back(chunk->GetValue(col, row).ToString());
        result.rows.push_back(row_data);
        result.num_rows++;
      }
    }
    return result;
  }
#endif

  auto duckdb_result = conn->Query(sql);
  if (duckdb_result->HasError()) {
    throw std::runtime_error("Query failed: " + duckdb_result->GetError());
  }
  //  auto intermediate_results = std::move(duckdb_result->Collection());

  // Get columns
  result.num_columns = duckdb_result->ColumnCount();
  for (size_t i = 0; i < result.num_columns; i++) {
    result.column_names.push_back(duckdb_result->ColumnName(i));
  }

  // Get rows
  result.num_rows = 0;
  while (true) {
    auto chunk = duckdb_result->Fetch();
    if (!chunk || 0 == chunk->size())
      break;

    for (size_t row = 0; row < chunk->size(); row++) {
      std::vector<std::string> row_data;
      row_data.reserve(result.num_columns);
      for (size_t col = 0; col < result.num_columns; col++) {
        row_data.push_back(chunk->GetValue(col, row).ToString());
      }
      result.rows.push_back(std::move(row_data));
      result.num_rows++;
    }
  }
  return result;
}

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card, bool enable_timing) {
  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing)
    timer = chrono_tic();
  auto prepared = conn->Prepare(sql);
  if (prepared->HasError()) {
    throw std::runtime_error("[DuckDB] Prepare failed: " +
                             prepared->GetError());
  }

#ifdef HAVE_LLVM
  // JIT: compile filters from the pending IR and register before execution.
  if (jit_pending_ir_ && prepared->data && prepared->data->physical_plan) {
    jit_compiler_.reset(); // fresh compiler per sub-plan execution
    RegisterJITFilters(prepared->data->physical_plan->Root(), *jit_pending_ir_);
    jit_pending_ir_ = nullptr;
  }
#endif

  duckdb::vector<duckdb::Value> bound_values;
  auto subquery_result = prepared->ExecuteRow(bound_values, false);
  if (enable_timing) {
    auto execute_sub_sql_time =
        chrono_toc(&timer, "Execute sub-SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_sub_sql_time / 1000.0) << ", ";
    log_file.close();
  }

  int64_t chunk_size = subquery_result->Count();
  auto data_chunk_index = planner->binder->GenerateTableIndex();

  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;

  // Build column names (same dedup logic as before)
  temp_table_types = subquery_result->Types();
  auto &result_names = prepared->GetNames();
  duckdb::case_insensitive_set_t used_column_names;
  std::vector<std::string> column_names;
  for (duckdb::idx_t i = 0; i < temp_table_types.size(); i++) {
    std::string column_name =
        (i < result_names.size() && !result_names[i].empty())
            ? result_names[i]
            : "col_" + std::to_string(i);

    // Handle duplicate column names
    std::string unique_column_name = column_name;
    duckdb::idx_t suffix = 1;
    while (used_column_names.count(unique_column_name) > 0) {
      unique_column_name = column_name + "_" + std::to_string(suffix);
      suffix++;
    }
    used_column_names.insert(unique_column_name);
    column_names.push_back(unique_column_name);
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  unique_column_name);
  }

  // Store the ColumnDataCollection in temp_collections_ (zero-copy)
  chunk_col_names_[data_chunk_index] = column_names;
  StoredTempResult stored;
  stored.collection = std::move(subquery_result);
  stored.column_names = std::move(column_names);
  temp_collections_[temp_table_name] = std::move(stored);

  temp_table_card_.emplace(temp_table_name, chunk_size);

  if (enable_timing) {
    auto extra_materialize_time =
        chrono_toc(&timer, "Extra materialize time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extra_materialize_time / 1000.0) << ", ";
    log_file.close();
  }
}
#else
void DuckDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card, bool enable_timing) {
  auto prepared = conn->Prepare(sql);
  if (prepared->HasError()) {
    throw std::runtime_error("[DuckDB] Prepare failed: " +
                             prepared->GetError());
  }
  duckdb::vector<duckdb::Value> bound_values;
  auto subquery_result = prepared->ExecuteRow(bound_values, false);
  int64_t chunk_size = subquery_result->Count();
  auto data_chunk_index = planner->binder->GenerateTableIndex();

  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;

  auto context = GetClientContext();
  // create a table from data chunk
  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  auto &types = subquery_result->Types();
  auto info = duckdb::make_uniq<duckdb::CreateTableInfo>(
      TEMP_CATALOG, DEFAULT_SCHEMA, temp_table_name);
  info->temporary = true;
  info->on_conflict = duckdb::OnCreateConflict::ERROR_ON_CONFLICT;

  // Use actual column names from SQL result (matches alias convention)
  auto &result_names = prepared->GetNames();
  duckdb::case_insensitive_set_t used_column_names;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    std::string column_name =
        (i < result_names.size() && !result_names[i].empty())
            ? result_names[i]
            : "col_" + std::to_string(i);

    // Handle duplicate column names
    std::string unique_column_name = column_name;
    duckdb::idx_t suffix = 1;
    while (used_column_names.count(unique_column_name) > 0) {
      unique_column_name = column_name + "_" + std::to_string(suffix);
      suffix++;
    }
    used_column_names.insert(unique_column_name);
    info->columns.AddColumn(
        duckdb::ColumnDefinition(unique_column_name, types[i]));
    chunk_col_names_[data_chunk_index].push_back(unique_column_name);
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  std::move(unique_column_name));
  }

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  auto created_table = catalog.CreateTable(*context, std::move(info));
  auto &created_table_entry = created_table->Cast<duckdb::TableCatalogEntry>();
  temp_table_card_.emplace(temp_table_name, chunk_size);
  const duckdb::vector<duckdb::unique_ptr<duckdb::BoundConstraint>>
      bound_constraints = planner->binder->BindConstraints(created_table_entry);

  auto &storage = created_table_entry.GetStorage();
  storage.LocalAppend(created_table_entry, *context, *subquery_result,
                      bound_constraints, nullptr);
  //  storage.LocalAppend(created_table_entry, *context, *subquery_result);

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}
#endif

void DuckDBAdapter::CreateTempTable(const std::string &table_name,
                                    const QueryResult &result) {
  //  auto context = GetClientContext();
  //  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  //  auto info = duckdb::make_uniq<duckdb::CreateTableInfo>(TEMP_CATALOG,
  //  DEFAULT_SCHEMA, chunk_name); info->temporary = true; info->on_conflict =
  //  duckdb::OnCreateConflict::REPLACE_ON_CONFLICT; auto &types =
  //  result.Types(); auto data_chunk_index =
  //  planner.binder->GenerateTableIndex();
  //
  //  duckdb::case_insensitive_set_t used_column_names;
  //  for (size_t i = 0; i < result.column_names.size(); i++) {
  //    std::string column_name = result.column_names[i];
  //
  //    // Handle duplicate column names
  //    std::string unique_column_name = column_name;
  //    duckdb::idx_t suffix = 1;
  //    while (used_column_names.count(unique_column_name) > 0) {
  //      unique_column_name = column_name + "_" + std::to_string(suffix);
  //      suffix++;
  //    }
  //    used_column_names.insert(unique_column_name);
  //    info->columns.AddColumn(ColumnDefinition(unique_column_name, types[i]));
  //    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
  //    std::move(unique_column_name));
  //  }
  //
  //  auto created_table = catalog.CreateTable(*context, std::move(info));
  //  auto &created_table_entry =
  //  created_table->Cast<duckdb::TableCatalogEntry>(); int64_t
  //  created_table_size = subquery_result->Count();
  //  temp_table_card_.emplace(intermediate_table_name, created_table_size);
  //
  //  auto &storage = created_table_entry.GetStorage();
  //  storage.LocalAppend(created_table_entry, *context, *subquery_result);
}

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::DropTempTable(const std::string &table_name) {
  temp_collections_.erase(table_name);
}

bool DuckDBAdapter::TempTableExists(const std::string &table_name) {
  return temp_collections_.count(table_name) > 0;
}
#else
void DuckDBAdapter::DropTempTable(const std::string &chunk_name) {
  ExecuteSQL("DROP TABLE IF EXISTS " + chunk_name);
}

bool DuckDBAdapter::TempTableExists(const std::string &chunk_name) {
  try {
    auto result = ExecuteSQL(
        "SELECT count(*) FROM information_schema.tables WHERE chunk_name = '" +
        chunk_name + "'");
    return result.num_rows > 0 && result.rows[0][0] != "0";
  } catch (...) {
    return false;
  }
}
#endif

uint64_t
DuckDBAdapter::GetTempTableCardinality(const std::string &temp_table_name) {
  if (temp_table_card_.count(temp_table_name)) {
    return temp_table_card_[temp_table_name];
  }
  return 0; // Default if not found
}

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::SetTempTableCardinality(const std::string &temp_table_name,
                                            uint64_t cardinality) {
  // Set override cardinality on the stored collection
  auto it = temp_collections_.find(temp_table_name);
  if (it != temp_collections_.end()) {
    it->second.has_override_cardinality = true;
    it->second.override_cardinality = cardinality;
  }

  // Update temp_table_card_ for consistency
  temp_table_card_[temp_table_name] = cardinality;

#ifndef NDEBUG
  std::cout << "[DuckDB] SetTempTableCardinality: " << temp_table_name << " = "
            << cardinality << std::endl;
#endif
}
#else
void DuckDBAdapter::SetTempTableCardinality(const std::string &temp_table_name,
                                            uint64_t cardinality) {
  auto context = GetClientContext();

  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  auto &table_entry = catalog.GetEntry<duckdb::TableCatalogEntry>(
      *context, DEFAULT_SCHEMA, temp_table_name);
  auto &storage = table_entry.GetStorage();
  // Inject the override cardinality into DataTableInfo so that
  // DataTable::GetTotalRows() (and thus TableScanCardinality) returns
  // this value instead of the real row count.
  storage.GetDataTableInfo()->cardinality_override.store(cardinality);

  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }

  temp_table_card_[temp_table_name] = cardinality;

#ifndef NDEBUG
  std::cout << "[DuckDB] SetTempTableCardinality: " << temp_table_name << " = "
            << cardinality << std::endl;
#endif
}
#endif

// todo: if the middleware cannot access the duckdb's source code, it should run
//  `EXPLAIN` as the other engines
std::pair<double, double>
DuckDBAdapter::GetEstimatedCost(const std::string &sql) {
  // Use EXPLAIN to get estimated cost and rows
  // DuckDB's EXPLAIN output format: we'll parse the cardinality from it
  try {

    auto context = GetClientContext();

    // Begin transaction if in auto-commit mode
    if (context->transaction.IsAutoCommit()) {
      context->transaction.BeginTransaction();
    }

    auto cardest_plan = conn->ExtractPlan(sql);
    if (!cardest_plan) {
      throw std::runtime_error("couldn't extract plan!");
    }

    double estimated_rows = (double)cardest_plan->estimated_cardinality;
    double estimated_cost = estimated_rows;

    // Commit transaction if we started one
    if (context->transaction.IsAutoCommit()) {
      context->transaction.Commit();
    }

    return {estimated_cost, estimated_rows};

  } catch (const std::exception &e) {
    std::cerr << "[DuckDB] GetEstimatedCost exception: " << e.what()
              << std::endl;
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }
}

void DuckDBAdapter::CleanUp() {
#if IN_MEM_TMP_TABLE
  temp_collections_.clear();
#endif
  plan.reset();
  planner.reset();
  conn.reset();
  db.reset();
  table_column_mappings.clear();
  intermediate_table_map.clear();
  temp_table_card_.clear();
}

duckdb::ClientContext *DuckDBAdapter::GetClientContext() {
  return conn->context.get();
}

#ifdef HAVE_LLVM
// Recursively find the first AQPStmt node of type FilterNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstFilterNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir) return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *f = FindFirstFilterNode(child.get()))
      return f;
  return nullptr;
}

// Recursively find the first table index referenced by any leaf attribute
// in an AQPExpr tree.  Returns UINT_MAX if no attribute is found.
static unsigned int FirstTableIdxFromExpr(
    const ir_sql_converter::AQPExpr *expr) {
  if (!expr) return UINT_MAX;
  using ir_sql_converter::SimplestNodeType;
  switch (expr->GetNodeType()) {
    case SimplestNodeType::VarConstComparisonNode: {
      auto *c = static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
      return c->attr ? c->attr->GetTableIndex() : UINT_MAX;
    }
    case SimplestNodeType::IsNullExprNode: {
      auto *n = static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
      return n->attr ? n->attr->GetTableIndex() : UINT_MAX;
    }
    case SimplestNodeType::InExprNode: {
      auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
      return i->attr ? i->attr->GetTableIndex() : UINT_MAX;
    }
    case SimplestNodeType::LogicalExprNode: {
      auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
      unsigned int t = FirstTableIdxFromExpr(l->left_expr.get());
      if (t != UINT_MAX) return t;
      return FirstTableIdxFromExpr(l->right_expr.get());
    }
    default: return UINT_MAX;
  }
}

void DuckDBAdapter::RegisterJITFilters(duckdb::PhysicalOperator &op,
                                       const ir_sql_converter::AQPStmt &ir) {
  using duckdb::PhysicalOperatorType;

  if (op.type == PhysicalOperatorType::FILTER && !op.children.empty()) {
    auto &child = op.children[0].get();

    // Determine table_idx from the filter's own expressions (not target_list,
    // which may belong to an ancestor aggregate in multi-table final IRs).
    unsigned int ir_table_idx = 0;
    for (const auto &q : ir.qual_vec) {
      unsigned int t = FirstTableIdxFromExpr(q.get());
      if (t != UINT_MAX) { ir_table_idx = t; break; }
    }
    if (ir_table_idx == 0 && !ir.target_list.empty())
      ir_table_idx = ir.target_list[0]->GetTableIndex(); // fallback

    // Build ColSchema: chunk position → (table_idx, col_idx, dtype)
    std::vector<aqp_jit::ColSchema> schema;
    if (child.type == PhysicalOperatorType::TABLE_SCAN) {
      auto &scan = static_cast<duckdb::PhysicalTableScan &>(child);
      for (size_t i = 0; i < scan.column_ids.size(); i++) {
        aqp_jit::ColSchema cs;
        cs.table_idx = ir_table_idx;
        cs.col_idx   = static_cast<unsigned int>(scan.column_ids[i].GetPrimaryIndex());
        cs.dtype     = duckdb::ToDtype(child.types[i].InternalType());
        schema.push_back(cs);
      }
    } else {
      // Non-scan child: position-based fallback (col_idx = chunk position)
      for (size_t i = 0; i < child.types.size(); i++) {
        aqp_jit::ColSchema cs;
        cs.table_idx = ir_table_idx;
        cs.col_idx   = static_cast<unsigned int>(i);
        cs.dtype     = duckdb::ToDtype(child.types[i].InternalType());
        schema.push_back(cs);
      }
    }

    if (!schema.empty()) {
      // CompileFilter needs a Filter IR node (with qual_vec), not the root IR
      // (which may be an Aggregate or Projection with empty qual_vec).
      const ir_sql_converter::AQPStmt *filter_ir = FindFirstFilterNode(&ir);
      if (!filter_ir) {
        std::cerr << "[AQP-JIT] no Filter IR node found in tree, skipping\n";
        for (auto &child_ref : op.children)
          RegisterJITFilters(child_ref.get(), ir);
        return;
      }
      // Use the persistent compiler so the LLJIT instance (and its compiled
      // code memory) stays alive until jit_compiler_ is reset next query.
      if (!jit_compiler_)
        jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(/*use_o3=*/false);
      // CompileFilter returns global ::AQPExprFn (from aqp_jit_abi.h);
      // duckdb::AQPExprFn has identical layout — reinterpret_cast is safe.
      ::AQPExprFn raw_fn = jit_compiler_->CompileFilter(*filter_ir, schema);
      if (raw_fn) {
        duckdb::AQPExprFn fn =
            reinterpret_cast<duckdb::AQPExprFn>(reinterpret_cast<void*>(raw_fn));
        auto *ctx = GetClientContext();
        if (!ctx->aqp_jit_context)
          ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        uint64_t eid = duckdb::ExpressionID(op);
        ctx->aqp_jit_context->expr_fns[eid] = fn;
        ctx->aqp_jit_context->flags |= duckdb::AQPJIT_EXPR;
        std::cerr << "[AQP-JIT] compiled filter eid=0x" << std::hex << eid << std::dec
                  << "  cols=" << schema.size() << "\n";
      } else {
        uint64_t eid = duckdb::ExpressionID(op);
        std::ostringstream msg;
        msg << "[AQP-JIT] JIT compilation failed (eid=0x" << std::hex << eid
            << std::dec << "  cols=" << schema.size()
            << "). Stopping. Use --no-jit to fall back to interpreter.";
        throw std::runtime_error(msg.str());
      }
    }
  }

  // Recurse into children
  for (auto &child_ref : op.children)
    RegisterJITFilters(child_ref.get(), ir);
}
#endif

duckdb::Binder &DuckDBAdapter::GetBinder() { return *planner->binder; }

duckdb::unique_ptr<duckdb::LogicalOperator> DuckDBAdapter::TakePlan() {
  return std::move(plan);
}

void DuckDBAdapter::SetPlan(duckdb::unique_ptr<duckdb::LogicalOperator> p) {
  plan = std::move(p);
}

void DuckDBAdapter::RegisterExternalTempTable(
    const std::string &temp_name,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names) {

  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_name;
  temp_table_index_ = data_chunk_index;
  temp_table_types = types;
  chunk_col_names_[data_chunk_index] = col_names;
}

} // namespace middleware
