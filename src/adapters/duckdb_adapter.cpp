/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#include "adapters/duckdb_adapter.h"

#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_set>
#include <stdexcept>

#ifdef HAVE_LLVM
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "simplest_ir.h"
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
      // Reset JIT context (stale function pointers would crash if dispatched).
      // Keep jit_compiler_ alive — reusing the LLJIT instance avoids re-registering
      // 15+ runtime symbols and re-initializing the JIT on every sub-plan.
      GetClientContext()->aqp_jit_context.reset();
      std::cerr << "[AQP-JIT-TRACE] ExecuteSQL: reset aqp_jit_context (compiler reused)\n";
      RegisterJITFilters(prepared->data->physical_plan->Root(), *jit_pending_ir_);
      auto *ctx = GetClientContext();
      if (ctx->aqp_jit_context) {
        std::cerr << "[AQP-JIT] summary: flags=0x" << std::hex << ctx->aqp_jit_context->flags
                  << std::dec << " expr_fns=" << ctx->aqp_jit_context->expr_fns.size()
                  << " op_fns=" << ctx->aqp_jit_context->op_fns.size()
                  << " proj_maps=" << ctx->aqp_jit_context->proj_col_maps.size()
                  << " agg_fns=" << ctx->aqp_jit_context->agg_fns.size()
                  << " pipeline_fns=" << ctx->aqp_jit_context->pipeline_fns.size()
                  << " scan_filter=" << ctx->aqp_jit_context->scan_filter_fns.size() << "\n";
      }
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
    // Reset JIT context (stale function pointers would crash if dispatched).
    // Keep jit_compiler_ alive — reusing LLJIT avoids re-init overhead.
    GetClientContext()->aqp_jit_context.reset();
    std::cerr << "[AQP-JIT-TRACE] ExecuteSQLandCreateTempTable: reset aqp_jit_context (compiler reused)\n";
    RegisterJITFilters(prepared->data->physical_plan->Root(), *jit_pending_ir_);

    // Level 4: Sub-plan compilation — compile the entire sub-IR tree
    if (jit_flags_ & AQP_JIT_SUBPLAN) {
      if (!jit_compiler_)
        jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
            /*use_o3=*/(jit_flags_ & AQP_JIT_OPT3) != 0,
            /*use_simd=*/(jit_flags_ & AQP_JIT_SIMD) != 0);
      void *subplan_result = jit_compiler_->CompileSubPlan(*jit_pending_ir_);
      if (subplan_result) {
        auto *ctx_sp = GetClientContext();
        if (ctx_sp->aqp_jit_context) {
          ctx_sp->aqp_jit_context->flags |= duckdb::AQPJIT_SUBPLAN;
          std::cerr << "[AQP-JIT] sub-plan compiled for this sub-IR\n";
        }
      }
    }

    auto *ctx2 = GetClientContext();
    if (ctx2->aqp_jit_context) {
      std::cerr << "[AQP-JIT] summary: flags=0x" << std::hex << ctx2->aqp_jit_context->flags
                << std::dec << " expr_fns=" << ctx2->aqp_jit_context->expr_fns.size()
                << " op_fns=" << ctx2->aqp_jit_context->op_fns.size()
                << " proj_maps=" << ctx2->aqp_jit_context->proj_col_maps.size()
                << " agg_fns=" << ctx2->aqp_jit_context->agg_fns.size()
                << " pipeline_fns=" << ctx2->aqp_jit_context->pipeline_fns.size()
                << " scan_filter=" << ctx2->aqp_jit_context->scan_filter_fns.size() << "\n";
    }
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

// Recursively find the first ProjectionNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstProjectionNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir) return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ProjectionNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *p = FindFirstProjectionNode(child.get()))
      return p;
  return nullptr;
}

// Recursively find the first HashNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstHashNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir) return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::HashNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *h = FindFirstHashNode(child.get()))
      return h;
  return nullptr;
}

// Recursively find the first JoinNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstJoinNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir) return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *j = FindFirstJoinNode(child.get()))
      return j;
  return nullptr;
}

// Recursively find the first AggregateNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstAggregateNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir) return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::AggregateNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *a = FindFirstAggregateNode(child.get()))
      return a;
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

// Walk the IR tree and build a map: table_name (lowercase) → table_index.
// Also populates `ambiguous` with names that appear with multiple different indices
// (e.g. two comp_cast_type aliases with index 1 and 2). JIT is skipped for those.
static void CollectTableNameToIndex(
    const ir_sql_converter::AQPStmt *ir,
    std::unordered_map<std::string, unsigned int> &out,
    std::unordered_set<std::string> &ambiguous) {
  if (!ir) return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    std::string name = scan->GetTableName();
    for (auto &c : name) c = (char)tolower((unsigned char)c);
    auto it = out.find(name);
    if (it != out.end() && it->second != scan->GetTableIndex())
      ambiguous.insert(name);  // same physical table, different IR indices
    out[name] = scan->GetTableIndex();
  }
  for (const auto &child : ir->children)
    CollectTableNameToIndex(child.get(), out, ambiguous);
}

// Collect ALL FilterNodes in DFS order.
static void CollectAllFilterNodes(
    const ir_sql_converter::AQPStmt *ir,
    std::vector<const ir_sql_converter::AQPStmt *> &out) {
  if (!ir) return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode)
    out.push_back(ir);
  for (const auto &child : ir->children)
    CollectAllFilterNodes(child.get(), out);
}


// Returns true if the expr (recursively) references a (table_idx, col_idx) pair
// that exists in schema.  Used to skip JIT for filters whose predicates don't
// map to any column in the physical operator's output chunk.
static bool ExprHasColInSchema(
    const ir_sql_converter::AQPExpr *expr,
    const std::vector<aqp_jit::ColSchema> &schema) {
  if (!expr) return false;
  using ir_sql_converter::SimplestNodeType;
  auto attrInSchema = [&](unsigned int t, unsigned int col) -> bool {
    for (auto &cs : schema)
      if (cs.table_idx == t && cs.col_idx == col) return true;
    return false;
  };
  switch (expr->GetNodeType()) {
    case SimplestNodeType::VarConstComparisonNode: {
      auto *c = static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
      return c->attr && attrInSchema(
          static_cast<unsigned int>(c->attr->GetTableIndex()),
          static_cast<unsigned int>(c->attr->GetColumnIndex()));
    }
    case SimplestNodeType::IsNullExprNode: {
      auto *n = static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
      return n->attr && attrInSchema(
          static_cast<unsigned int>(n->attr->GetTableIndex()),
          static_cast<unsigned int>(n->attr->GetColumnIndex()));
    }
    case SimplestNodeType::InExprNode: {
      auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
      return i->attr && attrInSchema(
          static_cast<unsigned int>(i->attr->GetTableIndex()),
          static_cast<unsigned int>(i->attr->GetColumnIndex()));
    }
    case SimplestNodeType::LogicalExprNode: {
      auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
      return ExprHasColInSchema(l->left_expr.get(), schema) ||
             ExprHasColInSchema(l->right_expr.get(), schema);
    }
    default: return false;
  }
}

static bool HasApplicablePredicate(
    const ir_sql_converter::AQPStmt *filter_ir,
    const std::vector<aqp_jit::ColSchema> &schema) {
  for (const auto &q : filter_ir->qual_vec)
    if (ExprHasColInSchema(q.get(), schema)) return true;
  return false;
}

void DuckDBAdapter::RegisterJITFilters(duckdb::PhysicalOperator &op,
                                       const ir_sql_converter::AQPStmt &ir) {
  using duckdb::PhysicalOperatorType;

  // Diagnostic: show every node in the physical plan tree
  std::cerr << "[AQP-JIT-TRACE] visit op=" << (int)op.type
            << " children=" << op.children.size()
            << " addr=" << (void*)&op << "\n";

  if (op.type == PhysicalOperatorType::FILTER && !op.children.empty()) {
    auto &child = op.children[0].get();
    uint64_t eid = duckdb::ExpressionID(op);

    std::cerr << "[AQP-JIT-TRACE] FILTER eid=0x" << std::hex << eid << std::dec
              << "  child_type=" << (int)child.type
              << "  child_cols=" << child.types.size() << "\n";

    // Dump all IR FilterNodes once per FILTER op (to show the full IR structure)
    {
      std::vector<const ir_sql_converter::AQPStmt *> all_filters;
      CollectAllFilterNodes(&ir, all_filters);
      std::cerr << "[AQP-JIT-TRACE] IR has " << all_filters.size()
                << " FilterNode(s):\n";
      for (size_t fi = 0; fi < all_filters.size(); fi++) {
        std::cerr << "[AQP-JIT-TRACE]   IR filter[" << fi
                  << "] qual_vec.size=" << all_filters[fi]->qual_vec.size() << "\n";
        for (size_t qi = 0; qi < all_filters[fi]->qual_vec.size(); qi++) {
          std::string qs = all_filters[fi]->qual_vec[qi]->Print(false);
          std::cerr << "[AQP-JIT-TRACE]     qual[" << qi << "]: " << qs << "\n";
        }
      }
    }

    // Determine the IR table_idx for this physical scan by matching the DuckDB
    // table name against IR ScanNodes.  This is reliable because each physical
    // TABLE_SCAN in DuckDB corresponds to exactly one logical table whose name
    // is stored in both the DuckDB bind_data and the IR SimplestScan node.
    const ir_sql_converter::AQPStmt *filter_ir = nullptr;
    unsigned int ir_table_idx = UINT_MAX;

    // Build table-name → IR table_index map from IR ScanNodes
    std::unordered_map<std::string, unsigned int> ir_table_name_to_idx;
    std::unordered_set<std::string> ir_ambiguous_names;
    CollectTableNameToIndex(&ir, ir_table_name_to_idx, ir_ambiguous_names);
    std::cerr << "[AQP-JIT-TRACE] IR table name map:\n";
    for (auto &kv : ir_table_name_to_idx)
      std::cerr << "[AQP-JIT-TRACE]   \"" << kv.first << "\" -> idx=" << kv.second << "\n";

    std::vector<aqp_jit::ColSchema> schema_prelim;

    if (child.type == PhysicalOperatorType::TABLE_SCAN) {
      auto &scan = static_cast<duckdb::PhysicalTableScan &>(child);

      // Get table name from DuckDB scan bind_data (works for regular storage scans)
      std::string duckdb_table_name;
      if (scan.bind_data) {
        auto *tsbd = dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
        if (tsbd) {
          duckdb_table_name = tsbd->table.name;
          for (auto &c : duckdb_table_name) c = (char)tolower((unsigned char)c);
        }
      }
      // Fall back: use function name (for temp table scans)
      if (duckdb_table_name.empty()) {
        duckdb_table_name = scan.function.name;
        for (auto &c : duckdb_table_name) c = (char)tolower((unsigned char)c);
      }
      std::cerr << "[AQP-JIT-TRACE]   duckdb table_name=\"" << duckdb_table_name << "\"\n";

      // Look up IR table_idx by name
      auto it = ir_table_name_to_idx.find(duckdb_table_name);
      if (it != ir_table_name_to_idx.end()) {
        ir_table_idx = it->second;
        std::cerr << "[AQP-JIT-TRACE]   resolved ir_table_idx=" << ir_table_idx << "\n";
      } else {
        std::cerr << "[AQP-JIT-TRACE]   WARNING: no IR table matches \"" << duckdb_table_name << "\"\n";
      }

      // Skip JIT for tables that appear under multiple different IR indices
      // (e.g. self-join on comp_cast_type with alias indices 1 and 2).
      // The name→index map has only one entry so one side would get the wrong filter.
      // Leave schema_prelim empty and filter_ir as nullptr → JIT skipped below.
      if (ir_ambiguous_names.count(duckdb_table_name)) {
        std::cerr << "[AQP-JIT] ambiguous table \"" << duckdb_table_name
                  << "\" → skipping JIT for this filter\n";
      } else {
        for (size_t i = 0; i < scan.column_ids.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = ir_table_idx;
          cs.col_idx   = static_cast<unsigned int>(scan.column_ids[i].GetPrimaryIndex());
          cs.dtype     = duckdb::ToDtype(child.types[i].InternalType());
          schema_prelim.push_back(cs);
          std::cerr << "[AQP-JIT-TRACE]   scan col[" << i
                    << "] raw_col=" << cs.col_idx << " dtype=" << cs.dtype
                    << " table_idx=" << cs.table_idx << "\n";
        }

        // Find the FilterNode that references ir_table_idx
        filter_ir = FindFirstFilterNode(&ir); // fallback
        if (ir_table_idx != UINT_MAX) {
          std::vector<const ir_sql_converter::AQPStmt *> all_filters;
          CollectAllFilterNodes(&ir, all_filters);
          bool found = false;
          for (auto *f : all_filters) {
            for (const auto &q : f->qual_vec) {
              unsigned int t = FirstTableIdxFromExpr(q.get());
              if (t == ir_table_idx) { filter_ir = f; found = true; break; }
            }
            if (found) break;
          }
        }
      }
    } else if (child.type == PhysicalOperatorType::PROJECTION &&
               !child.children.empty() &&
               child.children[0].get().type == PhysicalOperatorType::TABLE_SCAN) {
      // PROJECTION → TABLE_SCAN: trace projection expressions back to original
      // scan column indices.  PhysicalProjection::select_list[i] is a
      // BoundReferenceExpression with index=j meaning "output col i = scan
      // input col j", so the original table col idx = scan.column_ids[j].
      auto &proj = static_cast<duckdb::PhysicalProjection &>(child);
      auto &scan = static_cast<duckdb::PhysicalTableScan &>(child.children[0].get());

      // Get table name → ir_table_idx (same logic as the TABLE_SCAN path)
      std::string duckdb_table_name;
      if (scan.bind_data) {
        auto *tsbd = dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
        if (tsbd) {
          duckdb_table_name = tsbd->table.name;
          for (auto &c : duckdb_table_name) c = (char)tolower((unsigned char)c);
        }
      }
      if (duckdb_table_name.empty()) {
        duckdb_table_name = scan.function.name;
        for (auto &c : duckdb_table_name) c = (char)tolower((unsigned char)c);
      }
      std::cerr << "[AQP-JIT-TRACE]   proj→scan table_name=\"" << duckdb_table_name << "\"\n";

      auto it = ir_table_name_to_idx.find(duckdb_table_name);
      if (it != ir_table_name_to_idx.end()) {
        ir_table_idx = it->second;
        std::cerr << "[AQP-JIT-TRACE]   resolved ir_table_idx=" << ir_table_idx << "\n";
      } else {
        std::cerr << "[AQP-JIT-TRACE]   WARNING: proj→scan no IR table matches \"" << duckdb_table_name << "\"\n";
      }

      if (ir_ambiguous_names.count(duckdb_table_name)) {
        std::cerr << "[AQP-JIT] ambiguous table \"" << duckdb_table_name
                  << "\" (proj path) → skipping JIT for this filter\n";
        // schema_prelim stays empty, filter_ir stays nullptr → JIT skipped below
      } else {

      // Build schema by tracing each projection expression back to scan col
      for (size_t i = 0; i < proj.select_list.size(); i++) {
        aqp_jit::ColSchema cs;
        cs.table_idx = ir_table_idx;
        cs.col_idx   = UINT_MAX; // set below if traceable
        cs.dtype     = duckdb::ToDtype(child.types[i].InternalType());

        auto &expr = *proj.select_list[i];
        if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
          if (ref.index < scan.column_ids.size()) {
            cs.col_idx = static_cast<unsigned int>(
                scan.column_ids[ref.index].GetPrimaryIndex());
          }
        }
        schema_prelim.push_back(cs);
        std::cerr << "[AQP-JIT-TRACE]   proj col[" << i
                  << "] raw_col=" << cs.col_idx << " dtype=" << cs.dtype
                  << " table_idx=" << cs.table_idx << "\n";
      }

      // Find the FilterNode that references ir_table_idx
      filter_ir = FindFirstFilterNode(&ir); // fallback
      if (ir_table_idx != UINT_MAX) {
        std::vector<const ir_sql_converter::AQPStmt *> all_filters;
        CollectAllFilterNodes(&ir, all_filters);
        bool found = false;
        for (auto *f : all_filters) {
          for (const auto &q : f->qual_vec) {
            unsigned int t = FirstTableIdxFromExpr(q.get());
            if (t == ir_table_idx) { filter_ir = f; found = true; break; }
          }
          if (found) break;
        }
      }
      } // end else (not ambiguous) for proj path
    } else {
      // Unknown child type (HASH_JOIN result, etc.): skip JIT.
      // schema_prelim stays empty → HasApplicablePredicate won't match → JIT skipped.
      std::cerr << "[AQP-JIT-TRACE]   child_type=" << (int)child.type
                << " not JIT-able → interpreter\n";
    }

    if (!filter_ir) {
      std::cerr << "[AQP-JIT] FILTER eid=0x" << std::hex << eid << std::dec
                << ": no Filter IR node found in tree, using interpreter\n";
      for (auto &child_ref : op.children)
        RegisterJITFilters(child_ref.get(), ir);
      return;
    }

    std::cerr << "[AQP-JIT-TRACE] filter_ir node_type=" << (int)filter_ir->GetNodeType()
              << "  qual_vec.size=" << filter_ir->qual_vec.size()
              << "  ir_table_idx=" << ir_table_idx << "\n";
    for (size_t qi = 0; qi < filter_ir->qual_vec.size(); qi++) {
      std::string qs = filter_ir->qual_vec[qi]->Print(false);
      std::cerr << "[AQP-JIT-TRACE]   selected qual[" << qi << "]: " << qs << "\n";
    }

    // Build final ColSchema from schema_prelim.
    // For TABLE_SCAN and PROJECTION→TABLE_SCAN: schema_prelim already has correct
    // (table_idx, col_idx, dtype) from the DuckDB table-name → IR table_idx lookup.
    // For unknown child types: schema_prelim is empty → JIT skipped below.
    std::vector<aqp_jit::ColSchema> schema = schema_prelim;
    for (size_t i = 0; i < schema.size(); i++)
      std::cerr << "[AQP-JIT-TRACE]   schema[" << i << "] table=" << schema[i].table_idx
                << " col=" << schema[i].col_idx << " dtype=" << schema[i].dtype << "\n";

    // Note: VARCHAR predicates (LIKE, NOT LIKE, Equal, NotEqual) are now
    // handled by the JIT via aqp_like_match / aqp_str_eq runtime helpers.

    // Only register JIT if at least one predicate references a column in the
    // schema.  If none do (e.g. the predicate was already applied at scan level
    // and the physical FILTER's child only projects a different column), skip
    // JIT for this operator and let DuckDB's interpreter handle it.
    if (!schema.empty() && HasApplicablePredicate(filter_ir, schema)) {
      // Use the persistent compiler so the LLJIT instance (and its compiled
      // code memory) stays alive until jit_compiler_ is reset next query.
      if (!jit_compiler_)
        jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
            /*use_o3=*/(jit_flags_ & AQP_JIT_OPT3) != 0,
            /*use_simd=*/(jit_flags_ & AQP_JIT_SIMD) != 0);
      ::AQPExprFn raw_fn = jit_compiler_->CompileFilter(*filter_ir, schema);
      if (raw_fn) {
        duckdb::AQPExprFn fn =
            reinterpret_cast<duckdb::AQPExprFn>(reinterpret_cast<void*>(raw_fn));
        auto *ctx = GetClientContext();
        if (!ctx->aqp_jit_context)
          ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        ctx->aqp_jit_context->expr_fns[eid] = fn;
        ctx->aqp_jit_context->flags |= duckdb::AQPJIT_EXPR;

        // Scan+Filter fusion: also register filter on the TABLE_SCAN operator
        // so it can be applied at scan level (pre-filtered chunks).
        if (child.type == PhysicalOperatorType::TABLE_SCAN) {
          uint64_t scan_eid = duckdb::ExpressionID(child);
          ctx->aqp_jit_context->scan_filter_fns[scan_eid] = fn;
          std::cerr << "[AQP-JIT] scan+filter fusion: scan_eid=0x" << std::hex
                    << scan_eid << std::dec << "\n";
        }

        std::cerr << "[AQP-JIT] compiled filter eid=0x" << std::hex << eid << std::dec
                  << "  cols=" << schema.size()
                  << "  table_idx=" << ir_table_idx << "\n";
      } else {
        std::ostringstream msg;
        msg << "[AQP-JIT] JIT compilation failed (eid=0x" << std::hex << eid
            << std::dec << "  cols=" << schema.size()
            << "  table_idx=" << ir_table_idx
            << "). Stopping. Use --no-jit to fall back to interpreter.";
        throw std::runtime_error(msg.str());
      }
    } else if (!schema.empty()) {
      std::cerr << "[AQP-JIT] skipping filter eid=0x" << std::hex << eid << std::dec
                << ": no predicates reference schema columns → interpreter\n";
    }

    // Level 3: Fuse Filter → Projection into a single pipeline function.
    // Gated by AQP_JIT_PIPELINE flag. Registered by filter operator eid.
    if ((jit_flags_ & AQP_JIT_PIPELINE) && filter_ir && !schema.empty()) {
      const ir_sql_converter::AQPStmt *proj_ir = FindFirstProjectionNode(&ir);

      if (!jit_compiler_)
        jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
            /*use_o3=*/(jit_flags_ & AQP_JIT_OPT3) != 0,
            /*use_simd=*/(jit_flags_ & AQP_JIT_SIMD) != 0);

      ::AQPPipelineFn pipe_fn = jit_compiler_->CompilePipeline(
          filter_ir, proj_ir, schema);
      if (pipe_fn) {
        auto fn = reinterpret_cast<duckdb::AQPPipelineFn>(
            reinterpret_cast<void*>(pipe_fn));
        auto *ctx = GetClientContext();
        if (!ctx->aqp_jit_context)
          ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        ctx->aqp_jit_context->pipeline_fns[eid] = fn;
        ctx->aqp_jit_context->flags |= duckdb::AQPJIT_PIPELINE;
        std::cerr << "[AQP-JIT] compiled pipeline eid=0x" << std::hex << eid
                  << std::dec << " (filter"
                  << (proj_ir ? "+projection" : "") << ")\n";
      }
    }
  }

  // Level 2: Compile projection operators when AQPJIT_OPERATOR is set.
  if ((jit_flags_ & AQP_JIT_OPERATOR) &&
      op.type == PhysicalOperatorType::PROJECTION) {
    uint64_t eid = duckdb::ExpressionID(op);
    std::cerr << "[AQP-JIT-TRACE] PROJECTION eid=0x" << std::hex << eid
              << std::dec << "\n";

    // Find the first ProjectionNode in the IR tree (static function, no heap alloc)
    const ir_sql_converter::AQPStmt *proj_ir = FindFirstProjectionNode(&ir);

    if (proj_ir && !proj_ir->target_list.empty() && !op.children.empty()) {
      // Build input schema from child operator's output types
      auto &child = op.children[0].get();
      std::vector<aqp_jit::ColSchema> in_schema;

      // Use the same table-name resolution as the filter path
      std::unordered_map<std::string, unsigned int> ir_table_name_to_idx;
      std::unordered_set<std::string> ir_ambiguous_names;
      CollectTableNameToIndex(&ir, ir_table_name_to_idx, ir_ambiguous_names);

      unsigned int ir_table_idx = UINT_MAX;
      if (child.type == PhysicalOperatorType::TABLE_SCAN) {
        auto &scan = static_cast<duckdb::PhysicalTableScan &>(child);
        std::string duckdb_table_name;
        if (scan.bind_data) {
          auto *tsbd = dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
          if (tsbd) {
            duckdb_table_name = tsbd->table.name;
            for (auto &c : duckdb_table_name) c = (char)tolower((unsigned char)c);
          }
        }
        if (duckdb_table_name.empty()) {
          duckdb_table_name = scan.function.name;
          for (auto &c : duckdb_table_name) c = (char)tolower((unsigned char)c);
        }
        auto it = ir_table_name_to_idx.find(duckdb_table_name);
        if (it != ir_table_name_to_idx.end())
          ir_table_idx = it->second;

        for (size_t i = 0; i < scan.column_ids.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = ir_table_idx;
          cs.col_idx   = static_cast<unsigned int>(scan.column_ids[i].GetPrimaryIndex());
          cs.dtype     = duckdb::ToDtype(child.types[i].InternalType());
          in_schema.push_back(cs);
        }
      }

      if (!in_schema.empty()) {
        if (!jit_compiler_)
          jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
              /*use_o3=*/(jit_flags_ & AQP_JIT_OPT3) != 0);
        // Build column mapping: out_col_i → in_col_i
        duckdb::vector<int> col_map;
        for (const auto &attr : proj_ir->target_list) {
          int found = -1;
          for (int ci = 0; ci < (int)in_schema.size(); ci++) {
            if (in_schema[ci].table_idx == attr->GetTableIndex() &&
                in_schema[ci].col_idx  == attr->GetColumnIndex()) {
              found = ci;
              break;
            }
          }
          col_map.push_back(found);
        }

        ::AQPOperatorFn raw_fn = jit_compiler_->CompileProjection(*proj_ir, in_schema);
        if (raw_fn) {
          auto fn = reinterpret_cast<duckdb::AQPOperatorFn>(
              reinterpret_cast<void*>(raw_fn));
          auto *ctx = GetClientContext();
          if (!ctx->aqp_jit_context)
            ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
          ctx->aqp_jit_context->op_fns[eid] = fn;  // for Level 3 pipeline fusion
          ctx->aqp_jit_context->proj_col_maps[eid] = std::move(col_map);  // for Level 2 zero-copy
          ctx->aqp_jit_context->flags |= duckdb::AQPJIT_OPERATOR;
          std::cerr << "[AQP-JIT] compiled projection eid=0x" << std::hex << eid
                    << std::dec << "  in_cols=" << in_schema.size()
                    << "  out_cols=" << proj_ir->target_list.size() << "\n";
        }
      }
    }
  }

  // Level 2: Compile aggregate operators when AQPJIT_OPERATOR is set.
  // Ungrouped aggregates: dispatched at Level 2 (compiled loop is faster than
  // DuckDB's generic ExpressionExecutor). Grouped: compiled for Level 3 only.
  if ((jit_flags_ & AQP_JIT_OPERATOR) &&
      (op.type == PhysicalOperatorType::HASH_GROUP_BY ||
       op.type == PhysicalOperatorType::UNGROUPED_AGGREGATE)) {
    uint64_t eid = duckdb::ExpressionID(op);
    std::cerr << "[AQP-JIT-TRACE] AGGREGATE eid=0x" << std::hex << eid
              << std::dec << " type=" << (int)op.type << "\n";

    const ir_sql_converter::AQPStmt *agg_ir = FindFirstAggregateNode(&ir);
    if (agg_ir && !op.children.empty()) {
      auto &child = op.children[0].get();
      std::vector<aqp_jit::ColSchema> in_schema;

      // Build schema from the IR aggregate node's child target_list.
      // The agg_fns reference IR table/column indices, so the schema must
      // use those same indices (not sequential col_idx with UINT_MAX table).
      const ir_sql_converter::AQPStmt *agg_child_ir = nullptr;
      if (!agg_ir->children.empty())
        agg_child_ir = agg_ir->children[0].get();

      if (agg_child_ir && !agg_child_ir->target_list.empty()) {
        // Use IR child's target_list for table/column indices
        for (size_t i = 0; i < agg_child_ir->target_list.size() &&
                           i < child.types.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = agg_child_ir->target_list[i]->GetTableIndex();
          cs.col_idx   = agg_child_ir->target_list[i]->GetColumnIndex();
          cs.dtype     = duckdb::ToDtype(child.types[i].InternalType());
          in_schema.push_back(cs);
        }
      } else {
        // Fallback: use target_list from agg node itself
        for (size_t i = 0; i < agg_ir->target_list.size() &&
                           i < child.types.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = agg_ir->target_list[i]->GetTableIndex();
          cs.col_idx   = agg_ir->target_list[i]->GetColumnIndex();
          cs.dtype     = duckdb::ToDtype(child.types[i].InternalType());
          in_schema.push_back(cs);
        }
      }

      std::cerr << "[AQP-JIT-TRACE] agg in_schema:";
      for (size_t i = 0; i < in_schema.size(); i++)
        std::cerr << " [" << i << "]=(t=" << in_schema[i].table_idx
                  << ",c=" << in_schema[i].col_idx
                  << ",d=" << in_schema[i].dtype << ")";
      std::cerr << "\n";

      if (!jit_compiler_)
        jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
            /*use_o3=*/(jit_flags_ & AQP_JIT_OPT3) != 0,
            /*use_simd=*/(jit_flags_ & AQP_JIT_SIMD) != 0);
      void *raw_fn = jit_compiler_->CompileAggUpdate(*agg_ir, in_schema);
      if (raw_fn) {
        // Compute state size (same logic as CompileAggUpdate)
        auto *agg = dynamic_cast<const ir_sql_converter::SimplestAggregate *>(agg_ir);
        uint32_t state_size = 0;
        if (agg) {
          for (const auto &fp : agg->agg_fns) {
            if (fp.second == ir_sql_converter::SimplestAggFnType::Average)
              state_size += 16;
            else
              state_size += 8;
          }
        }

        auto *ctx = GetClientContext();
        if (!ctx->aqp_jit_context)
          ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        // Store in agg_fns (typed) + op_fns (for Level 3)
        auto agg_fn = reinterpret_cast<duckdb::AQPJITContext::AQPAggUpdateFn>(raw_fn);
        ctx->aqp_jit_context->agg_fns[eid] = agg_fn;
        ctx->aqp_jit_context->agg_state_sizes[eid] = state_size;
        ctx->aqp_jit_context->op_fns[eid] =
            reinterpret_cast<duckdb::AQPOperatorFn>(raw_fn);
        ctx->aqp_jit_context->flags |= duckdb::AQPJIT_OPERATOR;
        std::cerr << "[AQP-JIT] compiled agg eid=0x" << std::hex << eid
                  << std::dec << " state_bytes=" << state_size
                  << " ungrouped=" << (agg && agg->groups.empty() ? "yes" : "no") << "\n";
      }
    }
  }

  // Level 2: Compile hash join operators when AQPJIT_OPERATOR is set.
  // Like aggregate, compiled for Level 3 pipeline fusion. DuckDB's JoinHashTable
  // handles Level 2 execution (parallel build, spilling, outer join state, etc.).
  if ((jit_flags_ & AQP_JIT_OPERATOR) &&
      op.type == PhysicalOperatorType::HASH_JOIN) {
    uint64_t eid = duckdb::ExpressionID(op);
    std::cerr << "[AQP-JIT-TRACE] HASH_JOIN eid=0x" << std::hex << eid
              << std::dec << "\n";

    // Find JoinNode in IR. DuckDB path doesn't generate HashNode, so we
    // derive hash keys from the JoinNode's join_conditions instead.
    const ir_sql_converter::AQPStmt *join_ir = FindFirstJoinNode(&ir);
    const ir_sql_converter::AQPStmt *hash_ir = FindFirstHashNode(&ir);

    if (join_ir) {
      auto *join = dynamic_cast<const ir_sql_converter::SimplestJoin *>(join_ir);
      if (join && !join->join_conditions.empty()) {
        // Build side is typically children[1] in the IR JoinNode.
        // Its target_list defines the build schema.
        const ir_sql_converter::AQPStmt *build_child =
            (join_ir->children.size() > 1) ? join_ir->children[1].get() : nullptr;

        // If HashNode exists (PostgreSQL path), use its child instead
        if (hash_ir)
          build_child = hash_ir->children.empty() ? build_child : hash_ir->children[0].get();

        if (build_child && !build_child->target_list.empty()) {
          // Build schema from build child's target_list
          std::vector<aqp_jit::ColSchema> build_schema;
          for (const auto &attr : build_child->target_list) {
            aqp_jit::ColSchema cs;
            cs.table_idx = attr->GetTableIndex();
            cs.col_idx   = attr->GetColumnIndex();
            switch (attr->GetType()) {
            case ir_sql_converter::IntVar:    cs.dtype = AQP_DTYPE_INT32; break;
            case ir_sql_converter::FloatVar:  cs.dtype = AQP_DTYPE_DOUBLE; break;
            case ir_sql_converter::StringVar: cs.dtype = AQP_DTYPE_VARCHAR; break;
            case ir_sql_converter::BoolVar:   cs.dtype = AQP_DTYPE_BOOL; break;
            case ir_sql_converter::Date:      cs.dtype = AQP_DTYPE_DATE; break;
            default: cs.dtype = AQP_DTYPE_OTHER; break;
            }
            build_schema.push_back(cs);
          }

          // If no HashNode, create a synthetic one from join conditions.
          // The build-side key columns are the join condition attrs that match
          // the build_child's target_list.
          std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> synth_keys;
          for (const auto &cond : join->join_conditions) {
            // Check which side of the condition is in the build schema
            bool left_in_build = false, right_in_build = false;
            for (const auto &cs : build_schema) {
              if (cs.table_idx == cond->left_attr->GetTableIndex() &&
                  cs.col_idx  == cond->left_attr->GetColumnIndex())
                left_in_build = true;
              if (cs.table_idx == cond->right_attr->GetTableIndex() &&
                  cs.col_idx  == cond->right_attr->GetColumnIndex())
                right_in_build = true;
            }
            if (left_in_build) {
              synth_keys.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(
                  *cond->left_attr));
            } else if (right_in_build) {
              synth_keys.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(
                  *cond->right_attr));
            }
          }

          if (!synth_keys.empty()) {
            // Create a temporary SimplestHash with synthetic keys for compilation
            auto synth_base = std::make_unique<ir_sql_converter::AQPStmt>(
                std::vector<std::unique_ptr<ir_sql_converter::AQPStmt>>{},
                ir_sql_converter::SimplestNodeType::HashNode);
            ir_sql_converter::SimplestHash synth_hash(
                std::move(synth_base), std::move(synth_keys));

            if (!jit_compiler_)
              jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
                  /*use_o3=*/(jit_flags_ & AQP_JIT_OPT3) != 0);

            void *build_fn = jit_compiler_->CompileHashBuild(synth_hash, build_schema);
            if (build_fn) {
              auto *ctx = GetClientContext();
              if (!ctx->aqp_jit_context)
                ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
              ctx->aqp_jit_context->op_fns[eid] =
                  reinterpret_cast<duckdb::AQPOperatorFn>(build_fn);
              ctx->aqp_jit_context->flags |= duckdb::AQPJIT_OPERATOR;
              std::cerr << "[AQP-JIT] compiled hash build eid=0x" << std::hex << eid
                        << std::dec << " keys=" << join->join_conditions.size() << "\n";
            }
          }
        }
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
