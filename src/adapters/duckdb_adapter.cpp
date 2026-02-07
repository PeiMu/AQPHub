/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#include "adapters/duckdb_adapter.h"

#include <iostream>
#include <stdexcept>

namespace middleware {
DuckDBAdapter::DuckDBAdapter(const std::string &db_path) {
  db = std::make_unique<duckdb::DuckDB>(db_path);
  conn = std::make_unique<duckdb::Connection>(*db);
}

DuckDBAdapter::~DuckDBAdapter() { CleanUp(); }

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
}

void DuckDBAdapter::PreOptimizePlan() {
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
  auto optimized_plan = optimizer.PreOptimize(std::move(plan));

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

std::unique_ptr<ir_sql_converter::SimplestStmt>
DuckDBAdapter::ConvertPlanToIR() {
  auto context = GetClientContext();
  auto logical_plan = std::move(plan);
  auto ir = ir_sql_converter::ConvertDuckDBPlanToIR(
      *planner->binder, *context, logical_plan.get(), intermediate_table_map);

  return std::move(ir);
}

QueryResult DuckDBAdapter::ExecuteSQL(const std::string &sql) {
  QueryResult result;

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

void DuckDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name) {
  auto context = GetClientContext();

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  auto duckdb_result = conn->Query(sql);
  if (duckdb_result->HasError()) {
    throw std::runtime_error("Query failed: " + duckdb_result->GetError());
  }
  auto subquery_result = duckdb_result->Collection();
  auto data_chunk_index = planner->binder->GenerateTableIndex();

  subquery_index++;
  intermediate_table_map[data_chunk_index] = temp_table_name;
  // create a table from data chunk
  auto &default_entry = context->client_data->catalog_search_path->GetDefault();
  auto current_catalog = default_entry.catalog;
  auto current_schema = default_entry.schema;
  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  auto &types = subquery_result.Types();
  auto info = duckdb::make_uniq<duckdb::CreateTableInfo>(
      TEMP_CATALOG, DEFAULT_SCHEMA, temp_table_name);
  info->temporary = true;
  info->on_conflict = duckdb::OnCreateConflict::ERROR_ON_CONFLICT;

  // Track used column names to handle duplicates
  duckdb::case_insensitive_set_t used_column_names;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    std::string column_name = "col_" + std::to_string(i);

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
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  std::move(unique_column_name));
  }

  auto created_table = catalog.CreateTable(*context, std::move(info));
  auto &created_table_entry = created_table->Cast<duckdb::TableCatalogEntry>();
  int64_t created_table_size = subquery_result.Count();
  temp_table_card_.emplace(temp_table_name, created_table_size);
  //  const duckdb::vector<duckdb::unique_ptr<duckdb::BoundConstraint>>
  //      bound_constraints =
  //      planner->binder->BindConstraints(created_table_entry);

  auto &storage = created_table_entry.GetStorage();
  //  storage.LocalAppend(created_table_entry, *context, subquery_result,
  //                      bound_constraints, nullptr);
  storage.LocalAppend(created_table_entry, *context, subquery_result);

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

void DuckDBAdapter::CreateTempTable(const std::string &table_name,
                                    const QueryResult &result) {
  //  auto context = GetClientContext();
  //  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  //  auto info = duckdb::make_uniq<duckdb::CreateTableInfo>(TEMP_CATALOG,
  //  DEFAULT_SCHEMA, table_name); info->temporary = true; info->on_conflict =
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

void DuckDBAdapter::DropTempTable(const std::string &table_name) {
  ExecuteSQL("DROP TABLE IF EXISTS " + table_name);
}

bool DuckDBAdapter::TempTableExists(const std::string &table_name) {
  try {
    auto result = ExecuteSQL(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = '" +
        table_name + "'");
    return result.num_rows > 0 && result.rows[0][0] != "0";
  } catch (...) {
    return false;
  }
}

uint64_t
DuckDBAdapter::GetTempTableCardinality(const std::string &temp_table_name) {
  if (temp_table_card_.count(temp_table_name)) {
    return temp_table_card_[temp_table_name];
  }
  return 0; // Default if not found
}

std::pair<double, double>
DuckDBAdapter::GetEstimatedCost(const std::string &sql) {
  // Use EXPLAIN to get estimated cost and rows
  // DuckDB's EXPLAIN output format: we'll parse the cardinality from it
  try {
    std::string explain_sql = "EXPLAIN " + sql;
    auto duckdb_result = conn->Query(explain_sql);

    if (duckdb_result->HasError()) {
      std::cerr << "[DuckDB] EXPLAIN failed: " << duckdb_result->GetError()
                << std::endl;
      return {std::numeric_limits<double>::max(),
              std::numeric_limits<double>::max()};
    }

    // DuckDB doesn't expose cost directly like PostgreSQL
    // We'll estimate based on the plan - use cardinality as proxy
    // Parse the EXPLAIN output to find estimated cardinality

    double estimated_rows = 0.0;
    double estimated_cost = 0.0;

    // Collect all output lines
    std::string explain_output;
    while (true) {
      auto chunk = duckdb_result->Fetch();
      if (!chunk || chunk->size() == 0)
        break;

      for (size_t row = 0; row < chunk->size(); row++) {
        explain_output += chunk->GetValue(0, row).ToString() + "\n";
      }
    }

    // Parse cardinality from EXPLAIN output
    // DuckDB format shows "~X" for estimated rows
    // Look for patterns like "~1000" or "EC: 1000"
    size_t pos = 0;
    while ((pos = explain_output.find("~", pos)) != std::string::npos) {
      pos++;
      size_t end = pos;
      while (
          end < explain_output.size() &&
          (std::isdigit(explain_output[end]) || explain_output[end] == '.')) {
        end++;
      }
      if (end > pos) {
        double rows = std::stod(explain_output.substr(pos, end - pos));
        if (rows > estimated_rows) {
          estimated_rows = rows; // Take the max cardinality as estimate
        }
      }
    }

    // If we couldn't parse cardinality, use a default
    if (estimated_rows == 0.0) {
      estimated_rows = 1000.0; // Default fallback
    }

    // DuckDB doesn't have cost in the same way as PostgreSQL
    // Use estimated_rows as a proxy for cost
    estimated_cost = estimated_rows;

    return {estimated_cost, estimated_rows};

  } catch (const std::exception &e) {
    std::cerr << "[DuckDB] GetEstimatedCost exception: " << e.what()
              << std::endl;
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }
}

void DuckDBAdapter::CleanUp() {
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
} // namespace middleware