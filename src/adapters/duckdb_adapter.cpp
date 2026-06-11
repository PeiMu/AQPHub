/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#include "adapters/duckdb_adapter.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "duckdb/common/enums/physical_operator_type.hpp"

#ifdef HAVE_LLVM
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "jit/aqp_jit_hashtable.h"
#include "qjit/query_jit_steps.h"
#include "simplest_ir.h"
#include "storage/storage_plan.h"
#include "duckdb/common/types/hash.hpp"

// libpg_query's pg_functions.hpp does `#define fprintf(...)`, silently
// deleting every fprintf in this file; restore the real function.
#ifdef fprintf
#undef fprintf
#endif

// Resolve jit_flags bitfields to typed enums for IrToLlvmCompiler
static aqp_jit::SimdISA ResolveSimdISA(uint32_t flags) {
  // Legacy AQP_JIT_SIMD (0x20) maps to AUTO
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
#endif

// Write one JIT timing value (in ms) as a CSV column.
static void WriteJitTimingColumn(long us) {
  std::ofstream log_file;
  log_file.open("time_log.csv", std::ios_base::app);
  log_file << std::fixed << std::setprecision(3) << (us / 1000.0) << ", ";
  log_file.close();
}

namespace middleware {

#if IN_MEM_TMP_TABLE
// Anonymous namespace for table function internal state
namespace {

// Bind data: holds pointer to collection and override cardinality
struct TempCollectionFunctionData : public duckdb::FunctionData {
  duckdb::ColumnDataCollection *collection = nullptr;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;
  std::string table_name;
  // Non-owning pointer to column stats stored in StoredTempResult
  std::vector<duckdb::BaseStatistics*> column_stats;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
    auto result = duckdb::make_uniq<TempCollectionFunctionData>();
    result->collection = collection;
    result->has_override_cardinality = has_override_cardinality;
    result->override_cardinality = override_cardinality;
    result->table_name = table_name;
    result->column_stats = column_stats;
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
  duckdb::vector<duckdb::column_t> column_ids;
};

// Bind data for kernel temp table scan
struct KernelTempFunctionData : public duckdb::FunctionData {
  const middleware::storage::FlatTable *flat_table = nullptr;
  std::string table_name;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
    auto result = duckdb::make_uniq<KernelTempFunctionData>();
    result->flat_table = flat_table;
    result->table_name = table_name;
    return std::move(result);
  }

  bool Equals(const duckdb::FunctionData &other_p) const override {
    auto &other = other_p.Cast<KernelTempFunctionData>();
    return flat_table == other.flat_table;
  }
};

struct KernelTempGlobalState : public duckdb::GlobalTableFunctionState {
  uint64_t current_row = 0;
  duckdb::vector<duckdb::column_t> column_ids;
};

// TableFunctionInfo for kernel temp tables
struct KernelTempScanInfo : public duckdb::TableFunctionInfo {
  explicit KernelTempScanInfo(
      std::unordered_map<std::string, const middleware::storage::FlatTable *> *tables)
      : kernel_temps(tables) {}
  std::unordered_map<std::string, const middleware::storage::FlatTable *> *kernel_temps;
};

#ifdef HAVE_LLVM
// Bind data for qjit temp table scan (scan_qjit_temp)
struct QjitTempFunctionData : public duckdb::FunctionData {
  const qjit::QjitTable *table = nullptr;
  std::string table_name;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
    auto result = duckdb::make_uniq<QjitTempFunctionData>();
    result->table = table;
    result->table_name = table_name;
    result->has_override_cardinality = has_override_cardinality;
    result->override_cardinality = override_cardinality;
    return std::move(result);
  }

  bool Equals(const duckdb::FunctionData &other_p) const override {
    auto &other = other_p.Cast<QjitTempFunctionData>();
    return table == other.table;
  }
};

struct QjitTempGlobalState : public duckdb::GlobalTableFunctionState {
  uint64_t current_row = 0;
  duckdb::vector<duckdb::column_t> column_ids;
};

// TableFunctionInfo for qjit temp tables
struct QjitTempScanInfo : public duckdb::TableFunctionInfo {
  QjitTempScanInfo(
      std::unordered_map<std::string, std::unique_ptr<qjit::QjitTable>> *tables,
      std::unordered_map<std::string, middleware::QjitTempMeta> *meta)
      : qjit_temps(tables), qjit_meta(meta) {}
  std::unordered_map<std::string, std::unique_ptr<qjit::QjitTable>> *qjit_temps;
  std::unordered_map<std::string, middleware::QjitTempMeta> *qjit_meta;
};
#endif

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
  func.projection_pushdown = true;
  func.function_info =
      duckdb::make_shared_ptr<TempCollectionScanInfo>(&temp_collections_);

  // Register the table functions in the catalog
  duckdb::CreateTableFunctionInfo info(func);
  auto &catalog = duckdb::Catalog::GetSystemCatalog(*context);
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }
  catalog.CreateTableFunction(*context, info);

  // Register the kernel temp table function
  duckdb::TableFunction kernel_func(
      "scan_kernel_temp", {duckdb::LogicalType::VARCHAR},
      KernelTempScanFunc, KernelTempBind, KernelTempInitGlobal);
  kernel_func.cardinality = KernelTempCardinality;
  kernel_func.projection_pushdown = true;
  kernel_func.function_info =
      duckdb::make_shared_ptr<KernelTempScanInfo>(&kernel_temp_tables_);

  duckdb::CreateTableFunctionInfo kernel_info(kernel_func);
  catalog.CreateTableFunction(*context, kernel_info);

#ifdef HAVE_LLVM
  // Register the qjit temp table function (serves query-jit results
  // directly from the in-memory QjitTable — no CDC copy)
  duckdb::TableFunction qjit_func(
      "scan_qjit_temp", {duckdb::LogicalType::VARCHAR}, QjitTempScanFunc,
      QjitTempBind, QjitTempInitGlobal);
  qjit_func.cardinality = QjitTempCardinality;
  qjit_func.projection_pushdown = true;
  qjit_func.function_info = duckdb::make_shared_ptr<QjitTempScanInfo>(
      &qjit_temps_, &qjit_temp_meta_);

  duckdb::CreateTableFunctionInfo qjit_info(qjit_func);
  catalog.CreateTableFunction(*context, qjit_info);
#endif

  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }

  // Register the replacement scan
  auto &db_config = duckdb::DBConfig::GetConfig(*context);
  auto scan_data = duckdb::make_uniq<TempCollectionScanData>(
      &temp_collections_, &kernel_temp_tables_, &qjit_temp_meta_);
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
  result->table_name = table_name;
  for (auto &stat : stored.column_stats) {
    result->column_stats.push_back(stat.get());
  }
  return std::move(result);
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
DuckDBAdapter::TempCollectionInitGlobal(duckdb::ClientContext &context,
                                        duckdb::TableFunctionInitInput &input) {
  auto state = duckdb::make_uniq<TempCollectionGlobalState>();
  state->column_ids = input.column_ids;
  return std::move(state);
}

void DuckDBAdapter::TempCollectionScanFunc(duckdb::ClientContext &context,
                                           duckdb::TableFunctionInput &data,
                                           duckdb::DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<TempCollectionFunctionData>();
  auto &state = data.global_state->Cast<TempCollectionGlobalState>();

  if (!state.initialized) {
    if (!state.column_ids.empty()) {
      bind_data.collection->InitializeScan(state.scan_state, state.column_ids);
    } else {
      bind_data.collection->InitializeScan(state.scan_state);
    }
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

duckdb::unique_ptr<duckdb::BaseStatistics>
DuckDBAdapter::TempCollectionStatistics(
    duckdb::ClientContext &context,
    duckdb::TableFunctionGetStatisticsInput &input) {
  auto &data = input.bind_data->Cast<TempCollectionFunctionData>();
  auto col_idx = input.column_index.GetPrimaryIndex();
  if (col_idx < data.column_stats.size() && data.column_stats[col_idx]) {
    return data.column_stats[col_idx]->ToUnique();
  }
  return nullptr;
}

// Replacement scan callback
duckdb::unique_ptr<duckdb::TableRef>
DuckDBAdapter::TempCollectionReplacementScan(
    duckdb::ClientContext &context, duckdb::ReplacementScanInput &input,
    duckdb::optional_ptr<duckdb::ReplacementScanData> data) {

  auto &scan_data = data->Cast<TempCollectionScanData>();
  const auto &table_name = input.table_name;

  // Check temp collections first
  if (scan_data.temp_collections->find(table_name) !=
      scan_data.temp_collections->end()) {
    auto table_ref = duckdb::make_uniq<duckdb::TableFunctionRef>();
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(
        duckdb::Value(table_name)));
    table_ref->function = duckdb::make_uniq<duckdb::FunctionExpression>(
        "scan_temp_collection", std::move(children));
    table_ref->alias = table_name;
    return std::move(table_ref);
  }

  // Check kernel temp tables
  if (scan_data.kernel_temps &&
      scan_data.kernel_temps->find(table_name) !=
          scan_data.kernel_temps->end()) {
    auto table_ref = duckdb::make_uniq<duckdb::TableFunctionRef>();
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(
        duckdb::Value(table_name)));
    table_ref->function = duckdb::make_uniq<duckdb::FunctionExpression>(
        "scan_kernel_temp", std::move(children));
    table_ref->alias = table_name;
    return std::move(table_ref);
  }

  // Check qjit temp tables (the map is only populated under HAVE_LLVM;
  // scan_qjit_temp is registered iff entries can exist)
  if (scan_data.qjit_meta &&
      scan_data.qjit_meta->find(table_name) != scan_data.qjit_meta->end()) {
    auto table_ref = duckdb::make_uniq<duckdb::TableFunctionRef>();
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(
        duckdb::Value(table_name)));
    table_ref->function = duckdb::make_uniq<duckdb::FunctionExpression>(
        "scan_qjit_temp", std::move(children));
    table_ref->alias = table_name;
    return std::move(table_ref);
  }

  return nullptr;
}

// Kernel temp table callbacks
duckdb::unique_ptr<duckdb::FunctionData> DuckDBAdapter::KernelTempBind(
    duckdb::ClientContext &context, duckdb::TableFunctionBindInput &input,
    duckdb::vector<duckdb::LogicalType> &return_types,
    duckdb::vector<duckdb::string> &names) {

  auto &info = input.info->Cast<KernelTempScanInfo>();
  auto table_name = input.inputs[0].GetValue<duckdb::string>();

  auto it = info.kernel_temps->find(table_name);
  if (it == info.kernel_temps->end()) {
    throw duckdb::BinderException("Kernel temp table '%s' not found",
                                  table_name);
  }

  const auto *flat = it->second;
  for (size_t i = 0; i < flat->columns.size(); i++) {
    if (flat->columns[i].type == storage::FlatColumnType::INT32)
      return_types.push_back(duckdb::LogicalType::INTEGER);
    else
      return_types.push_back(duckdb::LogicalType::VARCHAR);
    names.push_back(flat->column_names[i]);
  }

  auto result = duckdb::make_uniq<KernelTempFunctionData>();
  result->flat_table = flat;
  result->table_name = table_name;
  return std::move(result);
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
DuckDBAdapter::KernelTempInitGlobal(duckdb::ClientContext &context,
                                    duckdb::TableFunctionInitInput &input) {
  auto state = duckdb::make_uniq<KernelTempGlobalState>();
  state->column_ids = input.column_ids;
  return std::move(state);
}

void DuckDBAdapter::KernelTempScanFunc(duckdb::ClientContext &context,
                                       duckdb::TableFunctionInput &data,
                                       duckdb::DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<KernelTempFunctionData>();
  auto &state = data.global_state->Cast<KernelTempGlobalState>();
  const auto *flat = bind_data.flat_table;

  uint64_t remaining = flat->row_count - state.current_row;
  if (remaining == 0) {
    output.SetCardinality(0);
    return;
  }

  uint64_t count = std::min(remaining, (uint64_t)STANDARD_VECTOR_SIZE);

  for (duckdb::idx_t out_idx = 0; out_idx < state.column_ids.size();
       out_idx++) {
    auto col_idx = state.column_ids[out_idx];
    if (col_idx == duckdb::COLUMN_IDENTIFIER_ROW_ID)
      continue;

    const auto &col = flat->columns[col_idx];
    auto &vec = output.data[out_idx];

    if (col.type == storage::FlatColumnType::INT32) {
      auto *src = reinterpret_cast<const int32_t *>(col.data.get()) +
                  state.current_row;
      auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
      std::memcpy(dst, src, count * sizeof(int32_t));
    } else {
      auto *str_vec = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
      for (uint64_t r = 0; r < count; r++) {
        uint32_t len;
        const char *ptr = col.GetVarchar(state.current_row + r, len);
        str_vec[r] = duckdb::StringVector::AddString(vec, ptr, len);
      }
    }
  }

  output.SetCardinality(count);
  state.current_row += count;
}

duckdb::unique_ptr<duckdb::NodeStatistics>
DuckDBAdapter::KernelTempCardinality(duckdb::ClientContext &context,
                                     const duckdb::FunctionData *bind_data) {
  auto &data = bind_data->Cast<KernelTempFunctionData>();
  auto card = data.flat_table->row_count;
  return duckdb::make_uniq<duckdb::NodeStatistics>(card, card);
}

#ifdef HAVE_LLVM
// Qjit temp table callbacks (scan_qjit_temp)
duckdb::unique_ptr<duckdb::FunctionData> DuckDBAdapter::QjitTempBind(
    duckdb::ClientContext &context, duckdb::TableFunctionBindInput &input,
    duckdb::vector<duckdb::LogicalType> &return_types,
    duckdb::vector<duckdb::string> &names) {

  auto &info = input.info->Cast<QjitTempScanInfo>();
  auto table_name = input.inputs[0].GetValue<duckdb::string>();

  auto mit = info.qjit_meta->find(table_name);
  auto tit = info.qjit_temps->find(table_name);
  if (mit == info.qjit_meta->end() || tit == info.qjit_temps->end()) {
    throw duckdb::BinderException("Qjit temp table '%s' not found",
                                  table_name);
  }

  auto &meta = mit->second;
  return_types = meta.types;
  for (auto &col_name : meta.column_names) {
    names.push_back(col_name);
  }

  auto result = duckdb::make_uniq<QjitTempFunctionData>();
  result->table = tit->second.get();
  result->table_name = table_name;
  result->has_override_cardinality = meta.has_override_cardinality;
  result->override_cardinality = meta.override_cardinality;
  return std::move(result);
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
DuckDBAdapter::QjitTempInitGlobal(duckdb::ClientContext &context,
                                  duckdb::TableFunctionInitInput &input) {
  auto state = duckdb::make_uniq<QjitTempGlobalState>();
  state->column_ids = input.column_ids;
  return std::move(state);
}

void DuckDBAdapter::QjitTempScanFunc(duckdb::ClientContext &context,
                                     duckdb::TableFunctionInput &data,
                                     duckdb::DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<QjitTempFunctionData>();
  auto &state = data.global_state->Cast<QjitTempGlobalState>();
  const auto *t = bind_data.table;

  uint64_t remaining = t->NumRows() - state.current_row;
  if (remaining == 0) {
    output.SetCardinality(0);
    return;
  }

  uint64_t count = std::min(remaining, (uint64_t)STANDARD_VECTOR_SIZE);

  for (duckdb::idx_t out_idx = 0; out_idx < state.column_ids.size();
       out_idx++) {
    auto col_idx = state.column_ids[out_idx];
    if (col_idx == duckdb::COLUMN_IDENTIFIER_ROW_ID)
      continue;

    auto &vec = output.data[out_idx];
    switch (t->Col(col_idx).dtype) {
    case AQP_DTYPE_INT32: {
      auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
      const auto *src =
          static_cast<const int32_t *>(t->Data(col_idx)) + state.current_row;
      std::memcpy(dst, src, count * sizeof(int32_t));
      break;
    }
    case AQP_DTYPE_INT64: {
      auto *dst = duckdb::FlatVector::GetData<int64_t>(vec);
      const auto *src =
          static_cast<const int64_t *>(t->Data(col_idx)) + state.current_row;
      std::memcpy(dst, src, count * sizeof(int64_t));
      break;
    }
    case AQP_DTYPE_VARCHAR: {
      // QjitString is bit-compatible with duckdb::string_t; long strings
      // point into the QjitTable arena, which outlives the scan (temps are
      // cleared only at per-query reset) — zero-copy views, no AddString.
      static_assert(sizeof(QjitString) == sizeof(duckdb::string_t),
                    "QjitString must be bit-compatible with string_t");
      auto *dst = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
      const auto *src = static_cast<const QjitString *>(t->Data(col_idx)) +
                        state.current_row;
      std::memcpy(dst, src, count * sizeof(duckdb::string_t));
      break;
    }
    default:
      throw duckdb::InternalException("scan_qjit_temp: unsupported dtype");
    }

    const uint64_t *vmask = t->Validity(col_idx);
    if (vmask) {
      auto &validity = duckdb::FlatVector::Validity(vec);
      for (uint64_t r = 0; r < count; r++) {
        if (!qjit::RowValid(vmask, state.current_row + r))
          validity.SetInvalid(r);
      }
    }
  }

  output.SetCardinality(count);
  state.current_row += count;
}

duckdb::unique_ptr<duckdb::NodeStatistics>
DuckDBAdapter::QjitTempCardinality(duckdb::ClientContext &context,
                                   const duckdb::FunctionData *bind_data) {
  auto &data = bind_data->Cast<QjitTempFunctionData>();
  duckdb::idx_t cardinality;
  if (data.has_override_cardinality) {
    cardinality = data.override_cardinality;
  } else {
    cardinality = data.table->NumRows();
  }
  return duckdb::make_uniq<duckdb::NodeStatistics>(cardinality, cardinality);
}
#endif

void DuckDBAdapter::RegisterKernelTemp(const std::string &name,
                                       const storage::FlatTable *table) {
  kernel_temp_tables_[name] = table;
}

void DuckDBAdapter::ClearKernelTemps() { kernel_temp_tables_.clear(); }

const StoredTempResult *
DuckDBAdapter::GetStoredTempResult(const std::string &name) const {
  auto it = temp_collections_.find(name);
  if (it == temp_collections_.end())
    return nullptr;
  return &it->second;
}

void DuckDBAdapter::CreateTempFromFlatTable(
    const storage::FlatTable &flat, const std::string &temp_table_name) {

  auto *ctx = GetClientContext();

  // Build DuckDB types
  duckdb::vector<duckdb::LogicalType> types;
  for (const auto &col : flat.columns) {
    if (col.type == storage::FlatColumnType::INT32)
      types.push_back(duckdb::LogicalType::INTEGER);
    else
      types.push_back(duckdb::LogicalType::VARCHAR);
  }

  // Build a ColumnDataCollection from the flat arrays
  auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(*ctx, types);

  duckdb::DataChunk chunk;
  chunk.Initialize(*ctx, types);

  uint64_t offset = 0;
  while (offset < flat.row_count) {
    uint64_t count = std::min(flat.row_count - offset,
                              (uint64_t)STANDARD_VECTOR_SIZE);
    chunk.Reset();
    chunk.SetCardinality(count);

    for (size_t c = 0; c < flat.columns.size(); c++) {
      const auto &col = flat.columns[c];
      auto &vec = chunk.data[c];

      if (col.type == storage::FlatColumnType::INT32) {
        auto *src = reinterpret_cast<const int32_t *>(col.data.get()) + offset;
        auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
        std::memcpy(dst, src, count * sizeof(int32_t));
      } else {
        auto *str_vec = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        for (uint64_t r = 0; r < count; r++) {
          uint32_t len;
          const char *ptr = col.GetVarchar(offset + r, len);
          str_vec[r] = duckdb::StringVector::AddString(vec, ptr, len);
        }
      }
    }

    collection->Append(chunk);
    offset += count;
  }

  // Set up DuckDB internal state
  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;
  temp_table_types = types;

  chunk_col_names_[data_chunk_index] = flat.column_names;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  flat.column_names[i]);
  }

  StoredTempResult stored;
  stored.collection = std::move(collection);
  stored.column_names = flat.column_names;
  temp_collections_[temp_table_name] = std::move(stored);

  temp_table_card_.emplace(temp_table_name,
                           static_cast<int64_t>(flat.row_count));
}

void DuckDBAdapter::RegisterTempMetadata(
    const storage::FlatTable &flat, const std::string &temp_table_name) {

  duckdb::vector<duckdb::LogicalType> types;
  for (const auto &col : flat.columns) {
    if (col.type == storage::FlatColumnType::INT32)
      types.push_back(duckdb::LogicalType::INTEGER);
    else
      types.push_back(duckdb::LogicalType::VARCHAR);
  }

  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;
  temp_table_types = types;

  chunk_col_names_[data_chunk_index] = flat.column_names;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  flat.column_names[i]);
  }

  temp_table_card_.emplace(temp_table_name,
                           static_cast<int64_t>(flat.row_count));
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

void DuckDBAdapter::Optimize() {
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
  auto optimized_plan = optimizer.Optimize(std::move(plan));

  // Store the optimized plan
  plan = std::move(optimized_plan);

  // Commit transaction if in auto-commit mode
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

// Walk a logical plan tree and update intermediate_table_map for any
// LogicalGet nodes that reference temp tables via the scan_temp_collection
// table function.  After ReOptimizeIR re-parses SQL, DuckDB assigns fresh
// table indices; this function synchronises them with the map so that
// ConvertDuckDBPlanToIR can identify temp tables correctly.
static void RebuildTempTableIndices(
    duckdb::LogicalOperator *op,
    std::unordered_map<unsigned int, std::string> &intermediate_table_map,
    std::unordered_map<unsigned int, std::vector<std::string>> &chunk_col_names,
    const std::unordered_map<std::string, StoredTempResult> &temp_collections,
    const std::unordered_map<std::string, QjitTempMeta> &qjit_temp_meta) {
  if (!op)
    return;
  if (op->type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto &get_op = op->Cast<duckdb::LogicalGet>();
    if (get_op.function.name == "scan_temp_collection" &&
        !get_op.parameters.empty()) {
      auto temp_name = get_op.parameters[0].GetValue<std::string>();
      auto it = temp_collections.find(temp_name);
      if (it != temp_collections.end()) {
        intermediate_table_map[get_op.table_index] = temp_name;
        chunk_col_names[get_op.table_index] = it->second.column_names;
      }
    } else if (get_op.function.name == "scan_qjit_temp" &&
               !get_op.parameters.empty()) {
      auto temp_name = get_op.parameters[0].GetValue<std::string>();
      auto it = qjit_temp_meta.find(temp_name);
      if (it != qjit_temp_meta.end()) {
        intermediate_table_map[get_op.table_index] = temp_name;
        chunk_col_names[get_op.table_index] = it->second.column_names;
      }
    }
  }
  for (auto &child : op->children)
    RebuildTempTableIndices(child.get(), intermediate_table_map,
                            chunk_col_names, temp_collections, qjit_temp_meta);
}

std::unique_ptr<ir_sql_converter::AQPStmt>
DuckDBAdapter::ReOptimizeIR(std::unique_ptr<ir_sql_converter::AQPStmt> ir) {
  if (!ir)
    return ir;

  auto sql = GenerateSQL(*ir, subquery_index);

  ParseSQL(sql);

  // Run full optimization (PreOptimize + MiddleOptimize on the fork,
  // Optimize() on vanilla DuckDB) so join ordering uses actual cardinalities.
  auto context = GetClientContext();
  if (context->transaction.IsAutoCommit())
    context->transaction.BeginTransaction();

  if (planner && planner->binder && plan && plan->RequireOptimizer()) {
    duckdb::Optimizer optimizer(*planner->binder, *context);
    // Temporarily disable the split flag so Optimize() runs the full
    // pipeline including JOIN_ORDER (the fork skips it when the flag is on).
    // This makes the code forward-compatible with vanilla DuckDB where
    // Optimize() always includes JOIN_ORDER.
    auto &cfg = context->config;
    bool saved = cfg.enable_dbshaker_query_split;
    cfg.enable_dbshaker_query_split = false;
    plan = optimizer.Optimize(std::move(plan));
    cfg.enable_dbshaker_query_split = saved;
  }

  if (context->transaction.IsAutoCommit())
    context->transaction.Commit();

  // After re-parsing, DuckDB assigned fresh table indices to temp tables
  // (they come through as LogicalGet via replacement scan, not as
  // LogicalColumnDataGet).  Update intermediate_table_map so that
  // ConvertPlanToIR can identify them and create SimplestChunk nodes.
  if (plan) {
    intermediate_table_map.clear();
    chunk_col_names_.clear();
    RebuildTempTableIndices(plan.get(), intermediate_table_map,
                            chunk_col_names_, temp_collections_,
                            qjit_temp_meta_);
  }

  // Convert re-optimized plan back to IR
  return ConvertPlanToIR();
}

void *DuckDBAdapter::GetLogicalPlan() {
  return static_cast<void *>(plan.get());
}

// Collect ALL FilterNodes in DFS order.
static void
CollectAllFilterNodes(const ir_sql_converter::AQPStmt *ir,
                      std::vector<const ir_sql_converter::AQPStmt *> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode)
    out.push_back(ir);
  for (const auto &child : ir->children)
    CollectAllFilterNodes(child.get(), out);
}

// Collect ALL ScanNodes in DFS order.
static void
CollectAllScanNodes(const ir_sql_converter::AQPStmt *ir,
                    std::vector<const ir_sql_converter::AQPStmt *> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode)
    out.push_back(ir);
  for (const auto &child : ir->children)
    CollectAllScanNodes(child.get(), out);
}

std::unique_ptr<ir_sql_converter::AQPStmt> DuckDBAdapter::ConvertPlanToIR() {
  auto context = GetClientContext();

#ifndef NDEBUG
  std::cerr << "[AQP-JIT-TRACE] === Logical plan before IR conversion ===\n";
  if (plan)
    plan->Print();
#endif

  auto ir = ir_sql_converter::ConvertDuckDBPlanToIR(
      *planner->binder, *context, plan.get(), intermediate_table_map,
      false, &chunk_col_names_);

#ifndef NDEBUG
  if (ir) {
    std::cerr << "[AQP-JIT-TRACE] === IR after conversion ===\n";
    std::vector<const ir_sql_converter::AQPStmt *> all_filters;
    CollectAllFilterNodes(ir.get(), all_filters);
    std::cerr << "[AQP-JIT-TRACE] IR has " << all_filters.size()
              << " FilterNode(s)\n";
    for (size_t fi = 0; fi < all_filters.size(); fi++) {
      std::cerr << "[AQP-JIT-TRACE]   filter[" << fi
                << "] qual_vec.size=" << all_filters[fi]->qual_vec.size()
                << "\n";
      for (size_t qi = 0; qi < all_filters[fi]->qual_vec.size(); qi++) {
        std::string qs = all_filters[fi]->qual_vec[qi]->Print(false);
        std::cerr << "[AQP-JIT-TRACE]     qual[" << qi << "]: " << qs << "\n";
      }
    }
    // Also dump all ScanNodes and their qual_vecs
    std::vector<const ir_sql_converter::AQPStmt *> all_scans;
    CollectAllScanNodes(ir.get(), all_scans);
    std::cerr << "[AQP-JIT-TRACE] IR has " << all_scans.size()
              << " ScanNode(s)\n";
    for (size_t si = 0; si < all_scans.size(); si++) {
      auto *sn = all_scans[si];
      auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(sn);
      std::cerr << "[AQP-JIT-TRACE]   scan[" << si
                << "] table_idx=" << scan->GetTableIndex()
                << " target_list.size=" << sn->target_list.size()
                << " qual_vec.size=" << sn->qual_vec.size() << "\n";
      for (size_t qi = 0; qi < sn->qual_vec.size(); qi++) {
        std::string qs = sn->qual_vec[qi]->Print(false);
        std::cerr << "[AQP-JIT-TRACE]     qual[" << qi << "]: " << qs << "\n";
      }
    }
  }
#endif

  return std::move(ir);
}

#ifdef HAVE_LLVM
namespace {
// Query-jit only analyzes SELECT statements. Non-SELECT SQL reaching
// ExecuteSQL (e.g. the FK splitters' per-table ANALYZE, or multi-statement
// combined SQL) must take the legacy path unchanged — ParseSQL would throw
// and the timing-column layout must match the baseline interpreter run.
bool IsSelectStatement(const std::string &sql) {
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
} // namespace
#endif

QueryResult DuckDBAdapter::ExecuteSQL(const std::string &sql) {
  QueryResult result;

#ifdef HAVE_LLVM
#ifndef NDEBUG
  std::cerr << "[AQP-JIT-TRACE] ExecuteSQL: jit_pending_ir_="
            << (void *)jit_pending_ir_ << " jit_flags_=0x" << std::hex
            << jit_flags_ << std::dec << "\n";
#endif

  // When JIT is requested but no pending IR (e.g. --split=none
  // --jit-level=expr), build the IR pipeline: parse -> filter optimize ->
  // convert to IR.
  auto timer = chrono_tic();

  // Query-jit (--jit-level=query): analyze the sub-query IR, compile it if
  // it is inside the v1 coverage (scan/filter/project, inner hash joins,
  // ungrouped MIN/MAX/COUNT aggregates), execute the compiled entry over
  // the qjit runtime; otherwise the interpreter runs the prepared statement
  // from the analysis pass. Mutually exclusive with the other JIT levels
  // (enforced in ParamConfig), so jit_pending_ir_ is never set here.
  if (query_jit_ && IsSelectStatement(sql)) {
    auto prep = PrepareWithQueryJitAnalysis(sql, "result");
    auto prepared = std::move(prep.prepared);
    std::unique_ptr<QjitCompiled> compiled;
    if (prepared && !prepared->HasError() && prep.ir)
      compiled = TryCompileQueryJit(*prep.ir, prep.analysis, *prepared,
                                    "result");
    if (enable_timing_) {
      // jit_compile column: analysis + prepare + (Phase 2) codegen time.
      auto analyze_us =
          chrono_toc(&timer, "ExecuteSQL::qjit analyze time\n", false);
      WriteJitTimingColumn(analyze_us);
    }
    if (prepared->HasError())
      throw std::runtime_error("Query failed: " + prepared->GetError());
    if (compiled) {
      qjit::QjitTable qtable(compiled->out_descs,
                             qjit_executor_->NumWorkers());
      int64_t rows = qjit_executor_->Run(
          reinterpret_cast<QjitQueryFn>(compiled->fn), compiled->srcs,
          compiled->ht_tuple_sizes, compiled->agg_descs,
          compiled->agg_output_cells, qtable, compiled->ht_key0_offsets);
      if (rows >= 0) {
        if (enable_timing_) {
          auto run_us =
              chrono_toc(&timer, "ExecuteSQL::qjit run time\n", false);
          WriteJitTimingColumn(run_us);
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
            // Match duckdb::Value::ToString: NULL prints "NULL", INTEGER/
            // BIGINT print the decimal digits, VARCHAR prints the raw bytes.
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
      // Entry returned an error code: fall through to the interpreter (the
      // result table is discarded; nothing was emitted).
      fprintf(stderr, "[AQP-QJIT] fallback:run-error(rc=%lld) label=result\n",
              (long long)rows);
    }
    // Drop any stale JIT context from previous sub-plan iterations.
    GetClientContext()->aqp_jit_context.reset();
    if (prepared->data && prepared->data->physical_plan)
      InjectTempTableJoinStats(prepared->data->physical_plan->Root());
    duckdb::vector<duckdb::Value> bound;
    auto duckdb_result = prepared->Execute(bound, false);
    if (enable_timing_) {
      auto run_us =
          chrono_toc(&timer, "ExecuteSQL::Execute sub-SQL time\n", false);
      WriteJitTimingColumn(run_us);
    }
    if (duckdb_result->HasError())
      throw std::runtime_error("Query failed: " + duckdb_result->GetError());
    result.num_columns = duckdb_result->ColumnCount();
    for (size_t i = 0; i < result.num_columns; i++)
      result.column_names.push_back(duckdb_result->ColumnName(i));
    result.num_rows = 0;
    while (true) {
      auto chunk = duckdb_result->Fetch();
      if (!chunk || 0 == chunk->size())
        break;
      for (size_t row = 0; row < chunk->size(); row++) {
        std::vector<std::string> row_data;
        row_data.reserve(result.num_columns);
        for (size_t col = 0; col < result.num_columns; col++)
          row_data.push_back(chunk->GetValue(col, row).ToString());
        result.rows.push_back(std::move(row_data));
        result.num_rows++;
      }
    }
    return result;
  }

  bool jit_active = false;
  if (!jit_pending_ir_ && (jit_flags_ & AQP_JIT_LEVEL_MASK)) {
    jit_active = true;
    ParseSQL(sql);
    Optimize();
    // ConvertPlanToIR reads plan via raw pointer without mutating it.
    auto whole_ir = ConvertPlanToIR();
    if (whole_ir) {
      SetJITPendingIR(whole_ir.get(), jit_flags_, TakePlan());
      owned_jit_ir_ = std::move(whole_ir);
      jit_pending_ir_ = owned_jit_ir_.get();
    } else {
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] ConvertPlanToIR failed → interpreter fallback\n";
#endif
    }
  }

  // If JIT pending IR is set, use Prepare path so we can walk the physical
  // plan and register compiled filters before execution.
  if (jit_pending_ir_) {
    std::unique_ptr<duckdb::PreparedStatement> prepared;
    if (jit_pending_plan_) {
      prepared = conn->PrepareFromPlan(std::move(jit_pending_plan_),
                                       planner->names, planner->types);
    } else {
      prepared = conn->Prepare(sql);
    }
    if (!prepared->HasError() && prepared->data &&
        prepared->data->physical_plan) {
      // Reset JIT context (stale function pointers would crash if dispatched).
      // Keep jit_compiler_ alive — reusing the LLJIT instance avoids
      // re-registering 15+ runtime symbols and re-initializing the JIT on every
      // sub-plan. But we DO release all previously compiled modules so the
      // machine code / IR allocations / symbol-table entries from the
      // previous query don't accumulate in the JIT for the adapter's lifetime
      // — important for long-running servers and multi-query benchmark runs.
      //
      // Ordering: clear the function-pointer map FIRST, then ResetModules.
      // The map holds raw pointers into the JIT's code; freeing the code
      // while the map still references it would leave dangling pointers
      // dispatchable. Multi-thread note: this entire reset sequence assumes
      // no concurrent dispatch is in flight on another thread. Today the
      // executor is synchronous, but when async / parallel dispatch is
      // added a lock around (reset + ResetModules + RegisterJIT) vs each
      // dispatch site is required to avoid use-after-free.
      GetClientContext()->aqp_jit_context.reset();
      if (jit_compiler_)
        jit_compiler_->ResetModules();
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE] ExecuteSQL: reset aqp_jit_context "
                   "(compiler reused)\n";
#endif
      jit_consumed_ir_filters_.clear();
      jit_consumed_ir_joins_.clear();
      RegisterJIT(prepared->data->physical_plan->Root(), *jit_pending_ir_);

      // Ensure JIT context always exists when JIT level is active, even if
      // no filters were compiled. This enables JIT-gated optimizations like
      // hash join prefetching.
      auto *ctx = GetClientContext();
      if (!ctx->aqp_jit_context) {
        ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        ctx->aqp_jit_context->flags = duckdb::AQPJIT_PIPELINE;
      }

      if (enable_timing_) {
        // Record total JIT compile time (includes Prepare + RegisterJIT)
        auto jit_compile_time =
            chrono_toc(&timer, "ExecuteSQL::jit compile time\n", false);
        WriteJitTimingColumn(jit_compile_time);
      }
#ifndef NDEBUG
      if (ctx->aqp_jit_context) {
        std::cerr << "[AQP-JIT] summary: flags=0x" << std::hex
                  << ctx->aqp_jit_context->flags << std::dec
                  << " expr_fns=" << ctx->aqp_jit_context->expr_fns.size()
                  << " op_fns=" << ctx->aqp_jit_context->op_fns.size()
                  << " proj_maps=" << ctx->aqp_jit_context->proj_col_maps.size()
                  << " agg_fns=" << ctx->aqp_jit_context->agg_fns.size()
                  << " pipeline_fns="
                  << ctx->aqp_jit_context->pipeline_fns.size()
                  << " scan_filter="
                  << ctx->aqp_jit_context->scan_filter_fns.size() << "\n";
      }
#endif
    } else if (enable_timing_) {
      // Keep the timing CSV rectangular: JIT-level runs always emit the
      // jit_compile column, even when registration was skipped.
      WriteJitTimingColumn(0);
    }

    jit_pending_ir_ = nullptr;
    if (prepared->data && prepared->data->physical_plan) {
      InjectTempTableJoinStats(prepared->data->physical_plan->Root());
    }
    duckdb::vector<duckdb::Value> bound;
    auto duckdb_result = prepared->Execute(bound, false);
    if (enable_timing_) {
      auto run_us =
          chrono_toc(&timer, "ExecuteSQL::Execute sub-SQL time\n", false);
      WriteJitTimingColumn(run_us);
    }
    if (duckdb_result->HasError())
      throw std::runtime_error("Query failed: " + duckdb_result->GetError());
    result.num_columns = duckdb_result->ColumnCount();
    for (size_t i = 0; i < result.num_columns; i++)
      result.column_names.push_back(duckdb_result->ColumnName(i));
    result.num_rows = 0;
    while (true) {
      auto chunk = duckdb_result->Fetch();
      if (!chunk || 0 == chunk->size())
        break;
      for (size_t row = 0; row < chunk->size(); row++) {
        std::vector<std::string> row_data;
        for (size_t col = 0; col < result.num_columns; col++)
          row_data.push_back(chunk->GetValue(col, row).ToString());
        result.rows.push_back(row_data);
        result.num_rows++;
      }
    }
  } else
#endif
  {
#ifdef HAVE_LLVM
    // Keep the timing CSV rectangular: JIT-level runs always emit the
    // jit_compile column, even when no IR was registered (e.g. the
    // cross-product guard skipped SetJITPendingIR).
    if (enable_timing_ && (jit_flags_ & AQP_JIT_LEVEL_MASK))
      WriteJitTimingColumn(0);
#endif
    // No JIT registration for this statement: drop any stale JIT context
    // from previous sub-plan iterations. EIDs are operator memory addresses
    // which get reused across plans, so leftover compiled fns / bloom
    // filters would mis-dispatch on this plan.
    GetClientContext()->aqp_jit_context.reset();
    auto duckdb_result = conn->Query(sql);
    if (enable_timing_) {
      auto run_us =
          chrono_toc(&timer, "ExecuteSQL::Execute sub-SQL time\n", false);
      WriteJitTimingColumn(run_us);
    }
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
  }
  return result;
}

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card) {
  std::chrono::high_resolution_clock::time_point timer;
  bool jit_active = false;
  if (enable_timing_) {
    timer = chrono_tic();
  }
  std::unique_ptr<duckdb::PreparedStatement> prepared;
#ifdef HAVE_LLVM
  std::unique_ptr<QjitCompiled> qjit_compiled;
  // Keepalive for a spec-jit HIT: holds the plan + IR the compiled fn's
  // metadata points into, plus the result types/names (no Prepare happens
  // on the main connection in that case).
  std::unique_ptr<QjitSpecCompiled> qjit_spec;
  if (query_jit_) {
    if (qjit_spec_hit_) {
      // Spec-jit HIT: the bg pool already compiled this sub-query on a spec
      // compiler. Resolve sources now (the consumed temp exists only since
      // the previous store) and skip Prepare entirely — the compiled fn
      // owns the result. Resolve failure falls through to the inline path
      // (same as a MISS).
      qjit_spec = std::move(qjit_spec_hit_);
      std::string reason;
      if (ResolveQjitSources(qjit_spec->plan, *qjit_spec->compiled, reason)) {
        qjit_compiled = std::move(qjit_spec->compiled);
#ifndef NDEBUG
        fprintf(stderr, "[AQP-QJIT] spec-hit label=%s\n",
                temp_table_name.c_str());
#endif
      } else {
        fprintf(stderr, "[AQP-QJIT] spec-hit-resolve-failed:%s label=%s\n",
                reason.c_str(), temp_table_name.c_str());
        qjit_spec.reset();
      }
    }
    if (!qjit_compiled) {
      // Query-jit: analyze + compile; execution below branches on
      // qjit_compiled. The interpreter path is unchanged and uses the same
      // prepared statement.
      auto prep = PrepareWithQueryJitAnalysis(sql, temp_table_name);
      prepared = std::move(prep.prepared);
      if (prepared && !prepared->HasError() && prep.ir)
        qjit_compiled = TryCompileQueryJit(*prep.ir, prep.analysis, *prepared,
                                           temp_table_name);
    }
    if (enable_timing_) {
      // jit_compile column: analysis + prepare + codegen time (≈0 on a spec
      // HIT — the compile happened on the bg pool).
      auto analyze_us = chrono_toc(
          &timer, "ExecuteSQLandCreateTempTable::qjit analyze time\n", false);
      WriteJitTimingColumn(analyze_us);
    }
  } else if (jit_pending_plan_) {
    jit_pending_plan_->ResolveOperatorTypes();
    auto plan_types = jit_pending_plan_->types;
    duckdb::vector<duckdb::string> plan_names;
    for (duckdb::idx_t i = 0; i < plan_types.size(); i++)
      plan_names.push_back("col" + std::to_string(i));
    prepared = conn->PrepareFromPlan(std::move(jit_pending_plan_),
                                     std::move(plan_names), std::move(plan_types));
  } else
#endif
  {
    prepared = conn->Prepare(sql);
  }
  // prepared is null only on a query-jit spec HIT (no main-connection
  // Prepare; the compiled fn owns the result).
  if (prepared && prepared->HasError()) {
    throw std::runtime_error("[DuckDB] Prepare failed: " +
                             prepared->GetError());
  }

#ifdef HAVE_LLVM
  // JIT: compile filters from the pending IR and register before execution.
  if (jit_pending_ir_ && prepared && prepared->data &&
      prepared->data->physical_plan) {
    jit_active = true;
    // Reset JIT context (stale function pointers would crash if dispatched).
    // Keep jit_compiler_ alive — reusing LLJIT avoids re-init overhead.
    // Release previously compiled modules (see ExecuteSQL for the ordering
    // invariant and the multi-thread caveat).
    GetClientContext()->aqp_jit_context.reset();
    if (jit_compiler_)
      jit_compiler_->ResetModules();
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] ExecuteSQLandCreateTempTable: reset "
                 "aqp_jit_context (compiler reused)\n";
#endif
    jit_consumed_ir_filters_.clear();
    jit_consumed_ir_joins_.clear();
    RegisterJIT(prepared->data->physical_plan->Root(), *jit_pending_ir_);

    // Ensure JIT context always exists when JIT level is active
    auto *ctx2 = GetClientContext();
    if (!ctx2->aqp_jit_context) {
      ctx2->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
      ctx2->aqp_jit_context->flags = duckdb::AQPJIT_PIPELINE;
    }

    if (enable_timing_) {
      // Record total JIT compile time (includes Prepare + RegisterJIT)
      auto jit_compile_time = chrono_toc(
          &timer, "ExecuteSQLandCreateTempTable::jit compile time\n", false);
      WriteJitTimingColumn(jit_compile_time);
    }
#ifndef NDEBUG
    if (ctx2->aqp_jit_context) {
      std::cerr << "[AQP-JIT] summary: flags=0x" << std::hex
                << ctx2->aqp_jit_context->flags << std::dec
                << " expr_fns=" << ctx2->aqp_jit_context->expr_fns.size()
                << " op_fns=" << ctx2->aqp_jit_context->op_fns.size()
                << " proj_maps=" << ctx2->aqp_jit_context->proj_col_maps.size()
                << " agg_fns=" << ctx2->aqp_jit_context->agg_fns.size()
                << " pipeline_fns="
                << ctx2->aqp_jit_context->pipeline_fns.size() << " scan_filter="
                << ctx2->aqp_jit_context->scan_filter_fns.size() << "\n";
    }
#endif
    jit_pending_ir_ = nullptr;
  } else if (jit_flags_ & AQP_JIT_LEVEL_MASK) {
    // Registration skipped (e.g. cross-product guard). Drop the stale JIT
    // context from the previous iteration: EIDs are operator memory
    // addresses which get reused across sub-plans, so leftover compiled
    // fns would mis-dispatch on this plan and corrupt results.
    GetClientContext()->aqp_jit_context.reset();
    if (enable_timing_) {
      // Keep the timing CSV rectangular: JIT-level runs always emit the
      // jit_compile column, even when registration was skipped.
      WriteJitTimingColumn(0);
    }
  }
#endif

  // Always clear stale bloom filters before executing a sub-plan.
  // EIDs are derived from operator memory addresses which get reused across
  // sub-plans, so leftover BFs would match wrong scans.
#ifdef HAVE_LLVM
  {
    auto *ctx3 = GetClientContext();
    if (ctx3->aqp_jit_context) {
      ctx3->aqp_jit_context->bloom_scan_filters.clear();
    }
  }
  // Register pending bloom filters (independent of JIT).
  if (!pending_bloom_filters_.empty() && prepared && prepared->data &&
      prepared->data->physical_plan) {
    auto *ctx3 = GetClientContext();
    if (!ctx3->aqp_jit_context) {
      ctx3->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
    }
    RegisterBloomFilters(prepared->data->physical_plan->Root(),
                         GetClientContext(), pending_bloom_filters_);
    pending_bloom_filters_.clear();
  }
#endif

  // Inject temp table column stats into PhysicalHashJoin nodes to enable
  // perfect hash join (direct array lookup instead of hash table).
  if (prepared && prepared->data && prepared->data->physical_plan) {
    InjectTempTableJoinStats(prepared->data->physical_plan->Root());
  }

  // Allocate the chunk index and build deduplicated column names BEFORE
  // execution: the speculative-compile hook needs the exact identity the
  // result temp table will have (the real path below reuses the same values).
  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;

#ifdef HAVE_LLVM
  // Spec HIT: no main-connection prepared statement — the payload carries
  // the spec Prepare's types/names (cross-checked against the compiled plan
  // by BuildQjitOutputDescs in the bg task).
  temp_table_types = qjit_spec ? qjit_spec->out_types : prepared->GetTypes();
  const auto &result_names =
      qjit_spec ? qjit_spec->out_names : prepared->GetNames();
#else
  temp_table_types = prepared->GetTypes();
  auto &result_names = prepared->GetNames();
#endif
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
  chunk_col_names_[data_chunk_index] = column_names;

#ifdef HAVE_LLVM
  if (post_prepare_hook_) {
    duckdb::idx_t est_card = 0;
    if (prepared && prepared->data && prepared->data->physical_plan)
      est_card = prepared->data->physical_plan->Root().estimated_cardinality;
    else if (qjit_spec)
      est_card = qjit_spec->est_card;
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       column_names, est_card, /*post_execute=*/false);
  }
#endif

  duckdb::unique_ptr<duckdb::ColumnDataCollection> subquery_result;
#ifdef HAVE_LLVM
  // §5.7: qjit results are served to later sub-queries straight from the
  // QjitTable via scan_qjit_temp — no CDC copy. Exception: a speculative
  // placeholder's bind data holds a raw pointer to the placeholder
  // collection, so that path keeps the CDC write-back (Combine below).
  bool qjit_stored = false;
  int64_t qjit_rows = 0;
  if (qjit_compiled) {
    auto qtable = std::make_unique<qjit::QjitTable>(
        qjit_compiled->out_descs, qjit_executor_->NumWorkers());
    int64_t rows =
        qjit_executor_->Run(reinterpret_cast<QjitQueryFn>(qjit_compiled->fn),
                            qjit_compiled->srcs, qjit_compiled->ht_tuple_sizes,
                            qjit_compiled->agg_descs,
                            qjit_compiled->agg_output_cells, *qtable,
                            qjit_compiled->ht_key0_offsets);
    if (rows >= 0) {
#ifndef NDEBUG
      fprintf(stderr, "[AQP-QJIT] exec label=%s rows=%lld\n",
              temp_table_name.c_str(), (long long)rows);
#endif
      auto pit = temp_collections_.find(temp_table_name);
      bool has_placeholder =
          pit != temp_collections_.end() && pit->second.is_placeholder;
      if (has_placeholder) {
        // temp_table_types == prepared types (or the spec payload's types on
        // a HIT, where prepared is null).
        subquery_result = QjitTableToCollection(*qtable, temp_table_types);
      } else {
        qjit_stored = true;
        qjit_rows = rows;
      }
      qjit_temps_[temp_table_name] = std::move(qtable);
    } else {
      fprintf(stderr, "[AQP-QJIT] fallback:run-error(rc=%lld) label=%s\n",
              (long long)rows, temp_table_name.c_str());
    }
  }
  if (!subquery_result && !qjit_stored)
#else
  if (!subquery_result)
#endif
  {
#ifdef HAVE_LLVM
    // Spec-HIT run error: no prepared statement exists — create the
    // interpreter fallback now.
    if (!prepared) {
      prepared = conn->Prepare(sql);
      if (prepared->HasError())
        throw std::runtime_error("[DuckDB] Prepare failed: " +
                                 prepared->GetError());
    }
#endif
    duckdb::vector<duckdb::Value> bound_values;
    subquery_result = prepared->ExecuteRow(bound_values, false);
  }
  if (enable_timing_) {
    auto run_us = chrono_toc(
        &timer, "ExecuteSQLandCreateTempTable::Execute sub-SQL time\n", false);
    WriteJitTimingColumn(run_us);
  }

  int64_t chunk_size;
#ifdef HAVE_LLVM
  if (qjit_stored) {
    // temp_table_types already holds prepared->GetTypes() (set above); the
    // compile gate cross-checked the QjitTable layout against it.
    chunk_size = qjit_rows;
  } else
#endif
  {
    chunk_size = subquery_result->Count();
    temp_table_types = subquery_result->Types();
  }

  // Compute per-column statistics before moving the collection
  std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> col_stats;
#ifdef HAVE_LLVM
  if (!qjit_stored)
#endif
  {
    auto &types = subquery_result->Types();
    col_stats.resize(types.size());
    for (size_t ci = 0; ci < types.size(); ci++) {
      col_stats[ci] = duckdb::BaseStatistics::CreateEmpty(types[ci]).ToUnique();
    }
    duckdb::ColumnDataScanState scan_state;
    subquery_result->InitializeScan(scan_state);
    duckdb::DataChunk chunk;
    chunk.Initialize(*GetClientContext(), types);
    bool has_rows = false;
    while (subquery_result->Scan(scan_state, chunk)) {
      if (chunk.size() == 0) break;
      has_rows = true;
      chunk.Flatten();
      for (size_t ci = 0; ci < types.size(); ci++) {
        auto &vec = chunk.data[ci];
        auto &validity = duckdb::FlatVector::Validity(vec);
        bool col_has_null = false;
        bool col_has_valid = false;
        for (duckdb::idx_t r = 0; r < chunk.size(); r++) {
          if (!validity.RowIsValid(r)) {
            col_has_null = true;
            continue;
          }
          col_has_valid = true;
        }
        if (col_has_null) col_stats[ci]->SetHasNullFast();
        if (col_has_valid) col_stats[ci]->SetHasNoNullFast();

        auto pt = types[ci].InternalType();
        switch (pt) {
        case duckdb::PhysicalType::INT8: {
          auto *data = duckdb::FlatVector::GetData<int8_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int8_t>(data[r]);
          break;
        }
        case duckdb::PhysicalType::INT16: {
          auto *data = duckdb::FlatVector::GetData<int16_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int16_t>(data[r]);
          break;
        }
        case duckdb::PhysicalType::INT32: {
          auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int32_t>(data[r]);
          break;
        }
        case duckdb::PhysicalType::INT64: {
          auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int64_t>(data[r]);
          break;
        }
        default:
          break;
        }
      }
      chunk.Reset();
    }
    (void)has_rows;
  }

  // Store the result. qjit-stored temps keep only binder metadata here
  // (scan_qjit_temp reads rows from qjit_temps_); everything else stores
  // the ColumnDataCollection in temp_collections_ (zero-copy). If a
  // speculative placeholder was registered for this temp, Combine the
  // result into it: a speculative Prepare's bind data holds a raw pointer to
  // the placeholder collection, so the object must survive.
#ifdef HAVE_LLVM
  if (qjit_stored) {
    QjitTempMeta meta;
    meta.column_names = std::move(column_names);
    meta.types = temp_table_types;
    qjit_temp_meta_[temp_table_name] = std::move(meta);
  } else
#endif
  {
    auto existing_it = temp_collections_.find(temp_table_name);
    if (existing_it != temp_collections_.end() &&
        existing_it->second.is_placeholder) {
      existing_it->second.collection->Combine(*subquery_result);
      existing_it->second.column_names = std::move(column_names);
      existing_it->second.column_stats = std::move(col_stats);
      existing_it->second.has_override_cardinality = false;
      existing_it->second.is_placeholder = false;
    } else {
      StoredTempResult stored;
      stored.collection = std::move(subquery_result);
      stored.column_names = std::move(column_names);
      stored.column_stats = std::move(col_stats);
      temp_collections_[temp_table_name] = std::move(stored);
    }
  }

  temp_table_card_.emplace(temp_table_name, chunk_size);

#ifdef HAVE_LLVM
  // Post-execute: the actual cardinality is now known and the placeholder
  // holds the real data + stats. Re-invoke the hook so the speculation layer
  // can keep the post-Prepare compile (estimate was close) or relaunch with
  // an accurately-planned speculative Prepare.
  if (post_prepare_hook_) {
    const auto &stored_names =
        qjit_stored ? qjit_temp_meta_[temp_table_name].column_names
                    : temp_collections_[temp_table_name].column_names;
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       stored_names, static_cast<duckdb::idx_t>(chunk_size),
                       /*post_execute=*/true);
  }
#endif

  if (enable_timing_) {
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

#ifdef HAVE_LLVM
void DuckDBAdapter::ExecuteSpeculativeAndCreateTempTable(
    duckdb::PreparedStatement &prepared, duckdb::Connection &spec_conn,
    const std::string &temp_table_name, bool update_temp_card) {
  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_) {
    timer = chrono_tic();
    WriteJitTimingColumn(0);
  }

  // Inject temp table stats into spec connection's physical plan
  if (prepared.data && prepared.data->physical_plan)
    InjectTempTableJoinStats(prepared.data->physical_plan->Root());

  // Allocate chunk index + dedup names BEFORE execution (see
  // ExecuteSQLandCreateTempTable) so the hook can launch the next speculation.
  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;

  temp_table_types = prepared.GetTypes();
  auto &result_names = prepared.GetNames();
  duckdb::case_insensitive_set_t used_column_names;
  std::vector<std::string> column_names;
  for (duckdb::idx_t i = 0; i < temp_table_types.size(); i++) {
    std::string column_name =
        (i < result_names.size() && !result_names[i].empty())
            ? result_names[i]
            : "col_" + std::to_string(i);
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
  chunk_col_names_[data_chunk_index] = column_names;

  if (post_prepare_hook_) {
    duckdb::idx_t est_card = 0;
    if (prepared.data && prepared.data->physical_plan)
      est_card = prepared.data->physical_plan->Root().estimated_cardinality;
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       column_names, est_card, /*post_execute=*/false);
  }

  duckdb::vector<duckdb::Value> bound_values;
  auto subquery_result = prepared.ExecuteRow(bound_values, false);
  if (enable_timing_) {
    auto run_us = chrono_toc(
        &timer, "ExecuteSpeculative::Execute sub-SQL time\n", false);
    WriteJitTimingColumn(run_us);
  }

  int64_t chunk_size = subquery_result->Count();
  temp_table_types = subquery_result->Types();

  std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> col_stats;
  {
    auto &types = subquery_result->Types();
    col_stats.resize(types.size());
    for (size_t ci = 0; ci < types.size(); ci++)
      col_stats[ci] = duckdb::BaseStatistics::CreateEmpty(types[ci]).ToUnique();
    duckdb::ColumnDataScanState scan_state;
    subquery_result->InitializeScan(scan_state);
    duckdb::DataChunk chunk;
    chunk.Initialize(*spec_conn.context, types);
    while (subquery_result->Scan(scan_state, chunk)) {
      if (chunk.size() == 0) break;
      chunk.Flatten();
      for (size_t ci = 0; ci < types.size(); ci++) {
        auto &vec = chunk.data[ci];
        auto &validity = duckdb::FlatVector::Validity(vec);
        bool col_has_null = false, col_has_valid = false;
        for (duckdb::idx_t r = 0; r < chunk.size(); r++) {
          if (!validity.RowIsValid(r)) { col_has_null = true; continue; }
          col_has_valid = true;
        }
        if (col_has_null) col_stats[ci]->SetHasNullFast();
        if (col_has_valid) col_stats[ci]->SetHasNoNullFast();
        auto pt = types[ci].InternalType();
        switch (pt) {
        case duckdb::PhysicalType::INT8: {
          auto *data = duckdb::FlatVector::GetData<int8_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int8_t>(data[r]);
          break;
        }
        case duckdb::PhysicalType::INT16: {
          auto *data = duckdb::FlatVector::GetData<int16_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int16_t>(data[r]);
          break;
        }
        case duckdb::PhysicalType::INT32: {
          auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int32_t>(data[r]);
          break;
        }
        case duckdb::PhysicalType::INT64: {
          auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
          for (duckdb::idx_t r = 0; r < chunk.size(); r++)
            if (validity.RowIsValid(r)) col_stats[ci]->UpdateNumericStats<int64_t>(data[r]);
          break;
        }
        default:
          break;
        }
      }
      chunk.Reset();
    }
  }

  auto existing_it = temp_collections_.find(temp_table_name);
  if (existing_it != temp_collections_.end() &&
      existing_it->second.is_placeholder) {
    existing_it->second.collection->Combine(*subquery_result);
    existing_it->second.column_names = std::move(column_names);
    existing_it->second.column_stats = std::move(col_stats);
    existing_it->second.has_override_cardinality = false;
    existing_it->second.is_placeholder = false;
  } else {
    StoredTempResult stored;
    stored.collection = std::move(subquery_result);
    stored.column_names = std::move(column_names);
    stored.column_stats = std::move(col_stats);
    temp_collections_[temp_table_name] = std::move(stored);
  }
  temp_table_card_.emplace(temp_table_name, chunk_size);

  // Post-execute hook: see ExecuteSQLandCreateTempTable.
  if (post_prepare_hook_) {
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       temp_collections_[temp_table_name].column_names,
                       static_cast<duckdb::idx_t>(chunk_size),
                       /*post_execute=*/true);
  }

  if (enable_timing_) {
    auto extra_materialize_time =
        chrono_toc(&timer, "Extra materialize time is\n", false);
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extra_materialize_time / 1000.0) << ", ";
    log_file.close();
  }
}

void DuckDBAdapter::RegisterPlaceholderTemp(
    const std::string &temp_table_name,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names, duckdb::idx_t est_card) {
  if (temp_collections_.count(temp_table_name) > 0)
    return; // already registered (real result or earlier placeholder)
  if (qjit_temp_meta_.count(temp_table_name) > 0)
    return; // real qjit-stored temp exists — an empty placeholder would
            // shadow it in the replacement scan (temp_collections_ is
            // checked first) and interpreter consumers would read 0 rows
  StoredTempResult stored;
  stored.collection =
      duckdb::make_uniq<duckdb::ColumnDataCollection>(*conn->context, types);
  stored.column_names = col_names;
  stored.has_override_cardinality = true;
  stored.override_cardinality = est_card == 0 ? 1 : est_card;
  stored.is_placeholder = true;
  temp_collections_[temp_table_name] = std::move(stored);
}
#endif

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
  qjit_temp_meta_.erase(table_name);
#ifdef HAVE_LLVM
  qjit_temps_.erase(table_name);
#endif
}

bool DuckDBAdapter::TempTableExists(const std::string &table_name) {
  return temp_collections_.count(table_name) > 0 ||
         qjit_temp_meta_.count(table_name) > 0;
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
  auto qit = qjit_temp_meta_.find(temp_table_name);
  if (qit != qjit_temp_meta_.end()) {
    qit->second.has_override_cardinality = true;
    qit->second.override_cardinality = cardinality;
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

std::string DuckDBAdapter::GetColumnName(const std::string &table_name,
                                         unsigned col_idx) {
  try {
    auto result = conn->Query("SELECT column_name FROM information_schema.columns "
                              "WHERE table_name = '" + table_name + "' "
                              "ORDER BY ordinal_position");
    if (!result->HasError()) {
      unsigned row = 0;
      while (true) {
        auto chunk = result->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (duckdb::idx_t i = 0; i < chunk->size(); i++) {
          if (row == col_idx) {
            return chunk->GetValue(0, i).ToString();
          }
          row++;
        }
      }
    }
  } catch (...) {
  }
  return "";
}

std::unordered_map<size_t, std::pair<int64_t, int64_t>>
DuckDBAdapter::GetTempTableMinMax(const std::string &temp_table_name) {
  std::unordered_map<size_t, std::pair<int64_t, int64_t>> result;
#if IN_MEM_TMP_TABLE
#ifdef HAVE_LLVM
  // qjit-stored temps (§5.7) have no CDC; compute from the QjitTable.
  if (!temp_collections_.count(temp_table_name)) {
    auto qit = qjit_temps_.find(temp_table_name);
    if (qit != qjit_temps_.end()) {
      const qjit::QjitTable &t = *qit->second;
      for (size_t col_idx = 0; col_idx < t.NumCols(); col_idx++) {
        const int32_t dt = t.Col(col_idx).dtype;
        if (dt != AQP_DTYPE_INT32 && dt != AQP_DTYPE_INT64)
          continue;
        int64_t min_val = std::numeric_limits<int64_t>::max();
        int64_t max_val = std::numeric_limits<int64_t>::min();
        bool found = false;
        for (uint64_t r = 0; r < t.NumRows(); r++) {
          if (!t.ValueValid(col_idx, r))
            continue;
          int64_t v = dt == AQP_DTYPE_INT32 ? t.GetI32(col_idx, r)
                                            : t.GetI64(col_idx, r);
          if (v < min_val) min_val = v;
          if (v > max_val) max_val = v;
          found = true;
        }
        if (found)
          result[col_idx] = {min_val, max_val};
      }
      return result;
    }
  }
#endif
  auto it = temp_collections_.find(temp_table_name);
  if (it == temp_collections_.end() || !it->second.collection)
    return result;

  auto &collection = *it->second.collection;
  auto &types = collection.Types();

  for (size_t col_idx = 0; col_idx < types.size(); col_idx++) {
    auto pt = types[col_idx].InternalType();
    if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
      continue;

    int64_t min_val = std::numeric_limits<int64_t>::max();
    int64_t max_val = std::numeric_limits<int64_t>::min();
    bool found = false;

    duckdb::ColumnDataScanState scan_state;
    collection.InitializeScan(scan_state);
    duckdb::DataChunk chunk;
    chunk.Initialize(*GetClientContext(), types);
    while (collection.Scan(scan_state, chunk)) {
      if (chunk.size() == 0)
        break;
      chunk.Flatten();
      auto &vec = chunk.data[col_idx];
      auto &validity = duckdb::FlatVector::Validity(vec);
      if (pt == duckdb::PhysicalType::INT32) {
        auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
        for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
          if (validity.RowIsValid(i)) {
            int64_t v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            found = true;
          }
        }
      } else {
        auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
        for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
          if (validity.RowIsValid(i)) {
            int64_t v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            found = true;
          }
        }
      }
      chunk.Reset();
    }
    if (found)
      result[col_idx] = {min_val, max_val};
  }
#endif
  return result;
}

std::vector<int64_t>
DuckDBAdapter::GetTempTableDistinctValues(const std::string &temp_table_name,
                                          size_t col_idx, size_t max_distinct) {
  std::vector<int64_t> result;
#if IN_MEM_TMP_TABLE
  auto it = temp_collections_.find(temp_table_name);
  if (it == temp_collections_.end() || !it->second.collection)
    return result;

  auto &collection = *it->second.collection;
  auto &types = collection.Types();
  if (col_idx >= types.size())
    return result;

  auto pt = types[col_idx].InternalType();
  if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
    return result;

  std::unordered_set<int64_t> distinct_set;
  duckdb::ColumnDataScanState scan_state;
  collection.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  chunk.Initialize(*GetClientContext(), types);
  bool exceeded = false;
  while (collection.Scan(scan_state, chunk)) {
    if (chunk.size() == 0) break;
    chunk.Flatten();
    auto &vec = chunk.data[col_idx];
    auto &validity = duckdb::FlatVector::Validity(vec);
    if (pt == duckdb::PhysicalType::INT32) {
      auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          distinct_set.insert(static_cast<int64_t>(data[i]));
          if (distinct_set.size() > max_distinct) { exceeded = true; break; }
        }
      }
    } else {
      auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          distinct_set.insert(data[i]);
          if (distinct_set.size() > max_distinct) { exceeded = true; break; }
        }
      }
    }
    chunk.Reset();
    if (exceeded) break;
  }
  if (!exceeded) {
    result.assign(distinct_set.begin(), distinct_set.end());
  }
#endif
  return result;
}

uint64_t
DuckDBAdapter::GetBaseTableCardinality(const std::string &table_name) {
  static std::unordered_map<std::string, uint64_t> cache;
  auto it = cache.find(table_name);
  if (it != cache.end())
    return it->second;

  try {
    auto duck_result = conn->Query("SELECT COUNT(*) FROM \"" + table_name + "\"");
    if (duck_result && !duck_result->HasError()) {
      auto chunk = duck_result->Fetch();
      if (chunk && chunk->size() > 0) {
        uint64_t rows = chunk->GetValue(0, 0).GetValue<uint64_t>();
        cache[table_name] = rows;
        return rows;
      }
    }
  } catch (...) {}
  cache[table_name] = 0;
  return 0;
}

#ifdef HAVE_LLVM
// Bloom filter over a qjit-stored temp (§5.7: no CDC to scan). Same sizing
// and bit-insert scheme as the ColumnDataCollection variant below.
static DuckDBAdapter::BloomFilterInfo
BuildBloomFilterFromQjitTable(const qjit::QjitTable &table, size_t col_idx) {
  DuckDBAdapter::BloomFilterInfo info;
  if (col_idx >= table.NumCols())
    return info;
  const int32_t dt = table.Col(col_idx).dtype;
  if (dt != AQP_DTYPE_INT32 && dt != AQP_DTYPE_INT64)
    return info;

  uint64_t temp_card = table.NumRows();

  constexpr uint64_t kMinBits = 512;
  constexpr uint64_t kBitsPerKey = 12;
  constexpr uint64_t kMaxSectors = (1ULL << 26);
  uint64_t min_bits = std::max(kMinBits, temp_card * kBitsPerKey);
  uint64_t num_sectors = 1;
  while (num_sectors < (min_bits >> 6)) num_sectors <<= 1;
  if (num_sectors > kMaxSectors) num_sectors = kMaxSectors;

  info.bf_data.resize(num_sectors, 0);
  info.bitmask = num_sectors - 1;

  constexpr uint64_t kShiftMask = 0x3F3F3F3F3F3F3F3F;
  constexpr int kNBits = 4;

  auto insert_one = [&](uint64_t hash) {
    uint64_t offset = hash & info.bitmask;
    uint64_t shifts = hash & kShiftMask;
    auto shifts_8 = reinterpret_cast<const uint8_t *>(&shifts);
    uint64_t mask = 0;
    for (int i = 8 - kNBits; i < 8; i++) {
      mask |= (1ULL << shifts_8[i]);
    }
    info.bf_data[offset] |= mask;
  };

  for (uint64_t r = 0; r < table.NumRows(); r++) {
    if (!table.ValueValid(col_idx, r))
      continue;
    if (dt == AQP_DTYPE_INT32)
      insert_one(duckdb::Hash<int32_t>(table.GetI32(col_idx, r)));
    else
      insert_one(duckdb::Hash<int64_t>(table.GetI64(col_idx, r)));
  }
  return info;
}
#endif

DuckDBAdapter::BloomFilterInfo
DuckDBAdapter::BuildBloomFilter(const std::string &temp_table_name,
                                size_t col_idx,
                                uint64_t temp_card) {
  BloomFilterInfo info;
  (void)temp_card;
#if IN_MEM_TMP_TABLE
#ifdef HAVE_LLVM
  if (!temp_collections_.count(temp_table_name)) {
    auto qit = qjit_temps_.find(temp_table_name);
    if (qit != qjit_temps_.end())
      return BuildBloomFilterFromQjitTable(*qit->second, col_idx);
  }
#endif
  auto it = temp_collections_.find(temp_table_name);
  if (it == temp_collections_.end() || !it->second.collection)
    return info;
  info = BuildBloomFilter(*it->second.collection, col_idx, *GetClientContext());
#endif
  return info;
}

DuckDBAdapter::BloomFilterInfo
DuckDBAdapter::BuildBloomFilter(duckdb::ColumnDataCollection &collection,
                                size_t col_idx, duckdb::ClientContext &ctx) {
  BloomFilterInfo info;
  auto &types = collection.Types();
  if (col_idx >= types.size())
    return info;

  auto pt = types[col_idx].InternalType();
  if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
    return info;

  uint64_t temp_card = collection.Count();

  // Size the Bloom filter: 12 bits per key, rounded up to power-of-2 sectors.
  // Each sector is 64 bits. This gives ~1.5% false positive rate with 4 hash bits.
  constexpr uint64_t kMinBits = 512;
  constexpr uint64_t kBitsPerKey = 12;
  constexpr uint64_t kMaxSectors = (1ULL << 26);
  uint64_t min_bits = std::max(kMinBits, temp_card * kBitsPerKey);
  uint64_t num_sectors = 1;
  while (num_sectors < (min_bits >> 6)) num_sectors <<= 1;
  if (num_sectors > kMaxSectors) num_sectors = kMaxSectors;

  info.bf_data.resize(num_sectors, 0);
  info.bitmask = num_sectors - 1;

  constexpr uint64_t kShiftMask = 0x3F3F3F3F3F3F3F3F;
  constexpr int kNBits = 4;

  auto insert_one = [&](uint64_t hash) {
    uint64_t offset = hash & info.bitmask;
    uint64_t shifts = hash & kShiftMask;
    auto shifts_8 = reinterpret_cast<const uint8_t *>(&shifts);
    uint64_t mask = 0;
    for (int i = 8 - kNBits; i < 8; i++) {
      mask |= (1ULL << shifts_8[i]);
    }
    info.bf_data[offset] |= mask;
  };

  duckdb::ColumnDataScanState scan_state;
  collection.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  chunk.Initialize(ctx, types);
  while (collection.Scan(scan_state, chunk)) {
    if (chunk.size() == 0) break;
    chunk.Flatten();
    auto &vec = chunk.data[col_idx];
    auto &validity = duckdb::FlatVector::Validity(vec);
    if (pt == duckdb::PhysicalType::INT32) {
      auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          insert_one(duckdb::Hash<int32_t>(data[i]));
        }
      }
    } else {
      auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          insert_one(duckdb::Hash<int64_t>(data[i]));
        }
      }
    }
    chunk.Reset();
  }
  return info;
}

DuckDBAdapter::BloomFilterInfo
DuckDBAdapter::BuildBloomFilterFromCollection(
    duckdb::ColumnDataCollection &collection,
    size_t col_idx, duckdb::ClientContext &ctx) {
  BloomFilterInfo info;
  auto &types = collection.Types();
  if (col_idx >= types.size()) return info;

  uint64_t temp_card = collection.Count();
  if (temp_card == 0) return info;

  constexpr uint64_t kMinBits = 512;
  constexpr uint64_t kBitsPerKey = 12;
  constexpr uint64_t kMaxSectors = (1ULL << 26);
  uint64_t min_bits = std::max(kMinBits, temp_card * kBitsPerKey);
  uint64_t num_sectors = 1;
  while (num_sectors < (min_bits >> 6)) num_sectors <<= 1;
  if (num_sectors > kMaxSectors) num_sectors = kMaxSectors;

  info.bf_data.resize(num_sectors, 0);
  info.bitmask = num_sectors - 1;

  constexpr uint64_t kShiftMask = 0x3F3F3F3F3F3F3F3F;
  constexpr int kNBits = 4;

  auto insert_one = [&](uint64_t hash) {
    uint64_t offset = hash & info.bitmask;
    uint64_t shifts = hash & kShiftMask;
    auto shifts_8 = reinterpret_cast<const uint8_t *>(&shifts);
    uint64_t mask = 0;
    for (int i = 8 - kNBits; i < 8; i++)
      mask |= (1ULL << shifts_8[i]);
    info.bf_data[offset] |= mask;
  };

  duckdb::ColumnDataScanState scan_state;
  collection.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  chunk.Initialize(ctx, types);
  duckdb::Vector hashes(duckdb::LogicalType::HASH);
  while (collection.Scan(scan_state, chunk)) {
    if (chunk.size() == 0) break;
    duckdb::VectorOperations::Hash(chunk.data[col_idx], hashes, chunk.size());
    auto hash_data = duckdb::FlatVector::GetData<duckdb::hash_t>(hashes);
    for (duckdb::idx_t i = 0; i < chunk.size(); i++)
      insert_one(hash_data[i]);
    chunk.Reset();
  }
  return info;
}

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
  qjit_temp_meta_.clear();
#endif
#ifdef HAVE_LLVM
  qjit_temps_.clear();
  qjit_spec_hit_.reset();
#endif
  plan.reset();
  planner.reset();
  conn.reset();
  db.reset();
  table_column_mappings.clear();
  intermediate_table_map.clear();
  temp_table_card_.clear();
}

void DuckDBAdapter::ResetQueryState() {
#if IN_MEM_TMP_TABLE
  temp_collections_.clear();
  qjit_temp_meta_.clear();
#endif
#ifdef HAVE_LLVM
  qjit_temps_.clear();
  qjit_spec_hit_.reset();
#endif
  kernel_temp_tables_.clear();
  plan.reset();
  planner.reset();
  table_column_mappings.clear();
  intermediate_table_map.clear();
  chunk_col_names_.clear();
  temp_table_card_.clear();
  subquery_index = 0;
  temp_table_index_ = 0;
#ifdef HAVE_LLVM
  jit_pending_ir_ = nullptr;
  owned_jit_ir_.reset();
  jit_pending_plan_.reset();
  jit_consumed_ir_filters_.clear();
  jit_consumed_ir_joins_.clear();
  temp_col_ranges_.clear();
  pending_bloom_filters_.clear();
  if (jit_compiler_)
    jit_compiler_->ResetModules();
  auto ctx = GetClientContext();
  if (ctx)
    ctx->aqp_jit_context.reset();
#endif
}

void DuckDBAdapter::LoadTablesFromCSV(const std::string &schema_path,
                                       const std::string &csv_dir) {
  std::ifstream schema_file(schema_path);
  if (!schema_file.is_open())
    throw std::runtime_error("Cannot open schema file: " + schema_path);

  // Read the full schema file and extract table names + CREATE TABLE statements
  std::string schema_sql((std::istreambuf_iterator<char>(schema_file)),
                          std::istreambuf_iterator<char>());

  std::vector<std::string> table_names;
  std::string::size_type pos = 0;
  while ((pos = schema_sql.find("CREATE TABLE", pos)) != std::string::npos) {
    auto start = pos + 13;
    while (start < schema_sql.size() && schema_sql[start] == ' ')
      start++;
    auto end = start;
    while (end < schema_sql.size() && schema_sql[end] != ' ' && schema_sql[end] != '(')
      end++;
    if (end > start)
      table_names.push_back(schema_sql.substr(start, end - start));
    pos = end;
  }

  std::cout << "[AQP] Loading " << table_names.size()
            << " tables from CSV into memory..." << std::endl;
  auto load_start = std::chrono::high_resolution_clock::now();

  // Create tables with proper schema (types, constraints)
  auto create_result = conn->Query(schema_sql);
  if (create_result->HasError())
    throw std::runtime_error("Schema creation failed: " +
                             create_result->GetError());

  // Load data from CSV into each table
  for (const auto &table_name : table_names) {
    std::string csv_path = csv_dir;
    if (!csv_path.empty() && csv_path.back() != '/')
      csv_path += '/';
    csv_path += table_name + ".csv";

    std::string sql = "COPY " + table_name + " FROM '" + csv_path +
                      "' (HEADER, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
    auto result = conn->Query(sql);
    if (result->HasError())
      throw std::runtime_error("CSV load failed for " + table_name + ": " +
                               result->GetError());
  }

  auto load_end = std::chrono::high_resolution_clock::now();
  auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     load_end - load_start).count();
  std::cout << "[AQP] CSV loading complete in " << load_ms << " ms"
            << std::endl;
}

duckdb::ClientContext *DuckDBAdapter::GetClientContext() {
  return conn->context.get();
}

#ifdef HAVE_LLVM
// Recursively find the first AQPStmt node of type FilterNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstFilterNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return nullptr;
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
  if (!ir)
    return nullptr;
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
  if (!ir)
    return nullptr;
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
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *j = FindFirstJoinNode(child.get()))
      return j;
  return nullptr;
}

// Walk the physical plan downward to find a leaf PhysicalTableScan.
static duckdb::PhysicalTableScan *
FindBuildSideTableScan(duckdb::PhysicalOperator &op) {
  if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN)
    return &op.Cast<duckdb::PhysicalTableScan>();
  for (auto &child : op.children)
    if (auto *s = FindBuildSideTableScan(child.get()))
      return s;
  return nullptr;
}

static duckdb::PhysicalColumnDataScan *
FindBuildSideColumnDataScan(duckdb::PhysicalOperator &op) {
  if (op.type == duckdb::PhysicalOperatorType::COLUMN_DATA_SCAN)
    return &op.Cast<duckdb::PhysicalColumnDataScan>();
  for (auto &child : op.children)
    if (auto *s = FindBuildSideColumnDataScan(child.get()))
      return s;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Query-jit (--jit-level=query): build-side annotation + Phase 0 analysis.
// ---------------------------------------------------------------------------

// Collect logical_table_index of every PhysicalTableScan in a subtree.
static void CollectPhysScanIndices(duckdb::PhysicalOperator &op,
                                   std::unordered_set<unsigned int> &out) {
  if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
    auto &scan = op.Cast<duckdb::PhysicalTableScan>();
    out.insert(static_cast<unsigned int>(scan.logical_table_index));
  }
  for (auto &child : op.children)
    CollectPhysScanIndices(child.get(), out);
}

// Collect table_index of every ScanNode/ChunkNode in an IR subtree.
static void CollectIRSourceIndices(const ir_sql_converter::AQPStmt *ir,
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

// Collect mutable JoinNodes in DFS order.
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

namespace {

bool PlanContainsOpType(const duckdb::LogicalOperator &op,
                        duckdb::LogicalOperatorType type) {
  if (op.type == type)
    return true;
  for (auto &child : op.children)
    if (PlanContainsOpType(*child, type))
      return true;
  return false;
}

// Collect IR source indices, keeping only those present in `universe` (the
// physical TABLE_SCAN indices). IR sources outside the universe — e.g. the
// ColumnDataGet chunk an IN-list rewrite joins against, which is a
// COLUMN_DATA_SCAN without a table index physically — are stripped so the
// set comparison still works.
void CollectIRSourceIndicesFiltered(
    const ir_sql_converter::AQPStmt *ir,
    const std::unordered_set<unsigned int> &universe,
    std::unordered_set<unsigned int> &out) {
  std::unordered_set<unsigned int> raw;
  CollectIRSourceIndices(ir, raw);
  for (auto v : raw)
    if (universe.count(v))
      out.insert(v);
}

void AnnotateBuildSidesRec(duckdb::PhysicalOperator &op,
                           ir_sql_converter::AQPStmt &ir,
                           const std::unordered_set<unsigned int> &universe) {
  for (auto &child : op.children)
    AnnotateBuildSidesRec(child.get(), ir, universe);
  if (op.type != duckdb::PhysicalOperatorType::HASH_JOIN ||
      op.children.size() != 2)
    return;

  std::unordered_set<unsigned int> probe_set, build_set;
  CollectPhysScanIndices(op.children[0].get(), probe_set);
  CollectPhysScanIndices(op.children[1].get(), build_set);
  // Joins against table-index-less sources (e.g. IN-list chunks) cannot be
  // matched by table-index sets; they stay unannotated.
  if (probe_set.empty() || build_set.empty())
    return;

  std::vector<ir_sql_converter::SimplestJoin *> joins;
  CollectIRJoinNodes(&ir, joins);
  for (auto *join : joins) {
    if (join->GetBuildChild() != -1 || join->children.size() != 2)
      continue;
    std::unordered_set<unsigned int> c0_set, c1_set;
    CollectIRSourceIndicesFiltered(join->children[0].get(), universe, c0_set);
    CollectIRSourceIndicesFiltered(join->children[1].get(), universe, c1_set);
    if (c0_set.empty() || c1_set.empty())
      continue;
    if (c0_set == probe_set && c1_set == build_set) {
      join->SetBuildChild(1); // physical kept logical orientation
      return;
    }
    if (c0_set == build_set && c1_set == probe_set) {
      join->SetBuildChild(0); // physical planner flipped the sides
      return;
    }
  }
#ifndef NDEBUG
  auto dump_set = [](const std::unordered_set<unsigned int> &s) {
    std::string r = "{";
    for (auto v : s)
      r += std::to_string(v) + ",";
    r += "}";
    return r;
  };
  std::cerr << "[AQP-QJIT] AnnotateBuildSides: no IR JoinNode matched a "
               "physical HASH_JOIN probe="
            << dump_set(probe_set) << " build=" << dump_set(build_set) << "\n";
  for (auto *join : joins) {
    std::unordered_set<unsigned int> c0_set, c1_set;
    if (join->children.size() == 2) {
      CollectIRSourceIndices(join->children[0].get(), c0_set);
      CollectIRSourceIndices(join->children[1].get(), c1_set);
    }
    std::cerr << "[AQP-QJIT]   IR join build_child=" << join->GetBuildChild()
              << " c0=" << dump_set(c0_set) << " c1=" << dump_set(c1_set)
              << "\n";
  }
#endif
}

} // namespace

// Annotate each IR JoinNode with the build side chosen by DuckDB's physical
// planner. Requires FRESH IR: the IR and the physical plan must come from
// the same optimized logical plan (one binder), so each physical HASH_JOIN
// corresponds to exactly one IR JoinNode with identical table-index sets
// under its children — only the build/probe orientation can differ.
// Unmatched joins keep build_child = -1 (analysis then rejects).
void DuckDBAdapter::AnnotateBuildSides(duckdb::PhysicalOperator &op,
                                       ir_sql_converter::AQPStmt &ir) {
  std::unordered_set<unsigned int> universe;
  CollectPhysScanIndices(op, universe);
  AnnotateBuildSidesRec(op, ir, universe);
}

DuckDBAdapter::QueryJitPrep
DuckDBAdapter::PrepareWithQueryJitAnalysis(const std::string &sql,
                                           const std::string &label) {
  QueryJitPrep prep;
  // Save the adapter's planning state. The splitter owns planner/plan across
  // sub-queries (GenerateTableIndex must keep drawing from the ORIGINAL
  // binder's number space), and intermediate_table_map / chunk_col_names_
  // describe the original binder's indices. The fresh parse below uses its
  // own binder whose indices restart at 0, so all its effects on these
  // members must stay local to this function.
  auto saved_planner = std::move(planner);
  auto saved_plan = std::move(plan);
  auto saved_map = intermediate_table_map;
  auto saved_chunk_names = chunk_col_names_;

  std::unique_ptr<duckdb::PreparedStatement> prepared;
  const char *stage = "parse";
  try {
    ParseSQL(sql);
    stage = "optimize";
    Optimize();
    // Fresh binder ⇒ temp-table scans got fresh indices; rebuild the maps
    // (restored below) so ConvertPlanToIR resolves ChunkNodes.
    intermediate_table_map.clear();
    chunk_col_names_.clear();
    stage = "rebuild-temp-indices";
    RebuildTempTableIndices(plan.get(), intermediate_table_map,
                            chunk_col_names_, temp_collections_,
                            qjit_temp_meta_);
    // ConvertPlanToIR prints "Do not support yet" (and D_ASSERTs) on plans
    // the converter can't represent — the only kind seen on JOB is
    // EMPTY_RESULT (optimizer proved a sub-query empty). Reject before
    // conversion to keep stdout identical to the baseline run.
    bool plan_empty_result =
        PlanContainsOpType(*plan, duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT);
    std::unique_ptr<ir_sql_converter::AQPStmt> ir;
    if (!plan_empty_result) {
      stage = "convert-ir";
      ir = ConvertPlanToIR();
    }
    stage = "prepare-from-plan";
    prepared = conn->PrepareFromPlan(TakePlan(), planner->names,
                                     planner->types);
    if (prepared && !prepared->HasError() && prepared->data &&
        prepared->data->physical_plan) {
      if (ir) {
        AnnotateBuildSides(prepared->data->physical_plan->Root(), *ir);
        prep.analysis = qjit::AnalyzeQueryJit(*ir, label);
        prep.ir = std::move(ir);
      }
#ifndef NDEBUG
      else if (plan_empty_result) {
        fprintf(stderr, "[AQP-QJIT] reject:plan-empty-result label=%s\n",
                label.c_str());
      } else {
        fprintf(stderr, "[AQP-QJIT] reject:ir-conversion-failed label=%s\n",
                label.c_str());
      }
    } else {
      fprintf(stderr, "[AQP-QJIT] reject:prepare-failed label=%s\n",
              label.c_str());
    }
#else
    }
#endif
  } catch (std::exception &e) {
    fprintf(stderr, "[AQP-QJIT] reject:exception(stage=%s: %s) label=%s\n",
            stage, e.what(), label.c_str());
    prepared.reset();
  }

  planner = std::move(saved_planner);
  plan = std::move(saved_plan);
  intermediate_table_map = std::move(saved_map);
  chunk_col_names_ = std::move(saved_chunk_names);

  // The analysis path failed before producing a usable prepared statement:
  // fall back to a plain Prepare so the interpreter still runs the sub-query.
  if (!prepared || prepared->HasError()) {
    prepared = conn->Prepare(sql);
    // This prepared statement was NOT the one the IR was annotated/analyzed
    // against — never compile from it.
    prep.ir.reset();
    prep.analysis = qjit::QjitAnalysisResult{};
  }
  prep.prepared = std::move(prepared);
  return prep;
}

bool DuckDBAdapter::BuildQjitOutputDescs(const qjit::QjitQueryPlan &plan,
                                         duckdb::PreparedStatement &prepared,
                                         QjitCompiled &compiled,
                                         std::string &reason) {
  const qjit::QjitStep &last = plan.steps.back();

  // Output cross-check against the prepared statement — under query-jit the
  // compiled code OWNS the result, so shape/type disagreement with DuckDB's
  // binding is fatal and must fall back.
  const auto &types = prepared.GetTypes();
  const auto &names = prepared.GetNames();
  auto out_name = [&](size_t i) {
    return (i < names.size() && !names[i].empty()) ? names[i]
                                                   : "col_" + std::to_string(i);
  };
  if (plan.has_agg) {
    if (types.size() != plan.agg_output_cells.size()) {
      reason = "output:count-mismatch";
      return false;
    }
    for (size_t i = 0; i < types.size(); i++) {
      const qjit::QjitAggCellPlan &cell =
          last.agg_cells[(size_t)plan.agg_output_cells[i]];
      int32_t dt;
      if (cell.fn == qjit::QjitAggFn::Count ||
          cell.fn == qjit::QjitAggFn::CountStar) {
        if (types[i].id() != duckdb::LogicalTypeId::BIGINT) {
          reason = "output:agg-type";
          return false;
        }
        dt = AQP_DTYPE_INT64;
      } else if (cell.arg.dtype == AQP_DTYPE_INT32) {
        // MIN/MAX(INTEGER) keeps INTEGER. SUM(INTEGER) is HUGEINT in
        // DuckDB and rejects here (v1).
        if (types[i].id() != duckdb::LogicalTypeId::INTEGER) {
          reason = "output:agg-type";
          return false;
        }
        dt = AQP_DTYPE_INT32;
      } else if (cell.arg.dtype == AQP_DTYPE_VARCHAR) {
        if (types[i].id() != duckdb::LogicalTypeId::VARCHAR) {
          reason = "output:agg-type";
          return false;
        }
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
    if (types.size() != last.outputs.size()) {
      reason = "output:count-mismatch";
      return false;
    }
    for (size_t i = 0; i < types.size(); i++) {
      int32_t dt = last.outputs[i].dtype;
      if ((dt == AQP_DTYPE_INT32 &&
           types[i].id() != duckdb::LogicalTypeId::INTEGER) ||
          (dt == AQP_DTYPE_VARCHAR &&
           types[i].id() != duckdb::LogicalTypeId::VARCHAR) ||
          (dt != AQP_DTYPE_INT32 && dt != AQP_DTYPE_VARCHAR)) {
        reason = "output:type-mismatch";
        return false;
      }
      compiled.out_descs.push_back({dt, out_name(i)});
    }
  }
  return true;
}

bool DuckDBAdapter::ResolveQjitSources(const qjit::QjitQueryPlan &plan,
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
      const qjit::QjitTable *tmp = GetOrLoadQjitTemp(st.source_table, reason);
      if (!tmp)
        return false;
      if (!qjit_executor_->ResolveTempSource(*tmp, st.cols, compiled.srcs[k],
                                             reason))
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

void DuckDBAdapter::RegisterQjitRuntimeSymbols(
    aqp_jit::IrToLlvmCompiler *comp) {
  comp->RegisterRuntimeSymbol("qjit_parallel_for", (void *)&qjit_parallel_for);
  comp->RegisterRuntimeSymbol("qjit_buffer_grow", (void *)&qjit_buffer_grow);
  comp->RegisterRuntimeSymbol("qjit_ht_append", (void *)&qjit_ht_append);
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
}

std::unique_ptr<DuckDBAdapter::QjitCompiled>
DuckDBAdapter::TryCompileQueryJit(const ir_sql_converter::AQPStmt &ir,
                                  const qjit::QjitAnalysisResult &analysis,
                                  duckdb::PreparedStatement &prepared,
                                  const std::string &label) {
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
    return nullptr; // AnalyzeQueryJit already traced the reject reason
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded())
    return fallback("no-storage-plan");

  qjit::QjitQueryPlan plan;
  std::string reason;
  if (!qjit::BuildExecutionSteps(ir, plan, reason))
    return fallback(reason);

  auto compiled = std::make_unique<QjitCompiled>();
  if (!BuildQjitOutputDescs(plan, prepared, *compiled, reason))
    return fallback(reason);

  if (!ResolveQjitSources(plan, *compiled, reason))
    return fallback(reason);

  compiled->ht_tuple_sizes.reserve(plan.hts.size());
  compiled->ht_key0_offsets.reserve(plan.hts.size());
  for (const auto &ht : plan.hts) {
    compiled->ht_tuple_sizes.push_back(ht.tuple_size);
    compiled->ht_key0_offsets.push_back(ht.prefix_bytes); // key 0 lives first
  }

  EnsureJITCompiler();
  if (!jit_compiler_)
    return fallback("no-jit-compiler");
  if (!qjit_syms_registered_) {
    RegisterQjitRuntimeSymbols(jit_compiler_.get());
    qjit_syms_registered_ = true;
  }
  // Drop the previous sub-query's compiled module. Safe: Run() blocks until
  // the pool is quiescent, so no compiled code is in flight here, and
  // query-jit is mutually exclusive with the other JIT levels (no
  // aqp_jit_context pointers reference these modules).
  jit_compiler_->ResetModules();

  compiled->fn = jit_compiler_->CompileQuerySteps(plan);
  if (!compiled->fn)
    return fallback("compile-failed");
  return compiled;
}

std::unique_ptr<DuckDBAdapter::QjitSpecCompiled>
DuckDBAdapter::SpeculativeQueryJitCompile(const std::string &sql,
                                          const std::string &label,
                                          duckdb::Connection &spec_conn,
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

  auto *spec_ctx = spec_conn.context.get();
  try {
    // Parse + plan + optimize with bg-local state (mirrors ParseSQL/Optimize;
    // the adapter's planner/plan members belong to the main thread). The spec
    // connection's ClientConfig has the same compile-time
    // enable_dbshaker_query_split default as the main connection, so the
    // optimizer behaves identically — plan parity by construction.
    std::unique_ptr<duckdb::Planner> local_planner;
    duckdb::unique_ptr<duckdb::LogicalOperator> local_plan;
    bool auto_commit = spec_ctx->transaction.IsAutoCommit();
    if (auto_commit)
      spec_ctx->transaction.BeginTransaction();
    try {
      duckdb::Parser parser(spec_ctx->GetParserOptions());
      parser.ParseQuery(sql);
      if (parser.statements.empty() ||
          parser.statements[0]->type !=
              duckdb::StatementType::SELECT_STATEMENT)
        throw std::runtime_error("not a single SELECT statement");
      local_planner = std::make_unique<duckdb::Planner>(*spec_ctx);
      local_planner->CreatePlan(std::move(parser.statements[0]));
      local_plan = std::move(local_planner->plan);
      if (!local_plan)
        throw std::runtime_error("failed to create logical plan");
      if (local_plan->RequireOptimizer()) {
        duckdb::Optimizer optimizer(*local_planner->binder, *spec_ctx);
        local_plan = optimizer.Optimize(std::move(local_plan));
      }
    } catch (...) {
      if (auto_commit)
        spec_ctx->transaction.Commit();
      throw;
    }
    if (auto_commit)
      spec_ctx->transaction.Commit();

    if (PlanContainsOpType(*local_plan,
                           duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT))
      return reject("plan-empty-result");

    // Fresh binder ⇒ fresh temp-scan indices; bg-local maps so the converter
    // can resolve ChunkNodes. This reads the live temp maps — the same
    // discipline as the spec Prepare's replacement-scan binding: launches
    // are post-execute, so every consumed temp is already stored, and a
    // placeholder Combine mutates entry fields, not the map shape.
    std::unordered_map<unsigned int, std::string> local_map;
    std::unordered_map<unsigned int, std::vector<std::string>> local_names;
    RebuildTempTableIndices(local_plan.get(), local_map, local_names,
                            temp_collections_, qjit_temp_meta_);

    auto ir = ir_sql_converter::ConvertDuckDBPlanToIR(
        *local_planner->binder, *spec_ctx, local_plan.get(), local_map, false,
        &local_names);
    if (!ir)
      return reject("ir-conversion-failed");

    auto prepared = spec_conn.PrepareFromPlan(
        std::move(local_plan), local_planner->names, local_planner->types);
    if (!prepared || prepared->HasError() || !prepared->data ||
        !prepared->data->physical_plan)
      return reject("prepare-failed");

    AnnotateBuildSides(prepared->data->physical_plan->Root(), *ir);
    auto analysis = qjit::AnalyzeQueryJit(*ir, label);
    if (!analysis.accepted)
      return nullptr; // AnalyzeQueryJit already traced the reject reason

    auto payload = std::make_unique<QjitSpecCompiled>();
    std::string reason;
    if (!qjit::BuildExecutionSteps(*ir, payload->plan, reason))
      return reject(reason);
    payload->compiled = std::make_unique<QjitCompiled>();
    if (!BuildQjitOutputDescs(payload->plan, *prepared, *payload->compiled,
                              reason))
      return reject(reason);
    payload->compiled->ht_tuple_sizes.reserve(payload->plan.hts.size());
    payload->compiled->ht_key0_offsets.reserve(payload->plan.hts.size());
    for (const auto &ht : payload->plan.hts) {
      payload->compiled->ht_tuple_sizes.push_back(ht.tuple_size);
      payload->compiled->ht_key0_offsets.push_back(ht.prefix_bytes);
    }

    // The caller (bg task) ran spec_comp->ResetModules() and the qjit
    // runtime symbols were registered at spec-compiler creation.
    payload->compiled->fn = spec_comp->CompileQuerySteps(payload->plan);
    if (!payload->compiled->fn)
      return reject("compile-failed");

    payload->ir = std::move(ir);
    payload->out_types = prepared->GetTypes();
    payload->out_names = prepared->GetNames();
    payload->est_card =
        prepared->data->physical_plan->Root().estimated_cardinality;
    return payload;
  } catch (std::exception &e) {
    return reject(std::string("exception:") + e.what());
  }
}

const qjit::QjitTable *
DuckDBAdapter::GetOrLoadQjitTemp(const std::string &name,
                                 std::string &reason) {
  auto it = qjit_temps_.find(name);
  if (it != qjit_temps_.end())
    return it->second.get();
  auto tc = temp_collections_.find(name);
  if (tc == temp_collections_.end() || !tc->second.collection) {
    reason = "source:tmp-missing:" + name;
    return nullptr;
  }
  // Speculative placeholders are Combine()d in-place later — not final.
  if (tc->second.is_placeholder) {
    reason = "source:tmp-placeholder:" + name;
    return nullptr;
  }
  auto table = CollectionToQjitTable(tc->second, reason);
  if (!table)
    return nullptr;
  const qjit::QjitTable *raw = table.get();
  qjit_temps_[name] = std::move(table);
  return raw;
}

std::unique_ptr<qjit::QjitTable>
DuckDBAdapter::CollectionToQjitTable(const StoredTempResult &stored,
                                     std::string &reason) {
  auto &collection = *stored.collection;
  const auto &types = collection.Types();
  std::vector<qjit::QjitTable::ColumnDesc> descs;
  descs.reserve(types.size());
  for (size_t i = 0; i < types.size(); i++) {
    int32_t dt;
    switch (types[i].id()) {
    case duckdb::LogicalTypeId::INTEGER:
      dt = AQP_DTYPE_INT32;
      break;
    case duckdb::LogicalTypeId::BIGINT:
      dt = AQP_DTYPE_INT64;
      break;
    case duckdb::LogicalTypeId::VARCHAR:
      dt = AQP_DTYPE_VARCHAR;
      break;
    default:
      reason = "source:tmp-cdc-dtype";
      return nullptr;
    }
    std::string name = i < stored.column_names.size()
                           ? stored.column_names[i]
                           : "col_" + std::to_string(i);
    descs.push_back({dt, std::move(name)});
  }

  auto table = std::make_unique<qjit::QjitTable>(std::move(descs), 1);
  for (auto &chunk : collection.Chunks()) {
    const duckdb::idx_t n = chunk.size();
    for (duckdb::idx_t r = 0; r < n; r++) {
      for (duckdb::idx_t c = 0; c < chunk.ColumnCount(); c++) {
        auto &vec = chunk.data[c];
        if (!duckdb::FlatVector::Validity(vec).RowIsValid(r)) {
          table->AppendNull(0, c);
          continue;
        }
        switch (table->Col(c).dtype) {
        case AQP_DTYPE_INT32:
          table->AppendI32(0, c,
                           duckdb::FlatVector::GetData<int32_t>(vec)[r]);
          break;
        case AQP_DTYPE_INT64:
          table->AppendI64(0, c,
                           duckdb::FlatVector::GetData<int64_t>(vec)[r]);
          break;
        default: { // AQP_DTYPE_VARCHAR
          const duckdb::string_t &s =
              duckdb::FlatVector::GetData<duckdb::string_t>(vec)[r];
          table->AppendStr(
              0, c,
              qjit::MakeString(s.GetData(), (uint32_t)s.GetSize()));
          break;
        }
        }
      }
      table->FinishRow(0);
    }
  }
  table->Finalize();
  return table;
}

duckdb::unique_ptr<duckdb::ColumnDataCollection>
DuckDBAdapter::QjitTableToCollection(
    const qjit::QjitTable &table,
    const duckdb::vector<duckdb::LogicalType> &types) {
  auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(
      *GetClientContext(), types);
  duckdb::DataChunk chunk;
  chunk.Initialize(*GetClientContext(), types);
  const uint64_t nrows = table.NumRows();
  for (uint64_t base = 0; base < nrows; base += STANDARD_VECTOR_SIZE) {
    const uint64_t n =
        std::min<uint64_t>(STANDARD_VECTOR_SIZE, nrows - base);
    chunk.Reset();
    for (size_t col = 0; col < table.NumCols(); col++) {
      auto &vec = chunk.data[col];
      auto &validity = duckdb::FlatVector::Validity(vec);
      if (table.Col(col).dtype == AQP_DTYPE_INT32) {
        auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
        const auto *src = static_cast<const int32_t *>(table.Data(col));
        std::memcpy(dst, src + base, n * sizeof(int32_t));
        for (uint64_t r = 0; r < n; r++)
          if (!table.ValueValid(col, base + r))
            validity.SetInvalid(r);
      } else if (table.Col(col).dtype == AQP_DTYPE_INT64) { // COUNT outputs
        auto *dst = duckdb::FlatVector::GetData<int64_t>(vec);
        const auto *src = static_cast<const int64_t *>(table.Data(col));
        std::memcpy(dst, src + base, n * sizeof(int64_t));
        for (uint64_t r = 0; r < n; r++)
          if (!table.ValueValid(col, base + r))
            validity.SetInvalid(r);
      } else { // AQP_DTYPE_VARCHAR (gated at compile)
        auto *dst = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        for (uint64_t r = 0; r < n; r++) {
          if (!table.ValueValid(col, base + r)) {
            validity.SetInvalid(r);
            continue;
          }
          QjitString s = table.GetStr(col, base + r);
          dst[r] = duckdb::StringVector::AddString(
              vec, qjit::StringData(s), qjit::StringLen(s));
        }
      }
    }
    chunk.SetCardinality(n);
    collection->Append(chunk);
  }
  return collection;
}

// Find a ScanNode with the given table_index in an IR subtree.
static const ir_sql_converter::AQPStmt *
FindScanByTableIndex(const ir_sql_converter::AQPStmt *ir, unsigned int tidx) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan =
        static_cast<const ir_sql_converter::SimplestScan *>(ir);
    if (scan->GetTableIndex() == tidx)
      return ir;
  }
  for (const auto &child : ir->children)
    if (auto *s = FindScanByTableIndex(child.get(), tidx))
      return s;
  return nullptr;
}

// Find a ScanNode with the given table_index that is a direct descendant
// (not through any nested JoinNode).  This walks through Filter/Projection
// wrappers but stops at JoinNode boundaries.
static const ir_sql_converter::AQPStmt *
FindDirectScanByTableIndex(const ir_sql_converter::AQPStmt *ir,
                           unsigned int tidx) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    if (scan->GetTableIndex() == tidx)
      return ir;
  }
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    return nullptr;
  for (const auto &child : ir->children)
    if (auto *s = FindDirectScanByTableIndex(child.get(), tidx))
      return s;
  return nullptr;
}

// Recursive helper: find an unconsumed IR JoinNode whose conditions
// contain a pair (build_tidx, probe_tidx) in either order.
static const ir_sql_converter::AQPStmt *
FindJoinNodeByTablePair(
    const ir_sql_converter::AQPStmt *ir, unsigned int tidx_a,
    unsigned int tidx_b,
    const std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode &&
      !consumed.count(ir) && ir->children.size() > 1) {
    auto *join =
        static_cast<const ir_sql_converter::SimplestJoin *>(ir);
    for (const auto &cond : join->join_conditions) {
      unsigned lt = cond->left_attr->GetTableIndex();
      unsigned rt = cond->right_attr->GetTableIndex();
      if ((lt == tidx_a && rt == tidx_b) ||
          (lt == tidx_b && rt == tidx_a))
        return ir;
    }
  }
  for (const auto &child : ir->children)
    if (auto *j = FindJoinNodeByTablePair(child.get(), tidx_a, tidx_b,
                                          consumed))
      return j;
  return nullptr;
}

// Recursive helper: find an unconsumed IR JoinNode whose conditions
// reference build_tidx and whose direct child (not through nested
// JoinNodes) contains a ScanNode for build_tidx.
static const ir_sql_converter::AQPStmt *
FindJoinNodeByDirectScan(
    const ir_sql_converter::AQPStmt *ir, unsigned int build_tidx,
    const std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode &&
      !consumed.count(ir) && ir->children.size() > 1) {
    bool direct_child = FindDirectScanByTableIndex(ir->children[0].get(),
                                                   build_tidx) ||
                        FindDirectScanByTableIndex(ir->children[1].get(),
                                                   build_tidx);
    if (direct_child) {
      auto *join =
          static_cast<const ir_sql_converter::SimplestJoin *>(ir);
      for (const auto &cond : join->join_conditions) {
        if (cond->left_attr->GetTableIndex() == build_tidx ||
            cond->right_attr->GetTableIndex() == build_tidx)
          return ir;
      }
    }
  }
  for (const auto &child : ir->children)
    if (auto *j = FindJoinNodeByDirectScan(child.get(), build_tidx, consumed))
      return j;
  return nullptr;
}

// Find the IR JoinNode for a physical HASH_JOIN.  Matching strategy:
//  1. If both build and probe table indices are known, find the JoinNode
//     whose conditions reference both (exact match).
//  2. Fall back to build_tidx-only with direct-child constraint.
static const ir_sql_converter::AQPStmt *
FindJoinNodeByBuildTableIndex(
    const ir_sql_converter::AQPStmt *ir, unsigned int build_tidx,
    const std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed,
    unsigned int probe_tidx = UINT_MAX) {
  // Phase 1: exact match by table pair.
  if (probe_tidx != UINT_MAX) {
    if (auto *exact = FindJoinNodeByTablePair(ir, build_tidx, probe_tidx,
                                              consumed))
      return exact;
  }
  // Phase 2: direct-child + condition match.
  if (auto *direct = FindJoinNodeByDirectScan(ir, build_tidx, consumed))
    return direct;
  // No looser fallback: a subtree-only match can pick a JoinNode whose
  // conditions reference unrelated tables (the synth_keys mismatch bug).
  return nullptr;
}

// Recursively find the first AggregateNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstAggregateNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::AggregateNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *a = FindFirstAggregateNode(child.get()))
      return a;
  return nullptr;
}

// Recursively find the first table index referenced by any leaf attribute
// in an AQPExpr tree.  Returns UINT_MAX if no attribute is found.
static unsigned int
FirstTableIdxFromExpr(const ir_sql_converter::AQPExpr *expr) {
  if (!expr)
    return UINT_MAX;
  using ir_sql_converter::SimplestNodeType;
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
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
    if (t != UINT_MAX)
      return t;
    return FirstTableIdxFromExpr(l->right_expr.get());
  }
  default:
    return UINT_MAX;
  }
}

// Walk the IR tree and collect the set of table_index values from ScanNodes.
static void
CollectIRTableIndices(const ir_sql_converter::AQPStmt *ir,
                      std::unordered_set<unsigned int> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    out.insert(scan->GetTableIndex());
  }
  for (const auto &child : ir->children)
    CollectIRTableIndices(child.get(), out);
}

// Build table_name → IR table_index map.  If a name appears more than once
// (self-join within the sub-plan), it is moved to `ambiguous` so callers
// know not to use name-based matching for that table.
static void
CollectIRTableNameToIndex(const ir_sql_converter::AQPStmt *ir,
                          std::unordered_map<std::string, unsigned int> &out,
                          std::unordered_set<std::string> &ambiguous) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    const auto &name = scan->GetTableName();
    auto it = out.find(name);
    if (it != out.end()) {
      if (it->second != scan->GetTableIndex())
        ambiguous.insert(name);
    } else {
      out[name] = scan->GetTableIndex();
    }
  }
  for (const auto &child : ir->children)
    CollectIRTableNameToIndex(child.get(), out, ambiguous);
}

// Collect (table_idx, col_idx) pairs referenced by an IR filter's quals.
static void
CollectIRFilterCols(const ir_sql_converter::AQPStmt *filter_ir,
                    std::set<std::pair<unsigned int, unsigned int>> &out) {
  using ir_sql_converter::SimplestNodeType;
  std::function<void(const ir_sql_converter::AQPExpr *)> walk =
      [&](const ir_sql_converter::AQPExpr *expr) {
        if (!expr)
          return;
        switch (expr->GetNodeType()) {
        case SimplestNodeType::VarConstComparisonNode: {
          auto *c =
              static_cast<const ir_sql_converter::SimplestVarConstComparison *>(
                  expr);
          if (c->attr)
            out.emplace(static_cast<unsigned int>(c->attr->GetTableIndex()),
                        static_cast<unsigned int>(c->attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::VarComparisonNode: {
          auto *v =
              static_cast<const ir_sql_converter::SimplestVarComparison *>(
                  expr);
          if (v->left_attr)
            out.emplace(
                static_cast<unsigned int>(v->left_attr->GetTableIndex()),
                static_cast<unsigned int>(v->left_attr->GetColumnIndex()));
          if (v->right_attr)
            out.emplace(
                static_cast<unsigned int>(v->right_attr->GetTableIndex()),
                static_cast<unsigned int>(v->right_attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::IsNullExprNode: {
          auto *n =
              static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
          if (n->attr)
            out.emplace(static_cast<unsigned int>(n->attr->GetTableIndex()),
                        static_cast<unsigned int>(n->attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::InExprNode: {
          auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
          if (i->attr)
            out.emplace(static_cast<unsigned int>(i->attr->GetTableIndex()),
                        static_cast<unsigned int>(i->attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::LogicalExprNode: {
          auto *l =
              static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
          walk(l->left_expr.get());
          walk(l->right_expr.get());
          break;
        }
        default:
          break;
        }
      };
  for (const auto &q : filter_ir->qual_vec)
    walk(q.get());
}

// Collect chunk column indices referenced by a DuckDB Expression tree.
static void CollectBoundRefIndices(const duckdb::Expression &expr,
                                   std::set<duckdb::idx_t> &out) {
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_REF) {
    auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
    out.insert(ref.index);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
      expr, [&](const duckdb::Expression &child) {
        CollectBoundRefIndices(child, out);
      });
}

// Returns true if the expr (recursively) references a (table_idx, col_idx) pair
// that exists in schema.  Used to skip JIT for filters whose predicates don't
// map to any column in the physical operator's output chunk.
static bool ExprHasColInSchema(const ir_sql_converter::AQPExpr *expr,
                               const std::vector<aqp_jit::ColSchema> &schema) {
  if (!expr)
    return false;
  using ir_sql_converter::SimplestNodeType;
  auto attrInSchema = [&](unsigned int t, unsigned int col) -> bool {
    for (auto &cs : schema)
      if (cs.table_idx == t && cs.col_idx == col)
        return true;
    return false;
  };
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    return c->attr &&
           attrInSchema(static_cast<unsigned int>(c->attr->GetTableIndex()),
                        static_cast<unsigned int>(c->attr->GetColumnIndex()));
  }
  case SimplestNodeType::IsNullExprNode: {
    auto *n = static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    return n->attr &&
           attrInSchema(static_cast<unsigned int>(n->attr->GetTableIndex()),
                        static_cast<unsigned int>(n->attr->GetColumnIndex()));
  }
  case SimplestNodeType::InExprNode: {
    auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    return i->attr &&
           attrInSchema(static_cast<unsigned int>(i->attr->GetTableIndex()),
                        static_cast<unsigned int>(i->attr->GetColumnIndex()));
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return ExprHasColInSchema(l->left_expr.get(), schema) ||
           ExprHasColInSchema(l->right_expr.get(), schema);
  }
  default:
    return false;
  }
}

static bool
HasApplicablePredicate(const ir_sql_converter::AQPStmt *filter_ir,
                       const std::vector<aqp_jit::ColSchema> &schema) {
  for (const auto &q : filter_ir->qual_vec)
    if (ExprHasColInSchema(q.get(), schema))
      return true;
  return false;
}

static bool ExprHasLike(const ir_sql_converter::AQPExpr *expr) {
  if (!expr) return false;
  using ir_sql_converter::SimplestNodeType;
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    auto et = c->GetSimplestExprType();
    return et == ir_sql_converter::SimplestExprType::TextLike ||
           et == ir_sql_converter::SimplestExprType::Text_Not_Like;
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return ExprHasLike(l->left_expr.get()) ||
           ExprHasLike(l->right_expr.get());
  }
  default:
    return false;
  }
}

static bool FilterHasLike(const ir_sql_converter::AQPStmt *filter_ir) {
  for (const auto &q : filter_ir->qual_vec)
    if (ExprHasLike(q.get()))
      return true;
  return false;
}

static bool ExprIsOnlyLike(const ir_sql_converter::AQPExpr *expr) {
  if (!expr) return true;
  using ir_sql_converter::SimplestNodeType;
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    auto et = c->GetSimplestExprType();
    return et == ir_sql_converter::SimplestExprType::TextLike ||
           et == ir_sql_converter::SimplestExprType::Text_Not_Like;
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return ExprIsOnlyLike(l->left_expr.get()) &&
           ExprIsOnlyLike(l->right_expr.get());
  }
  default:
    return false;
  }
}

static bool FilterIsOnlyLike(const ir_sql_converter::AQPStmt *filter_ir) {
  if (!filter_ir || filter_ir->qual_vec.empty()) return false;
  for (const auto &q : filter_ir->qual_vec)
    if (!ExprIsOnlyLike(q.get())) return false;
  return true;
}

// Returns true only if ALL columns referenced by the expression are present in
// the schema.  If any column is missing, the compiled filter would silently
// replace that sub-expression with "pass-all" (true), corrupting the filter
// logic (e.g. "gender='m' OR (gender='f' AND name LIKE 'B%')" becomes
// "gender='m' OR gender='f'" = all rows pass when 'name' col is absent).
static bool
AllColsInExprAvailable(const ir_sql_converter::AQPExpr *expr,
                       const std::vector<aqp_jit::ColSchema> &schema) {
  if (!expr)
    return true;
  using ir_sql_converter::SimplestNodeType;
  auto inSchema = [&](unsigned int t, unsigned int col) -> bool {
    for (auto &cs : schema)
      if (cs.table_idx == t && cs.col_idx == col)
        return true;
    return false;
  };
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    return c->attr &&
           inSchema(static_cast<unsigned int>(c->attr->GetTableIndex()),
                    static_cast<unsigned int>(c->attr->GetColumnIndex()));
  }
  case SimplestNodeType::IsNullExprNode: {
    auto *n = static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    return n->attr &&
           inSchema(static_cast<unsigned int>(n->attr->GetTableIndex()),
                    static_cast<unsigned int>(n->attr->GetColumnIndex()));
  }
  case SimplestNodeType::InExprNode: {
    auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    return i->attr &&
           inSchema(static_cast<unsigned int>(i->attr->GetTableIndex()),
                    static_cast<unsigned int>(i->attr->GetColumnIndex()));
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return AllColsInExprAvailable(l->left_expr.get(), schema) &&
           AllColsInExprAvailable(l->right_expr.get(), schema);
  }
  case SimplestNodeType::VarComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    return c->left_attr &&
           inSchema(
               static_cast<unsigned int>(c->left_attr->GetTableIndex()),
               static_cast<unsigned int>(c->left_attr->GetColumnIndex())) &&
           c->right_attr &&
           inSchema(static_cast<unsigned int>(c->right_attr->GetTableIndex()),
                    static_cast<unsigned int>(c->right_attr->GetColumnIndex()));
  }
  default:
    return true;
  }
}

static bool
FilterAllColsAvailable(const ir_sql_converter::AQPStmt *filter_ir,
                       const std::vector<aqp_jit::ColSchema> &schema) {
  for (const auto &q : filter_ir->qual_vec)
    if (!AllColsInExprAvailable(q.get(), schema))
      return false;
  return true;
}

void DuckDBAdapter::EnsureJITCompiler() {
  if (!jit_compiler_) {
    jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
        jit_debug_, ResolveSimdISA(jit_flags_));
  }
  jit_compiler_->SetPrefetch(jit_prefetch_, jit_prefetch_distance_);
  jit_compiler_->SetProbePrefetchDistances(jit_prefetch_entry_distance_,
                                           jit_prefetch_row_distance_);
  jit_compiler_->SetBatchProbe(jit_batch_probe_);
  jit_compiler_->SetSkipHashCmp(jit_skip_hash_cmp_);
  if (jit_cache_)
    jit_compiler_->SetCache(true);
}

aqp_jit::IrToLlvmCompiler *DuckDBAdapter::GetJitCompiler() {
  EnsureJITCompiler();
  return jit_compiler_.get();
}

void DuckDBAdapter::RegisterJIT(duckdb::PhysicalOperator &op,
                                const ir_sql_converter::AQPStmt &ir) {
  EnsureJITCompiler();
  jit_consumed_ir_filters_.clear();
  jit_consumed_ir_joins_.clear();
  RegisterJITImpl(op, ir, GetClientContext(), jit_compiler_.get(),
                  jit_consumed_ir_filters_, jit_consumed_ir_joins_);
}

DuckDBAdapter::TempCollectionSnapshot
DuckDBAdapter::SnapshotTempCollections() const {
  TempCollectionSnapshot snap;
  for (const auto &kv : intermediate_table_map) {
    auto it = temp_collections_.find(kv.second);
    if (it != temp_collections_.end() && it->second.collection &&
        !it->second.is_placeholder)
      snap[kv.first] = it->second.collection.get();
  }
  return snap;
}

void DuckDBAdapter::RegisterJITImpl(
    duckdb::PhysicalOperator &op, const ir_sql_converter::AQPStmt &ir,
    duckdb::ClientContext *jit_ctx, aqp_jit::IrToLlvmCompiler *jit_comp,
    std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed_filters,
    std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed_joins,
    const TempCollectionSnapshot *bf_temp_snapshot, bool is_build_side) {
  using duckdb::PhysicalOperatorType;

  // Diagnostic: show every node in the physical plan tree
#ifndef NDEBUG
  std::cerr << "[AQP-JIT-TRACE] visit op=" << (int)op.type
            << " children=" << op.children.size() << " addr=" << (void *)&op
            << "\n";
#endif

  if (op.type == PhysicalOperatorType::FILTER && !op.children.empty()) {
    auto &child = op.children[0].get();
    uint64_t eid = duckdb::ExpressionID(op);

#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] FILTER eid=0x" << std::hex << eid << std::dec
              << "  child_type=" << (int)child.type
              << "  child_cols=" << child.types.size() << "\n";
#endif

    // Dump all IR FilterNodes once per FILTER op (to show the full IR
    // structure)
    {
      std::vector<const ir_sql_converter::AQPStmt *> all_filters;
      CollectAllFilterNodes(&ir, all_filters);
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE] IR has " << all_filters.size()
                << " FilterNode(s):\n";
      for (size_t fi = 0; fi < all_filters.size(); fi++) {
        std::cerr << "[AQP-JIT-TRACE]   IR filter[" << fi
                  << "] qual_vec.size=" << all_filters[fi]->qual_vec.size()
                  << "\n";
        for (size_t qi = 0; qi < all_filters[fi]->qual_vec.size(); qi++) {
          std::string qs = all_filters[fi]->qual_vec[qi]->Print(false);
          std::cerr << "[AQP-JIT-TRACE]     qual[" << qi << "]: " << qs << "\n";
        }
      }
#endif
    }

    // Determine the IR table_idx for this physical scan by matching the DuckDB
    // table name against IR ScanNodes.  This is reliable because each physical
    // TABLE_SCAN in DuckDB corresponds to exactly one logical table whose name
    // is stored in both the DuckDB bind_data and the IR SimplestScan node.
    const ir_sql_converter::AQPStmt *filter_ir = nullptr;
    unsigned int ir_table_idx = UINT_MAX;

    // Collect known IR table indices for debug validation
    std::unordered_set<unsigned int> ir_table_indices;
    CollectIRTableIndices(&ir, ir_table_indices);
    // Name→index fallback for node-based split where sub-SQL gets fresh
    // logical_table_index values that don't match the original IR's indices.
    std::unordered_map<std::string, unsigned int> ir_name_to_idx;
    std::unordered_set<std::string> ir_ambiguous_names;
    CollectIRTableNameToIndex(&ir, ir_name_to_idx, ir_ambiguous_names);
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] IR table indices:";
    for (auto idx : ir_table_indices)
      std::cerr << " " << idx;
    std::cerr << "\n";
#endif

    std::vector<aqp_jit::ColSchema> schema_prelim;

    // schema/column index mapping

    // The scan's column_ids[i] directly gives the physical column index.
    // Build ColSchema from those.
    if (child.type == PhysicalOperatorType::TABLE_SCAN) {
      auto &scan = static_cast<duckdb::PhysicalTableScan &>(child);

      // Use logical_table_index propagated from the planner — unique per
      // table reference, so self-joins are handled correctly.
      ir_table_idx =
          static_cast<unsigned int>(scan.logical_table_index);
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE]   scan logical_table_index="
                << ir_table_idx << "\n";
#endif
      // Fallback: node-based split re-parses sub-SQL which assigns fresh
      // table indices.  Match by table name when the index isn't in the IR.
      if (!ir_table_indices.count(ir_table_idx) && scan.bind_data) {
        auto *tsbd =
            dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
        if (tsbd) {
          const auto &tname = tsbd->table.name;
          if (!ir_ambiguous_names.count(tname)) {
            auto it = ir_name_to_idx.find(tname);
            if (it != ir_name_to_idx.end()) {
              ir_table_idx = it->second;
#ifndef NDEBUG
              std::cerr << "[AQP-JIT-TRACE]   name fallback: \"" << tname
                        << "\" → ir_table_idx=" << ir_table_idx << "\n";
#endif
            }
          }
        }
      }
      if (!ir_table_indices.count(ir_table_idx)) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT-TRACE]   WARNING: no IR table matches "
                      "logical_table_index="
                  << ir_table_idx << "\n";
#endif
      }

      {
        // child.types may be shorter than scan.column_ids when DuckDB's
        // filter-prune / projection-pushdown strips columns that are only
        // needed for table filters but not returned in the output chunk.
        //
        // IMPORTANT: child.types[i] represents the ACTUAL data type at chunk
        // position i and must be used for dtype.  However, scan.column_ids[i]
        // may be misaligned with child.types in some physical plans (the
        // column_ids and types vectors can have different orderings).  When
        // that happens, we must find the correct base column ID by matching
        // the actual type against the catalog types of the scanned columns.
        for (size_t i = 0; i < child.types.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = ir_table_idx;
          cs.dtype = duckdb::ToDtype(child.types[i].InternalType());
          cs.col_idx =
              static_cast<unsigned int>(scan.column_ids[i].GetPrimaryIndex());

          if (scan.bind_data) {
            auto *tsbd =
                dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
            if (tsbd) {
              // Check if column_ids[i] catalog type matches child.types[i].
              // If not, search for the correct column_id.
              auto n_logical = tsbd->table.GetColumns().LogicalColumnCount();
              if (cs.col_idx < n_logical) {
                auto catalog_dtype = duckdb::ToDtype(
                    tsbd->table.GetColumns()
                        .GetColumn(duckdb::LogicalIndex(cs.col_idx))
                        .Type()
                        .InternalType());
                if (catalog_dtype != cs.dtype) {
                  // Misaligned: find the column_id whose catalog type matches
                  // child.types[i].
                  for (size_t j = 0; j < scan.column_ids.size(); j++) {
                    unsigned int cid = static_cast<unsigned int>(
                        scan.column_ids[j].GetPrimaryIndex());
                    if (cid < n_logical && cid != cs.col_idx) {
                      auto c_dtype = duckdb::ToDtype(
                          tsbd->table.GetColumns()
                              .GetColumn(duckdb::LogicalIndex(cid))
                              .Type()
                              .InternalType());
                      if (c_dtype == cs.dtype) {
                        cs.col_idx = cid;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
          schema_prelim.push_back(cs);
#ifndef NDEBUG
          // Look up column name from DuckDB catalog for diagnostics
          std::string duckdb_col_name;
          if (scan.bind_data) {
            auto *tsbd =
                dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
            if (tsbd &&
                cs.col_idx < tsbd->table.GetColumns().LogicalColumnCount()) {
              duckdb_col_name = tsbd->table.GetColumns()
                                    .GetColumn(duckdb::LogicalIndex(cs.col_idx))
                                    .Name();
            }
          }
          std::cerr << "[AQP-JIT-TRACE]   scan col[" << i
                    << "] raw_col=" << cs.col_idx << " dtype=" << cs.dtype
                    << " table_idx=" << cs.table_idx << " duckdb_col_name=\""
                    << duckdb_col_name << "\"\n";
#endif
        }

        // Find the IR FilterNode that best matches this DuckDB FILTER.
        // Strategy: collect the chunk column indices referenced by the DuckDB
        // expression, map them to (table_idx, col_idx) pairs, then find the
        // IR filter whose qual columns overlap with those pairs.
        filter_ir = nullptr;
        if (ir_table_idx != UINT_MAX) {
          std::vector<const ir_sql_converter::AQPStmt *> all_filters;
          CollectAllFilterNodes(&ir, all_filters);

          // Columns referenced by DuckDB filter expression → schema col pairs
          auto &filt_op = static_cast<duckdb::PhysicalFilter &>(op);
          std::set<std::pair<unsigned int, unsigned int>> duckdb_cols;
          if (filt_op.expression) {
            std::set<duckdb::idx_t> ref_indices;
            CollectBoundRefIndices(*filt_op.expression, ref_indices);
            for (auto idx : ref_indices) {
              if (idx < schema_prelim.size())
                duckdb_cols.emplace(schema_prelim[idx].table_idx,
                                    schema_prelim[idx].col_idx);
            }
          }

          // Find best-matching IR filter: prefer the one whose cols overlap
          // with the DuckDB expression's cols, falling back to table_idx match.
          // Safety: every column referenced by the IR filter must appear in the
          // DuckDB filter's column set.  If the IR references columns that the
          // physical filter doesn't use, the IR filter is a different predicate
          // (e.g., the optimizer rebinds or transforms columns) and compiling it
          // would corrupt results.
          const ir_sql_converter::AQPStmt *best = nullptr;
          size_t best_overlap = 0;
          for (auto *f : all_filters) {
            if (consumed_filters.count(f))
              continue;
            bool table_match = false;
            for (const auto &q : f->qual_vec) {
              unsigned int t = FirstTableIdxFromExpr(q.get());
              if (t == ir_table_idx) {
                table_match = true;
                break;
              }
            }
            if (!table_match)
              continue;
            if (duckdb_cols.empty()) {
              best = f;
              break;
            }
            std::set<std::pair<unsigned int, unsigned int>> ir_cols;
            CollectIRFilterCols(f, ir_cols);
            size_t overlap = 0;
            bool all_in_duckdb = true;
            for (auto &p : ir_cols) {
              if (duckdb_cols.count(p))
                overlap++;
              else
                all_in_duckdb = false;
            }
            if (!all_in_duckdb)
              continue;
            if (overlap > best_overlap) {
              best_overlap = overlap;
              best = f;
            }
          }
          filter_ir = best;
        }
      }
    } else if (child.type == PhysicalOperatorType::PROJECTION &&
               !child.children.empty() &&
               child.children[0].get().type ==
                   PhysicalOperatorType::TABLE_SCAN) {
      // PROJECTION → TABLE_SCAN: trace projection expressions back to original
      // scan column indices.  PhysicalProjection::select_list[i] is a
      // BoundReferenceExpression with index=j meaning "output col i = scan
      // input col j", so the original table col idx = scan.column_ids[j].
      auto &proj = static_cast<duckdb::PhysicalProjection &>(child);
      auto &scan =
          static_cast<duckdb::PhysicalTableScan &>(child.children[0].get());

      // Use logical_table_index from the inner scan (same as TABLE_SCAN path)
      ir_table_idx =
          static_cast<unsigned int>(scan.logical_table_index);
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE]   proj→scan logical_table_index="
                << ir_table_idx << "\n";
#endif
      // Fallback: same as TABLE_SCAN path — match by name for node-based split.
      if (!ir_table_indices.count(ir_table_idx) && scan.bind_data) {
        auto *tsbd =
            dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
        if (tsbd) {
          const auto &tname = tsbd->table.name;
          if (!ir_ambiguous_names.count(tname)) {
            auto it = ir_name_to_idx.find(tname);
            if (it != ir_name_to_idx.end()) {
              ir_table_idx = it->second;
#ifndef NDEBUG
              std::cerr << "[AQP-JIT-TRACE]   proj→scan name fallback: \""
                        << tname << "\" → ir_table_idx=" << ir_table_idx
                        << "\n";
#endif
            }
          }
        }
      }
      if (!ir_table_indices.count(ir_table_idx)) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT-TRACE]   WARNING: proj→scan no IR table "
                      "matches logical_table_index="
                  << ir_table_idx << "\n";
#endif
      }

      {

        // Build schema by tracing each projection expression back to scan col
        for (size_t i = 0; i < proj.select_list.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = ir_table_idx;
          cs.col_idx = UINT_MAX; // set below if traceable
          // Projection output type is correct for the actual chunk layout.
          cs.dtype = duckdb::ToDtype(child.types[i].InternalType());

          auto &expr = *proj.select_list[i];
          if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
            auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
            if (ref.index < scan.column_ids.size()) {
              cs.col_idx = static_cast<unsigned int>(
                  scan.column_ids[ref.index].GetPrimaryIndex());
              // column_ids may be misaligned with the actual chunk data.
              // Verify the catalog type matches the projection output type;
              // if not, find the correct column_id.
              if (scan.bind_data) {
                auto *tsbd = dynamic_cast<duckdb::TableScanBindData *>(
                    scan.bind_data.get());
                if (tsbd) {
                  auto n_logical =
                      tsbd->table.GetColumns().LogicalColumnCount();
                  if (cs.col_idx < n_logical) {
                    auto catalog_dtype = duckdb::ToDtype(
                        tsbd->table.GetColumns()
                            .GetColumn(duckdb::LogicalIndex(cs.col_idx))
                            .Type()
                            .InternalType());
                    if (catalog_dtype != cs.dtype) {
                      for (size_t j = 0; j < scan.column_ids.size(); j++) {
                        unsigned int cid = static_cast<unsigned int>(
                            scan.column_ids[j].GetPrimaryIndex());
                        if (cid < n_logical && cid != cs.col_idx) {
                          auto c_dtype = duckdb::ToDtype(
                              tsbd->table.GetColumns()
                                  .GetColumn(duckdb::LogicalIndex(cid))
                                  .Type()
                                  .InternalType());
                          if (c_dtype == cs.dtype) {
                            cs.col_idx = cid;
                            break;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          schema_prelim.push_back(cs);
#ifndef NDEBUG
          std::cerr << "[AQP-JIT-TRACE]   proj col[" << i
                    << "] raw_col=" << cs.col_idx << " dtype=" << cs.dtype
                    << " table_idx=" << cs.table_idx << "\n";
#endif
        }

        // Find best-matching IR filter (same column-overlap strategy).
        filter_ir = nullptr;
        if (ir_table_idx != UINT_MAX) {
          std::vector<const ir_sql_converter::AQPStmt *> all_filters;
          CollectAllFilterNodes(&ir, all_filters);

          auto &filt_op = static_cast<duckdb::PhysicalFilter &>(op);
          std::set<std::pair<unsigned int, unsigned int>> duckdb_cols;
          if (filt_op.expression) {
            std::set<duckdb::idx_t> ref_indices;
            CollectBoundRefIndices(*filt_op.expression, ref_indices);
            for (auto idx : ref_indices) {
              if (idx < schema_prelim.size())
                duckdb_cols.emplace(schema_prelim[idx].table_idx,
                                    schema_prelim[idx].col_idx);
            }
          }

          const ir_sql_converter::AQPStmt *best = nullptr;
          size_t best_overlap = 0;
          for (auto *f : all_filters) {
            if (consumed_filters.count(f))
              continue;
            bool table_match = false;
            for (const auto &q : f->qual_vec) {
              unsigned int t = FirstTableIdxFromExpr(q.get());
              if (t == ir_table_idx) {
                table_match = true;
                break;
              }
            }
            if (!table_match)
              continue;
            if (duckdb_cols.empty()) {
              best = f;
              break;
            }
            std::set<std::pair<unsigned int, unsigned int>> ir_cols;
            CollectIRFilterCols(f, ir_cols);
            size_t overlap = 0;
            bool all_in_duckdb = true;
            for (auto &p : ir_cols) {
              if (duckdb_cols.count(p))
                overlap++;
              else
                all_in_duckdb = false;
            }
            if (!all_in_duckdb)
              continue;
            if (overlap > best_overlap) {
              best_overlap = overlap;
              best = f;
            }
          }
          filter_ir = best;
        }
      } // end proj path schema/filter matching
    } else {
      // Unknown child type (HASH_JOIN result, etc.): skip JIT.
      // schema_prelim stays empty → HasApplicablePredicate won't match → JIT
      // skipped.
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE]   child_type=" << (int)child.type
                << " not JIT-able → interpreter\n";
#endif
    }

    if (!filter_ir) {
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] FILTER eid=0x" << std::hex << eid << std::dec
                << ": no Filter IR node found in tree, using interpreter\n";
#endif
      for (auto &child_ref : op.children)
        RegisterJITImpl(child_ref.get(), ir, jit_ctx, jit_comp,
                        consumed_filters, consumed_joins, bf_temp_snapshot);
      return;
    }

#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] filter_ir node_type="
              << (int)filter_ir->GetNodeType()
              << "  qual_vec.size=" << filter_ir->qual_vec.size()
              << "  ir_table_idx=" << ir_table_idx << "\n";
    for (size_t qi = 0; qi < filter_ir->qual_vec.size(); qi++) {
      std::string qs = filter_ir->qual_vec[qi]->Print(false);
      std::cerr << "[AQP-JIT-TRACE]   selected qual[" << qi << "]: " << qs
                << "\n";
    }
#endif

    // Build final ColSchema from schema_prelim.
    // For TABLE_SCAN and PROJECTION→TABLE_SCAN: schema_prelim already has
    // correct (table_idx, col_idx, dtype) from the DuckDB table-name → IR
    // table_idx lookup. For unknown child types: schema_prelim is empty → JIT
    // skipped below.
    std::vector<aqp_jit::ColSchema> schema = schema_prelim;
#ifndef NDEBUG
    for (size_t i = 0; i < schema.size(); i++)
      std::cerr << "[AQP-JIT-TRACE]   schema[" << i
                << "] table=" << schema[i].table_idx
                << " col=" << schema[i].col_idx << " dtype=" << schema[i].dtype
                << "\n";
#endif

    // Note: VARCHAR predicates (LIKE, NOT LIKE, Equal, NotEqual) are now
    // handled by the JIT via aqp_like_match / aqp_str_eq runtime helpers.

    // Only register JIT if at least one predicate references a column in the
    // schema AND every referenced column is available.  If any column is
    // missing (e.g. DuckDB pruned it from the scan output), the compiled
    // filter would silently replace that sub-expression with "pass-all",
    // corrupting the filter logic.  Fall back to interpreter in that case.
    // Keep expr-level JIT for LIKE filters: the selection-vector path benefits
    // mixed filters (LIKE + non-LIKE predicates). Scan+filter fusion and
    // pipeline filter fn are skipped for LIKE below (guards at operator/pipeline
    // level) because the per-row JIT LIKE is ~2x slower than DuckDB's vectorized
    // LIKE on large tables.
    bool skip_like = FilterIsOnlyLike(filter_ir);
    if (!schema.empty() && HasApplicablePredicate(filter_ir, schema) &&
        FilterAllColsAvailable(filter_ir, schema) &&
        !skip_like) {
      auto t_filter = chrono_tic();
      ::AQPExprFn raw_fn = jit_comp->CompileFilter(*filter_ir, schema);
      if (enable_timing_ && BREAK_DOWN_COMPILE_TIME) {
        chrono_toc(&t_filter, "RegisterJIT::CompileFilter\n", false);
      }
      if (raw_fn) {
        duckdb::AQPExprFn fn = reinterpret_cast<duckdb::AQPExprFn>(
            reinterpret_cast<void *>(raw_fn));
        if (!jit_ctx->aqp_jit_context)
          jit_ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        consumed_filters.insert(filter_ir);

        // Scan+Filter fusion (operator-level): apply filter at scan time
        // and mark the PhysicalFilter as a pass-through to avoid
        // double-application.
        // Skip fusion when filter contains LIKE predicates: the JIT
        // evaluates LIKE per-row (scalar aqp_like_match_segments), which
        // is ~2x slower than DuckDB's native vectorized LIKE on large
        // tables. Let DuckDB handle LIKE filters as standalone operators.
        bool has_like = FilterHasLike(filter_ir);
        if ((jit_flags_ & AQP_JIT_OPERATOR) &&
            child.type == PhysicalOperatorType::TABLE_SCAN &&
            !has_like) {
          uint64_t scan_eid = duckdb::ExpressionID(child);
          jit_ctx->aqp_jit_context->scan_filter_fns[scan_eid] = fn;
          jit_ctx->aqp_jit_context->fused_scan_filter_eids.insert(eid);
#ifndef NDEBUG
          std::cerr << "[AQP-JIT] scan+filter fusion: scan_eid=0x" << std::hex
                    << scan_eid << std::dec << "\n";
#endif
        } else {
          jit_ctx->aqp_jit_context->expr_fns[eid] = fn;
          jit_ctx->aqp_jit_context->flags |= duckdb::AQPJIT_EXPR;
        }

#ifndef NDEBUG
        std::cerr << "[AQP-JIT] compiled filter eid=0x" << std::hex << eid
                  << std::dec << "  cols=" << schema.size()
                  << "  table_idx=" << ir_table_idx << "\n";
#endif
      } else {
        std::ostringstream msg;
        msg << "[AQP-JIT] JIT compilation failed (eid=0x" << std::hex << eid
            << std::dec << "  cols=" << schema.size()
            << "  table_idx=" << ir_table_idx
            << "). Stopping. Use --no-jit to fall back to interpreter.";
        throw std::runtime_error(msg.str());
      }
    } else if (!schema.empty()) {
#ifndef NDEBUG
      if (!FilterAllColsAvailable(filter_ir, schema)) {
        std::cerr << "[AQP-JIT] skipping filter eid=0x" << std::hex << eid
                  << std::dec
                  << ": not all columns available in chunk → interpreter\n";
      } else {
        std::cerr << "[AQP-JIT] skipping filter eid=0x" << std::hex << eid
                  << std::dec
                  << ": no predicates reference schema columns → interpreter\n";
      }
#endif
    }

    // Level 3: Compile a pipeline function for the Filter operator.
    // VARCHAR columns are safe: aqp_copy_string deep-copies non-inline
    // strings into the output Vector's string heap via callback.
    // Guard: all filter-referenced columns must be in the schema, otherwise
    // FindColIdx returns -1 and the filter silently becomes pass-all.
    // Skip pipeline filter compilation for LIKE predicates: DuckDB's native
    // vectorized LIKE is faster than the JIT scalar implementation.
    if ((kernel_path_ == KernelPath::PIPELINE ||
         (jit_flags_ & AQP_JIT_PIPELINE_JIT)) &&
        filter_ir && !schema.empty() &&
        HasApplicablePredicate(filter_ir, schema) &&
        FilterAllColsAvailable(filter_ir, schema) &&
        !FilterHasLike(filter_ir)) {
      const ir_sql_converter::AQPStmt *proj_ir = nullptr;

      auto t_pipe = chrono_tic();
      ::AQPPipelineFn pipe_fn =
          jit_comp->CompilePipeline(filter_ir, proj_ir, schema);
      if (enable_timing_ && BREAK_DOWN_COMPILE_TIME) {
        chrono_toc(&t_pipe, "RegisterJIT::CompilePipeline\n", false);
      }
      if (pipe_fn) {
        auto fn = reinterpret_cast<duckdb::AQPPipelineFn>(
            reinterpret_cast<void *>(pipe_fn));
        if (!jit_ctx->aqp_jit_context)
          jit_ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        jit_ctx->aqp_jit_context->pipeline_fns[eid] = fn;
        jit_ctx->aqp_jit_context->flags |= duckdb::AQPJIT_PIPELINE;
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] compiled pipeline eid=0x" << std::hex << eid
                  << std::dec << " (filter" << (proj_ir ? "+projection" : "")
                  << ")\n";
#endif
      }
    }
  }

  // Level 2: Compile projection operators when AQPJIT_OPERATOR is set.
  if ((jit_flags_ & AQP_JIT_OPERATOR) &&
      op.type == PhysicalOperatorType::PROJECTION) {
    uint64_t eid = duckdb::ExpressionID(op);
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] PROJECTION eid=0x" << std::hex << eid
              << std::dec << "\n";
#endif

    // Build column mapping directly from the physical projection's
    // select_list.  Each entry should be a BoundReferenceExpression whose
    // index field gives the input column position — no IR lookup needed.
    auto &proj = static_cast<duckdb::PhysicalProjection &>(op);
    if (!proj.select_list.empty() && !op.children.empty()) {
      auto &child = op.children[0].get();
      duckdb::vector<int> col_map;
      bool all_refs = true;
      for (auto &expr : proj.select_list) {
        if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          auto &ref = expr->Cast<duckdb::BoundReferenceExpression>();
          col_map.push_back(static_cast<int>(ref.index));
        } else {
          all_refs = false;
          break;
        }
      }

      if (all_refs) {
        if (!jit_ctx->aqp_jit_context)
          jit_ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        jit_ctx->aqp_jit_context->proj_col_maps[eid] = std::move(col_map);
        jit_ctx->aqp_jit_context->flags |= duckdb::AQPJIT_OPERATOR;
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] compiled projection eid=0x" << std::hex << eid
                  << std::dec << "  out_cols=" << proj.select_list.size()
                  << "  (direct physical mapping)\n";
#endif
      }
    }
  }

  // Level 2: Aggregate JIT — disabled for JOB benchmark.
  // JOB uses only MIN on VARCHAR columns (name, title, info); no query has
  // all-numeric aggregates.  Enabling this wastes compile time for zero
  // benefit.  Re-enable with -DDISABLE_AGG_JIT=0 for benchmarks like TPC-H
  // that have SUM/COUNT/AVG on numeric columns.
#ifndef DISABLE_AGG_JIT
#define DISABLE_AGG_JIT 1
#endif
#if !DISABLE_AGG_JIT
  // Build AggOp directly from the physical plan — no IR matching needed.
  if ((jit_flags_ & AQP_JIT_OPERATOR) &&
      op.type == PhysicalOperatorType::UNGROUPED_AGGREGATE &&
      !op.children.empty()) {
    uint64_t eid = duckdb::ExpressionID(op);
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] AGGREGATE eid=0x" << std::hex << eid
              << std::dec << " type=" << (int)op.type << "\n";
#endif

    auto &child = op.children[0].get();
    auto &phys_aggs =
        static_cast<duckdb::PhysicalUngroupedAggregate &>(op).aggregates;

    // Build AggOp descriptors from physical aggregate expressions
    std::vector<aqp_jit::AggOp> agg_ops;
    unsigned state_offset = 0;
    bool all_ok = true;

    for (auto &phys_expr : phys_aggs) {
      auto &aggr = phys_expr->Cast<duckdb::BoundAggregateExpression>();
      aqp_jit::AggOp op;
      op.state_offset = state_offset;

      const auto &fn_name = aggr.function.name;
      if (fn_name == "sum") {
        op.agg_type = 3;
      } else if (fn_name == "min") {
        op.agg_type = 1;
      } else if (fn_name == "max") {
        op.agg_type = 2;
      } else if (fn_name == "count_star") {
        op.agg_type = 6;
      } else if (fn_name == "count") {
        op.agg_type = 5;
      } else if (fn_name == "avg") {
        op.agg_type = 4;
      } else {
        all_ok = false;
        break;
      }

      if (op.agg_type == 6) { // CountStar
        op.col_idx = -1;
        op.dtype = AQP_DTYPE_INT64;
        op.state_offset = state_offset;
        state_offset += 8;
        agg_ops.push_back(op);
        continue;
      }

      if (aggr.children.empty()) {
        all_ok = false;
        break;
      }
      auto &child_expr = *aggr.children[0];
      if (child_expr.GetExpressionClass() !=
          duckdb::ExpressionClass::BOUND_REF) {
        all_ok = false;
        break;
      }
      auto &ref = child_expr.Cast<duckdb::BoundReferenceExpression>();
      op.col_idx = static_cast<int>(ref.index);
      if (op.col_idx < 0 ||
          (size_t)op.col_idx >= child.types.size()) {
        all_ok = false;
        break;
      }
      op.dtype = duckdb::ToDtype(child.types[op.col_idx].InternalType());

      // Only numeric types supported (state is 8 bytes, no string_t)
      if (op.dtype == AQP_DTYPE_VARCHAR || op.dtype == AQP_DTYPE_OTHER) {
        all_ok = false;
        break;
      }

      state_offset += (op.agg_type == 4) ? 16 : 8;
      agg_ops.push_back(op);
    }

    if (all_ok && !agg_ops.empty()) {
      auto t_agg = chrono_tic();
      void *raw_fn =
          jit_comp->CompileAggUpdateDirect(agg_ops, state_offset);
      if (enable_timing_ && BREAK_DOWN_COMPILE_TIME) {
        chrono_toc(&t_agg, "RegisterJIT::CompileAggUpdateDirect\n", false);
      }
      if (raw_fn) {
        if (!jit_ctx->aqp_jit_context)
          jit_ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        auto agg_fn =
            reinterpret_cast<duckdb::AQPJITContext::AQPAggUpdateFn>(raw_fn);
        jit_ctx->aqp_jit_context->agg_fns[eid] = agg_fn;
        jit_ctx->aqp_jit_context->agg_state_sizes[eid] = state_offset;
        jit_ctx->aqp_jit_context->op_fns[eid] =
            reinterpret_cast<duckdb::AQPOperatorFn>(raw_fn);
        jit_ctx->aqp_jit_context->flags |= duckdb::AQPJIT_OPERATOR;

        // Build metadata directly from AggOp
        std::vector<duckdb::AQPJITContext::AQPAggMeta> meta_vec;
        for (const auto &aop : agg_ops) {
          duckdb::AQPJITContext::AQPAggMeta m;
          m.agg_type = aop.agg_type;
          m.state_offset = aop.state_offset;
          m.dtype = aop.dtype;
          m.state_bytes = (aop.agg_type == 4) ? 16 : 8;
          meta_vec.push_back(m);
        }
        jit_ctx->aqp_jit_context->agg_meta[eid] = std::move(meta_vec);
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] compiled agg eid=0x" << std::hex << eid
                  << std::dec << " state_bytes=" << state_offset << "\n";
#endif
      }
    } else if (!all_ok) {
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] agg: skipping (VARCHAR or unsupported agg)\n";
#endif
    }
  }
#endif // !DISABLE_AGG_JIT

  // Level 3: Pipeline-JIT hash join compilation. Direct-HT path emits IR
  // that probes DuckDB's JoinHashTable directly via AQPJoinHTView, so the
  // previous "IR schema vs. DuckDB chunk layout" mismatch is resolved.
  // Note: kernel-path=pipeline uses PipelineKernel's own HashJoinTable, not
  // DuckDB's JoinHashTable, so this gate is pipeline-jit only.
  if ((jit_flags_ & AQP_JIT_PIPELINE_JIT) &&
      op.type == PhysicalOperatorType::HASH_JOIN) {
    uint64_t eid = duckdb::ExpressionID(op);

    // Multi-probe chain detection: walk children[0] to find consecutive HJs.
    // DFS visits outermost first, so chain detection runs at the outer HJ.
    // Skip when is_build_side: the chain's probe pipeline would feed into
    // a parent HJ's Sink, and the fused output types would cause a type
    // mismatch at that Sink.
    // Skip if this HJ is already part of a fused chain (passthrough or innermost).
    bool already_in_chain = false;
    if (jit_ctx && jit_ctx->aqp_jit_context) {
      already_in_chain =
          jit_ctx->aqp_jit_context->multi_probe_passthrough_eids.count(eid) ||
          jit_ctx->aqp_jit_context->multi_probe_fns.count(eid);
    }
    if (getenv("AQP_MP_TRACE")) {
      fprintf(stderr, "[MP-WALK] hj eid=%llx in_chain=%d build_side=%d child0=%s\n",
              (unsigned long long)eid, (int)already_in_chain, (int)is_build_side,
              op.children.empty() ? "none"
                  : duckdb::PhysicalOperatorToString(op.children[0].get().type).c_str());
    }
    if (!already_in_chain && !is_build_side && op.children.size() >= 1 &&
        op.children[0].get().type == PhysicalOperatorType::HASH_JOIN) {
      std::vector<duckdb::PhysicalOperator *> chain_ops;
      chain_ops.push_back(&op);
      {
        auto *cursor = &op.children[0].get();
        while (cursor->type == PhysicalOperatorType::HASH_JOIN) {
          chain_ops.push_back(cursor);
          if (cursor->children.empty()) break;
          cursor = &cursor->children[0].get();
        }
      }
      if (chain_ops.size() >= 2) {
        // Collect ProbeStageInfo for each HJ in reverse (inner first).
        // Uses physical-plan-only data (no IR needed for multi-probe).
        auto LtToAqpDtype_mp =
            [](const duckdb::LogicalType &lt) -> int32_t {
          switch (lt.id()) {
          case duckdb::LogicalTypeId::BOOLEAN: return AQP_DTYPE_BOOL;
          case duckdb::LogicalTypeId::TINYINT:
          case duckdb::LogicalTypeId::UTINYINT: return AQP_DTYPE_INT8;
          case duckdb::LogicalTypeId::SMALLINT:
          case duckdb::LogicalTypeId::USMALLINT: return AQP_DTYPE_INT16;
          case duckdb::LogicalTypeId::INTEGER:
          case duckdb::LogicalTypeId::UINTEGER: return AQP_DTYPE_INT32;
          case duckdb::LogicalTypeId::BIGINT:
          case duckdb::LogicalTypeId::UBIGINT:
          case duckdb::LogicalTypeId::HUGEINT: return AQP_DTYPE_INT64;
          case duckdb::LogicalTypeId::FLOAT: return AQP_DTYPE_FLOAT;
          case duckdb::LogicalTypeId::DOUBLE: return AQP_DTYPE_DOUBLE;
          case duckdb::LogicalTypeId::VARCHAR: return AQP_DTYPE_VARCHAR;
          case duckdb::LogicalTypeId::DATE: return AQP_DTYPE_DATE;
          default: return AQP_DTYPE_OTHER;
          }
        };

        std::vector<aqp_jit::ProbeStageInfo> stages;
        bool all_eligible = true;

        for (int ci = (int)chain_ops.size() - 1; ci >= 0; --ci) {
          auto &hop = *chain_ops[ci];
          auto &hj = hop.Cast<duckdb::PhysicalHashJoin>();
          aqp_jit::ProbeStageInfo info;
          info.hj = &hj;
          info.eid = duckdb::ExpressionID(hop);

          // Join type check
          if (hj.join_type != duckdb::JoinType::INNER) {
            all_eligible = false;
            break;
          }

          // Build payload_schema from build-side (children[1]) physical types
          const auto &build_types = hop.children[1].get().GetTypes();
          for (size_t bi = 0; bi < build_types.size(); ++bi) {
            aqp_jit::ColSchema cs;
            cs.table_idx = 0;
            cs.col_idx = (unsigned)bi;
            cs.dtype = LtToAqpDtype_mp(build_types[bi]);
            info.payload_schema.push_back(cs);
          }

          // Payload pruning
          if (jit_payload_prune_ && !hj.payload_columns.col_idxs.empty()) {
            std::vector<aqp_jit::ColSchema> pruned;
            for (auto idx : hj.payload_columns.col_idxs) {
              if ((int)idx < (int)info.payload_schema.size())
                pruned.push_back(info.payload_schema[idx]);
            }
            info.payload_schema = std::move(pruned);
          }

          // payload_row_indices = identity
          info.payload_row_indices.reserve(info.payload_schema.size());
          for (size_t k = 0; k < info.payload_schema.size(); ++k)
            info.payload_row_indices.push_back((int)k);

          // probe_schema: only for stage 0 (innermost), from scan input
          if (ci == (int)chain_ops.size() - 1) {
            const auto &probe_types = hop.children[0].get().GetTypes();
            for (size_t pi = 0; pi < probe_types.size(); ++pi) {
              aqp_jit::ColSchema cs;
              cs.table_idx = 0;
              cs.col_idx = (unsigned)pi;
              cs.dtype = LtToAqpDtype_mp(probe_types[pi]);
              info.probe_schema.push_back(cs);
            }
          }

          // Output column mappings
          for (auto ci2 : hj.lhs_output_columns.col_idxs)
            info.lhs_output_idxs.push_back((int)ci2);
          for (auto ci2 : hj.rhs_output_columns.col_idxs)
            info.rhs_output_layout_idxs.push_back((int)ci2);
          for (auto &lt : hj.lhs_output_columns.col_types)
            info.lhs_output_dtypes.push_back(LtToAqpDtype_mp(lt));
          for (auto &lt : hj.rhs_output_columns.col_types)
            info.rhs_output_dtypes.push_back(LtToAqpDtype_mp(lt));
#ifndef NDEBUG
          {
            std::cerr << "  [chain-stage] ci=" << ci << " eid=0x" << std::hex << info.eid << std::dec
                      << " lhs_types=[";
            for (size_t k = 0; k < hj.lhs_output_columns.col_types.size(); k++)
              std::cerr << (k?",":"") << hj.lhs_output_columns.col_types[k].ToString();
            std::cerr << "] rhs_types=[";
            for (size_t k = 0; k < hj.rhs_output_columns.col_types.size(); k++)
              std::cerr << (k?",":"") << hj.rhs_output_columns.col_types[k].ToString();
            std::cerr << "] key_types=[";
            for (auto &c : hj.conditions)
              std::cerr << c.left->return_type.ToString() << " ";
            std::cerr << "]\n";
          }
#endif

          // Key extraction
          bool keys_ok = true;
          for (auto &cond : hj.conditions) {
            if (!cond.left || cond.left->GetExpressionClass() !=
                                  duckdb::ExpressionClass::BOUND_REF) {
              keys_ok = false;
              break;
            }
            auto &bref = cond.left->Cast<duckdb::BoundReferenceExpression>();
            info.lhs_key_chunk_idxs.push_back((int)bref.index);
            info.lhs_key_dtypes.push_back(
                LtToAqpDtype_mp(cond.left->return_type));
          }
          if (!keys_ok) {
            all_eligible = false;
            break;
          }

          // skip_hash_cmp eligibility
          info.skip_hash_cmp_eligible = true;
          for (auto d : info.lhs_key_dtypes) {
            if (d != AQP_DTYPE_INT8 && d != AQP_DTYPE_INT16 &&
                d != AQP_DTYPE_INT32 && d != AQP_DTYPE_INT64) {
              info.skip_hash_cmp_eligible = false;
              break;
            }
          }

          info.eligible = true;
          stages.push_back(std::move(info));
        }

        if (getenv("AQP_MP_TRACE")) {
          fprintf(stderr, "[MP-DETECT] chain_len=%zu all_eligible=%d stages=%zu\n",
                  chain_ops.size(), (int)all_eligible, stages.size());
        }
        // AQPMultiProbeState::views[] holds at most 4 stages (JOB max is 3).
        if (all_eligible && stages.size() >= 2 && stages.size() <= 4) {
          void *fused_fn = jit_comp->CompileMultiProbeChain(stages);
          if (getenv("AQP_MP_TRACE")) {
            fprintf(stderr, "[MP-COMPILE] fused_fn=%p innermost=%llx\n", fused_fn,
                    (unsigned long long)stages[0].eid);
          }
          if (fused_fn) {
            auto *ctx = jit_ctx;
            if (!ctx->aqp_jit_context)
              ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();

            auto innermost_eid = stages[0].eid;

            ctx->aqp_jit_context->multi_probe_fns[innermost_eid] =
                reinterpret_cast<duckdb::AQPPipelineFn>(fused_fn);

            // Register HT views for ALL stages including the innermost
            // (si=0): the inner HJ skips single-probe registration via
            // already_in_chain, so nothing else creates its view, and the
            // dispatch-time chain validation requires one per member.
            for (size_t si = 0; si < stages.size(); si++) {
              if (!ctx->aqp_jit_context->join_ht_views.count(stages[si].eid))
                ctx->aqp_jit_context->join_ht_views[stages[si].eid] =
                    duckdb::make_uniq<duckdb::AQPJoinHTView>();
            }

            auto mps = duckdb::make_uniq<duckdb::AQPJITContext::AQPMultiProbeState>();
            mps->num_stages = (uint32_t)stages.size();
            ctx->aqp_jit_context->multi_probe_states[innermost_eid] =
                std::move(mps);

            std::vector<duckdb::PhysicalHashJoin *> hj_chain;
            for (auto &s : stages)
              hj_chain.push_back(s.hj);
            ctx->aqp_jit_context->multi_probe_hj_chain[innermost_eid] =
                std::move(hj_chain);

            auto &outermost_hj = *stages.back().hj;
            ctx->aqp_jit_context->multi_probe_outer_types[innermost_eid] =
                outermost_hj.GetTypes();

            for (size_t si = 1; si < stages.size(); si++) {
              ctx->aqp_jit_context->multi_probe_passthrough_eids[stages[si].eid] =
                  innermost_eid;
            }

            // Record chain members and (optionally) force the regular
            // hash-table path so PhysicalHashJoin::Finalize bypasses the
            // perfect (array) hash join for them — otherwise the fused
            // probe cannot dispatch (it reads JoinHashTable directly).
            for (auto &s : stages)
              ctx->aqp_jit_context->multi_probe_chain_members.insert(s.eid);
            if (jit_single_col_int_join_)
              ctx->aqp_jit_context->force_hash_join_for_chains = true;

            ctx->aqp_jit_context->flags |= duckdb::AQPJIT_PIPELINE;

#ifndef NDEBUG
            std::cerr << "[AQP-JIT-MULTI] fused " << stages.size()
                      << "-probe chain: eids=[";
            for (size_t si = 0; si < stages.size(); si++) {
              if (si > 0) std::cerr << ",";
              std::cerr << "0x" << std::hex << stages[si].eid << std::dec;
            }
            std::cerr << "] innermost=0x" << std::hex << innermost_eid
                      << std::dec << " out_cols="
                      << stages.back().hj->GetTypes().size() << "\n";
            for (size_t si = 0; si < stages.size(); si++) {
              auto &s = stages[si];
              std::cerr << "  stage[" << si << "] eid=0x" << std::hex << s.eid << std::dec
                        << " lhs_out=[";
              for (size_t k = 0; k < s.lhs_output_idxs.size(); k++)
                std::cerr << (k?",":"") << s.lhs_output_idxs[k];
              std::cerr << "] rhs_out=[";
              for (size_t k = 0; k < s.rhs_output_layout_idxs.size(); k++)
                std::cerr << (k?",":"") << s.rhs_output_layout_idxs[k];
              std::cerr << "] keys=[";
              for (size_t k = 0; k < s.lhs_key_chunk_idxs.size(); k++)
                std::cerr << (k?",":"") << s.lhs_key_chunk_idxs[k];
              std::cerr << "] probe_ncols=" << s.probe_schema.size()
                        << " payload_ncols=" << s.payload_schema.size() << "\n";
            }
#endif
            // Skip single-probe for inner eids (they'll see pipeline_fns set)
            // but DON'T skip for this (outer) eid — it still needs single-probe
            // as fallback if multi-probe bails at runtime.
            // Actually: skip entirely. The outer HJ is passthrough at dispatch.
            // The inner HJ dispatches the fused function.
          }
        }
      }
    }

    // Skip single-probe if this HJ is already part of a multi-probe chain
    if (already_in_chain)
      goto aqp_skip_single_probe;

    { // Single-probe compilation block — skipped by goto aqp_skip_single_probe
    // Collect IR table indices and name→index map for phys→IR remapping.
    // Split sub-queries re-parse SQL which assigns fresh logical_table_index
    // values; this map lets us translate back to the IR's original indices.
    std::unordered_set<unsigned int> ir_table_indices;
    CollectIRTableIndices(&ir, ir_table_indices);
    std::unordered_map<std::string, unsigned int> ir_name_to_idx;
    std::unordered_set<std::string> ir_ambiguous_names;
    CollectIRTableNameToIndex(&ir, ir_name_to_idx, ir_ambiguous_names);

    auto RemapToIRIndex = [&](unsigned int phys_idx,
                              duckdb::PhysicalTableScan *phys_scan) -> unsigned int {
      if (phys_idx == UINT_MAX || ir_table_indices.count(phys_idx))
        return phys_idx;
      if (!phys_scan || !phys_scan->bind_data)
        return phys_idx;
      auto *tsbd = dynamic_cast<duckdb::TableScanBindData *>(
          phys_scan->bind_data.get());
      if (!tsbd)
        return phys_idx;
      const auto &tname = tsbd->table.name;
      if (ir_ambiguous_names.count(tname))
        return phys_idx;
      auto it = ir_name_to_idx.find(tname);
      return (it != ir_name_to_idx.end()) ? it->second : phys_idx;
    };

    // Match this physical HASH_JOIN to the correct IR JoinNode.
    // Get both build-side and probe-side table indices from the physical
    // plan, then find the IR JoinNode whose conditions reference both.
    auto *build_phys_scan = FindBuildSideTableScan(op.children[1].get());
    unsigned int build_tidx =
        build_phys_scan
            ? static_cast<unsigned int>(build_phys_scan->logical_table_index)
            : UINT_MAX;
    auto *probe_phys_scan = FindBuildSideTableScan(op.children[0].get());
    unsigned int probe_tidx =
        probe_phys_scan
            ? static_cast<unsigned int>(probe_phys_scan->logical_table_index)
            : UINT_MAX;
    unsigned int ir_build_tidx = RemapToIRIndex(build_tidx, build_phys_scan);
    unsigned int ir_probe_tidx = RemapToIRIndex(probe_tidx, probe_phys_scan);
    const ir_sql_converter::AQPStmt *join_ir =
        (ir_build_tidx != UINT_MAX)
            ? FindJoinNodeByBuildTableIndex(&ir, ir_build_tidx,
                                            consumed_joins,
                                            ir_probe_tidx)
            : FindFirstJoinNode(&ir);
    // Validate: when using FindFirstJoinNode fallback, verify the matched
    // IR JoinNode's condition count matches the physical operator's. A
    // mismatch means the IR JoinNode doesn't correspond to this physical
    // HASH_JOIN — the build-side schema/keys would be wrong, producing
    // silent incorrect results (wrong payload offsets, zero matches).
    if (ir_build_tidx == UINT_MAX && join_ir) {
      auto *join_check =
          dynamic_cast<const ir_sql_converter::SimplestJoin *>(join_ir);
      auto &hj_check = op.Cast<duckdb::PhysicalHashJoin>();
      if (!join_check ||
          join_check->join_conditions.size() != hj_check.conditions.size()) {
        join_ir = nullptr;
      }
    }
    const ir_sql_converter::AQPStmt *hash_ir = FindFirstHashNode(&ir);

#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] HASH_JOIN: build_tidx=" << build_tidx
              << " probe_tidx=" << probe_tidx
              << " ir_build_tidx=" << ir_build_tidx
              << " ir_probe_tidx=" << ir_probe_tidx
              << " join_ir=" << (void *)join_ir << "\n";
#endif
    if (join_ir) {
      consumed_joins.insert(join_ir);
      auto *join =
          dynamic_cast<const ir_sql_converter::SimplestJoin *>(join_ir);
      if (join && !join->join_conditions.empty()) {
        // Find the build-side ScanNode in the matched JoinNode's subtree.
        // The physical optimizer may swap probe/build sides relative to the
        // IR (logical plan).  When the build table is in IR children[0]
        // instead of children[1], skip compilation — the probe_schema and
        // key column mapping don't match the physical layout.
        const ir_sql_converter::AQPStmt *build_child = nullptr;
        bool ir_sides_swapped = false;
#ifndef NDEBUG
        std::cerr << "[AQP-JIT-TRACE] join_ir->children.size()="
                  << join_ir->children.size()
                  << " ir_build_tidx=" << ir_build_tidx << "\n";
#endif
        if (ir_build_tidx != UINT_MAX && join_ir->children.size() > 1) {
          build_child =
              FindScanByTableIndex(join_ir->children[1].get(), ir_build_tidx);
          if (!build_child &&
              FindScanByTableIndex(join_ir->children[0].get(), ir_build_tidx))
            ir_sides_swapped = true;
        }
        if (ir_sides_swapped) {
          consumed_joins.erase(join_ir);
#ifndef NDEBUG
          std::cerr << "[AQP-JIT-TRACE] skipped: ir_sides_swapped build_tidx="
                    << build_tidx << "\n";
#endif
        }
        if (!ir_sides_swapped) {
        // When build side is a temp table (COLUMN_DATA_SCAN, build_tidx ==
        // UINT_MAX), use the IR join's children[1] directly — it describes
        // the temp table's schema.
        if (!build_child && ir_build_tidx == UINT_MAX &&
            join_ir->children.size() > 1)
          build_child = join_ir->children[1].get();

        // If HashNode exists (PostgreSQL path), use its child instead
        if (hash_ir && !hash_ir->children.empty())
          build_child = hash_ir->children[0].get();

        // Walk through FilterNode wrappers to reach the ScanNode
        while (build_child &&
               build_child->GetNodeType() ==
                   ir_sql_converter::SimplestNodeType::FilterNode &&
               !build_child->children.empty())
          build_child = build_child->children[0].get();

#ifndef NDEBUG
        std::cerr << "[AQP-JIT-TRACE] build_child="
                  << (void *)build_child
                  << " target_list.size="
                  << (build_child ? build_child->target_list.size() : 0)
                  << " nodeType="
                  << (build_child ? (int)build_child->GetNodeType() : -1)
                  << "\n";
#endif
        if (build_child && !build_child->target_list.empty()) {
          // Build schema from build child's target_list
          std::vector<aqp_jit::ColSchema> build_schema;
          for (const auto &attr : build_child->target_list) {
            aqp_jit::ColSchema cs;
            cs.table_idx = attr->GetTableIndex();
            cs.col_idx = attr->GetColumnIndex();
            switch (attr->GetType()) {
            case ir_sql_converter::IntVar:
              cs.dtype = AQP_DTYPE_INT32;
              break;
            case ir_sql_converter::FloatVar:
              cs.dtype = AQP_DTYPE_DOUBLE;
              break;
            case ir_sql_converter::StringVar:
              cs.dtype = AQP_DTYPE_VARCHAR;
              break;
            case ir_sql_converter::BoolVar:
              cs.dtype = AQP_DTYPE_BOOL;
              break;
            case ir_sql_converter::Date:
              cs.dtype = AQP_DTYPE_DATE;
              break;
            default:
              cs.dtype = AQP_DTYPE_OTHER;
              break;
            }
            build_schema.push_back(cs);
          }

          // If no HashNode, create a synthetic one from join conditions.
          // The build-side key columns are the join condition attrs that match
          // the build_child's target_list.
          std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
              synth_keys;
          for (const auto &cond : join->join_conditions) {
            // Check which side of the condition is in the build schema
            bool left_in_build = false, right_in_build = false;
            for (const auto &cs : build_schema) {
              if (cs.table_idx == cond->left_attr->GetTableIndex() &&
                  cs.col_idx == cond->left_attr->GetColumnIndex())
                left_in_build = true;
              if (cs.table_idx == cond->right_attr->GetTableIndex() &&
                  cs.col_idx == cond->right_attr->GetColumnIndex())
                right_in_build = true;
            }
            if (left_in_build) {
              synth_keys.push_back(
                  std::make_unique<ir_sql_converter::SimplestAttr>(
                      *cond->left_attr));
            } else if (right_in_build) {
              synth_keys.push_back(
                  std::make_unique<ir_sql_converter::SimplestAttr>(
                      *cond->right_attr));
            }
          }

#ifndef NDEBUG
          std::cerr << "[AQP-JIT-TRACE] synth_keys.size()=" << synth_keys.size()
                    << " join_conditions.size()=" << join->join_conditions.size()
                    << " build_schema.size()=" << build_schema.size() << "\n";
          for (size_t dbgi = 0; dbgi < build_schema.size(); ++dbgi)
            std::cerr << "  build_schema[" << dbgi << "] tidx="
                      << build_schema[dbgi].table_idx << " cidx="
                      << build_schema[dbgi].col_idx << "\n";
          for (size_t dbgi = 0; dbgi < join->join_conditions.size(); ++dbgi)
            std::cerr << "  cond[" << dbgi << "] left=("
                      << join->join_conditions[dbgi]->left_attr->GetTableIndex()
                      << "," << join->join_conditions[dbgi]->left_attr->GetColumnIndex()
                      << ") right=("
                      << join->join_conditions[dbgi]->right_attr->GetTableIndex()
                      << "," << join->join_conditions[dbgi]->right_attr->GetColumnIndex()
                      << ")\n";
#endif
          if (!synth_keys.empty()) {
            // Hash-build JIT is intentionally NOT performed here. Rationale:
            //
            // DuckDB's native build path (PrepareKeys ->
            // TupleDataCollection::Append -> InsertHashesLoop) is already a
            // tight, parallelized C++ pipeline with no per-row virtual
            // dispatch. The places a JIT could specialize (typed key
            // extract, NULL-skip when NOT NULL, typed Murmur) sum to ~5-7%
            // of build cost, and build is only ~20-30% of execute time on
            // join-heavy queries -- so the achievable win is ~1-2% of total
            // query time.
            //
            // To capture *more* than that the JIT would have to inline
            // TupleDataCollection::Append (per-thread row-format encoding,
            // string heap management, radix partitioning, parallel append
            // protocol). That is months of engineering replicating
            // DuckDB's internals, and bypasses the very abstractions that
            // make DuckDB's HT fast and bloom-filter-friendly.
            //
            // The probe side is JIT'd for the opposite reason: probe is the
            // hot loop (O(N x K)), and removing the AQPHashTable +
            // ScanStructure abstraction layers yields a measured win.
            // Build has no analogous layer to strip.
            //
            // Build-side *filter expressions* are still JIT'd: they run via
            // the standalone filter pipeline path
            // (physical_filter.cpp:pipeline_fns dispatch). Only the
            // hash-build operator itself stays native.
            //
            // synth_keys above is retained as a gating condition (the join
            // has at least one key referenced in build_schema) — it used to
            // feed CompileFilterHashBuildFusion / CompileHashBuild, both
            // removed.

            // Level 3: attempt Filter+Probe+Projection fusion (probe pipeline).
            // Activated by jit-level=pipeline only (kernel-path uses PipelineKernel).
            if (jit_flags_ & AQP_JIT_PIPELINE_JIT) {
              // Payload pruning: use DuckDB's payload_columns to determine
              // which build-side columns are actually needed downstream.
              // Scoped here (not above) because needed_payload is only read
              // by the probe-fusion path; the build-JIT site that used to
              // consume it has been removed.
              std::vector<int> needed_payload;
              if (jit_payload_prune_) {
                auto &hj = op.Cast<duckdb::PhysicalHashJoin>();
                if (!hj.payload_columns.col_idxs.empty()) {
                  for (auto idx : hj.payload_columns.col_idxs) {
                    if ((int)idx < (int)build_schema.size())
                      needed_payload.push_back((int)idx);
                  }
#ifndef NDEBUG
                  std::cerr << "[AQP-JIT] payload pruning: "
                            << build_schema.size() << " → "
                            << needed_payload.size() << " columns\n";
#endif
                }
              }

              auto jt = join->GetSimplestJoinType();
              if (jt != ir_sql_converter::Right &&
                  jt != ir_sql_converter::Full &&
                  jt != ir_sql_converter::Anti) {
                const ir_sql_converter::AQPStmt *probe_child = nullptr;
                if (!join_ir->children.empty())
                  probe_child = join_ir->children[0].get();
                // Build probe schema: prefer IR target_list, fall back to
                // DuckDB's physical types for temp-table joins where IR
                // probe child has no target_list.
                auto LtToAqpDtype2 =
                    [](const duckdb::LogicalType &lt) -> int32_t {
                  switch (lt.id()) {
                  case duckdb::LogicalTypeId::BOOLEAN:
                    return AQP_DTYPE_BOOL;
                  case duckdb::LogicalTypeId::TINYINT:
                  case duckdb::LogicalTypeId::UTINYINT:
                    return AQP_DTYPE_INT8;
                  case duckdb::LogicalTypeId::SMALLINT:
                  case duckdb::LogicalTypeId::USMALLINT:
                    return AQP_DTYPE_INT16;
                  case duckdb::LogicalTypeId::INTEGER:
                  case duckdb::LogicalTypeId::UINTEGER:
                    return AQP_DTYPE_INT32;
                  case duckdb::LogicalTypeId::BIGINT:
                  case duckdb::LogicalTypeId::UBIGINT:
                  case duckdb::LogicalTypeId::HUGEINT:
                    return AQP_DTYPE_INT64;
                  case duckdb::LogicalTypeId::FLOAT:
                    return AQP_DTYPE_FLOAT;
                  case duckdb::LogicalTypeId::DOUBLE:
                    return AQP_DTYPE_DOUBLE;
                  case duckdb::LogicalTypeId::VARCHAR:
                    return AQP_DTYPE_VARCHAR;
                  case duckdb::LogicalTypeId::DATE:
                    return AQP_DTYPE_DATE;
                  default:
                    return AQP_DTYPE_OTHER;
                  }
                };
                {
                  std::vector<aqp_jit::ColSchema> probe_schema;
                  bool from_ir = (probe_child && !probe_child->target_list.empty());
                  if (from_ir) {
                    for (const auto &attr : probe_child->target_list) {
                      aqp_jit::ColSchema cs;
                      cs.table_idx = attr->GetTableIndex();
                      cs.col_idx = attr->GetColumnIndex();
                      switch (attr->GetType()) {
                      case ir_sql_converter::IntVar:
                        cs.dtype = AQP_DTYPE_INT32;
                        break;
                      case ir_sql_converter::FloatVar:
                        cs.dtype = AQP_DTYPE_DOUBLE;
                        break;
                      case ir_sql_converter::StringVar:
                        cs.dtype = AQP_DTYPE_VARCHAR;
                        break;
                      case ir_sql_converter::BoolVar:
                        cs.dtype = AQP_DTYPE_BOOL;
                        break;
                      case ir_sql_converter::Date:
                        cs.dtype = AQP_DTYPE_DATE;
                        break;
                      default:
                        cs.dtype = AQP_DTYPE_OTHER;
                        break;
                      }
                      probe_schema.push_back(cs);
                    }
                  } else {
                    const auto &pt = op.children[0].get().GetTypes();
                    for (size_t pi = 0; pi < pt.size(); ++pi) {
                      aqp_jit::ColSchema cs;
                      cs.table_idx = 0;
                      cs.col_idx = (unsigned)pi;
                      cs.dtype = LtToAqpDtype2(pt[pi]);
                      probe_schema.push_back(cs);
                    }
                  }
                if (!probe_schema.empty()) {

                  // Guard: AQP IR's probe_schema MUST match DuckDB's actual
                  // physical chunk shape (size + per-column dtype). The JIT
                  // emits col_data[i] loads indexed by probe_schema position;
                  // if the AQP IR view diverges from DuckDB's chunk schema,
                  // those loads read uninit AQPColView entries (sized to
                  // probe_schema.size(), but DuckDB only fills input.ncols)
                  // and produce garbage data pointers that crash when later
                  // dereferenced for hash/key/output. Skip pipeline-JIT for
                  // this eid and let the interpreter handle it.
                  const auto &phys_types = op.children[0].get().GetTypes();
                  bool schema_match = (probe_schema.size() == phys_types.size());
                  if (schema_match) {
                    for (size_t pi = 0; pi < probe_schema.size(); ++pi) {
                      int32_t phys_dtype = LtToAqpDtype2(phys_types[pi]);
                      if (phys_dtype != probe_schema[pi].dtype) {
                        schema_match = false;
                        break;
                      }
                    }
                  }
                  if (schema_match) {

                  // Build payload schema: the build-side columns actually
                  // stored in the hash table (respecting payload pruning).
                  std::vector<aqp_jit::ColSchema> payload_schema;
                  if (!needed_payload.empty()) {
                    for (int ci : needed_payload) {
                      if (ci >= 0 && ci < (int)build_schema.size())
                        payload_schema.push_back(build_schema[ci]);
                    }
                  } else {
                    payload_schema = build_schema;
                  }

                  // Find probe-side filter from IR.
                  // Don't use global projection — it references columns from
                  // all tables, but each intermediate join can only output
                  // its own probe + build columns. Pass nullptr to output all.
                  const ir_sql_converter::AQPStmt *probe_filter_ir =
                      FindFirstFilterNode(probe_child);
                  const ir_sql_converter::AQPStmt *probe_proj_ir = nullptr;

                  // Compute key and payload widths for AQP hash table
                  auto DtypeWidth = [](int32_t dtype) -> unsigned {
                    switch (dtype) {
                    case AQP_DTYPE_BOOL:
                    case AQP_DTYPE_INT8:
                      return 1;
                    case AQP_DTYPE_INT16:
                      return 2;
                    case AQP_DTYPE_INT32:
                    case AQP_DTYPE_DATE:
                    case AQP_DTYPE_FLOAT:
                      return 4;
                    case AQP_DTYPE_INT64:
                    case AQP_DTYPE_DOUBLE:
                      return 8;
                    case AQP_DTYPE_VARCHAR:
                      return 16;
                    default:
                      return 8;
                    }
                  };

                  unsigned build_key_width = 0;
                  for (const auto &cond : join->join_conditions) {
                    for (const auto &cs : build_schema) {
                      if ((cs.table_idx == cond->left_attr->GetTableIndex() &&
                           cs.col_idx == cond->left_attr->GetColumnIndex()) ||
                          (cs.table_idx == cond->right_attr->GetTableIndex() &&
                           cs.col_idx == cond->right_attr->GetColumnIndex())) {
                        build_key_width += DtypeWidth(cs.dtype);
                        break;
                      }
                    }
                  }

                  unsigned build_payload_width = 0;
                  for (const auto &cs : payload_schema)
                    build_payload_width += DtypeWidth(cs.dtype);

                  if (build_key_width > 0 && build_payload_width > 0) {
                    if (!jit_ctx->aqp_jit_context)
                      jit_ctx->aqp_jit_context =
                          duckdb::make_uniq<duckdb::AQPJITContext>();

                    // Pipeline-JIT direct-HT path: register an empty
                    // AQPJoinHTView. Fields are filled in
                    // PhysicalHashJoin::ExecuteInternal at probe time after
                    // the build side has Finalize()d. The probe-fn consumes
                    // this view to touch DuckDB's JoinHashTable directly.
                    jit_ctx->aqp_jit_context->join_ht_views[eid] =
                        duckdb::make_uniq<duckdb::AQPJoinHTView>();
                    // pipeline_states points at the same view — passed as the
                    // 3rd arg to the JIT'd AQPPipelineFn at dispatch time.
                    jit_ctx->aqp_jit_context->pipeline_states[eid] =
                        static_cast<void *>(
                            jit_ctx->aqp_jit_context->join_ht_views[eid].get());

                    // payload_row_indices[k] = k: payload_schema is already
                    // built in the order DuckDB stores payload columns in the
                    // row layout (needed_payload order matches
                    // hj.payload_columns.col_idxs which is what DuckDB stores).
                    std::vector<int> payload_row_indices;
                    payload_row_indices.reserve(payload_schema.size());
                    for (size_t k = 0; k < payload_schema.size(); ++k)
                      payload_row_indices.push_back(static_cast<int>(k));

                    // Output subsets matching the DuckDB hash-join output chunk
                    // shape [lhs subset, rhs subset]. lhs idxs are positions
                    // into probe_schema (probe-side child output). rhs idxs are
                    // indices into the HT layout = [keys, payload].
                    auto &hj = op.Cast<duckdb::PhysicalHashJoin>();
                    std::vector<int> lhs_output_idxs;
                    lhs_output_idxs.reserve(hj.lhs_output_columns.col_idxs.size());
                    for (auto ci : hj.lhs_output_columns.col_idxs)
                      lhs_output_idxs.push_back(static_cast<int>(ci));
                    std::vector<int> rhs_output_layout_idxs;
                    rhs_output_layout_idxs.reserve(hj.rhs_output_columns.col_idxs.size());
                    for (auto ci : hj.rhs_output_columns.col_idxs)
                      rhs_output_layout_idxs.push_back(static_cast<int>(ci));

                    // DuckDB-authoritative output dtypes. AQP IR's
                    // probe_schema/payload_schema may have a different ordering
                    // than DuckDB's actual chunk schema, so we MUST use
                    // hj.lhs/rhs_output_columns.col_types for elem_size.
                    auto LtToAqpDtype =
                        [](const duckdb::LogicalType &lt) -> int32_t {
                      switch (lt.id()) {
                      case duckdb::LogicalTypeId::BOOLEAN:
                        return AQP_DTYPE_BOOL;
                      case duckdb::LogicalTypeId::TINYINT:
                      case duckdb::LogicalTypeId::UTINYINT:
                        return AQP_DTYPE_INT8;
                      case duckdb::LogicalTypeId::SMALLINT:
                      case duckdb::LogicalTypeId::USMALLINT:
                        return AQP_DTYPE_INT16;
                      case duckdb::LogicalTypeId::INTEGER:
                      case duckdb::LogicalTypeId::UINTEGER:
                        return AQP_DTYPE_INT32;
                      case duckdb::LogicalTypeId::BIGINT:
                      case duckdb::LogicalTypeId::UBIGINT:
                      case duckdb::LogicalTypeId::HUGEINT:
                        return AQP_DTYPE_INT64;
                      case duckdb::LogicalTypeId::FLOAT:
                        return AQP_DTYPE_FLOAT;
                      case duckdb::LogicalTypeId::DOUBLE:
                        return AQP_DTYPE_DOUBLE;
                      case duckdb::LogicalTypeId::VARCHAR:
                        return AQP_DTYPE_VARCHAR;
                      case duckdb::LogicalTypeId::DATE:
                        return AQP_DTYPE_DATE;
                      default:
                        return AQP_DTYPE_OTHER;
                      }
                    };
                    std::vector<int32_t> lhs_output_dtypes;
                    lhs_output_dtypes.reserve(
                        hj.lhs_output_columns.col_types.size());
                    for (auto &lt : hj.lhs_output_columns.col_types)
                      lhs_output_dtypes.push_back(LtToAqpDtype(lt));
                    std::vector<int32_t> rhs_output_dtypes;
                    rhs_output_dtypes.reserve(
                        hj.rhs_output_columns.col_types.size());
                    for (auto &lt : hj.rhs_output_columns.col_types)
                      rhs_output_dtypes.push_back(LtToAqpDtype(lt));

                    // DuckDB-authoritative LHS join-key chunk positions:
                    // PhysicalComparisonJoin::conditions[i].left is the LHS
                    // expression; when it's a BoundReferenceExpression, .index
                    // is the chunk column position. AQP IR's positional lookup
                    // (table_idx, col_idx) → probe_schema position can pick
                    // the wrong column when ordering diverges from DuckDB's
                    // physical chunk even if dtypes coincidentally match —
                    // that produces wrong hashes (crash or silent miss).
                    std::vector<int> lhs_key_chunk_idxs;
                    std::vector<int32_t> lhs_key_dtypes;
                    bool keys_ok = true;
                    lhs_key_chunk_idxs.reserve(hj.conditions.size());
                    lhs_key_dtypes.reserve(hj.conditions.size());
                    for (auto &cond : hj.conditions) {
                      if (!cond.left || cond.left->GetExpressionClass() !=
                                            duckdb::ExpressionClass::BOUND_REF) {
                        keys_ok = false;
                        break;
                      }
                      auto &bref =
                          cond.left->Cast<duckdb::BoundReferenceExpression>();
                      lhs_key_chunk_idxs.push_back(static_cast<int>(bref.index));
                      lhs_key_dtypes.push_back(
                          LtToAqpDtype(cond.left->return_type));
                    }
                    if (!keys_ok) {
                      lhs_key_chunk_idxs.clear();
                      lhs_key_dtypes.clear();
                    }

                    auto t_fpp = chrono_tic();
                    void *probe_fused_fn =
                        jit_comp->CompileFilterProbeProjectFusion(
                            probe_filter_ir, join_ir, probe_proj_ir,
                            probe_schema, payload_schema,
                            payload_row_indices, lhs_output_idxs,
                            rhs_output_layout_idxs, lhs_output_dtypes,
                            rhs_output_dtypes, lhs_key_chunk_idxs,
                            lhs_key_dtypes);
                    if (enable_timing_ && BREAK_DOWN_COMPILE_TIME) {
                      chrono_toc(
                          &t_fpp,
                          "RegisterJIT::CompileFilterProbeProjectFusion\n",
                          false);
                    }
                    if (probe_fused_fn) {
                      jit_ctx->aqp_jit_context->pipeline_fns[eid] =
                          reinterpret_cast<duckdb::AQPPipelineFn>(
                              probe_fused_fn);
                      jit_ctx->aqp_jit_context->flags |= duckdb::AQPJIT_PIPELINE;
#ifndef NDEBUG
                      std::cerr
                          << "[AQP-JIT-COMPILE] direct-HT probe eid=0x"
                          << std::hex << eid << std::dec
                          << " probe_cols=" << probe_schema.size()
                          << " payload_cols=" << payload_schema.size()
                          << " lhs_out=" << lhs_output_idxs.size()
                          << " rhs_out=" << rhs_output_layout_idxs.size()
                          << " keys=" << lhs_key_chunk_idxs.size()
                          << " key_chunk_idx[0]="
                          << (lhs_key_chunk_idxs.empty()
                                  ? -1
                                  : lhs_key_chunk_idxs[0])
                          << "\n";
#endif
                      // Build Bloom filter from build-side temp table
                      // for probe pre-filtering. Only for temp tables
                      // (identified by build_tidx in intermediate_table_map)
                      // with single integer key columns.
                      // On the speculative bg thread, resolve the collection
                      // through the launch-time snapshot instead of the live
                      // maps (which the main thread may mutate concurrently).
                      if (build_phys_scan && build_tidx != UINT_MAX &&
                          lhs_key_chunk_idxs.size() == 1) {
                        duckdb::ColumnDataCollection *bf_coll = nullptr;
                        if (bf_temp_snapshot) {
                          auto sit = bf_temp_snapshot->find(build_tidx);
                          if (sit != bf_temp_snapshot->end())
                            bf_coll = sit->second;
                        } else {
                          auto tit = intermediate_table_map.find(build_tidx);
                          if (tit != intermediate_table_map.end()) {
                            auto cit = temp_collections_.find(tit->second);
                            if (cit != temp_collections_.end() &&
                                cit->second.collection)
                              bf_coll = cit->second.collection.get();
                          }
                        }
                        if (bf_coll) {
                            // Build BF using column 0 of the join key in the build side.
                            // Find which column in the temp table corresponds to
                            // the build-side join key.
                            size_t bf_col_idx = 0;
                            for (const auto &cond : join->join_conditions) {
                              bool found = false;
                              for (size_t ci = 0; ci < build_schema.size(); ci++) {
                                if ((build_schema[ci].table_idx == cond->left_attr->GetTableIndex() &&
                                     build_schema[ci].col_idx == cond->left_attr->GetColumnIndex()) ||
                                    (build_schema[ci].table_idx == cond->right_attr->GetTableIndex() &&
                                     build_schema[ci].col_idx == cond->right_attr->GetColumnIndex())) {
                                  bf_col_idx = ci;
                                  found = true;
                                  break;
                                }
                              }
                              if (found) break;
                            }
                            auto bf_info = BuildBloomFilter(*bf_coll, bf_col_idx, *jit_ctx);
                            if (!bf_info.bf_data.empty()) {
                              auto jbf = duckdb::make_uniq<duckdb::AQPJITContext::AQPJoinBloomFilter>();
                              jbf->bf_data = std::move(bf_info.bf_data);
                              jbf->bitmask = bf_info.bitmask;
                              jit_ctx->aqp_jit_context->join_bloom_filters[eid] = std::move(jbf);
                            }
                        }
                      }
                    }
                  }
                  } // schema_match
                } // !probe_schema.empty()
                } // probe_schema block
              }
            }
          }
        }
        } // !ir_sides_swapped
      }
    }
    } // end single-probe block
    aqp_skip_single_probe:;
  }


  // Recurse into children.
  // children[1] of a HASH_JOIN is the build side — pass is_build_side=true
  // so multi-probe fusion is disabled for chains on build pipelines.
  for (size_t ci = 0; ci < op.children.size(); ci++) {
    bool child_is_build = (op.type == PhysicalOperatorType::HASH_JOIN && ci == 1);
    RegisterJITImpl(op.children[ci].get(), ir, jit_ctx, jit_comp,
                    consumed_filters, consumed_joins, bf_temp_snapshot,
                    child_is_build);
  }
}

void DuckDBAdapter::InjectTempTableJoinStats(duckdb::PhysicalOperator &op) {
  using duckdb::PhysicalOperatorType;

  if (op.type == PhysicalOperatorType::HASH_JOIN) {
    auto &join = static_cast<duckdb::PhysicalHashJoin &>(op);

    if (join.join_type == duckdb::JoinType::INNER &&
        join.conditions.size() == 1 &&
        join.conditions[0].comparison == duckdb::ExpressionType::COMPARE_EQUAL &&
        join.join_stats.empty()) {

      auto &build_child = join.children[1].get();
      auto key_type = join.conditions[0].right->return_type;
      if (!duckdb::TypeIsIntegral(key_type.InternalType()))
        goto recurse;

      // Guard: no FILTER on build path (filters change effective key range)
      std::function<bool(duckdb::PhysicalOperator &)> has_filter =
          [&](duckdb::PhysicalOperator &node) -> bool {
        if (node.type == PhysicalOperatorType::FILTER) return true;
        if (node.type == PhysicalOperatorType::TABLE_SCAN) return false;
        for (auto &child : node.children)
          if (has_filter(child.get())) return true;
        return false;
      };
      if (has_filter(build_child))
        goto recurse;

      // Find temp table scan on build path
      std::function<std::string(duckdb::PhysicalOperator &)> find_temp_scan =
          [&](duckdb::PhysicalOperator &node) -> std::string {
        if (node.type == PhysicalOperatorType::TABLE_SCAN) {
          auto &scan = static_cast<duckdb::PhysicalTableScan &>(node);
          auto *bd = scan.bind_data
              ? dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get())
              : nullptr;
          if (bd && bd->table.name.find("temp") == 0)
            return bd->table.name;
        }
        for (auto &child : node.children) {
          auto name = find_temp_scan(child.get());
          if (!name.empty()) return name;
        }
        return "";
      };

      std::string tt_name = find_temp_scan(build_child);
      if (tt_name.empty())
        goto recurse;

      static constexpr int64_t MAX_BUILD_SIZE = 1048576;
      uint64_t tt_card = GetTempTableCardinality(tt_name);
      if (tt_card > static_cast<uint64_t>(MAX_BUILD_SIZE))
        goto recurse;

      auto min_max = GetTempTableMinMax(tt_name);
      // Find which column in the build scan corresponds to the join key.
      // Use the right-side condition expression to find the column index.
      // For simplicity, try all integer columns and check if any has a
      // range < MAX_BUILD_SIZE.
      for (auto it = min_max.begin(); it != min_max.end(); ++it) {
        int64_t range_min = it->second.first;
        int64_t range_max = it->second.second;
        if (range_max < range_min) continue;
        int64_t range = range_max - range_min;
        if (range > MAX_BUILD_SIZE) continue;

        // Create stats and inject
        auto stats = duckdb::BaseStatistics::CreateEmpty(key_type);
        duckdb::NumericStats::SetMin(stats, duckdb::Value::Numeric(key_type, range_min));
        duckdb::NumericStats::SetMax(stats, duckdb::Value::Numeric(key_type, range_max));
        stats.SetHasNoNull();

        join.join_stats.resize(2);
        join.join_stats[1] = stats.ToUnique();
        break;
      }
    }
  }

recurse:
  for (auto &child_ref : op.children)
    InjectTempTableJoinStats(child_ref.get());
}

void DuckDBAdapter::RegisterBloomFilters(duckdb::PhysicalOperator &op,
                                         duckdb::ClientContext *ctx,
                                         std::vector<BloomFilterInfo> &bfs) {
  using duckdb::PhysicalOperatorType;
  if (bfs.empty()) return;

  if (op.type == PhysicalOperatorType::TABLE_SCAN) {
    auto &scan = static_cast<duckdb::PhysicalTableScan &>(op);
    auto *tsbd = scan.bind_data
        ? dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get())
        : nullptr;
    if (!tsbd) return;
    const auto &tname = tsbd->table.name;

    for (auto &bf_info : bfs) {
      if (bf_info.bf_data.empty() || bf_info.base_table_name != tname)
        continue;

      // Find which chunk column index corresponds to bf_info.base_col_name.
      // With filter_prune, the chunk has projection_ids.size() columns
      // (a subset of column_ids). We must map column_ids position → chunk
      // position via projection_ids.
      auto &cols = tsbd->table.GetColumns();
      uint32_t chunk_col_idx = UINT32_MAX;
      int32_t dtype = AQP_DTYPE_INT32;
      for (size_t i = 0; i < scan.column_ids.size() && i < scan.types.size(); i++) {
        auto physical_col = scan.column_ids[i].GetPrimaryIndex();
        if (physical_col >= cols.LogicalColumnCount()) continue;
        auto &col = cols.GetColumn(duckdb::LogicalIndex(physical_col));
        if (col.Name() != bf_info.base_col_name) continue;
        auto pt = col.Type().InternalType();
        if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
          break;
        dtype = (pt == duckdb::PhysicalType::INT64)
                    ? AQP_DTYPE_INT64
                    : AQP_DTYPE_INT32;
        if (!scan.projection_ids.empty()) {
          for (size_t j = 0; j < scan.projection_ids.size(); j++) {
            if (scan.projection_ids[j] == i) {
              chunk_col_idx = static_cast<uint32_t>(j);
              break;
            }
          }
        } else {
          chunk_col_idx = static_cast<uint32_t>(i);
        }
        break;
      }
      if (chunk_col_idx == UINT32_MAX) continue;

      if (!ctx->aqp_jit_context) {
        ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
      }
      uint64_t scan_eid = duckdb::ExpressionID(scan);
      auto bloom = duckdb::make_uniq<duckdb::AQPJITContext::AQPBloomScanFilter>();
      bloom->bf_data = std::move(bf_info.bf_data);
      bloom->bitmask = bf_info.bitmask;
      bloom->col_idx = chunk_col_idx;
      bloom->dtype = dtype;
      // Only one BF per (scan, column) pair. Multiple BFs on the same column
      // from different temp columns are AND'd, which is too restrictive
      // (e.g., self-join: subject_id and status_id both map to comp_cast_type.id).
      auto &bf_vec = ctx->aqp_jit_context->bloom_scan_filters[scan_eid];
      bool dup = false;
      for (auto &existing : bf_vec) {
        if (existing->col_idx == chunk_col_idx) { dup = true; break; }
      }
      if (!dup) {
        bf_vec.push_back(std::move(bloom));
      }
#ifndef NDEBUG
      std::cerr << "[AQP-BF] Registered bloom filter for " << tname
                << "." << bf_info.base_col_name
                << " col_idx=" << chunk_col_idx
                << " scan_eid=0x" << std::hex << scan_eid << std::dec
                << (dup ? " (SKIPPED dup)" : "") << "\n";
#endif
    }
    return;
  }

  for (auto &child_ref : op.children)
    RegisterBloomFilters(child_ref.get(), ctx, bfs);
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
