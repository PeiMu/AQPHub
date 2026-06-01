/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adapters/db_adapter.h"
#include "storage/flat_table.h"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/column/column_data_scan_states.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/query_split/query_split.hpp"
#include "duckdb/optimizer/query_split/subquery_preparer.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_constraint.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

#ifdef HAVE_LLVM
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/aqp_jit.hpp"
#include "jit/ir_to_llvm.h"
#endif

#define IN_MEM_TMP_TABLE true

namespace duckdb {
class DuckDB;
class Connection;
class ClientContext;
class LogicalOperator;
class Planner;
class Optimizer;
} // namespace duckdb

namespace middleware {

#if IN_MEM_TMP_TABLE
// Stored result for replacement-scan-based temp tables
struct StoredTempResult {
  duckdb::unique_ptr<duckdb::ColumnDataCollection> collection;
  std::vector<std::string> column_names;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;
  std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> column_stats;
};

// ReplacementScanData subclass: holds pointer to temp_collections_ map
struct TempCollectionScanData : public duckdb::ReplacementScanData {
  TempCollectionScanData(
      std::unordered_map<std::string, StoredTempResult> *collections,
      std::unordered_map<std::string, const storage::FlatTable *> *kernel_temps = nullptr)
      : temp_collections(collections), kernel_temps(kernel_temps) {}
  std::unordered_map<std::string, StoredTempResult> *temp_collections;
  std::unordered_map<std::string, const storage::FlatTable *> *kernel_temps;
};

// TableFunctionInfo subclass: holds pointer to temp_collections_ map
struct TempCollectionScanInfo : public duckdb::TableFunctionInfo {
  explicit TempCollectionScanInfo(
      std::unordered_map<std::string, StoredTempResult> *collections)
      : temp_collections(collections) {}
  std::unordered_map<std::string, StoredTempResult> *temp_collections;
};
#endif

class DuckDBAdapter : public EngineAdapter {
public:
  explicit DuckDBAdapter(const std::string &db_path = ":memory :");
  ~DuckDBAdapter() override;

  void ResetQueryState() override;
  void LoadTablesFromCSV(const std::string &schema_path,
                         const std::string &csv_dir);

  // Parse SQL and return logical plan
  void ParseSQL(const std::string &sql) override;

  // Optimizer
  void Optimize();
  void FilterOptimize();
  void PostOptimizePlan();

  // Re-optimize IR by re-planning through DuckDB's full optimizer.
  // Generates SQL from the IR, parses it (DuckDB binds temp tables from
  // catalog), runs the full Optimize(), and converts back to IR.
  std::unique_ptr<ir_sql_converter::AQPStmt>
  ReOptimizeIR(std::unique_ptr<ir_sql_converter::AQPStmt> ir) override;

  void *GetLogicalPlan();

  void PrintLogicalPlan() { plan->Print(); };

  // Convert logical plan to IR
  std::unique_ptr<ir_sql_converter::AQPStmt> ConvertPlanToIR() override;

  // Execute SQL query
  QueryResult ExecuteSQL(const std::string &sql) override;
  void ExecuteSQLandCreateTempTable(const std::string &sql,
                                    const std::string &temp_table_name,
                                    bool update_temp_card) override;

  // Temp table management
  void CreateTempTable(const std::string &table_name,
                       const QueryResult &result) override;

  void DropTempTable(const std::string &table_name) override;

  bool TempTableExists(const std::string &table_name) override;

  uint64_t GetTempTableCardinality(const std::string &temp_table_name) override;

  void SetTempTableCardinality(const std::string &temp_table_name,
                               uint64_t cardinality) override;

  // Compute min/max for integer columns of a temp table from in-memory data.
  // Returns map: column_index -> (min_value, max_value) for INT32/INT64 cols.
  std::unordered_map<size_t, std::pair<int64_t, int64_t>>
  GetTempTableMinMax(const std::string &temp_table_name);

  // Get row count of a base (non-temp) table from the main catalog.
  uint64_t GetBaseTableCardinality(const std::string &table_name);

  // Collect distinct integer values from a temp table column.
  // Returns empty vector if column is not INT32/INT64, table not found,
  // or distinct count exceeds max_distinct.
  std::vector<int64_t>
  GetTempTableDistinctValues(const std::string &temp_table_name,
                             size_t col_idx, size_t max_distinct);

  // Look up a column name by table name and column index from DuckDB catalog.
  std::string GetColumnName(const std::string &table_name, unsigned col_idx);

  // Temp table column min/max info. Set by IRQuerySplitter after each
  // temp table materialization; consumed by RegisterJIT to compile range
  // filters on base table scans that join with these temp table columns.
  struct TempColRange {
    std::string temp_table_name;  // e.g. "temp1"
    unsigned col_idx;             // column position in temp table
    int64_t min_val;
    int64_t max_val;
  };
  void SetTempColRanges(std::vector<TempColRange> ranges) {
    temp_col_ranges_ = std::move(ranges);
  }

  // Bloom filter built from a temp table's join key column.
  // Used to filter base table scans in subsequent sub-plans.
  struct BloomFilterInfo {
    std::string base_table_name;  // e.g. "cast_info"
    std::string base_col_name;    // e.g. "id"
    std::vector<uint64_t> bf_data;
    uint64_t bitmask;
  };
  void SetPendingBloomFilters(std::vector<BloomFilterInfo> filters) {
    pending_bloom_filters_ = std::move(filters);
  }
  const std::vector<BloomFilterInfo> &GetPendingBloomFilters() const {
    return pending_bloom_filters_;
  }

  // Build a Bloom filter from a temp table's integer column.
  // Returns empty bf_data if column is not INT32/INT64 or table not found.
  BloomFilterInfo BuildBloomFilter(const std::string &temp_table_name,
                                   size_t col_idx,
                                   uint64_t temp_card);

  // Build a BF from a ColumnDataCollection directly (supports any hashable type).
  static BloomFilterInfo BuildBloomFilterFromCollection(
      duckdb::ColumnDataCollection &collection,
      size_t col_idx, duckdb::ClientContext &ctx);

  // Get estimated cost and rows for a query using EXPLAIN
  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;

  std::string GetEngineName() const override { return "DuckDB"; }

  void CleanUp() override;

  // Get context and binder for IR conversion
  duckdb::ClientContext *GetClientContext();

  // Get the DuckDB connection (for StoragePlan loading)
  duckdb::Connection &GetConnection() { return *conn; }

#ifdef HAVE_LLVM
  // JIT: set the sub-IR and flags for compilation before the next
  // ExecuteSQLandCreateTempTable call. Called by IRQuerySplitter.
  void SetJITPendingIR(const ir_sql_converter::AQPStmt *ir, uint32_t flags = 0,
                       duckdb::unique_ptr<duckdb::LogicalOperator> plan = nullptr) {
    jit_pending_ir_ = ir;
    jit_flags_ = flags;
    jit_pending_plan_ = std::move(plan);
  }

  // Set JIT flags independently (used in no-split path where SetJITPendingIR
  // is not called by the splitter).
  void SetJITFlags(uint32_t flags) { jit_flags_ = flags; }

  void SetJITOptFlags(bool fusion_probe, bool inline_hash,
                      bool payload_prune, bool prefetch, int prefetch_dist,
                      bool batch_probe, bool cache,
                      const std::string &cache_dir) {
    jit_fusion_probe_ = fusion_probe;
    jit_inline_hash_ = inline_hash;
    jit_payload_prune_ = payload_prune;
    jit_prefetch_ = prefetch;
    jit_prefetch_distance_ = prefetch_dist;
    jit_batch_probe_ = batch_probe;
    jit_cache_ = cache;
    jit_cache_dir_ = cache_dir;
  }

  // Phase 6: ROF probe-side look-ahead distances. 0 disables that level.
  void SetJITProbePrefetchDistances(int entry_dist, int row_dist) {
    jit_prefetch_entry_distance_ = entry_dist;
    jit_prefetch_row_distance_ = row_dist;
  }
#endif

  // NodeBasedSplitter support
  // Return the chunk table index allocated by the last
  // ExecuteSQLandCreateTempTable call. SubqueryPreparer::SetNewTableIndex must
  // use this exact index so that ConvertDuckDBPlanToIR can resolve the
  // resulting CHUNK_GET node via intermediate_table_map (which was already
  // populated by ExecuteSQLandCreateTempTable).
  duckdb::idx_t GetTempTableIndex() const { return temp_table_index_; }

  // Register a temp table created by an external execution engine.
  // Allocates a fresh DuckDB table index and maps it to temp_name so that
  // NodeBasedSplitter::UpdateRemainingIR can call SetNewTableIndex correctly.
  // Does NOT execute any SQL.
  void
  RegisterExternalTempTable(const std::string &temp_name,
                            const duckdb::vector<duckdb::LogicalType> &types,
                            const std::vector<std::string> &col_names);

  // Kernel temp table management (CSR executor results)
  void RegisterKernelTemp(const std::string &name,
                          const storage::FlatTable *table);
  void ClearKernelTemps();

  // Create a temp table from a FlatTable (for kernel results).
  // Sets up DuckDB internal state (data_chunk_index, chunk_col_names_,
  // temp_collections_, temp_table_index_) identically to
  // ExecuteSQLandCreateTempTable, but from flat arrays instead of SQL.
  void CreateTempFromFlatTable(const storage::FlatTable &flat,
                               const std::string &temp_table_name);

  // Lightweight metadata-only registration for kernel-produced temps.
  // Sets temp_table_index_, intermediate_table_map, temp_table_types,
  // chunk_col_names_, table_column_mappings, and temp_table_card_ WITHOUT
  // copying data into ColumnDataCollection. DuckDB SQL can still read the
  // data via scan_kernel_temp replacement scan.
  void RegisterTempMetadata(const storage::FlatTable &flat,
                            const std::string &temp_table_name);

  // Access a StoredTempResult (for loading into FlatTable)
  const StoredTempResult *GetStoredTempResult(const std::string &name) const;

  // Get reference to binder (for NodeBasedSplitter to create Optimizer /
  // QuerySplit / SubqueryPreparer)
  duckdb::Binder &GetBinder();

  // Take/return plan ownership (NodeBasedSplitter drives the plan through the
  // loop; SetPlan(sub_plan) + ConvertPlanToIR() converts each sub-plan to IR)
  duckdb::unique_ptr<duckdb::LogicalOperator> TakePlan();
  void SetPlan(duckdb::unique_ptr<duckdb::LogicalOperator> p);

  struct pair_hash {
    template <class T1, class T2>
    uint64_t operator()(const std::pair<T1, T2> &p) const {
      auto h1 = std::hash<T1>{}(p.first);
      auto h2 = std::hash<T2>{}(p.second);

      // Mainly for demonstration purposes, i.e. works but is overly simple
      // In the real world, use sth. like boost.hash_combine
      return h1 ^ h2;
    }
  };

  duckdb::vector<duckdb::LogicalType> temp_table_types;

private:
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> conn;
  std::unique_ptr<duckdb::Planner> planner;
  duckdb::unique_ptr<duckdb::LogicalOperator> plan;

  std::unordered_map<std::pair<uint64_t, uint64_t>, std::string, pair_hash>
      table_column_mappings;

  // <temp%, subquery_dd_index>
  std::unordered_map<unsigned int, std::string> intermediate_table_map;

  // Index allocated by the most recent ExecuteSQLandCreateTempTable call;
  // used by ExecuteSplitLoopNodeBased to call sp.SetNewTableIndex correctly.
  duckdb::idx_t temp_table_index_ = 0;

#ifdef HAVE_LLVM
#define BREAK_DOWN_COMPILE_TIME false
//  // Per-phase JIT compilation timing (microseconds), accumulated by RegisterJIT.
//  struct JitTimingStats {
//    long compile_filter_us = 0;
//    long compile_projection_us = 0;
//    long compile_aggregate_us = 0;
//    long compile_hash_build_us = 0;
//    long compile_hash_probe_us = 0;
//    long compile_pipeline_us = 0; // filter+project, filter+agg, filter+hashbuild, filter+probe+proj
//    long compile_sql_us = 0;
//    long register_jit_us = 0;
//    long run_us = 0;
//    void Reset() { memset(this, 0, sizeof(JitTimingStats)); }
//  };
//  JitTimingStats jit_timing_;

  // Pending sub-IR for JIT compilation (set before ExecuteSQLandCreateTempTable)
  const ir_sql_converter::AQPStmt *jit_pending_ir_ = nullptr;
  uint32_t jit_flags_ = 0;  // AQPJIT_* bitmask from param_config

  // Per-optimization toggles (from ParamConfig)
  bool jit_fusion_probe_ = true;
  bool jit_inline_hash_ = true;
  bool jit_payload_prune_ = true;
  bool jit_prefetch_ = true;
  int  jit_prefetch_distance_ = 8;
  // Phase 6: ROF probe look-ahead distances (consumer-side prefetch in
  // CompileFilterProbeProjectFusion). Plan §10.4 defaults.
  int  jit_prefetch_entry_distance_ = 24;
  int  jit_prefetch_row_distance_ = 12;
  bool jit_batch_probe_ = true;
  bool jit_cache_ = false;
  std::string jit_cache_dir_;
  // Owned IR built in the no-split JIT path; must outlive jit_pending_ir_.
  std::unique_ptr<ir_sql_converter::AQPStmt> owned_jit_ir_;
  // Pre-built logical plan for PrepareFromPlan (avoids redundant parse+optimize).
  duckdb::unique_ptr<duckdb::LogicalOperator> jit_pending_plan_;

  // Keeps the LLJIT instance alive until after query execution so that
  // compiled function pointers stored in AQPJITContext remain valid.
  std::unique_ptr<aqp_jit::IrToLlvmCompiler> jit_compiler_;

  // Lazily create and configure the JIT compiler with all flags.
  void EnsureJITCompiler();

  // Walk physical plan tree; compile IR filters and register in aqp_jit_context.
  void RegisterJIT(duckdb::PhysicalOperator &op,
                          const ir_sql_converter::AQPStmt &ir);
  // Walk physical plan tree; register pending bloom filters in aqp_jit_context.
  void RegisterBloomFilters(duckdb::PhysicalOperator &op);

  // Temp column ranges from temp table min/max (set before each sub-plan).
  std::vector<TempColRange> temp_col_ranges_;
  std::vector<BloomFilterInfo> pending_bloom_filters_;

  // IR filter nodes already matched to a DuckDB operator in this RegisterJIT
  // pass.  Prevents two DuckDB FILTERs on the same table from binding to the
  // same IR FilterNode.
  std::unordered_set<const ir_sql_converter::AQPStmt *>
      jit_consumed_ir_filters_;

  // IR join nodes already matched to a DuckDB HASH_JOIN operator.
  // Prevents multiple physical HASH_JOINs from binding to the same IR JoinNode.
  std::unordered_set<const ir_sql_converter::AQPStmt *>
      jit_consumed_ir_joins_;
#endif

  // Column names for each chunk table: data_chunk_index → column names as
  // stored in the temp table. Used by ConvertDuckDBPlanToIR to resolve correct
  // column names for CHUNK_GET nodes (DuckDB plan alias ≠ temp table alias).
  std::unordered_map<unsigned int, std::vector<std::string>> chunk_col_names_;


#if IN_MEM_TMP_TABLE
private:
  // Register the temp collection table function and replacement scan
  void RegisterTempCollectionScan();

  // Table function callbacks (static)
  static duckdb::unique_ptr<duckdb::FunctionData>
  TempCollectionBind(duckdb::ClientContext &context,
                     duckdb::TableFunctionBindInput &input,
                     duckdb::vector<duckdb::LogicalType> &return_types,
                     duckdb::vector<duckdb::string> &names);

  static duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
  TempCollectionInitGlobal(duckdb::ClientContext &context,
                           duckdb::TableFunctionInitInput &input);

  static void TempCollectionScanFunc(duckdb::ClientContext &context,
                                     duckdb::TableFunctionInput &data,
                                     duckdb::DataChunk &output);

  static duckdb::unique_ptr<duckdb::NodeStatistics>
  TempCollectionCardinality(duckdb::ClientContext &context,
                            const duckdb::FunctionData *bind_data);

  static duckdb::unique_ptr<duckdb::BaseStatistics>
  TempCollectionStatistics(duckdb::ClientContext &context,
                           duckdb::TableFunctionGetStatisticsInput &input);

  // Inject join_stats into PhysicalHashJoin nodes whose build side scans a
  // temp table, enabling DuckDB's perfect hash join optimization.
  void InjectTempTableJoinStats(duckdb::PhysicalOperator &op);

  // Replacement scan callback (static)
  static duckdb::unique_ptr<duckdb::TableRef> TempCollectionReplacementScan(
      duckdb::ClientContext &context, duckdb::ReplacementScanInput &input,
      duckdb::optional_ptr<duckdb::ReplacementScanData> data);
  // Replacement scan: in-memory temp table storage
  std::unordered_map<std::string, StoredTempResult> temp_collections_;

  // Kernel temp tables (CSR executor results, owned by IRQuerySplitter)
  std::unordered_map<std::string, const storage::FlatTable *>
      kernel_temp_tables_;

  // Table function callbacks for kernel temp tables (static)
  static duckdb::unique_ptr<duckdb::FunctionData>
  KernelTempBind(duckdb::ClientContext &context,
                 duckdb::TableFunctionBindInput &input,
                 duckdb::vector<duckdb::LogicalType> &return_types,
                 duckdb::vector<duckdb::string> &names);

  static duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
  KernelTempInitGlobal(duckdb::ClientContext &context,
                       duckdb::TableFunctionInitInput &input);

  static void KernelTempScanFunc(duckdb::ClientContext &context,
                                 duckdb::TableFunctionInput &data,
                                 duckdb::DataChunk &output);

  static duckdb::unique_ptr<duckdb::NodeStatistics>
  KernelTempCardinality(duckdb::ClientContext &context,
                        const duckdb::FunctionData *bind_data);
#endif
};
} // namespace middleware
