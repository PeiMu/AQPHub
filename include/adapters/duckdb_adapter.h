/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#pragma once

#include <functional>
#include <future>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adapters/db_adapter.h"
#include "storage/flat_table.h"
#include "util/param_config.h"

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
#include "duckdb_version_compat.h"

#ifdef HAVE_LLVM
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/aqp_jit.hpp"
#include "jit/ir_to_llvm.h"
#include "qjit/query_jit_executor.h"
#endif

namespace duckdb { class PhysicalHashJoin; }
namespace middleware { namespace storage { class StoragePlan; } }

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
  // True while this entry is a speculative placeholder (empty collection,
  // estimated cardinality). The real result is Combine()d into the same
  // collection object — a speculative Prepare's bind data holds a raw
  // pointer to it, so it must never be replaced.
  bool is_placeholder = false;
};

// Metadata for temps served directly from the in-memory qjit::QjitTable via
// the scan_qjit_temp table function (no CDC copy). Holds the binder-facing
// fields a StoredTempResult would have carried; the row data itself lives in
// qjit_temps_. Per-column BaseStatistics are deliberately absent: the CDC
// path computes them but never wires TableFunction::statistics, so they
// never reach the optimizer there either.
struct QjitTempMeta {
  std::vector<std::string> column_names;
  duckdb::vector<duckdb::LogicalType> types;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;
};

#ifdef HAVE_LLVM
struct CachedSubquery {
  bool is_query_jit = false;
  bool is_interpreter_fallback = false;

  std::string cache_key;
  std::string fn_name;
  std::vector<qjit::QjitColumnRef> source_cols;
  std::vector<std::string> source_tables;
  std::vector<bool> source_is_temp;
  std::vector<int> source_block_skip_cols;
  std::vector<size_t> step_col_counts;

  std::vector<uint32_t> ht_tuple_sizes;
  std::vector<uint32_t> ht_key0_offsets;
  std::vector<qjit::QjitAggCellDesc> agg_descs;
  std::vector<int> agg_output_cells;
  std::vector<qjit::QjitTable::ColumnDesc> out_descs;
  std::vector<uint8_t> params_buf;

  std::string sql;
  std::string temp_table_name;
  std::vector<std::string> column_names;
  duckdb::vector<duckdb::LogicalType> types;
};

struct CachedQueryPlan {
  std::vector<CachedSubquery> subqueries;
  CachedSubquery final_query;
};
#endif

// ReplacementScanData subclass: holds pointer to temp_collections_ map
struct TempCollectionScanData : public duckdb::ReplacementScanData {
  TempCollectionScanData(
      std::unordered_map<std::string, StoredTempResult> *collections,
      std::unordered_map<std::string, const storage::FlatTable *> *kernel_temps = nullptr,
      std::unordered_map<std::string, QjitTempMeta> *qjit_meta = nullptr,
      std::shared_timed_mutex *temp_state_mutex = nullptr)
      : temp_collections(collections), kernel_temps(kernel_temps),
        qjit_meta(qjit_meta), temp_state_mutex(temp_state_mutex) {}
  std::unordered_map<std::string, StoredTempResult> *temp_collections;
  std::unordered_map<std::string, const storage::FlatTable *> *kernel_temps;
  std::unordered_map<std::string, QjitTempMeta> *qjit_meta;
  std::shared_timed_mutex *temp_state_mutex;
};

// TableFunctionInfo subclass: holds pointer to temp_collections_ map
struct TempCollectionScanInfo : public duckdb::TableFunctionInfo {
  explicit TempCollectionScanInfo(
      std::unordered_map<std::string, StoredTempResult> *collections,
      std::shared_timed_mutex *temp_state_mutex = nullptr)
      : temp_collections(collections), temp_state_mutex(temp_state_mutex) {}
  std::unordered_map<std::string, StoredTempResult> *temp_collections;
  std::shared_timed_mutex *temp_state_mutex;
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

  // Untimed config statement (no timing columns). disabled_optimizers is a
  // GLOBAL DuckDB setting (DBConfig), so it also covers fresh spec/bg
  // connections created later from the same DatabaseInstance.
  void ApplyEngineSetting(const std::string &sql) override {
    auto res = conn->Query(sql);
    if (res->HasError())
      throw std::runtime_error("ApplyEngineSetting failed: " +
                               res->GetError());
  }

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
  // Overrides base class SQL-based default with direct ColumnDataCollection scan.
  std::unordered_map<size_t, std::pair<int64_t, int64_t>>
  GetTempTableMinMax(
      const std::string &temp_table_name,
      const std::vector<std::string> &column_names,
      const std::vector<ir_sql_converter::SimplestVarType> &column_types)
      override;

  // Get row count of a base (non-temp) table from the main catalog.
  uint64_t GetBaseTableCardinality(const std::string &table_name) override;

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
  std::vector<BloomFilterInfo> TakePendingBloomFilters() {
    return std::move(pending_bloom_filters_);
  }

  // Walk a physical plan tree; register the given bloom filters into ctx's
  // aqp_jit_context. Moves bf_data out of the entries. Safe to call from the
  // speculative bg thread when op/ctx are exclusively owned by it.
  void RegisterBloomFilters(duckdb::PhysicalOperator &op,
                            duckdb::ClientContext *ctx,
                            std::vector<BloomFilterInfo> &bfs);

  // Build a Bloom filter from a temp table's integer column.
  // Returns empty bf_data if column is not INT32/INT64 or table not found.
  BloomFilterInfo BuildBloomFilter(const std::string &temp_table_name,
                                   size_t col_idx,
                                   uint64_t temp_card);

  // Same as above but from a collection directly (thread-safe: no map
  // lookups). ctx is only used to allocate the scan chunk.
  static BloomFilterInfo BuildBloomFilter(duckdb::ColumnDataCollection &collection,
                                          size_t col_idx,
                                          duckdb::ClientContext &ctx);

  // Build a BF from a ColumnDataCollection directly (supports any hashable type).
  static BloomFilterInfo BuildBloomFilterFromCollection(
      duckdb::ColumnDataCollection &collection,
      size_t col_idx, duckdb::ClientContext &ctx);

  // Get estimated cost and rows for a query using EXPLAIN
  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;

  // EXPLAIN ANALYZE plan text for a sub-SQL
  std::string ExplainAnalyze(const std::string &sql) override;

  std::string GetEngineName() const override { return "DuckDB"; }

  void CleanUp() override;

  // Get context and binder for IR conversion
  duckdb::ClientContext *GetClientContext();

  // Get the DuckDB connection (for StoragePlan loading)
  duckdb::Connection &GetConnection() { return *conn; }

  // Get the DuckDB instance (for creating speculative connections)
  duckdb::DuckDB &GetDB() { return *db; }

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

  void SetKernelPath(KernelPath kp) { kernel_path_ = kp; }
  void SetJITDebug(bool debug) { jit_debug_ = debug; }

  void SetJITOptFlags(bool payload_prune, bool prefetch, int prefetch_dist,
                      bool batch_probe, int skip_hash_cmp,
                      bool single_col_int_join = true) {
    jit_payload_prune_ = payload_prune;
    jit_prefetch_ = prefetch;
    jit_prefetch_distance_ = prefetch_dist;
    jit_batch_probe_ = batch_probe;
    jit_skip_hash_cmp_ = skip_hash_cmp;
    jit_single_col_int_join_ = single_col_int_join;
  }

  void SetRangeGuard(bool v) { range_guard_ = v; }
  void SetBlockSkip(bool v) { block_skip_ = v; }
  void SetMembershipPreprobe(bool v) { membership_preprobe_ = v; }

  void SetBenchmarkMode(bool benchmark) { benchmark_mode_ = benchmark; }

  // JIT object cache mode (--jit-cache=..., default 0=off).
  void SetJITCache(int mode) { jit_cache_ = mode; }
  void SetJITCacheDir(const std::string &dir) { jit_cache_dir_ = dir; }

  // Compile mode (--compile-mode): 0=llvm (full quality), 2=tpde.
  // Must be set before the first compile — the backend is fixed at
  // IrToLlvmCompiler construction.
  void SetCompileMode(int mode) { compile_mode_ = mode; }

  // One-shot spec-jit miss action, armed by the splitter at the spec
  // check site and consumed by the NEXT ExecuteSQLandCreateTempTable call.
  // FAST_ONCE: compile this sub-query inline with TPDE — covers query-jit
  // AND the expr/operator/pipeline RegisterJIT path. SKIP_QJIT_ONCE: skip
  // the query-jit compile entirely (Prepare + interpreter run; the jit CSV
  // column still writes, keeping rows rectangular).
  enum class CompensateMissAction { NONE, FAST_ONCE, SKIP_QJIT_ONCE };
  void SetCompensateMissAction(CompensateMissAction action) {
    compensate_miss_action_ = action;
  }

  // Query-jit (--jit-level=query): lingo-db-style runtime. Set separately
  // from SetJITFlags because the splitter masks jit_flags with
  // AQP_JIT_LEVEL_MASK, which deliberately excludes AQP_JIT_QUERY_JIT.
  void SetQueryJit(bool enabled, int threads, int morsel_size) {
    query_jit_ = enabled;
    query_jit_threads_ = threads;
    query_jit_morsel_ = morsel_size;
  }
  bool GetQueryJit() const { return query_jit_; }

  // Phase 2: query-jit reads base tables through the FlatTable storage plan
  // (same scan layer as kernel-path). Not owned; must outlive the adapter's
  // query execution (it does: main owns it for the process lifetime).
  void SetQueryJitStoragePlan(const middleware::storage::StoragePlan *plan) {
    qjit_storage_plan_ = plan;
  }

  // Phase 6: ROF probe-side look-ahead distances. 0 disables that level.
  void SetJITProbePrefetchDistances(int entry_dist, int row_dist) {
    jit_prefetch_entry_distance_ = entry_dist;
    jit_prefetch_row_distance_ = row_dist;
  }

  aqp_jit::IrToLlvmCompiler *GetJitCompiler();
  // Pre-create JIT compilers for the given (compile_mode, simd_isa) combos
  // so that EnsureJITCompiler never pays the LLJIT construction cost inline.
  void PreCreateCompilers(
      const std::vector<std::pair<int, int>> &backend_simd_pairs);
  // Launch PreCreateCompilers on a background thread. EnsureJITCompiler
  // drains the future before checking the cache, so creation overlaps
  // preprocess + the first (interpreted) sub-query.
  void PreCreateCompilersAsync(
      const std::vector<std::pair<int, int>> &backend_simd_pairs);

  void ExecuteSpeculativeAndCreateTempTable(
      duckdb::PreparedStatement &prepared, duckdb::Connection &spec_conn,
      const std::string &temp_table_name, bool update_temp_card,
      const std::string &sql = "");

  // Invoked twice per ExecuteSQLandCreateTempTable:
  // 1. post_execute=false — after Prepare (chunk index, column names/types and
  //    ESTIMATED cardinality known), BEFORE ExecuteRow, so a speculative
  //    compile for the NEXT subquery can overlap this subquery's execution.
  // 2. post_execute=true — after ExecuteRow + result store, with the ACTUAL
  //    cardinality, so the speculation layer can keep or relaunch the compile
  //    with an accurately-planned Prepare.
  using PostPrepareHook = std::function<void(
      const std::string &temp_table_name, duckdb::idx_t chunk_index,
      const duckdb::vector<duckdb::LogicalType> &types,
      const std::vector<std::string> &col_names, duckdb::idx_t est_card,
      bool post_execute)>;
  void SetPostPrepareHook(PostPrepareHook hook) {
    post_prepare_hook_ = std::move(hook);
  }

  // Register an empty placeholder temp collection (exact names/types,
  // estimated cardinality) so a speculative Prepare on another connection can
  // bind temp_table_name before the real result exists. The real result is
  // later Combine()d into the placeholder by ExecuteSQLandCreateTempTable.
  void RegisterPlaceholderTemp(const std::string &temp_table_name,
                               const duckdb::vector<duckdb::LogicalType> &types,
                               const std::vector<std::string> &col_names,
                               duckdb::idx_t est_card);

  // Result of TryCompileQueryJit / SpeculativeQueryJitCompile: entry fn +
  // per-step resolved source views + hash-table/aggregate plan metadata +
  // result table layout. The fn is valid until the compiling
  // IrToLlvmCompiler's next ResetModules; src views point into FlatTables /
  // QjitTables that outlive the execution.
  struct QjitCompiled {
    void *fn = nullptr; // QjitQueryFn
    std::vector<qjit::QjitResolvedSource> srcs;
    std::vector<uint32_t> ht_tuple_sizes;
    std::vector<uint32_t> ht_key0_offsets; // = prefix_bytes (key0 row offset)
    std::vector<qjit::QjitAggCellDesc> agg_descs;
    std::vector<int> agg_output_cells;
    std::vector<qjit::QjitTable::ColumnDesc> out_descs;
    std::vector<qjit::QjitSortCol> order_by;
    int64_t limit = -1;
    std::vector<uint8_t> params_buf; // template cache mode 2: runtime constants
    // §7.3b plan-replay cache: source info for ResolveQjitSources
    std::string replay_cache_key;
    std::string replay_fn_name;
    std::vector<qjit::QjitColumnRef> replay_source_cols;
    std::vector<std::string> replay_source_tables;
    std::vector<bool> replay_source_is_temp;
    std::vector<int> replay_source_block_skip_cols;
    std::vector<size_t> replay_step_col_counts;
  };

  // Speculative query-jit compile payload: built on the bg pool, consumed on
  // the main thread at HIT time. `compiled->srcs` stays UNRESOLVED until the
  // hit — the consumed temp doesn't exist when the bg task runs. `ir` owns
  // the AQPExprs the plan's raw filter pointers reference (keepalive).
  struct QjitSpecCompiled {
    qjit::QjitQueryPlan plan;
    std::unique_ptr<ir_sql_converter::AQPStmt> ir;
    std::unique_ptr<QjitCompiled> compiled;
    duckdb::vector<duckdb::LogicalType> out_types;
    duckdb::vector<duckdb::string> out_names;
    duckdb::idx_t est_card = 0;
  };

  // Bg-thread mirror of PrepareWithQueryJitAnalysis + TryCompileQueryJit on
  // a speculative connection with bg-local planning state: parse → optimize
  // → IR → PrepareFromPlan → AnnotateBuildSides → AnalyzeQueryJit →
  // BuildExecutionSteps → output cross-check → CompileQuerySteps on
  // spec_comp. nullptr on any reject (the inline path would reject the same
  // sub-query identically, so learning the miss is correct). Source
  // resolution is deferred to the HIT (main thread, temps exist by then).
  std::unique_ptr<QjitSpecCompiled>
  SpeculativeQueryJitCompile(const std::string &sql, const std::string &label,
                             duckdb::Connection &spec_conn,
                             aqp_jit::IrToLlvmCompiler *spec_comp);

  // Hand a bg-compiled payload to the next ExecuteSQLandCreateTempTable call
  // (main thread, HIT only).
  void SetQjitSpecHit(std::unique_ptr<QjitSpecCompiled> hit) {
    qjit_spec_hit_ = std::move(hit);
  }

  // Pass pre-built IR from SplitIR to skip the redundant ParseSQL+Optimize+
  // ConvertPlanToIR round-trip in PrepareWithQueryJitAnalysis. IR is
  // borrowed (non-owning pointer); caller must keep it alive through
  // ExecuteSQLandCreateTempTable.
  // use_engine_plan=true (node-based): the adapter's plan member holds the
  // sub_plan from SplitIR (consumed by PrepareFromPlan in the fast path).
  // use_engine_plan=false (SDS v18): the plan member is STALE (it still
  // holds the initial FilterOptimize'd whole-query plan) — the fast path
  // must build the sub-plan from the IR via TryBuildBinaryPlanFromIR.
  void SetQjitPendingIR(const ir_sql_converter::AQPStmt *ir,
                        bool use_engine_plan = true) {
    qjit_pending_ir_ = ir;
    qjit_pending_use_plan_ = use_engine_plan;
  }

  // Spec-jit blocking-wait time charged by the splitter; added to the next
  // jit_compile CSV column write so the timing CSV stays equal to wall time
  // (the wait happens between the splitter's gen-sub-SQL toc and this
  // adapter's timer, an otherwise untimed gap).
  void AddSpecWaitTime(long us) { spec_wait_extra_us_ += us; }

  // Register the qjit_* runtime symbols into a compiler. NOT idempotent
  // (duplicate absoluteSymbols defines error) — call once per compiler
  // lifetime; the defines survive ResetModules.
  void RegisterQjitRuntimeSymbols(aqp_jit::IrToLlvmCompiler *comp);

  // Immutable snapshot of (table index -> temp collection) taken on the main
  // thread, so a bg compile thread can build join bloom filters without
  // racing on intermediate_table_map / temp_collections_. The collections
  // themselves are never mutated after registration (only placeholders are
  // Combine()d into, and those are excluded), and they outlive any compile:
  // speculations are drained before temp tables are cleared.
  using TempCollectionSnapshot =
      std::unordered_map<duckdb::idx_t, duckdb::ColumnDataCollection *>;
  TempCollectionSnapshot SnapshotTempCollections() const;

  // Core JIT registration logic, parameterized for speculative compilation.
  // bf_temp_snapshot: nullptr on the main thread (use the live maps);
  // non-null for bg thread calls (use the snapshot, thread-safe).
  void RegisterJITImpl(
      duckdb::PhysicalOperator &op, const ir_sql_converter::AQPStmt &ir,
      duckdb::ClientContext *jit_ctx, aqp_jit::IrToLlvmCompiler *jit_comp,
      std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed_filters,
      std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed_joins,
      const TempCollectionSnapshot *bf_temp_snapshot = nullptr,
      bool is_build_side = false);

  bool GetJitDebug() const { return jit_debug_; }
  uint32_t GetJitFlags() const { return jit_flags_; }
  bool GetJitPrefetch() const { return jit_prefetch_; }
  int GetJitPrefetchDistance() const { return jit_prefetch_distance_; }
  int GetJitPrefetchEntryDistance() const { return jit_prefetch_entry_distance_; }
  int GetJitPrefetchRowDistance() const { return jit_prefetch_row_distance_; }
  bool GetJitBatchProbe() const { return jit_batch_probe_; }
  int GetJitSkipHashCmp() const { return jit_skip_hash_cmp_; }
  int GetJitCache() const { return jit_cache_; }
  int GetCompileMode() const { return compile_mode_; }

  bool ResolveQjitSources(const qjit::QjitQueryPlan &plan,
                          QjitCompiled &compiled, std::string &reason);
  qjit::QjitExecutor *GetQjitExecutor() { return qjit_executor_.get(); }

  static std::unordered_map<std::string, CachedQueryPlan> &QueryPlanCache();
  void BeginPlanRecording() {
    plan_recording_active_ = true;
    plan_recording_.clear();
  }
  void EndPlanRecording() { plan_recording_active_ = false; }
  bool IsPlanRecording() const { return plan_recording_active_; }
  std::vector<CachedSubquery> &GetPlanRecording() { return plan_recording_; }

  std::vector<std::string> GetTempCollectionColumnNames(
      const std::string &name) const {
    auto it = temp_collections_.find(name);
    if (it != temp_collections_.end())
      return it->second.column_names;
    return {};
  }
  duckdb::vector<duckdb::LogicalType> GetTempCollectionTypes(
      const std::string &name) const {
    auto it = temp_collections_.find(name);
    if (it != temp_collections_.end() && it->second.collection)
      return it->second.collection->Types();
    return {};
  }

  int64_t ReplayQjitSubquery(const CachedSubquery &sub);
  QueryResult ReplayQjitFinal(const CachedSubquery &sub);
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
  void
  RegisterExternalTempTableWithIndex(
      const std::string &temp_name,
      const duckdb::vector<duckdb::LogicalType> &types,
      const std::vector<std::string> &col_names,
      duckdb::idx_t chunk_index);

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
  void SetPlanner(std::unique_ptr<duckdb::Planner> p) { planner = std::move(p); }

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
  KernelPath kernel_path_ = KernelPath::NONE;

  // Per-optimization toggles (from ParamConfig)
  bool jit_payload_prune_ = true;
  bool jit_prefetch_ = true;
  int  jit_prefetch_distance_ = 8;
  // Phase 6: ROF probe look-ahead distances (consumer-side prefetch in
  // CompileFilterProbeProjectFusion). Plan §10.4 defaults.
  int  jit_prefetch_entry_distance_ = 24;
  int  jit_prefetch_row_distance_ = 12;
  bool jit_batch_probe_ = true;
  int  jit_skip_hash_cmp_ = 2; // 0=off, 2=all
  bool jit_single_col_int_join_ = true;
  bool range_guard_ = true;
  bool block_skip_ = true;
  bool membership_preprobe_ = true;
  bool jit_debug_ = false;
  bool benchmark_mode_ = false;
  int jit_cache_ = 0;
  std::string jit_cache_dir_;
  int compile_mode_ = 0;
  CompensateMissAction compensate_miss_action_ = CompensateMissAction::NONE;
  // TPDE compiler for spec-jit miss recompiles. Only created when the main
  // compiler is NOT already TPDE; otherwise jit_compiler_ is reused.
  // Lifecycle mirrors jit_compiler_: ResetModules per use, compiled fn
  // pointers valid until its next reset. Cache-key safe: the backend tag
  // (F0/F1/F2) is serialized per compiler instance (§5.1).
  std::unique_ptr<aqp_jit::IrToLlvmCompiler> fast_jit_compiler_;
  bool fast_qjit_syms_registered_ = false;
  // Returns the TPDE compiler for miss recompiles (jit_compiler_ when it
  // is already TPDE).
  aqp_jit::IrToLlvmCompiler *EnsureFastJitCompiler();
  // Guards the SHAPE of temp_collections_/qjit_temps_/qjit_temp_meta_ and
  // placeholder field replacement at store time against concurrent reads
  // from bg spec-compile threads (replacement scan + bind callbacks +
  // RebuildTempTableIndices on the spec connection). Main thread is the
  // only writer (exclusive); bind-path readers take shared locks. Needed
  // because compensate-mode early launches (and retired-but-running zombie
  // specs) overlap Execute(i) and the subsequent store.
  mutable std::shared_timed_mutex temp_state_mutex_;
  // Owned IR built in the no-split JIT path; must outlive jit_pending_ir_.
  std::unique_ptr<ir_sql_converter::AQPStmt> owned_jit_ir_;
  // Pre-built logical plan for PrepareFromPlan (avoids redundant parse+optimize).
  duckdb::unique_ptr<duckdb::LogicalOperator> jit_pending_plan_;

  // Keeps the LLJIT instance alive until after query execution so that
  // compiled function pointers stored in AQPJITContext remain valid.
  std::unique_ptr<aqp_jit::IrToLlvmCompiler> jit_compiler_;
  // Compiler cache: avoids destroying + recreating LLJIT instances when
  // per-subquery tuning switches backends between sub-queries (~6 ms per
  // recreation).  Keyed on (compile_mode << 16 | simd_isa).
  struct JitCompilerCacheEntry {
    std::unique_ptr<aqp_jit::IrToLlvmCompiler> compiler;
    bool qjit_syms_registered = false;
  };
  std::unordered_map<uint32_t, JitCompilerCacheEntry> jit_compiler_cache_;
  std::future<void> compiler_precreate_future_;
  static uint32_t CompilerCacheKey(int fast, int simd) {
    return (uint32_t(fast) << 16) | uint32_t(simd);
  }

  // Speculative-compile kickoff hook (set by IRQuerySplitter when --spec-jit).
  PostPrepareHook post_prepare_hook_;

  // Query-jit state (--jit-level=query)
  bool query_jit_ = false;
  int query_jit_threads_ = 0;
  int query_jit_morsel_ = 20000;
  const middleware::storage::StoragePlan *qjit_storage_plan_ = nullptr;
  // Lazy: worker threads only spawn on the first compiled sub-query.
  std::unique_ptr<qjit::QjitExecutor> qjit_executor_;
  // qjit_* runtime symbols are registered once per jit_compiler_ lifetime
  // (absoluteSymbols defines survive ResetModules).
  bool qjit_syms_registered_ = false;
  // In-memory qjit results keyed by temp-table name (kept from Phase 2 on;
  // Phase 4 serves ChunkNode scans from here). Lifecycle mirrors
  // temp_collections_: cleared in CleanUp/ResetQueryState/DropTempTable.
  std::unordered_map<std::string, std::unique_ptr<qjit::QjitTable>>
      qjit_temps_;

  std::vector<CachedSubquery> plan_recording_;
  bool plan_recording_active_ = false;

  // Output cross-check against the prepared statement (the result
  // authority); fills compiled.out_descs + agg metadata. Shared by the
  // inline (TryCompileQueryJit) and speculative compile paths.
  bool BuildQjitOutputDescs(const qjit::QjitQueryPlan &plan,
                            duckdb::PreparedStatement &prepared,
                            QjitCompiled &compiled, std::string &reason);

  // Bg-compiled spec payload awaiting the next ExecuteSQLandCreateTempTable
  // call (set via SetQjitSpecHit on a spec HIT).
  std::unique_ptr<QjitSpecCompiled> qjit_spec_hit_;

  // Pre-built IR from SplitIR (non-owning). When set, the fast path in
  // ExecuteSQLandCreateTempTable uses TakePlan+PrepareFromPlan+AnnotateBuildSides
  // on this IR instead of the full ParseSQL+Optimize+ConvertPlanToIR round-trip.
  const ir_sql_converter::AQPStmt *qjit_pending_ir_ = nullptr;
  // Whether the plan member is the matching sub-plan (see SetQjitPendingIR).
  bool qjit_pending_use_plan_ = true;

  // Pending spec-jit wait time (µs) to fold into the next jit_compile CSV
  // column write. See AddSpecWaitTime.
  long spec_wait_extra_us_ = 0;
  long ConsumeSpecWaitUs() {
    long v = spec_wait_extra_us_;
    spec_wait_extra_us_ = 0;
    return v;
  }

  // Compile attempt for an analysis-accepted sub-query: BuildExecutionSteps
  // (strict whitelist; joins + ungrouped aggregates included), storage-plan
  // source resolution per step, and the output cross-check against
  // `prepared` (the result authority). Every failure traces
  // [AQP-QJIT] fallback:<reason> and returns nullptr (interpreter runs).
  // use_fast: compile on the fast tier (compensate=true miss path).
  std::unique_ptr<QjitCompiled>
  TryCompileQueryJit(const ir_sql_converter::AQPStmt &ir,
                     const qjit::QjitAnalysisResult &analysis,
                     duckdb::PreparedStatement &prepared,
                     const std::string &label, bool use_fast = false);

  // Temp-table step source: qjit_temps_ entry, or on-demand conversion of
  // the fallback-produced CDC (cached back into qjit_temps_). nullptr +
  // reason when the temp is missing, a speculative placeholder, or has an
  // unsupported column type.
  const qjit::QjitTable *GetOrLoadQjitTemp(const std::string &name,
                                           std::string &reason);

  // CDC -> QjitTable (INTEGER/BIGINT/VARCHAR; deep string copies into the
  // table arena). Single-threaded append (runs once per fallback temp).
  std::unique_ptr<qjit::QjitTable>
  CollectionToQjitTable(const StoredTempResult &stored, std::string &reason);

  // Convert a finalized QjitTable into a ColumnDataCollection with the
  // prepared statement's types (INT32/INT64/VARCHAR only — enforced at
  // compile gating). Chunked appends, deep string copies (DuckDB owns the
  // result).
  duckdb::unique_ptr<duckdb::ColumnDataCollection>
  QjitTableToCollection(const qjit::QjitTable &table,
                        const duckdb::vector<duckdb::LogicalType> &types);

  // Annotate IR JoinNodes with the build side chosen by DuckDB's physical
  // planner (SimplestJoin::build_child). Walks the physical plan, matches
  // each PhysicalHashJoin to an IR JoinNode via join-condition attrs.
  void AnnotateBuildSides(duckdb::PhysicalOperator &op,
                          ir_sql_converter::AQPStmt &ir,
                          bool include_chunk_scans = false);

  // v18 (SDS hybrid): closed IR→DuckDB-logical-plan constructor for the
  // pending-IR fast path when no pre-built plan exists (TOP_DOWN — the
  // node-based splitter sets `plan`, SDS only has the IR). Accepted shape:
  // Projection over ONE inner equi-join of two plain Scan/Chunk leaves with
  // conjunctive single-table filters. Base leaves bind through a fresh
  // Binder (catalog); temp leaves bind their chunk name through the temp
  // replacement scan. LogicalGet table indices are overwritten with the IR
  // indices so AnnotateBuildSides matches; FilterOptimize populates
  // table_filters (zone-map parity — the plan may also execute interpreted
  // on a qjit run error). Any other shape → nullptr + reject_reason; the
  // caller falls back to the SQL round trip (v17 behavior).
  duckdb::unique_ptr<duckdb::LogicalOperator>
  TryBuildBinaryPlanFromIR(const ir_sql_converter::AQPStmt &ir,
                           std::string &reject_reason);

  // Query-jit prepare result: the prepared statement (always usable as the
  // interpreter fallback), plus — when analysis succeeded — the fresh-binder
  // IR (build sides annotated) and the analysis verdict for the compile
  // attempt. The IR owns every AQPExpr the compiled step references, so it
  // must outlive TryCompileQueryJit (codegen only; not execution).
  struct QueryJitPrep {
    std::unique_ptr<duckdb::PreparedStatement> prepared;
    std::unique_ptr<ir_sql_converter::AQPStmt> ir;
    qjit::QjitAnalysisResult analysis;
  };

  // Query-jit hook: parse+optimize+IR+prepare the sub-query, annotate build
  // sides, run AnalyzeQueryJit ([AQP-QJIT] traces). On any failure `ir` is
  // null / `analysis.accepted` false and `prepared` falls back to a plain
  // Prepare so the interpreter still runs the sub-query.
  QueryJitPrep PrepareWithQueryJitAnalysis(const std::string &sql,
                                           const std::string &label);

  // Lazily create and configure the JIT compiler with all flags.
  void EnsureJITCompiler();

  // Walk physical plan tree; compile IR filters and register in aqp_jit_context.
  void RegisterJIT(duckdb::PhysicalOperator &op,
                          const ir_sql_converter::AQPStmt &ir);

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

  // Inject join_stats into PhysicalHashJoin nodes whose build side scans a
  // temp table, enabling DuckDB's perfect hash join optimization.
  void InjectTempTableJoinStats(duckdb::PhysicalOperator &op);

  // Replacement scan callback (static)
  static duckdb::unique_ptr<duckdb::TableRef> TempCollectionReplacementScan(
      duckdb::ClientContext &context, duckdb::ReplacementScanInput &input,
      duckdb::optional_ptr<duckdb::ReplacementScanData> data);
  // Replacement scan: in-memory temp table storage
  std::unordered_map<std::string, StoredTempResult> temp_collections_;

  // Temps served by scan_qjit_temp (row data in qjit_temps_). Disjoint from
  // temp_collections_ by construction: a temp is stored in exactly one of
  // the two (qjit-compiled result without a speculative placeholder ⇒ here;
  // everything else ⇒ CDC). Always empty without HAVE_LLVM.
  std::unordered_map<std::string, QjitTempMeta> qjit_temp_meta_;

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

#ifdef HAVE_LLVM
  // Table function callbacks for qjit temp tables (scan_qjit_temp, static)
  static duckdb::unique_ptr<duckdb::FunctionData>
  QjitTempBind(duckdb::ClientContext &context,
               duckdb::TableFunctionBindInput &input,
               duckdb::vector<duckdb::LogicalType> &return_types,
               duckdb::vector<duckdb::string> &names);

  static duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
  QjitTempInitGlobal(duckdb::ClientContext &context,
                     duckdb::TableFunctionInitInput &input);

  static void QjitTempScanFunc(duckdb::ClientContext &context,
                               duckdb::TableFunctionInput &data,
                               duckdb::DataChunk &output);

  static duckdb::unique_ptr<duckdb::NodeStatistics>
  QjitTempCardinality(duckdb::ClientContext &context,
                      const duckdb::FunctionData *bind_data);
#endif
#endif
};
} // namespace middleware
