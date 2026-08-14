/*
 * IR-level query splitter main pipeline
 */

#pragma once

#include "adapters/db_adapter.h"
#include "simplest_ir.h"
#include "split/fk_based_splitter.h"
#include "split/foreign_key_extractor.h"
#include "split/split_algorithm.h"
#include "split/topdown_splitter.h"
#ifdef HAVE_DUCKDB
#include "split/node_based_splitter.h"
#endif
#ifdef HAVE_POSTGRES
#include "adapters/postgres_adapter.h"
#endif
#ifdef HAVE_LLVM
namespace middleware { struct CachedQueryPlan; }
#endif
#include "storage/csr_index.h"
#include "storage/storage_plan.h"
#include "kernel/sub_query_plan.h"
#include "util/param_config.h"
#include "util/thread_pool.h"
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
#include "duckdb/main/connection.hpp"
#include "jit/ir_to_llvm.h"
#endif

namespace middleware {

struct CrossQueryPrepResult {
  // Engine-agnostic fields
  std::string sql;
  std::string query_name;

  std::unique_ptr<SubqueryExtraction> first_extraction;
  std::string first_sub_sql;

  // SDS (TOP_DOWN) cross-query prep fields (engine-agnostic)
  std::unique_ptr<ir_sql_converter::AQPStmt> whole_ir;
  std::map<unsigned int, double> td_table_card;
  std::map<unsigned int, std::string> td_table_index_to_name;
  unsigned int td_max_table_index = 0;
  std::map<unsigned int, std::pair<std::string, double>> td_mark_in;
  std::set<unsigned int> td_mark_locked;
  std::unordered_map<std::string, double> td_col_distinct_hints;
  JoinGraph td_join_graph;
  std::vector<std::pair<unsigned int, unsigned int>> td_current_join_pairs;
  std::vector<bool> td_is_relationship;
  int td_split_iteration = 0;
  std::set<unsigned int> td_executed_tables;

  bool success = false;
  std::string error;
  double prep_time_us = 0.0;

#ifdef HAVE_DUCKDB
  // DuckDB-specific fields
  std::unique_ptr<duckdb::Connection> bg_conn;
  std::unique_ptr<duckdb::Planner> bg_planner;

  duckdb::unique_ptr<duckdb::LogicalOperator> remaining_plan;
  std::unique_ptr<duckdb::QuerySplit> qs;
  std::unique_ptr<duckdb::SubqueryPreparer> sp;
  std::unique_ptr<duckdb::ReorderGet> reorder_get;
  duckdb::subquery_queue subqueries;
  duckdb::table_expr_info table_expr_queue;
  std::vector<duckdb::TableExpr> proj_expr;
  duckdb::unique_ptr<duckdb::LogicalOperator> last_sibling_node;
  bool merge_sibling_expr = false;
  duckdb::vector<duckdb::LogicalType> sub_plan_types;
#endif

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  std::unique_ptr<duckdb::PreparedStatement> prepared;
  bool has_prepare = false;
  std::unique_ptr<DuckDBAdapter::QjitSpecCompiled> qjit_spec;
  bool has_qjit = false;
#endif
};

// Mapping entry for column index updates
struct ColumnMapping {
  unsigned int old_table_idx;
  unsigned int old_col_idx;
  std::string column_name;

  ColumnMapping(unsigned int table_idx, unsigned int col_idx, std::string name)
      : old_table_idx(table_idx), old_col_idx(col_idx),
        column_name(std::move(name)) {}
};

// Temp table information after executing a subquery
struct TempTableInfo {
  std::string table_name;
  unsigned int table_index;
  uint64_t cardinality;
  std::vector<std::string> column_names;
  std::vector<ir_sql_converter::SimplestVarType> column_types;

  // Mapping from old (table_idx, col_idx) to position in this temp table
  // column_mappings[i] contains the original (table_idx, col_idx) for column i
  std::vector<ColumnMapping> column_mappings;

  TempTableInfo(std::string name, unsigned int idx, uint64_t card)
      : table_name(std::move(name)), table_index(idx), cardinality(card) {}

  // Find the new column index for a given old (table_idx, col_idx)
  // Returns -1 if not found
  int FindNewColumnIndex(unsigned int old_table_idx,
                         unsigned int old_col_idx) const {
    for (size_t i = 0; i < column_mappings.size(); i++) {
      if (column_mappings[i].old_table_idx == old_table_idx &&
          column_mappings[i].old_col_idx == old_col_idx) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
};

class IRQuerySplitter {
public:
  IRQuerySplitter(EngineAdapter *adapter, const ParamConfig &config,
                  storage::StoragePlan *storage_plan = nullptr);
  ~IRQuerySplitter();

  // Set query name (without .sql) and load per-subquery tune config if path is set.
  void SetQueryName(const std::string &name);

  // Main entry: Execute query with optional splitting
  QueryResult ExecuteWithSplit(const std::string &sql);

  // Statistics
  int GetIterationCount() const { return iteration_count_; }

  // §7.2 Cross-query latency hiding
  void SetCrossQueryPrep(std::unique_ptr<CrossQueryPrepResult> prep) {
    active_cross_query_prep_ = std::move(prep);
  }
#ifdef HAVE_DUCKDB
#if defined(HAVE_LLVM)
  static std::unique_ptr<CrossQueryPrepResult>
  PrepareNextQuery(const std::string &sql_path, duckdb::DuckDB &db_ref,
                   DuckDBAdapter *duck, const ParamConfig &config,
                   std::unique_ptr<aqp_jit::IrToLlvmCompiler> &bg_compiler,
                   uint32_t effective_jit_flags, int effective_compile_mode);
  // Resolve effective (jit_flags, compile_mode) for a query's sub_idx,
  // applying tune override if configured. Main thread only.
  static std::pair<uint32_t, int> ResolveTuneFlags(
      const ParamConfig &config, const std::string &query_name, int sub_idx);
  // Resolve per-query split strategy from tune JSON ("split" field).
  // Returns fallback if the JSON has no "split" field for this query.
  static SplitStrategy ResolveTuneSplit(const std::string &tune_config_path,
                                        const std::string &query_name,
                                        SplitStrategy fallback);
#else
  static std::unique_ptr<CrossQueryPrepResult>
  PrepareNextQuery(const std::string &sql_path, duckdb::DuckDB &db_ref,
                   DuckDBAdapter *duck, const ParamConfig &config);
#endif
#if defined(HAVE_LLVM)
  static std::unique_ptr<CrossQueryPrepResult> PrepareNextQueryTopDown(
      const std::string &sql_path, duckdb::DuckDB &db_ref, DuckDBAdapter *duck,
      const ParamConfig &config,
      std::unique_ptr<aqp_jit::IrToLlvmCompiler> &bg_compiler,
      uint32_t effective_jit_flags, int effective_compile_mode);
#else
  static std::unique_ptr<CrossQueryPrepResult>
  PrepareNextQueryTopDown(const std::string &sql_path, duckdb::DuckDB &db_ref,
                          DuckDBAdapter *duck, const ParamConfig &config);
#endif
#endif
#ifdef HAVE_POSTGRES
  static std::unique_ptr<CrossQueryPrepResult>
  PrepareNextQueryTopDownPG(const std::string &sql_path,
                            EngineAdapter *adapter,
                            const ParamConfig &config);
#endif

private:
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  QueryResult ReplayQueryPlan(const CachedQueryPlan &cached);
#endif
#if defined(HAVE_POSTGRES) && defined(HAVE_LLVM)
  QueryResult ReplayQueryPlanPG(const PgCachedQueryPlan &cached);
#endif

  // === IR-based Iterative Split-Execute Loop (all strategies) ===
  QueryResult
  ExecuteSplitLoop(std::unique_ptr<ir_sql_converter::AQPStmt> whole_ir);

  // Single iteration: extract → execute → update remaining IR
  bool ExecuteOneIteration(
      std::unique_ptr<ir_sql_converter::AQPStmt> &remaining_ir);

  // Execute a sub-IR and create temp table
  TempTableInfo
  ExecuteSubIR(std::unique_ptr<ir_sql_converter::AQPStmt> sub_ir,
               const std::set<unsigned int> &executed_table_indices);

  // === Shared Index Update Functions ===
  // Update table/column indices in remaining IR after creating temp table
  // Called after strategy-specific UpdateRemainingIR
  void
  UpdateRemainingIRIndices(ir_sql_converter::AQPStmt *remaining_ir,
                           const TempTableInfo &temp_table,
                           const std::set<unsigned int> &old_table_indices);

  // Helper: Update a single SimplestAttr if it references an executed table
  std::unique_ptr<ir_sql_converter::SimplestAttr>
  UpdateAttrIndices(const ir_sql_converter::SimplestAttr *attr,
                    const TempTableInfo &temp_table,
                    const std::set<unsigned int> &old_table_indices);

  // Helper: Recursively update all attributes in an IR node
  void UpdateNodeIndices(ir_sql_converter::AQPStmt *node,
                         const TempTableInfo &temp_table,
                         const std::set<unsigned int> &old_table_indices);

  // Helper: Recursively update attributes in an expression tree
  void UpdateExprIndices(ir_sql_converter::AQPExpr *expr,
                         const TempTableInfo &temp_table,
                         const std::set<unsigned int> &old_table_indices);

  // === Helper Functions ===
  std::string GenerateTempTableName();

  // Check if remaining IR is trivial (just a temp table reference)
  std::string GetTrivialTempTable(ir_sql_converter::AQPStmt *ir) const;

  // Check if SQL references any temp table known to have 0 rows
  bool SubPlanReferencesEmptyTemp(const std::string &sql) const;

  // Walk IR tree and return true iff every join is Inner (or Semi/Anti/Mark,
  // which also produce empty output from empty input).
  static bool AllJoinsPropagatEmpty(const ir_sql_converter::AQPStmt *ir);

  // Cross-sub-plan optimizations (range pred injection + bloom filter).
  // Outlined from ExecuteOneIteration to keep the hot path compact for
  // better instruction cache utilization (expert knowledge #9, #18).
  // Range preds change the SQL text; bloom filters are side-band (pending in
  // the adapter), so the spec-check path can skip building them on a HIT.
  void ApplyCrossSubPlanOptimizations(std::string &sub_sql,
                                      bool inject_range_preds = true,
                                      bool build_bloom_filters = true);

  EngineAdapter *adapter_;
  storage::StoragePlan *storage_plan_ = nullptr;
#ifdef HAVE_DUCKDB
  // Owned helper DuckDB adapter; non-null only when engine != DUCKDB and
  // strategy == NODE_BASED. Null when engine == DUCKDB (adapter_ is the DuckDB
  // adapter in that case).
  std::unique_ptr<DuckDBAdapter> owned_duckdb_adapter_;
  // Non-owning pointer to the DuckDB adapter used for planning.
  // Valid whenever strategy == NODE_BASED.
  DuckDBAdapter *duckdb_adapter_ = nullptr;
#endif
  ParamConfig config_;
  std::unique_ptr<AQPSplitter> splitter_;

  // Iteration tracking
  int iteration_count_ = 0;
  std::vector<TempTableInfo> temp_tables_;

  // Kernel temp tables (CSR executor results) — owned here
  std::unordered_map<std::string, std::unique_ptr<storage::FlatTable>>
      kernel_temps_;
  // Non-owning view for AnalyzeSubIR
  std::unordered_map<std::string, const storage::FlatTable *>
      kernel_temp_ptrs_;
  // Runtime CSR indexes built on temp table results (built lazily on demand)
  std::unordered_map<std::string, storage::CSRIndex> runtime_csrs_;

  // Background thread pool (1 worker) for async CSR builds; reusable for other tasks
  std::unique_ptr<ThreadPool> bg_pool_;
  std::unordered_map<std::string, std::future<storage::CSRIndex>> async_csrs_;

  // Temp tables known to have 0 rows (INNER JOIN → 0 results guaranteed)
  std::set<std::string> empty_temp_tables_;

  // Early termination: skip remaining splits + final JIT when a temp
  // returns 0 rows and all joins are inner (empty propagates to output).
  bool all_inner_joins_ = false;
  bool early_terminate_ = false;
  size_t original_output_col_count_ = 0;

  // Cached integer-column min/max per temp table (immutable once stored);
  // avoids re-scanning collections on repeated range-pred injection.
  std::unordered_map<std::string,
                     std::unordered_map<size_t, std::pair<int64_t, int64_t>>>
      temp_min_max_cache_;

  // Kernel decision logging (for threshold tuning, --tuning flag)
  int kernel_log_repeat_idx_ = 0;
  int current_repeat_ = 0;
  std::string tuning_log_file_;

  // Sub-plan combiner: collected (temp_name, sql) pairs
  std::vector<std::pair<std::string, std::string>> sub_plan_sqls_;

  // Per-subquery tune config: maps sub_idx → tunable flag set.
  // Loaded from tune JSON for the current query_name_.
  // Fields with -1 / unset use the global config (no override).
  struct TuneEntry {
    std::string config_label;
    uint32_t jit_flags = 0;
    bool query_jit = false;
    int compile_mode = 0;      // 0=llvm, 1=fastisel, 2=tpde
    bool jit_simd = false;
    int payload_prune = -1;    // -1=use global, 0=off, 1=on
    int prefetch = -1;         // -1=use global, 0=off, 1=on
    int batch_probe = -1;      // -1=use global, 0=off, 1=on
    int skip_hash_cmp = -1;    // -1=use global, 0=off, 2=all
  };
  std::string query_name_;
  std::unordered_map<int, TuneEntry> tune_entries_;
  static TuneEntry ParseTuneLabel(const std::string &label);
  void LoadTuneEntry(int idx, const nlohmann::json &val);
  void ApplyTuneOverride(int sub_idx);

  // Helper: Compute column alias using SQL generator's convention
  std::string ComputeColumnAlias(unsigned int table_idx,
                                 const std::string &col_name) const;

  // Helper: Build a combined CTE SQL from collected sub-plans + final SQL
  std::string BuildCombinedSQL(
      const std::vector<std::pair<std::string, std::string>> &sub_plans,
      const std::string &final_sql) const;

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  struct SpeculativeCompilation {
    std::future<bool> future;
    std::string speculative_sql;
    // Temp table + estimated cardinality the speculative Prepare planned
    // against. If the actual cardinality diverges, the frozen physical plan
    // (join sides, HT sizing, parallelism) may be slow — reject the hit.
    std::string assumed_temp_name;
    duckdb::idx_t assumed_card = 0;
    std::unique_ptr<duckdb::Connection> spec_conn;
    std::unique_ptr<duckdb::PreparedStatement> spec_prepared;
    // Speculative IR owned here so bg thread can read it safely.
    std::unique_ptr<ir_sql_converter::AQPStmt> spec_ir;
    // Phase B relaunch instead BORROWS the next iteration's precomputed
    // extraction IR. The owning extraction is destroyed at the end of the
    // iteration it drives, so ExecuteOneIteration must WaitSpecsBorrowingIR
    // on it first (kernel-path iterations never consult the spec, and a
    // MISS retires it to zombie_specs_ without waiting).
    const ir_sql_converter::AQPStmt *borrowed_ir = nullptr;
    // Bloom filters built on the main thread at launch (same set the inline
    // path would build); registered into the spec plan by the bg task.
    std::vector<DuckDBAdapter::BloomFilterInfo> bloom_filters;
    // Iteration this speculation targets (for cross-repeat miss learning).
    int target_iter = 0;
    // Snapshot of temp-table collections taken on the main thread at launch
    // so the bg RegisterJITImpl can build join bloom filters race-free.
    DuckDBAdapter::TempCollectionSnapshot temp_snapshot;
    // Query-jit spec payload (compiled plan + metadata, sources unresolved).
    // Null on the pipeline-jit spec path and when the bg compile rejected.
    std::unique_ptr<DuckDBAdapter::QjitSpecCompiled> qjit;
#ifdef HAVE_POSTGRES
    std::unique_ptr<PostgreSQLAdapter::QjitSpecCompiled> pg_qjit;
#endif
  };
  std::unique_ptr<ThreadPool> jit_compile_pool_;
  std::unique_ptr<SpeculativeCompilation> pending_spec_;
  // Stale speculations whose bg compile may still be running. Kept alive here
  // (instead of blocking the main thread on future.wait()) so the bg task's
  // raw pointer into the SpeculativeCompilation stays valid. Reaped
  // non-blockingly by RetirePendingSpec; drained (blocking) by DrainSpecs.
  std::vector<std::unique_ptr<SpeculativeCompilation>> zombie_specs_;
  // Move pending_spec_ into zombie_specs_ without blocking, then reap any
  // zombies whose futures are ready.
  void RetirePendingSpec();
  // Blocking: wait for any pending/zombie bg compile that borrowed `ir`
  // (Phase B relaunch) before the IR's owning extraction is destroyed.
  void WaitSpecsBorrowingIR(const ir_sql_converter::AQPStmt *ir);
  // Blocking: wait for pending + zombies (end of query / destructor).
  // charge_wait: charge the blocking time to the adapter's next jit_compile
  // CSV column (end-of-loop drain — otherwise it falls in an untimed gap
  // before the final query's timer).
  void DrainSpecs(bool charge_wait = false);
  // Long-lived compilers for speculative JIT — reused across iterations to
  // avoid LLVM LLJIT memory growth from creating/destroying instances.
  // Two instances, used alternately (ping-pong): on a HIT the next bg compile
  // is launched WHILE the hit's JIT code is executing, and its ResetModules
  // must not free that code — so it runs on the other compiler.
  std::unique_ptr<aqp_jit::IrToLlvmCompiler> spec_compilers_[2];
  int spec_compiler_idx_ = 0;
  int spec_hits_ = 0;
  int spec_misses_ = 0;
  int spec_card_misses_ = 0;
  int spec_not_ready_ = 0;
  int spec_bg_errors_ = 0;
  // Compensate-jit miss-policy applications this query (trace/summary only).
  int spec_compensate_fast_ = 0;
  int spec_compensate_interp_ = 0;
  // Compensate mode: iteration whose spec launch was skipped (learned miss)
  // or whose Phase-B validation mispredicted — the check site applies the
  // miss policy to it even though pending_spec_ is null. One-shot.
  int spec_learned_miss_iter_ = -1;
  // Total time this query the main thread blocked on bg-compile futures (µs).
  long spec_wait_us_ = 0;
  // Key into the cross-repeat miss-history map (the original query SQL).
  std::string spec_history_key_;

  // Phase A: peek at next subquery and launch bg Prepare+JIT. Invoked via
  // the adapter's post-Prepare hook — after Prepare(i) (when the result temp
  // table's identity is known and registered as a placeholder) and before
  // ExecuteRow(i), so the bg compile overlaps the whole execution.
  void LaunchSpeculativeCompile(const std::string &temp_table_name,
                                duckdb::idx_t chunk_index,
                                const duckdb::vector<duckdb::LogicalType> &types,
                                const std::vector<std::string> &col_names,
                                duckdb::idx_t est_card, bool post_execute);
#ifdef HAVE_POSTGRES
  void LaunchSpeculativeCompilePG(
      const std::string &temp_table_name,
      const std::vector<int32_t> &aqp_dtypes,
      const std::vector<std::string> &col_names,
      uint64_t est_card, bool post_execute);
#endif
  // Phase B: run real SplitIR(i+1) AFTER UpdateRemainingIR to produce
  // precomputed_extraction_ for the next iteration.
  void PrecomputeNextExtraction(
      std::unique_ptr<ir_sql_converter::AQPStmt> &remaining_ir);
  bool CheckSpeculativeResult(const std::string &actual_sql,
                              const std::string &temp_table_name);
  // Extraction pre-computed by PrecomputeNextExtraction (real SplitIR).
  // If non-null, the next iteration uses it instead of calling SplitIR.
  std::unique_ptr<SubqueryExtraction> precomputed_extraction_;
  // Main-thread time (µs) spent in PrecomputeNextExtraction; added to the
  // NEXT iteration's extract_next_sub-IR timer column so the breakdown CSV
  // stays complete (the work is that iteration's extraction, done early).
  double pending_extract_us_ = 0.0;
#endif

  std::unique_ptr<CrossQueryPrepResult> active_cross_query_prep_;

#ifdef HAVE_DUCKDB
  // Lazy CSR (7.3b): build FlatTable + CSR from DuckDB ColumnDataCollection
  // on demand, only when a kernel iteration actually needs the temp.
  void EnsureKernelTempReady(const std::string &temp_name);

  // Build FlatTable only (no CSR) for pipeline kernel path
  void EnsureKernelTempReadyNoCsr(const std::string &temp_name);

  // Collect all ChunkNode (temp table) names referenced in an IR tree.
  static void CollectChunkNames(const ir_sql_converter::AQPStmt *node,
                                std::set<std::string> &names);

  // Ensure all temps referenced by an IR tree are ready for kernel use.
  void EnsureReferencedTempsReady(const ir_sql_converter::AQPStmt *ir);

  // FlatTable only (no CSR) for pipeline kernel
  void EnsureReferencedTempsReadyNoCsr(const ir_sql_converter::AQPStmt *ir);
#endif
};

} // namespace middleware
