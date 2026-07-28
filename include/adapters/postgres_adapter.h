/*
 * PostgreSQL adapter for binding IR to the PostgreSQL engine
 * */

#pragma once

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include <libpq-fe.h>

#include "adapters/db_adapter.h"
#include "pg_query.h"

#ifdef HAVE_LLVM
#include "jit/ir_to_llvm.h"
#include "qjit/query_jit_executor.h"
#include "qjit/query_jit_steps.h"
#include "storage/storage_plan.h"
#endif

using json = nlohmann::json;

namespace middleware {

#ifdef HAVE_LLVM
struct PgCachedSubquery {
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
};

struct PgCachedQueryPlan {
  std::vector<PgCachedSubquery> subqueries;
  PgCachedSubquery final_query;
};
#endif

class PostgreSQLAdapter : public EngineAdapter {
public:
  explicit PostgreSQLAdapter(const std::string &connection_string);
  ~PostgreSQLAdapter() override;

  // Parse SQL and return logical plan
  void ParseSQL(const std::string &sql) override;

  json GetParseTree() { return parse_tree; }

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
                               uint64_t estimated_rows) override;

  // Get estimated cost and rows for a query using EXPLAIN
  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;

  // EXPLAIN (ANALYZE, VERBOSE, BUFFERS) plan text for a sub-SQL
  std::string ExplainAnalyze(const std::string &sql) override;

  // Batch EXPLAIN: send multiple EXPLAIN queries in one PQsendQuery round-trip
  std::vector<std::pair<double, double>>
  BatchGetEstimatedCosts(const std::vector<std::string> &sqls) override;

  std::string GetEngineName() const override { return "PostgreSQL"; }

  void CleanUp() override;

  void ResetQueryState() override;

  // Get connection handle
  PGconn *GetConnection() { return conn; }

  void CheckConnection();

#ifdef HAVE_LLVM
  void SetQueryJit(bool enable, int threads, int morsel) {
    query_jit_ = enable;
    if (enable)
      session_query_jit_ = true;
    query_jit_threads_ = threads;
    query_jit_morsel_ = morsel;
  }
  void SetQueryJitStoragePlan(
      const middleware::storage::StoragePlan *plan) {
    qjit_storage_plan_ = plan;
  }
  void SetCompileMode(int mode) { compile_mode_ = mode; }
  void SetSkipHashCmp(int mode) { skip_hash_cmp_ = mode; }
  void SetJitFlags(uint32_t flags) { jit_flags_ = flags; }
  void SetJITCache(int mode) { jit_cache_ = mode; }
  void SetJITCacheDir(const std::string &dir) { jit_cache_dir_ = dir; }
  void SetJITDebug(bool debug) { jit_debug_ = debug; }
  void SetJITPrefetch(bool enable, int distance) {
    jit_prefetch_ = enable;
    jit_prefetch_distance_ = distance;
  }

  int GetSkipHashCmp() const { return skip_hash_cmp_; }
  bool GetJitDebug() const { return jit_debug_; }
  bool GetJitPrefetch() const { return jit_prefetch_; }
  int GetJitPrefetchDistance() const { return jit_prefetch_distance_; }
  int GetCompileMode() const { return compile_mode_; }

  struct QjitCompiled {
    void *fn = nullptr;
    std::vector<qjit::QjitResolvedSource> srcs;
    std::vector<uint32_t> ht_tuple_sizes;
    std::vector<uint32_t> ht_key0_offsets;
    std::vector<qjit::QjitAggCellDesc> agg_descs;
    std::vector<int> agg_output_cells;
    std::vector<qjit::QjitTable::ColumnDesc> out_descs;
    std::vector<uint8_t> params_buf;
    std::string replay_cache_key;
    std::string replay_fn_name;
    std::vector<qjit::QjitColumnRef> replay_source_cols;
    std::vector<std::string> replay_source_tables;
    std::vector<bool> replay_source_is_temp;
    std::vector<int> replay_source_block_skip_cols;
    std::vector<size_t> replay_step_col_counts;
  };

  struct QjitSpecCompiled {
    qjit::QjitQueryPlan plan;
    std::unique_ptr<ir_sql_converter::AQPStmt> ir;
    std::unique_ptr<QjitCompiled> compiled;
  };

  using PostPrepareHook = std::function<void(
      const std::string &temp_table_name,
      const std::vector<int32_t> &aqp_dtypes,
      const std::vector<std::string> &col_names,
      uint64_t est_card,
      bool post_execute)>;
  void SetPostPrepareHook(PostPrepareHook hook) {
    post_prepare_hook_ = std::move(hook);
  }

  void SetQjitPendingIR(const ir_sql_converter::AQPStmt *ir) {
    qjit_pending_ir_ = ir;
  }

  void SetQjitSpecHit(std::unique_ptr<QjitSpecCompiled> hit) {
    qjit_spec_hit_ = std::move(hit);
  }

  void AddSpecWaitTime(long us) { spec_wait_extra_us_ += us; }

  long ConsumeSpecWaitUs() {
    long us = spec_wait_extra_us_;
    spec_wait_extra_us_ = 0;
    return us;
  }

  std::unique_ptr<QjitSpecCompiled>
  SpeculativeQueryJitCompile(
      const std::string &sql,
      const std::unordered_map<std::string, int64_t> &temp_card_snapshot,
      const std::string &label,
      unsigned int sub_plan_id,
      aqp_jit::IrToLlvmCompiler *spec_comp);

  void AnnotateBuildSidesByCard(ir_sql_converter::AQPStmt &ir);

  uint64_t EstimateIRCard(const ir_sql_converter::AQPStmt &ir) const;

  void RegisterQjitRuntimeSymbols(aqp_jit::IrToLlvmCompiler *comp);

  static std::unordered_map<std::string, PgCachedQueryPlan> &PgQueryPlanCache();
  void BeginPlanRecording() {
    plan_recording_active_ = true;
    plan_recording_.clear();
  }
  void EndPlanRecording() { plan_recording_active_ = false; }
  bool IsPlanRecording() const { return plan_recording_active_; }
  std::vector<PgCachedSubquery> &GetPlanRecording() {
    return plan_recording_;
  }
  int64_t ReplayQjitSubquery(const PgCachedSubquery &sub,
                              bool update_temp_card);
  QueryResult ReplayQjitFinal(const PgCachedSubquery &sub);
#endif

protected:
  PGconn *conn;
  json parse_tree;

private:

#ifdef HAVE_LLVM
  std::unique_ptr<QjitCompiled>
  TryCompileQueryJit(const ir_sql_converter::AQPStmt &ir,
                     const qjit::QjitAnalysisResult &analysis,
                     const std::string &label);

  bool ResolveQjitSources(const qjit::QjitQueryPlan &plan,
                          QjitCompiled &compiled, std::string &reason);

  bool BuildOutputDescsFromIR(const qjit::QjitQueryPlan &plan,
                              QjitCompiled &compiled, std::string &reason);

  int64_t ExecuteQueryJit(QjitCompiled &compiled,
                          const std::string &temp_table_name);

  QueryResult ExecuteQueryJitFinal(QjitCompiled &compiled);

  void MaterializeQjitTempToPostgreSQL(const std::string &name,
                                       bool update_temp_card);

  bool query_jit_ = false;
  bool session_query_jit_ = false;
  int query_jit_threads_ = 0;
  int query_jit_morsel_ = 20000;
  const middleware::storage::StoragePlan *qjit_storage_plan_ = nullptr;
  std::unique_ptr<qjit::QjitExecutor> qjit_executor_;
  std::unique_ptr<aqp_jit::IrToLlvmCompiler> jit_compiler_;
  bool qjit_syms_registered_ = false;
  int compile_mode_ = 0;
  int skip_hash_cmp_ = 2;
  uint32_t jit_flags_ = 0;
  int jit_cache_ = 0;
  std::string jit_cache_dir_;
  bool jit_debug_ = false;
  bool jit_prefetch_ = true;
  int jit_prefetch_distance_ = 8;
  std::unordered_map<std::string, std::unique_ptr<qjit::QjitTable>>
      qjit_temps_;
  const ir_sql_converter::AQPStmt *qjit_pending_ir_ = nullptr;
  PostPrepareHook post_prepare_hook_;
  std::unique_ptr<QjitSpecCompiled> qjit_spec_hit_;
  long spec_wait_extra_us_ = 0;
  std::vector<PgCachedSubquery> plan_recording_;
  bool plan_recording_active_ = false;
  void EnsureJITCompiler();
#endif
};
} // namespace middleware