#pragma once

#include "adapters/db_adapter.h"
#include "pg_query.h"
#include <nlohmann/json.hpp>

// pg_query.h defines _(a) as (a) (PostgreSQL i18n macro), which collides
// with lingo-db's use of _ as a variable name in lock_guard declarations.
#undef _

#include <lingodb/catalog/Catalog.h>
#include <lingodb/execution/Execution.h>
#include <lingodb/runtime/Session.h>

#ifdef HAVE_LLVM
#include "jit/ir_to_llvm.h"
#include "qjit/query_jit_executor.h"
#include "qjit/query_jit_steps.h"
namespace middleware { namespace storage { class StoragePlan; } }
#endif

namespace arrow { class Table; }

namespace middleware {

class LingoDBAdapter : public EngineAdapter {
public:
  explicit LingoDBAdapter(const std::string &db_path);
  ~LingoDBAdapter() override;

  void ParseSQL(const std::string &sql) override;
  std::unique_ptr<ir_sql_converter::AQPStmt> ConvertPlanToIR() override;

  QueryResult ExecuteSQL(const std::string &sql) override;
  void ExecuteSQLandCreateTempTable(const std::string &sql,
                                    const std::string &temp_table_name,
                                    bool update_temp_card) override;

  void ExecuteIRandCreateTempTable(
      ir_sql_converter::AQPStmt &ir,
      const std::string &temp_table_name,
      bool update_temp_card) override;

  QueryResult ExecuteIRQuery(ir_sql_converter::AQPStmt &ir) override;

  void CreateTempTable(const std::string &table_name,
                       const QueryResult &result) override;
  void DropTempTable(const std::string &table_name) override;
  bool TempTableExists(const std::string &table_name) override;

  uint64_t
  GetTempTableCardinality(const std::string &temp_table_name) override;
  void SetTempTableCardinality(const std::string &temp_table_name,
                               uint64_t cardinality) override;

  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;

  std::string GetEngineName() const override { return "LingoDB"; }

  void CleanUp() override;

  void LoadTablesFromCSV(const std::string &schema_path,
                         const std::string &csv_dir);

  void SetExecutionMode(const std::string &mode);

#ifdef HAVE_LLVM
  void SetQueryJit(bool enable, int threads, uint64_t morsel) {
    query_jit_ = enable;
    query_jit_threads_ = threads;
    query_jit_morsel_ = morsel;
  }
  void SetQueryJitStoragePlan(const storage::StoragePlan *plan) {
    qjit_storage_plan_ = plan;
  }
  void SetCompileMode(int mode) { compile_mode_ = mode; }
  void SetSkipHashCmp(int mode) { skip_hash_cmp_ = mode; }
  void SetJitFlags(uint32_t flags) { jit_flags_ = flags; }
  void SetJITCache(int mode) { jit_cache_ = (mode == 4) ? 3 : mode; }
  void SetJITCacheDir(const std::string &dir) { jit_cache_dir_ = dir; }
  void SetJITDebug(bool debug) { jit_debug_ = debug; }
  void SetJITPrefetch(bool enable, int distance) {
    jit_prefetch_ = enable;
    jit_prefetch_distance_ = distance;
  }
  void SetJITOptFlags(bool payload_prune, bool prefetch, int prefetch_dist,
                      bool batch_probe, int skip_hash_cmp,
                      int /*single_col_mode*/) {
    jit_payload_prune_ = payload_prune;
    jit_prefetch_ = prefetch;
    jit_prefetch_distance_ = prefetch_dist;
    jit_batch_probe_ = batch_probe;
    skip_hash_cmp_ = skip_hash_cmp;
  }

  bool IsQueryJitEnabled() const { return query_jit_; }
  int GetCompileMode() const { return compile_mode_; }
  int GetSkipHashCmp() const { return skip_hash_cmp_; }

  void ConfigureQueryJit(bool enable, int threads, uint64_t morsel) override {
    SetQueryJit(enable, threads, morsel);
  }
  void ConfigureQueryJitStoragePlan(const void *plan) override {
    SetQueryJitStoragePlan(static_cast<const storage::StoragePlan *>(plan));
  }
  void ConfigureJitCompileMode(int mode) override { SetCompileMode(mode); }
  void ConfigureJitCache(int mode) override { SetJITCache(mode); }
  void ConfigureJitCacheDir(const std::string &dir) override { SetJITCacheDir(dir); }
  void ConfigureJitFlags(uint32_t flags) override { SetJitFlags(flags); }
  void ConfigureJitDebug(bool debug) override { SetJITDebug(debug); }
  void ConfigureJitSkipHashCmp(int mode) override { SetSkipHashCmp(mode); }
  void ConfigureJitPrefetch(bool enable, int distance) override {
    SetJITPrefetch(enable, distance);
  }
  void ConfigureJitOptFlags(bool payload_prune, bool prefetch,
                            int prefetch_dist, bool batch_probe,
                            int skip_hash_cmp, int single_col_mode) override {
    SetJITOptFlags(payload_prune, prefetch, prefetch_dist, batch_probe,
                   skip_hash_cmp, single_col_mode);
  }

  void SetQjitPendingIR(const ir_sql_converter::AQPStmt *ir) {
    qjit_pending_ir_ = ir;
  }

  void AnnotateBuildSidesByCard(ir_sql_converter::AQPStmt &ir);
  void AnnotateBuildSidesFromMLIR(ir_sql_converter::AQPStmt &ir);
  uint64_t EstimateIRCard(const ir_sql_converter::AQPStmt &ir) const;
#endif

protected:
  std::shared_ptr<lingodb::runtime::Session> session_;
  lingodb::execution::ExecutionMode exec_mode_ =
      lingodb::execution::ExecutionMode::SPEED;

  QueryResult ExecuteSingleSQL(const std::string &sql);

  void CreateTempTableFromArrow(const std::string &table_name,
                                std::shared_ptr<arrow::Table> table);

  static void WriteLingoDBTimingRow(
      const std::unordered_map<std::string, double> &timing);

  static QueryResult ArrowTableToQueryResult(
      const std::shared_ptr<arrow::Table> &table);

private:
  nlohmann::json parse_tree_;
  bool scheduler_started_ = false;

#ifdef HAVE_LLVM
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
  };

  std::unique_ptr<QjitCompiled>
  TryCompileQueryJit(ir_sql_converter::AQPStmt &ir,
                     const qjit::QjitAnalysisResult &analysis,
                     const std::string &label);

  bool ResolveQjitSources(const qjit::QjitQueryPlan &plan,
                          QjitCompiled &compiled, std::string &reason);

  bool BuildOutputDescsFromIR(const qjit::QjitQueryPlan &plan,
                              QjitCompiled &compiled, std::string &reason);

  int64_t ExecuteQueryJit(QjitCompiled &compiled,
                          const std::string &temp_table_name);

  QueryResult ExecuteQueryJitFinal(QjitCompiled &compiled);

  void MaterializeQjitTempToLingoDB(const std::string &temp_table_name);

  void EnsureJITCompiler();
  void RegisterQjitRuntimeSymbols(aqp_jit::IrToLlvmCompiler *comp);

  bool query_jit_ = false;
  int query_jit_threads_ = 0;
  uint64_t query_jit_morsel_ = 20000;
  const storage::StoragePlan *qjit_storage_plan_ = nullptr;
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
  bool jit_payload_prune_ = true;
  bool jit_batch_probe_ = true;
  std::unordered_map<std::string, std::unique_ptr<qjit::QjitTable>>
      qjit_temps_;
  const ir_sql_converter::AQPStmt *qjit_pending_ir_ = nullptr;
#endif
};

} // namespace middleware
