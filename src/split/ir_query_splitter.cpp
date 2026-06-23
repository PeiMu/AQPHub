/*
 * IR-level query splitter main pipeline
 */

#include "split/ir_query_splitter.h"
#include "kernel/pipeline_kernel.h"
#include "jit/aqp_jit_abi.h"
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>

#ifdef HAVE_DUCKDB
#include "adapters/duckdb_adapter.h"
#endif

namespace middleware {

namespace {
void LogKernelDecision(const char *log_file_path,
                       int repeat, int iteration, const char *type,
                       bool valid, bool used,
                       const std::string &scan_table, uint64_t scan_rows,
                       size_t num_joins, size_t num_filters,
                       size_t num_output_cols, double exe_time_ms) {
  static const char *query_name = [] {
    const char *q = getenv("AQP_QUERY_NAME");
    return q ? q : "unknown";
  }();
  std::ofstream f(log_file_path, std::ios_base::app);
  f << query_name << "," << repeat << "," << iteration << ","
    << type << "," << (valid ? 1 : 0) << "," << (used ? 1 : 0) << ","
    << scan_table << "," << scan_rows << ","
    << num_joins << "," << num_filters << ","
    << num_output_cols << ","
    << std::fixed << std::setprecision(3) << exe_time_ms << "\n";
}
bool HasCrossProduct(const ir_sql_converter::AQPStmt *node) {
  if (!node) return false;
  if (node->GetNodeType() ==
      ir_sql_converter::SimplestNodeType::CrossProductNode)
    return true;
  for (const auto &child : node->children)
    if (HasCrossProduct(child.get())) return true;
  return false;
}
} // namespace

IRQuerySplitter::IRQuerySplitter(EngineAdapter *adapter,
                                 const ParamConfig &config,
                                 storage::StoragePlan *storage_plan)
    : adapter_(adapter), storage_plan_(storage_plan), config_(config),
      bg_pool_(std::make_unique<ThreadPool>(1))
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
      , jit_compile_pool_(std::make_unique<ThreadPool>(1))
#endif
{

  if (config_.enable_tuning) {
    const char *p = getenv("AQP_KERNEL_LOG_FILE");
    tuning_log_file_ = p ? p : "kernel_decision_log.csv";
  }

  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Initializing with strategy: "
              << config.GetStrategyName() << std::endl;
  }

  // Create the appropriate splitter based on strategy
  switch (config.strategy) {
  case SplitStrategy::TOP_DOWN:
    splitter_ =
        std::make_unique<TopDownSplitter>(adapter, config.enable_reorder_get);
    break;

  case SplitStrategy::MIN_SUBQUERY:
    splitter_ = std::make_unique<MinSubquerySplitter>(
        adapter, config.engine, config.enable_analyze, config.fkeys_path);
    break;

  case SplitStrategy::RELATIONSHIP_CENTER:
    splitter_ = std::make_unique<RelationshipCenterSplitter>(
        adapter, config.engine, config.enable_analyze, config.fkeys_path);
    break;

  case SplitStrategy::ENTITY_CENTER:
    splitter_ = std::make_unique<EntityCenterSplitter>(
        adapter, config.engine, config.enable_analyze, config.fkeys_path);
    break;

  case SplitStrategy::NODE_BASED: {
#ifdef HAVE_DUCKDB
    if (config_.engine == BackendEngine::DUCKDB) {
      duckdb_adapter_ = dynamic_cast<DuckDBAdapter *>(adapter);
    } else {
      // Create and OWN a helper DuckDB adapter for planning.
      owned_duckdb_adapter_ =
          std::make_unique<DuckDBAdapter>(config_.helper_db);
      duckdb_adapter_ = owned_duckdb_adapter_.get();
    }
    if (!duckdb_adapter_)
      throw std::runtime_error(
          "NODE_BASED strategy requires a DuckDB adapter for planning");
    splitter_ = std::make_unique<NodeBasedSplitter>(adapter, duckdb_adapter_,
                                                    config.enable_debug_print);
#else
    throw std::runtime_error("NODE_BASED strategy requires HAVE_DUCKDB");
#endif
    break;
  }

  case SplitStrategy::NONE:
  default:
    splitter_ = nullptr;
    break;
  }
}

IRQuerySplitter::~IRQuerySplitter() {
  for (auto &kv : async_csrs_)
    kv.second.wait();
  async_csrs_.clear();
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  {
    auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
    if (duck)
      duck->SetPostPrepareHook(nullptr); // hook captures this
  }
  DrainSpecs();
  jit_compile_pool_.reset();
  spec_compilers_[0].reset();
  spec_compilers_[1].reset();
#endif
  bg_pool_.reset();
}

IRQuerySplitter::TuneEntry
IRQuerySplitter::ParseTuneLabel(const std::string &label) {
  TuneEntry e;
  e.config_label = label;
  if (label == "interp") {
    // no JIT
  } else if (label == "expr") {
    e.jit_flags = AQP_JIT_EXPR;
  } else if (label == "expr_simd") {
    e.jit_flags = AQP_JIT_EXPR;
    e.jit_simd = true;
  } else if (label == "operator") {
    e.jit_flags = AQP_JIT_EXPR | AQP_JIT_OPERATOR;
  } else if (label == "operator_simd") {
    e.jit_flags = AQP_JIT_EXPR | AQP_JIT_OPERATOR;
    e.jit_simd = true;
  } else if (label == "pipeline") {
    e.jit_flags = AQP_JIT_EXPR | AQP_JIT_OPERATOR | AQP_JIT_PIPELINE_JIT;
  } else if (label == "pipeline_simd") {
    e.jit_flags = AQP_JIT_EXPR | AQP_JIT_OPERATOR | AQP_JIT_PIPELINE_JIT;
    e.jit_simd = true;
  } else if (label == "query_full") {
    e.query_jit = true;
  } else if (label == "query_fastisel") {
    e.query_jit = true;
    e.compile_mode = 1;
  } else if (label == "query_tpde") {
    e.query_jit = true;
    e.compile_mode = 2;
  } else if (label == "query_full_simd") {
    e.query_jit = true;
    e.jit_simd = true;
  } else if (label == "query_fastisel_simd") {
    e.query_jit = true;
    e.compile_mode = 1;
    e.jit_simd = true;
  } else {
    std::cerr << "[TUNE] unknown config label '" << label
              << "', treating as interp\n";
  }
  return e;
}

void IRQuerySplitter::LoadTuneEntry(int idx, const nlohmann::json &val) {
  std::string label = val.value("config", "interp");
  TuneEntry e = ParseTuneLabel(label);
  if (val.contains("compile_mode"))
    e.compile_mode = val["compile_mode"].get<int>();
  else if (val.contains("fast_compile"))
    e.compile_mode = val["fast_compile"].get<int>();
  if (val.contains("simd"))
    e.jit_simd = val["simd"].get<bool>();
  if (val.contains("payload_prune"))
    e.payload_prune = val["payload_prune"].get<bool>() ? 1 : 0;
  if (val.contains("prefetch"))
    e.prefetch = val["prefetch"].get<bool>() ? 1 : 0;
  if (val.contains("batch_probe"))
    e.batch_probe = val["batch_probe"].get<bool>() ? 1 : 0;
  if (val.contains("skip_hash_cmp")) {
    auto &v = val["skip_hash_cmp"];
    if (v.is_string()) {
      std::string s = v.get<std::string>();
      if (s == "off") e.skip_hash_cmp = 0;
      else if (s == "single") e.skip_hash_cmp = 1;
      else if (s == "all") e.skip_hash_cmp = 2;
    } else {
      e.skip_hash_cmp = v.get<bool>() ? 2 : 0;
    }
  }
  tune_entries_[idx] = e;
}

void IRQuerySplitter::SetQueryName(const std::string &name) {
  query_name_ = name;
  tune_entries_.clear();
  if (config_.tune_config_path.empty())
    return;

  static nlohmann::json tune_json;
  static std::string loaded_path;
  if (loaded_path != config_.tune_config_path) {
    std::ifstream f(config_.tune_config_path);
    if (!f.is_open()) {
      std::cerr << "[TUNE] cannot open " << config_.tune_config_path << "\n";
      return;
    }
    tune_json = nlohmann::json::parse(f);
    loaded_path = config_.tune_config_path;
  }

  auto it = tune_json.find(query_name_);
  if (it == tune_json.end())
    return;
  for (auto &[idx_str, val] : it->items()) {
    int idx = std::stoi(idx_str);
    LoadTuneEntry(idx, val);
  }
  if (config_.enable_debug_print && !tune_entries_.empty())
    std::cerr << "[TUNE] " << query_name_ << ": " << tune_entries_.size()
              << " sub-query overrides loaded\n";

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  if (!tune_entries_.empty() && config_.engine == BackendEngine::DUCKDB) {
    auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
    if (duck) {
      std::set<std::pair<int, int>> needed;
      for (const auto &[idx, te] : tune_entries_) {
        if (te.jit_flags == 0 && !te.query_jit)
          continue;
        int simd = te.jit_simd ? static_cast<int>(aqp_jit::SimdISA::AUTO)
                               : static_cast<int>(aqp_jit::SimdISA::OFF);
        needed.emplace(te.compile_mode, simd);
      }
      if (!needed.empty())
        duck->PreCreateCompilersAsync({needed.begin(), needed.end()});
    }
  }
#endif
}

void IRQuerySplitter::ApplyTuneOverride(int sub_idx) {
  auto it = tune_entries_.find(sub_idx);
  if (it == tune_entries_.end())
    return;
  const auto &te = it->second;

  uint32_t simd_bits = te.jit_simd ? AQP_JIT_SIMD_AUTO : AQP_JIT_SIMD_OFF;
  config_.jit_flags = te.jit_flags | simd_bits;
  if (te.query_jit)
    config_.jit_flags |= AQP_JIT_QUERY_JIT;
  config_.compile_mode = te.compile_mode;
  if (te.payload_prune >= 0)
    config_.jit_payload_prune = te.payload_prune != 0;
  if (te.prefetch >= 0)
    config_.jit_prefetch = te.prefetch != 0;
  if (te.batch_probe >= 0)
    config_.jit_batch_probe = te.batch_probe != 0;
  if (te.skip_hash_cmp >= 0)
    config_.jit_skip_hash_cmp = te.skip_hash_cmp;

#ifdef HAVE_DUCKDB
  if (config_.engine == BackendEngine::DUCKDB) {
    auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
    if (duck) {
#ifdef HAVE_LLVM
      duck->SetQueryJit(te.query_jit, config_.query_jit_threads,
                        config_.query_jit_morsel);
      duck->SetCompileMode(te.compile_mode);
      duck->SetJITFlags(config_.jit_flags);
      duck->SetJITOptFlags(
          config_.jit_payload_prune, config_.jit_prefetch,
          config_.jit_prefetch_distance, config_.jit_batch_probe,
          config_.jit_skip_hash_cmp, config_.single_col_int_join_mode);
#endif
    }
  }
#endif

  if (config_.enable_debug_print)
    std::cerr << "[TUNE] iter=" << sub_idx << " config=" << te.config_label
              << " jit_flags=0x" << std::hex << config_.jit_flags << std::dec
              << " compile_mode=" << te.compile_mode << "\n";
}

#ifdef HAVE_DUCKDB
std::unique_ptr<CrossQueryPrepResult>
IRQuerySplitter::PrepareNextQuery(const std::string &sql_path,
                                  duckdb::DuckDB &db_ref, bool debug) {
  auto result = std::make_unique<CrossQueryPrepResult>();
  auto t0 = std::chrono::high_resolution_clock::now();
  try {
    // Read SQL file
    {
      std::ifstream f(sql_path);
      if (!f.is_open()) {
        result->error = "failed to open " + sql_path;
        return result;
      }
      std::stringstream buf;
      buf << f.rdbuf();
      result->sql = buf.str();
    }

    // Extract query name (filename without .sql)
    {
      auto slash = sql_path.rfind('/');
      std::string fname =
          (slash != std::string::npos) ? sql_path.substr(slash + 1) : sql_path;
      auto dot = fname.rfind('.');
      if (dot != std::string::npos)
        fname = fname.substr(0, dot);
      result->query_name = std::move(fname);
    }

    // Create bg connection + parse + plan (mirrors DuckDBAdapter::ParseSQL)
    result->bg_conn = std::make_unique<duckdb::Connection>(db_ref);
    auto *ctx = result->bg_conn->context.get();

    bool auto_commit = ctx->transaction.IsAutoCommit();
    if (auto_commit)
      ctx->transaction.BeginTransaction();

    duckdb::Parser parser(ctx->GetParserOptions());
    parser.ParseQuery(result->sql);
    if (parser.statements.empty() ||
        parser.statements[0]->type !=
            duckdb::StatementType::SELECT_STATEMENT) {
      if (auto_commit)
        ctx->transaction.Commit();
      result->error = "not a single SELECT statement";
      return result;
    }

    result->bg_planner = std::make_unique<duckdb::Planner>(*ctx);
    result->bg_planner->CreatePlan(std::move(parser.statements[0]));
    auto plan = std::move(result->bg_planner->plan);
    if (!plan) {
      if (auto_commit)
        ctx->transaction.Commit();
      result->error = "failed to create logical plan";
      return result;
    }

    // FilterOptimize (mirrors DuckDBAdapter::FilterOptimize)
    if (plan->RequireOptimizer()) {
      duckdb::Optimizer optimizer(*result->bg_planner->binder, *ctx);
      plan = optimizer.FilterOptimize(std::move(plan));
    }

    if (auto_commit)
      ctx->transaction.Commit();

    // Preprocess (mirrors NodeBasedSplitter::Preprocess)
    result->qs = std::make_unique<duckdb::QuerySplit>(*ctx);
    result->sp = std::make_unique<duckdb::SubqueryPreparer>(
        *result->bg_planner->binder, *ctx);
    result->reorder_get = std::make_unique<duckdb::ReorderGet>(*ctx);

    // SplitIR first iteration (mirrors NodeBasedSplitter::SplitIR)
    // BLOCK 1: no pending subqueries on first iteration

    // RunMiddleOptimize
    auto_commit = ctx->transaction.IsAutoCommit();
    if (auto_commit)
      ctx->transaction.BeginTransaction();
    {
      duckdb::Optimizer optimizer(*result->bg_planner->binder, *ctx);
      plan = optimizer.MiddleOptimize(std::move(plan));
    }
    if (auto_commit)
      ctx->transaction.Commit();

    plan = result->qs->Clear(std::move(plan));
    plan = result->qs->Split(std::move(plan), true);
    auto subqueries = result->qs->GetSubqueries();
    auto table_expr_queue = result->qs->GetTableExprQueue();
    auto proj_expr = result->qs->GetProjExpr();

    // BLOCK 2
    result->reorder_get->ReorderTables(subqueries);
    result->sp->MergeSubquery(plan, std::move(subqueries));
    plan = result->sp->UpdateProjHead(std::move(plan), proj_expr);

    plan = result->qs->Clear(std::move(plan));
    plan = result->qs->Split(std::move(plan), true);
    subqueries = result->qs->GetSubqueries();
    table_expr_queue = result->qs->GetTableExprQueue();
    proj_expr = result->qs->GetProjExpr();

    // Early terminal cases
    if (subqueries.empty() || subqueries.size() == 1) {
      if (subqueries.size() == 1) {
        auto &child_node = subqueries.front()[0];
        bool merged = false;
        result->sp->MergeToSubquery(plan, child_node, merged);
      }
      // Terminal: the entire query is a single sub-plan. Convert to IR.
      auto extraction =
          std::make_unique<SubqueryExtraction>(std::set<unsigned int>{});
      extraction->is_final = true;

      // Convert plan to IR using bg planner's binder
      std::unordered_map<unsigned int, std::string> empty_map;
      auto ir = ir_sql_converter::ConvertDuckDBPlanToIR(
          *result->bg_planner->binder, *ctx, plan.get(), empty_map, false);
      extraction->sub_ir = std::move(ir);

      result->first_extraction = std::move(extraction);
      result->first_sub_sql.clear();
      result->success = true;
      auto t1 = std::chrono::high_resolution_clock::now();
      result->prep_time_us =
          std::chrono::duration<double, std::micro>(t1 - t0).count();
      return result;
    }

    // Normal: extract first subquery group as sub-plan
    result->last_sibling_node = nullptr;
    if (subqueries.front().size() > 1)
      result->last_sibling_node = std::move(subqueries.front()[1]);

    result->sp->ClearOldTableIndex();
    result->sp->AddOldTableIndex(subqueries.front()[0]);
    auto sub_plan = result->sp->GenerateProjHead(
        plan, std::move(subqueries.front()[0]), table_expr_queue, proj_expr,
        false);
    subqueries.pop_front();
    table_expr_queue.pop_front();

    sub_plan->ResolveOperatorTypes();
    result->sub_plan_types = sub_plan->types;

    // Convert sub-plan to IR
    std::unordered_map<unsigned int, std::string> empty_map;
    auto sub_ir = ir_sql_converter::ConvertDuckDBPlanToIR(
        *result->bg_planner->binder, *ctx, sub_plan.get(), empty_map, false);
    if (!sub_ir) {
      result->error = "sub-plan IR conversion failed";
      return result;
    }

    // Generate SQL from IR
    result->first_sub_sql =
        ir_sql_converter::ConvertIRToSQL(*sub_ir, 0);

    auto extraction =
        std::make_unique<SubqueryExtraction>(std::set<unsigned int>{});
    extraction->sub_ir = std::move(sub_ir);
    result->first_extraction = std::move(extraction);

    // Store remaining state for iterations 2+
    result->remaining_plan = std::move(plan);
    result->subqueries = std::move(subqueries);
    result->table_expr_queue = std::move(table_expr_queue);
    result->proj_expr = std::move(proj_expr);
    result->merge_sibling_expr = false;

    result->success = true;
  } catch (const std::exception &e) {
    result->error = e.what();
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  result->prep_time_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  return result;
}
#endif

QueryResult IRQuerySplitter::ExecuteWithSplit(const std::string &sql) {
  if (config_.enable_debug_print) {
    std::cout
        << "\n[IRQuerySplitter] ========== Starting Split Execution =========="
        << std::endl;
  }

  static int s_repeat_idx = 0;
  current_repeat_ = s_repeat_idx++;

  // Reset per-query state (but NOT subquery_index -- it must keep
  // incrementing across queries so temp table names stay unique)
  temp_tables_.clear();
  sub_plan_sqls_.clear();
  for (auto &kv : async_csrs_)
    kv.second.wait();
  async_csrs_.clear();
  runtime_csrs_.clear();
  kernel_temp_ptrs_.clear();
  kernel_temps_.clear();
  temp_min_max_cache_.clear();
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  DrainSpecs();
  spec_history_key_ = sql;
  precomputed_extraction_.reset();
  pending_extract_us_ = 0.0;
  spec_hits_ = 0;
  spec_misses_ = 0;
  spec_card_misses_ = 0;
  spec_not_ready_ = 0;
  spec_bg_errors_ = 0;
  spec_compensate_fast_ = 0;
  spec_compensate_interp_ = 0;
  spec_learned_miss_iter_ = -1;
  spec_wait_us_ = 0;
#endif
#ifdef HAVE_DUCKDB
  if (config_.engine == BackendEngine::DUCKDB) {
    auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
    if (duck) {
      duck->ClearKernelTemps();
#ifdef HAVE_LLVM
      duck->SetJITDebug(config_.enable_debug_print);
      duck->SetJITOptFlags(
          config_.jit_payload_prune,
          config_.jit_prefetch, config_.jit_prefetch_distance,
          config_.jit_batch_probe, config_.jit_skip_hash_cmp,
          config_.single_col_int_join_mode);
      duck->SetJITProbePrefetchDistances(
          config_.jit_prefetch_entry_distance,
          config_.jit_prefetch_row_distance);
      duck->SetBenchmarkMode(config_.benchmark_mode);
      duck->SetJITCache(config_.jit_cache);
      duck->SetCompileMode(config_.compile_mode);
      // Query-jit is plumbed separately: the SetJITFlags sites below mask
      // jit_flags with AQP_JIT_LEVEL_MASK, which excludes AQP_JIT_QUERY_JIT.
      duck->SetQueryJit((config_.jit_flags & AQP_JIT_QUERY_JIT) != 0,
                        config_.query_jit_threads, config_.query_jit_morsel);
      duck->SetQueryJitStoragePlan(storage_plan_);
#endif
    }
  }
#endif

  if (!config_.NeedsSplit() || !splitter_) {
    std::cout << "[IRQuerySplitter] No splitting needed, executing directly"
              << std::endl;
    return adapter_->ExecuteSQL(sql);
  }

  // === Phase 1: Parse SQL ===
  std::chrono::high_resolution_clock::time_point timer;
#ifdef HAVE_DUCKDB
  bool using_cross_query_prep =
      active_cross_query_prep_ && active_cross_query_prep_->success &&
      config_.strategy == SplitStrategy::NODE_BASED;
  if (using_cross_query_prep) {
    if (config_.enable_debug_print)
      std::cout << "[IRQuerySplitter] Phase 1+2: Skipped (cross-query prep)\n";
    // Inject bg planner so iterations 2+ can use GetBinder()/ConvertPlanToIR()
    duckdb_adapter_->SetPlanner(std::move(active_cross_query_prep_->bg_planner));
    if (config_.enable_timing) {
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << "0.000, "; // parse_sql placeholder
      log_file.close();
    }
  } else {
    active_cross_query_prep_.reset();
#endif
    if (config_.enable_debug_print)
      std::cout << "[IRQuerySplitter] Phase 1: Parsing SQL" << std::endl;
    if (config_.enable_timing)
      timer = chrono_tic();
#ifdef HAVE_DUCKDB
    if (config_.strategy == SplitStrategy::NODE_BASED) {
      duckdb_adapter_->ParseSQL(sql);
    } else
#endif
    {
      adapter_->ParseSQL(sql);
    }
    if (config_.enable_timing) {
      auto parse_sql_time = chrono_toc(&timer, "Parse SQL time is\n", false);
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (parse_sql_time / 1000.0) << ", ";
      log_file.close();
    }
#ifdef HAVE_DUCKDB
  }
#endif

  // === Phase 2: Pre-Optimize (ONLY for DuckDB, or node-based split) ===
#ifdef HAVE_DUCKDB
  if (!using_cross_query_prep) {
    DuckDBAdapter *pre_opt = nullptr;
    if (config_.strategy == SplitStrategy::NODE_BASED) {
      pre_opt = duckdb_adapter_; // always a DuckDBAdapter*
    } else if (config_.engine == BackendEngine::DUCKDB) {
      pre_opt = dynamic_cast<DuckDBAdapter *>(adapter_);
    }
    if (pre_opt) {
      if (config_.enable_debug_print)
        std::cout << "[IRQuerySplitter] Phase 2: Pre-Optimization\n";
      pre_opt->FilterOptimize();
      if (config_.enable_debug_print)
        pre_opt->PrintLogicalPlan();
    } else {
      if (config_.enable_debug_print)
        std::cout << "[IRQuerySplitter] Phase 2: Skipping Pre-Optimization\n";
    }
  }
#else
  if (config_.enable_debug_print)
    std::cout << "[IRQuerySplitter] Phase 2: Skipping Pre-Optimization\n";
#endif

  // === Phase 3: Convert to IR ===
  // NODE_BASED skips this: NodeBasedSplitter holds the DuckDB plan directly
  // via TakePlan() in Preprocess, so ConvertPlanToIR must not be called here.
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Phase 3: Converting to IR" << std::endl;
  }
  std::unique_ptr<ir_sql_converter::AQPStmt> whole_ir;
#ifdef HAVE_DUCKDB
  if (config_.strategy != SplitStrategy::NODE_BASED) {
#endif
    if (config_.enable_timing)
      timer = chrono_tic();
    whole_ir = adapter_->ConvertPlanToIR();
    if (config_.enable_timing) {
      auto convert_plan_to_ir_time =
          chrono_toc(&timer, "Convert Plan to IR time is\n", false);
      // save time to a file
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (convert_plan_to_ir_time / 1000.0) << ", ";
      log_file.close();
    }
    if (!whole_ir) {
      throw std::runtime_error("Failed to convert plan to IR");
    }
    if (config_.enable_debug_print) {
      std::cout << "\n=== Whole IR (before split) ===" << std::endl;
      whole_ir->Print();
    }
#ifdef HAVE_DUCKDB
  }
#endif

  // === Phase 4: Iterative Split-Execute Loop ===
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Phase 4: Iterative Split-Execute Loop"
              << std::endl;
  }
  auto result = ExecuteSplitLoop(std::move(whole_ir));

  if (config_.enable_debug_print) {
    std::cout
        << "[IRQuerySplitter] ========== Split Execution Complete =========="
        << std::endl;
    std::cout << "Total iterations: " << iteration_count_ << std::endl;
  }

  return result;
}

QueryResult IRQuerySplitter::ExecuteSplitLoop(
    std::unique_ptr<ir_sql_converter::AQPStmt> whole_ir) {

  iteration_count_ = 0;
  std::unique_ptr<ir_sql_converter::AQPStmt> remaining_ir = std::move(whole_ir);

  // === Strategy Preprocessing ===
  std::chrono::high_resolution_clock::time_point timer;
#ifdef HAVE_DUCKDB
  if (active_cross_query_prep_ && active_cross_query_prep_->success &&
      config_.strategy == SplitStrategy::NODE_BASED) {
    if (config_.enable_debug_print)
      std::cout << "[IRQuerySplitter] Preprocess: using cross-query prep\n";
    auto *node_splitter = dynamic_cast<NodeBasedSplitter *>(splitter_.get());
    node_splitter->InitFromCrossQueryPrep(*active_cross_query_prep_);
#if defined(HAVE_LLVM)
    precomputed_extraction_ =
        std::move(active_cross_query_prep_->first_extraction);
#endif
    if (config_.enable_timing) {
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << "0.000, "; // preprocess placeholder
      log_file.close();
    }
    if (config_.enable_debug_print) {
      double prep_us = active_cross_query_prep_->prep_time_us;
      std::cerr << "[CROSS-QUERY] query=" << query_name_
                << " bg_prep=" << std::fixed << std::setprecision(1)
                << (prep_us / 1000.0) << "ms\n";
    }
  } else {
    active_cross_query_prep_.reset();
#endif
    if (config_.enable_debug_print)
      std::cout << "[IRQuerySplitter] Strategy Preprocessing" << std::endl;
    if (config_.enable_timing)
      timer = chrono_tic();
    splitter_->Preprocess(remaining_ir);
    if (config_.enable_timing) {
      auto preprocess_time =
          chrono_toc(&timer, "Preprocess time is\n", false);
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (preprocess_time / 1000.0) << ", ";
      log_file.close();
    }
#ifdef HAVE_DUCKDB
  }
#endif

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  // Phase A wiring: the adapter fires this hook twice per iteration — after
  // Prepare(i) and again after Execute(i). Legacy spec mode launches the bg
  // compile for subquery i+1 at the post-execute invocation (post-Prepare is
  // trace-only); spec-jit launches at post-Prepare so the compile
  // overlaps Execute(i). Cleared in the destructor (the hook captures
  // `this`).
  if (config_.spec_jit != 0 &&
      config_.strategy == SplitStrategy::NODE_BASED &&
      config_.engine == BackendEngine::DUCKDB &&
      (config_.jit_flags & (AQP_JIT_LEVEL_MASK | AQP_JIT_QUERY_JIT))) {
    auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
    if (duck) {
      duck->SetPostPrepareHook(
          [this](const std::string &temp_name, duckdb::idx_t chunk_index,
                 const duckdb::vector<duckdb::LogicalType> &types,
                 const std::vector<std::string> &col_names,
                 duckdb::idx_t est_card, bool post_execute) {
            LaunchSpeculativeCompile(temp_name, chunk_index, types, col_names,
                                     est_card, post_execute);
          });
    }
  }
#endif

  // Main loop: while (graph has edges) { extract → execute → merge }
  // Also continue if precomputed_extraction_ holds a cached SplitIR result
  // from LaunchSpeculativeCompile — even if the splitter is now terminal.
  auto has_work = [&]() -> bool {
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
    if (precomputed_extraction_)
      return true;
#endif
    return !splitter_->IsComplete(remaining_ir.get());
  };
  while (has_work()) {
    iteration_count_++;

    if (config_.enable_debug_print) {
      std::cout << "\n========== Iteration " << iteration_count_
                << " ==========" << std::endl;
    }

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
    if (!precomputed_extraction_)
#endif
      splitter_->ReorderBeforeSplit(remaining_ir);

    if (!ExecuteOneIteration(remaining_ir)) {
      std::cerr << "[IRQuerySplitter] Warning: ExecuteOneIteration returned "
                   "false but IsComplete was false. Breaking loop."
                << std::endl;
      break;
    }

    if (config_.enable_debug_print) {
      std::cout << "Iteration " << iteration_count_ << " completed\n";
    }
  }

  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Split loop completed after "
              << iteration_count_ << " iteration(s)" << std::endl;
  }

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  // Wait for any pending/zombie speculative compilations and print summary.
  // The drain blocks on any unconsumed bg compile (e.g. one targeting the
  // final query, which never consumes specs) — charge that wait to the
  // final's jit_compile column so the CSV stays equal to wall time.
  DrainSpecs(/*charge_wait=*/true);
  if ((config_.enable_debug_print || config_.enable_timing) &&
      spec_hits_ + spec_misses_ + spec_card_misses_ + spec_not_ready_ +
              spec_bg_errors_ + spec_compensate_fast_ +
              spec_compensate_interp_ > 0) {
    std::cerr << "[AQP-SPECJIT] summary: hits=" << spec_hits_
              << " misses=" << spec_misses_
              << " card_misses=" << spec_card_misses_
              << " not_ready=" << spec_not_ready_
              << " bg_errors=" << spec_bg_errors_;
    if (config_.spec_jit != 0)
      std::cerr << " compensate_fast=" << spec_compensate_fast_
                << " compensate_interp=" << spec_compensate_interp_;
    std::cerr << " wait_ms=" << std::fixed << std::setprecision(3)
              << (spec_wait_us_ / 1000.0) << "\n";
  }
#endif

  // === Final Execution ===
  if (!remaining_ir) {
    throw std::runtime_error("Remaining IR is null after split loop");
  }

  if (config_.enable_debug_print) {
    std::cout << "\n=== Final Remaining IR ===" << std::endl;
    remaining_ir->Print();
  }

  if (config_.enable_timing)
    timer = chrono_tic();
  // Determine final SQL (trivial or non-trivial)
  std::string final_sql;
  std::string trivial_temp = GetTrivialTempTable(remaining_ir.get());
  if (!trivial_temp.empty()) {
    if (config_.enable_debug_print) {
      std::cout << "\n[IRQuerySplitter] Final IR is trivial (temp table: "
                << trivial_temp << "), returning directly" << std::endl;
    }
    final_sql = "SELECT * FROM " + trivial_temp;
  } else {
    // Non-trivial case: generate final SQL
    if (config_.enable_debug_print) {
      std::cout << "\n[IRQuerySplitter] Executing final remaining IR"
                << std::endl;
    }
    final_sql =
        adapter_->GenerateSQL(*remaining_ir, adapter_->subquery_index++);
    if (config_.print_sql || config_.enable_debug_print) {
      std::cout << "\n=== Final Generated Sub-SQL ===" << std::endl;
      std::cout << final_sql << std::endl;
    }
  }
  // Try kernel MIN aggregate execution path
  QueryResult query_result;
  bool kernel_final_executed = false;

  // Kernel decision log variables (final)
  bool log_final_valid = false;
  std::string log_final_scan_table;
  uint64_t log_final_scan_rows = 0;
  size_t log_final_num_joins = 0, log_final_num_filters = 0;
  size_t log_final_num_min_cols = 0;
  double log_final_exe_ms = 0.0;

#ifdef HAVE_DUCKDB
  // Mask out AQP_JIT_QUERY_JIT: query-jit's final query must reach the
  // adapter's ExecuteSQL (analysis trace + interpreter fallback), not the
  // kernel sorted-MIN aggregate path.
  if (storage_plan_ && storage_plan_->IsLoaded() &&
      config_.kernel_path != KernelPath::NONE &&
      config_.engine == BackendEngine::DUCKDB) {

    if (config_.kernel_path == KernelPath::PIPELINE)
      EnsureReferencedTempsReadyNoCsr(remaining_ir.get());
    else
      EnsureReferencedTempsReady(remaining_ir.get());

    auto lazy_builder = [&](const std::string &key) -> const storage::CSRIndex * {
      auto fit = async_csrs_.find(key);
      if (fit == async_csrs_.end()) return nullptr;
      runtime_csrs_[key] = fit->second.get();
      async_csrs_.erase(fit);
      return &runtime_csrs_[key];
    };

    auto final_plan = storage::AnalyzeFinalIR(
        remaining_ir.get(), storage_plan_,
        kernel_temp_ptrs_, runtime_csrs_,
        storage_plan_->GetDimensionCache(),
        lazy_builder);

    // Capture features for kernel decision log
    if (final_plan.valid) {
      log_final_valid = true;
      log_final_scan_table = final_plan.scan_table_name;
      log_final_scan_rows = final_plan.scan_table->row_count;
      log_final_num_joins = final_plan.join_steps.size();
      log_final_num_filters = final_plan.scan_filters.size();
      log_final_num_min_cols = final_plan.min_cols.size();
    }

    if (final_plan.valid && !config_.no_kernel) {
      if (config_.enable_timing) {
        auto gen_and_analyze_time =
            chrono_toc(&timer, "Generate final + AnalyzeFinalIR time\n", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (gen_and_analyze_time / 1000.0) << ", ";
        log_file << "0.000, "; // jit_compile_final = 0 (kernel path)
        log_file.close();
        timer = chrono_tic();
      }

      std::chrono::high_resolution_clock::time_point kernel_start;
      if (config_.enable_tuning)
        kernel_start = std::chrono::high_resolution_clock::now();
      query_result = storage::ExecuteFinalAggregate(final_plan);
      if (config_.enable_tuning)
        log_final_exe_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - kernel_start).count();
      kernel_final_executed = true;

      if (config_.enable_timing) {
        auto exec_time =
            chrono_toc(&timer, "ExecuteFinalAggregate time\n", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (exec_time / 1000.0) << ", ";
        log_file.close();
      }

      if (config_.enable_debug_print) {
        std::cout << "[SORTED-MIN] Kernel final aggregate: "
                  << final_plan.min_cols.size() << " MIN columns" << std::endl;
        for (size_t i = 0; i < final_plan.min_cols.size(); i++) {
          std::cout << "  " << final_plan.output_names[i] << " = "
                    << query_result.rows[0][i]
                    << (final_plan.min_cols[i].sorted ? " (sorted)" : " (running)")
                    << std::endl;
        }
      }
    }
  }
#endif

  if (!kernel_final_executed) {
    if (config_.enable_timing) {
      auto generate_final_sub_sql_time =
          chrono_toc(&timer, "Generate final sub-SQL time is\n", false);
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (generate_final_sub_sql_time / 1000.0) << ", ";
      log_file.close();
    }
    // Print combined sub-plan SQL if enabled (print only; result comes from
    // final_sql executed via the normal temp-table chain below)
    if (config_.enable_sub_plan_combiner && !sub_plan_sqls_.empty()) {
      std::string combined = BuildCombinedSQL(sub_plan_sqls_, final_sql);
      if (config_.enable_timing) {
        auto combine_sql_time =
            chrono_toc(&timer, "Combine SQL time is\n", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (combine_sql_time / 1000.0) << ", ";
        log_file.close();
      }
      if (config_.print_sql || config_.enable_debug_print) {
        std::cout << "\n=== Combined Sub-Plan SQL ===" << std::endl;
        std::cout << combined << std::endl;
      }
      // Drop temp tables created by the split loop so the combined SQL can
      // CREATE them fresh (avoiding "already exists" errors)
      for (const auto &plan : sub_plan_sqls_) {
        adapter_->DropTempTable(plan.first);
      }
      std::chrono::high_resolution_clock::time_point duckdb_final_start;
      if (config_.enable_tuning)
        duckdb_final_start = std::chrono::high_resolution_clock::now();
      query_result = adapter_->ExecuteSQL(combined);
      if (config_.enable_tuning)
        log_final_exe_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - duckdb_final_start).count();
    } else {
      if (!tune_entries_.empty())
        ApplyTuneOverride(iteration_count_);
      if (config_.enable_debug_print) {
        std::cerr << "[AQP-JIT-TRACE] final SQL path: jit_flags=0x" << std::hex
                  << config_.jit_flags << std::dec << " (";
        if (config_.jit_flags & AQP_JIT_EXPR)
          std::cerr << "EXPR ";
        if (config_.jit_flags & AQP_JIT_OPERATOR)
          std::cerr << "OPERATOR ";
        if (config_.kernel_path == KernelPath::PIPELINE)
          std::cerr << "KERNEL=PIPELINE ";
        else if (config_.kernel_path == KernelPath::QUERY)
          std::cerr << "KERNEL=QUERY ";
        if (config_.jit_flags & AQP_JIT_SIMD)
          std::cerr << "SIMD ";
        std::cerr << ") engine=" << (int)config_.engine << "\n";
      }
#ifdef HAVE_DUCKDB
#ifdef HAVE_LLVM
      {
        uint32_t duckdb_flags = config_.jit_flags & AQP_JIT_LEVEL_MASK;
        uint32_t adapter_flags = config_.jit_flags & (AQP_JIT_LEVEL_MASK | AQP_JIT_SIMD_MASK | AQP_JIT_SIMD);
        bool query_jit_level = (config_.jit_flags & AQP_JIT_QUERY_JIT) != 0;
        if (duckdb_flags && config_.engine == BackendEngine::DUCKDB) {
          auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
          if (config_.enable_debug_print) {
            std::cerr << "[AQP-JIT-TRACE] duck=" << (void *)duck
                      << " remaining_ir=" << (void *)remaining_ir.get() << "\n";
          }
          if (duck && remaining_ir && !HasCrossProduct(remaining_ir.get()))
            duck->SetJITPendingIR(remaining_ir.get(), adapter_flags);
          else if (duck)
            duck->SetJITFlags(adapter_flags);
        }
      }
#else
      std::cerr << "[AQP-JIT-TRACE] HAVE_LLVM NOT defined\n";
#endif
#else
      std::cerr << "[AQP-JIT-TRACE] HAVE_DUCKDB NOT defined\n";
#endif
      std::chrono::high_resolution_clock::time_point duckdb_final_start;
      if (config_.enable_tuning)
        duckdb_final_start = std::chrono::high_resolution_clock::now();
      query_result = adapter_->ExecuteSQL(final_sql);
      if (config_.enable_tuning)
        log_final_exe_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - duckdb_final_start).count();
    }
  }

  if (config_.enable_tuning)
    LogKernelDecision(tuning_log_file_.c_str(),
                      current_repeat_, iteration_count_, "final",
                      log_final_valid, kernel_final_executed,
                      log_final_scan_table, log_final_scan_rows,
                      log_final_num_joins, log_final_num_filters,
                      log_final_num_min_cols, log_final_exe_ms);

  // EXPLAIN ANALYZE the final sub-SQL for the web UI's "Show Plan" toggle
  // (outside the timed sections, like the per-iteration plan above).
  if (config_.enable_explain) {
    std::cout << "\n=== Final Sub-Query Plan ===\n"
              << adapter_->ExplainAnalyze(final_sql)
              << "\n=== End Sub-Query Plan ===" << std::endl;
  }
  
  return query_result;
}

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)

// ─── Speculative compile helpers ─────────────────────────────────────────────

// Iterations whose speculation missed for a given query, learned across
// in-process repeats (the split sequence is deterministic per query, so a
// miss on repeat 1 will miss on every repeat). Keyed by the original SQL.
// A wasted bg compile steals a core from the next iteration's execution, so
// skipping known-miss launches removes the speculation overhead entirely.
static std::unordered_map<std::string, std::set<int>> g_spec_miss_history;

// backend: FastCompileBackend value for the spec compiler. Always FULL (0) —
// the bg compile is full quality (LLVM O2), miss recompile uses TPDE via the
// adapter's EnsureFastJitCompiler.
static void EnsureSpecCompiler(
    std::unique_ptr<aqp_jit::IrToLlvmCompiler> &spec_compiler,
    DuckDBAdapter *duck, uint32_t jit_flags, int backend,
    bool need_qjit_syms) {
  aqp_jit::SimdISA simd = aqp_jit::SimdISA::OFF;
  if (jit_flags & AQP_JIT_SIMD_AVX2)
    simd = aqp_jit::SimdISA::AVX2;
  else if (jit_flags & AQP_JIT_SIMD_AVX512)
    simd = aqp_jit::SimdISA::AVX512;
  auto want_fast = static_cast<aqp_jit::FastCompileBackend>(backend);
  if (spec_compiler &&
      (spec_compiler->GetFastMode() != want_fast ||
       spec_compiler->GetSimdISA() != simd))
    spec_compiler.reset();
  if (!spec_compiler) {
    spec_compiler = std::make_unique<aqp_jit::IrToLlvmCompiler>(
        duck->GetJitDebug(), simd, want_fast);
    spec_compiler->SetPrefetch(duck->GetJitPrefetch(),
                               duck->GetJitPrefetchDistance());
    spec_compiler->SetProbePrefetchDistances(
        duck->GetJitPrefetchEntryDistance(),
        duck->GetJitPrefetchRowDistance());
    spec_compiler->SetBatchProbe(duck->GetJitBatchProbe());
    spec_compiler->SetSkipHashCmp(duck->GetJitSkipHashCmp());
    if (duck->GetJitCache())
      spec_compiler->SetCache(true);
  }
  if (need_qjit_syms)
    duck->RegisterQjitRuntimeSymbols(spec_compiler.get());
}

void IRQuerySplitter::RetirePendingSpec() {
  if (pending_spec_) {
    if (pending_spec_->future.valid() &&
        pending_spec_->future.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
      pending_spec_.reset(); // bg task done; safe to destroy now
    } else {
      zombie_specs_.push_back(std::move(pending_spec_));
    }
    pending_spec_.reset();
  }
  // Reap zombies whose bg task has finished.
  for (auto it = zombie_specs_.begin(); it != zombie_specs_.end();) {
    if (!(*it)->future.valid() ||
        (*it)->future.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
      it = zombie_specs_.erase(it);
    } else {
      ++it;
    }
  }
}

void IRQuerySplitter::WaitSpecsBorrowingIR(
    const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return;
  if (pending_spec_ && pending_spec_->borrowed_ir == ir &&
      pending_spec_->future.valid())
    pending_spec_->future.wait();
  for (auto &z : zombie_specs_)
    if (z->borrowed_ir == ir && z->future.valid())
      z->future.wait();
}

void IRQuerySplitter::DrainSpecs(bool charge_wait) {
  auto drain_start = std::chrono::high_resolution_clock::now();
  if (pending_spec_) {
    // Never consumed by a CheckSpeculativeResult (e.g. the next iteration ran
    // on the kernel path or the query ended) — the launch was wasted; learn
    // to skip it on later repeats.
    if (pending_spec_->target_iter > 0)
      g_spec_miss_history[spec_history_key_].insert(pending_spec_->target_iter);
    if (pending_spec_->future.valid())
      pending_spec_->future.wait();
  }
  pending_spec_.reset();
  for (auto &z : zombie_specs_)
    if (z->future.valid())
      z->future.wait();
  zombie_specs_.clear();
  if (charge_wait) {
    long wait_us = (long)std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - drain_start)
                       .count();
    if (wait_us > 0) {
      spec_wait_us_ += wait_us;
      if (auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_))
        duck->AddSpecWaitTime(wait_us);
    }
  }
}

// Phase A: peek-based speculative compile, invoked via the adapter's hook.
// The hook fires post-Prepare (trace-only, returns early below) and after
// Execute(i), where the launch happens: registers a placeholder for the temp
// iteration i just produced (so the bg Prepare can bind it), injects a
// matching speculative CHUNK_GET in the peek, then launches bg Prepare +
// RegisterJIT overlapping only the inter-iteration middleware window.
void IRQuerySplitter::LaunchSpeculativeCompile(
    const std::string &temp_table_name, duckdb::idx_t chunk_index,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names, duckdb::idx_t est_card,
    bool post_execute) {
  if (config_.spec_jit == 0)
    return;

  // Launch EARLY at post-Prepare so the full-quality bg compile overlaps
  // Execute(i); assumed_card is the Prepare estimate and the card guard in
  // CheckSpeculativeResult is the usability filter. A miss never relaunches
  // — it goes to the inline miss policy instead (recompile=TPDE or
  // interpret=skip).
  // Trace est (post-Prepare) and actual (post-execute) cardinality.
  if (!post_execute) {
    if (config_.enable_debug_print)
      std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                << " temp=" << temp_table_name << " prepare_est=" << est_card
                << "\n";
  } else {
    if (config_.enable_debug_print)
      std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                << " temp=" << temp_table_name << " actual_card=" << est_card
                << "\n";
    return; // already launched at post-Prepare
  }
  if (config_.strategy != SplitStrategy::NODE_BASED)
    return;
  auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
  if (!duck)
    return;

  // Tune look-ahead: the spec compile targets the NEXT iteration (1-based:
  // iteration_count_ + 1), whose 0-based tune key = iteration_count_. Use
  // that sub-query's config so the bg-compiled code matches what
  // ApplyTuneOverride will set on the inline path.
  int next_tune_key = iteration_count_;  // 0-based key for the next iteration
  uint32_t spec_jit_flags = config_.jit_flags;
  int spec_compile_mode = config_.compile_mode;
  auto tune_it = tune_entries_.find(next_tune_key);
  if (tune_it != tune_entries_.end()) {
    const auto &te = tune_it->second;
    uint32_t simd_bits = te.jit_simd ? AQP_JIT_SIMD_AUTO : AQP_JIT_SIMD_OFF;
    spec_jit_flags = te.jit_flags | simd_bits;
    if (te.query_jit)
      spec_jit_flags |= AQP_JIT_QUERY_JIT;
    spec_compile_mode = te.compile_mode;
  }

  uint32_t duckdb_flags = spec_jit_flags & AQP_JIT_LEVEL_MASK;
  bool query_jit = (spec_jit_flags & AQP_JIT_QUERY_JIT) != 0;
  if (!duckdb_flags && !query_jit)
    return;

  // Skip launches that an earlier in-process repeat learned will miss.
  {
    auto hist = g_spec_miss_history.find(spec_history_key_);
    if (hist != g_spec_miss_history.end() &&
        hist->second.count(iteration_count_ + 1)) {
      if (config_.enable_debug_print)
        std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                  << " skipping launch (learned miss for iter "
                  << iteration_count_ + 1 << ")\n";
      // The skipped iteration still gets the miss policy (recompile/interp)
      // at its check site — record it (no pending_spec_).
      spec_learned_miss_iter_ = iteration_count_ + 1;
      return;
    }
  }

  // Retire any stale speculation without blocking: the bg thread holds a raw
  // pointer (spec_raw) into the SpeculativeCompilation, so it is parked in
  // zombie_specs_ until its future completes.
  RetirePendingSpec();

  auto *node_splitter = dynamic_cast<NodeBasedSplitter *>(splitter_.get());
  if (!node_splitter || !node_splitter->HasNextSubquery())
    return;

  // Placeholder so the bg Prepare can bind temp_table_name before the real
  // result exists. The real result is Combine()d into it after ExecuteRow.
  duck->RegisterPlaceholderTemp(temp_table_name, types, col_names, est_card);

  auto spec_ir = node_splitter->PeekNextSubquery(chunk_index, types, est_card);
  if (!spec_ir)
    return;
  // Inline path skips JIT for cross-product IRs; don't speculate on them.
  if (HasCrossProduct(spec_ir.get()))
    return;

  int spec_idx = adapter_->subquery_index;
  std::string spec_sql = adapter_->GenerateSQL(*spec_ir, spec_idx);

  // Apply range predicates so a HIT is never inferior to the inline plan.
  // At early-launch (post-Prepare) temp(i)'s collection is still the empty
  // placeholder — we DON'T push it to temp_tables_ (bloom/min-max from zero
  // rows would be wrong). Divergence from the inline SQL is caught by the
  // SQL match and routed to the miss policy.
  if (query_jit)
    ApplyCrossSubPlanOptimizations(spec_sql, /*inject_range_preds=*/true,
                                   /*build_bloom_filters=*/false);
  else
    ApplyCrossSubPlanOptimizations(spec_sql);

  auto spec = std::make_unique<SpeculativeCompilation>();
  if (!query_jit)
    spec->bloom_filters = duck->TakePendingBloomFilters();
  spec->speculative_sql = std::move(spec_sql);
  spec->assumed_temp_name = temp_table_name;
  spec->assumed_card = est_card == 0 ? 1 : est_card;
  spec->target_iter = iteration_count_ + 1;
  spec->temp_snapshot = duck->SnapshotTempCollections();
  spec->spec_ir = std::move(spec_ir);
  auto *spec_ir_ptr = spec->spec_ir.get();

  auto *spec_raw = spec.get();
  auto &db_ref = duck->GetDB();

  // Use the next sub-query's tune config for the spec compiler so the
  // bg-compiled code matches the flags ApplyTuneOverride will set inline.
  // Bg spec compile always uses FULL quality (LLVM O2) — compile time is
  // free (overlaps execute(i)); miss recompile uses TPDE via the adapter.
  int spec_backend = 0;

  auto &compiler_slot = spec_compilers_[spec_compiler_idx_];
  spec_compiler_idx_ ^= 1;
  if (compiler_slot) {
    auto want_fast = static_cast<aqp_jit::FastCompileBackend>(spec_backend);
    aqp_jit::SimdISA want_simd = aqp_jit::SimdISA::OFF;
    if (spec_jit_flags & AQP_JIT_SIMD_AVX2) want_simd = aqp_jit::SimdISA::AVX2;
    else if (spec_jit_flags & AQP_JIT_SIMD_AVX512) want_simd = aqp_jit::SimdISA::AVX512;
    if (compiler_slot->GetFastMode() != want_fast ||
        compiler_slot->GetSimdISA() != want_simd) {
      for (auto &z : zombie_specs_)
        if (z->future.valid()) z->future.wait();
      zombie_specs_.clear();
    }
  }
  EnsureSpecCompiler(compiler_slot, duck, spec_jit_flags,
                     spec_backend, query_jit);
  auto *spec_comp = compiler_slot.get();

  if (query_jit) {
    // Query-jit bg task: full parse→optimize→IR→analyze→compile on the spec
    // connection (no DuckDB-embedded registration; the payload's sources are
    // resolved at HIT time on the main thread, after the consumed temp is
    // stored). A reject returns false — CheckSpeculativeResult counts it as
    // BG_ERROR and learns the miss, which is correct: the inline path would
    // reject the identical sub-query too.
    std::string label = "spec-iter" + std::to_string(iteration_count_ + 1);
    spec->future = jit_compile_pool_->Submit(
        [spec_raw, &db_ref, spec_comp, duck, label]() -> bool {
          try {
            spec_comp->ResetModules();
            spec_raw->spec_conn =
                std::make_unique<duckdb::Connection>(db_ref);
            spec_raw->qjit = duck->SpeculativeQueryJitCompile(
                spec_raw->speculative_sql, label, *spec_raw->spec_conn,
                spec_comp);
            return spec_raw->qjit != nullptr;
          } catch (...) {
            return false;
          }
        });
  } else {
    spec->future = jit_compile_pool_->Submit(
        [spec_raw, &db_ref, spec_comp, spec_ir_ptr, duck]() -> bool {
          try {
            spec_comp->ResetModules();
            spec_raw->spec_conn =
                std::make_unique<duckdb::Connection>(db_ref);
            spec_raw->spec_prepared =
                spec_raw->spec_conn->Prepare(spec_raw->speculative_sql);
            if (spec_raw->spec_prepared->HasError())
              return false;
            if (!spec_raw->spec_prepared->data ||
                !spec_raw->spec_prepared->data->physical_plan)
              return false;

            auto *spec_ctx = spec_raw->spec_conn->context.get();
            spec_ctx->aqp_jit_context =
                duckdb::make_uniq<duckdb::AQPJITContext>();

            std::unordered_set<const ir_sql_converter::AQPStmt *>
                consumed_filters, consumed_joins;
            duck->RegisterJITImpl(
                spec_raw->spec_prepared->data->physical_plan->Root(),
                *spec_ir_ptr, spec_ctx, spec_comp,
                consumed_filters, consumed_joins,
                &spec_raw->temp_snapshot);

            if (!spec_raw->bloom_filters.empty())
              duck->RegisterBloomFilters(
                  spec_raw->spec_prepared->data->physical_plan->Root(),
                  spec_ctx, spec_raw->bloom_filters);

            // Mirror the inline path: it only force-sets AQPJIT_PIPELINE when
            // RegisterJIT compiled nothing (to keep JIT-gated optimizations
            // like prefetch active). Unconditionally OR-ing it here would
            // diverge from inline behavior at expr/operator JIT levels.
            if (spec_ctx->aqp_jit_context->flags == 0)
              spec_ctx->aqp_jit_context->flags = duckdb::AQPJIT_PIPELINE;
            return true;
          } catch (...) {
            return false;
          }
        });
  }

  pending_spec_ = std::move(spec);

  if (config_.enable_debug_print)
    std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
              << " launched peek-based bg compile\n";
}

namespace {
bool NormalizedSqlEquals(const std::string &a, const std::string &b);
} // namespace

// Phase B: run real SplitIR(i+1) AFTER UpdateRemainingIR to produce
// precomputed_extraction_ for the next iteration. If the Phase A peek compile
// already targets the same SQL, keep it (it has been compiling since before
// Execute(i)); otherwise retire it and relaunch with the real SQL to salvage
// the remaining window.
void IRQuerySplitter::PrecomputeNextExtraction(
    std::unique_ptr<ir_sql_converter::AQPStmt> &remaining_ir) {
  if (config_.spec_jit == 0)
    return;
  if (config_.strategy != SplitStrategy::NODE_BASED)
    return;
  auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
  if (!duck)
    return;

  // Tune look-ahead: the precomputed extraction targets the NEXT iteration
  // (1-based: iteration_count_ + 1), whose 0-based tune key = iteration_count_.
  int next_tune_key = iteration_count_;
  uint32_t spec_jit_flags = config_.jit_flags;
  int spec_compile_mode = config_.compile_mode;
  auto tune_it = tune_entries_.find(next_tune_key);
  if (tune_it != tune_entries_.end()) {
    const auto &te = tune_it->second;
    uint32_t simd_bits = te.jit_simd ? AQP_JIT_SIMD_AUTO : AQP_JIT_SIMD_OFF;
    spec_jit_flags = te.jit_flags | simd_bits;
    if (te.query_jit)
      spec_jit_flags |= AQP_JIT_QUERY_JIT;
    spec_compile_mode = te.compile_mode;
  }

  uint32_t duckdb_flags = spec_jit_flags & AQP_JIT_LEVEL_MASK;
  bool query_jit = (spec_jit_flags & AQP_JIT_QUERY_JIT) != 0;
  if (!duckdb_flags && !query_jit)
    return;

  if (splitter_->IsComplete(remaining_ir.get()))
    return;

  splitter_->ReorderBeforeSplit(remaining_ir);
  auto extraction = splitter_->SplitIR(remaining_ir.get());
  if (!extraction || extraction->is_final) {
    precomputed_extraction_ = std::move(extraction);
    // Next iteration is final (no SQL execute) — any pending compile is moot.
    // Learn it so later repeats skip this peek launch entirely.
    if (pending_spec_ && pending_spec_->target_iter > 0)
      g_spec_miss_history[spec_history_key_].insert(pending_spec_->target_iter);
    RetirePendingSpec();
    return;
  }

  auto *spec_executable_ir = extraction->GetExecutableIR();
  if (!spec_executable_ir) {
    precomputed_extraction_ = std::move(extraction);
    return;
  }
  // Inline path skips JIT for cross-product IRs; don't speculate on them.
  if (HasCrossProduct(spec_executable_ir)) {
    RetirePendingSpec();
    precomputed_extraction_ = std::move(extraction);
    return;
  }

  int spec_idx = adapter_->subquery_index;
  std::string spec_sql = adapter_->GenerateSQL(*spec_executable_ir, spec_idx);

  // Inject the same range predicates the inline path would inject, so the
  // compiled plan is never inferior and the keep-on-match compare below is
  // optimized-vs-optimized. Bloom filters do not change the SQL text, so
  // defer building them until we know a relaunch actually happens (on the
  // kept-peek and learned-skip paths they would only be built and dropped).
  // temp_tables_ already contains the latest temp here (ExecuteSubIR pushed
  // it before UpdateRemainingIR ran).
  ApplyCrossSubPlanOptimizations(spec_sql, /*inject_range_preds=*/true,
                                 /*build_bloom_filters=*/false);

  precomputed_extraction_ = std::move(extraction);

  // Keep-on-match: speculative_sql is written on the main thread before
  // submit, so this comparison needs no future wait. The Phase A peek compile
  // has been running since Execute(i) finished — keeping it beats
  // relaunching.
  if (pending_spec_ &&
      (pending_spec_->speculative_sql == spec_sql ||
       NormalizedSqlEquals(pending_spec_->speculative_sql, spec_sql))) {
    if (config_.enable_debug_print)
      std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                << " Phase B: peek compile matches real SQL, kept\n";
    return;
  }

  // Peek predicted the wrong SQL for this iteration: learn it so later
  // repeats skip the peek launch. Otherwise the wasted peek compile steals a
  // core during execution AND delays our relaunch on the 1-worker pool.
  if (pending_spec_ && pending_spec_->target_iter > 0) {
    g_spec_miss_history[spec_history_key_].insert(pending_spec_->target_iter);
    if (config_.enable_debug_print)
      std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                << " Phase B: peek mismatch, learned miss for iter "
                << pending_spec_->target_iter << "\n";
  }

  // No Phase-B relaunch — a mispredict routes the next iteration to the
  // inline miss policy (TPDE recompile or interpret) instead of starting a
  // second full compile that could not finish in time. One bg launch per iter.
  if (pending_spec_) {
    spec_learned_miss_iter_ = pending_spec_->target_iter;
    RetirePendingSpec();
  }
}

namespace {
// Compare two generated sub-SQL strings modulo cosmetic differences that the
// next iteration's MiddleOptimize introduces (join children swapped): FROM
// item order and equality operand order. The SELECT list (which fixes the
// temp table's output schema) and any tail clauses must match exactly, so a
// normalized HIT executes the speculative prepared statement safely.
bool NormalizedSqlEquals(const std::string &a, const std::string &b) {
  struct Parts {
    std::string select_part;
    std::vector<std::string> from_items;
    std::vector<std::string> where_conjuncts;
    std::string tail;
  };
  auto parse = [](const std::string &sql, Parts &p) -> bool {
    size_t from_pos = sql.find("\nFROM ");
    if (from_pos == std::string::npos)
      return false;
    p.select_part = sql.substr(0, from_pos);
    size_t from_begin = from_pos + 6;
    size_t where_pos = sql.find("\nWHERE ", from_begin);
    size_t from_end = where_pos != std::string::npos ? where_pos : sql.size();
    std::string from_part = sql.substr(from_begin, from_end - from_begin);
    if (from_part.find('(') != std::string::npos)
      return false; // sub-selects in FROM: bail to exact compare
    size_t start = 0;
    while (start <= from_part.size()) {
      size_t comma = from_part.find(", ", start);
      if (comma == std::string::npos) {
        p.from_items.push_back(from_part.substr(start));
        break;
      }
      p.from_items.push_back(from_part.substr(start, comma - start));
      start = comma + 2;
    }
    if (where_pos == std::string::npos)
      return true;
    size_t where_begin = where_pos + 7;
    size_t where_end = sql.find('\n', where_begin);
    if (where_end == std::string::npos)
      where_end = sql.size();
    std::string where_part = sql.substr(where_begin, where_end - where_begin);
    p.tail = sql.substr(where_end);
    if (!where_part.empty() && where_part.back() == ';') {
      where_part.pop_back();
      p.tail = ";" + p.tail;
    }
    // Split top-level " AND " (respect parens and string literals).
    int depth = 0;
    size_t seg_start = 0;
    for (size_t i = 0; i < where_part.size(); i++) {
      char c = where_part[i];
      if (c == '(') {
        depth++;
      } else if (c == ')') {
        depth--;
      } else if (c == '\'') {
        i++;
        while (i < where_part.size() && where_part[i] != '\'')
          i++;
      } else if (depth == 0 && where_part.compare(i, 5, " AND ") == 0) {
        p.where_conjuncts.push_back(where_part.substr(seg_start, i - seg_start));
        i += 4;
        seg_start = i + 1;
      }
    }
    p.where_conjuncts.push_back(where_part.substr(seg_start));
    for (auto &conj : p.where_conjuncts) {
      // Canonicalize simple equality operand order.
      size_t eq = conj.find(" = ");
      if (eq != std::string::npos &&
          conj.find(" = ", eq + 3) == std::string::npos &&
          conj.find('(') == std::string::npos &&
          conj.find('\'') == std::string::npos) {
        std::string l = conj.substr(0, eq), r = conj.substr(eq + 3);
        if (l > r)
          conj = r + " = " + l;
      }
    }
    return true;
  };
  Parts pa, pb;
  if (!parse(a, pa) || !parse(b, pb))
    return false;
  if (pa.select_part != pb.select_part || pa.tail != pb.tail)
    return false;
  std::sort(pa.from_items.begin(), pa.from_items.end());
  std::sort(pb.from_items.begin(), pb.from_items.end());
  std::sort(pa.where_conjuncts.begin(), pa.where_conjuncts.end());
  std::sort(pb.where_conjuncts.begin(), pb.where_conjuncts.end());
  return pa.from_items == pb.from_items &&
         pa.where_conjuncts == pb.where_conjuncts;
}
} // namespace

bool IRQuerySplitter::CheckSpeculativeResult(
    const std::string &actual_sql, const std::string &temp_table_name) {
  if (!pending_spec_ || !pending_spec_->future.valid())
    return false;

  // Compare SQL FIRST: speculative_sql was written on the main thread before
  // the bg task was submitted, so no future wait is needed. Mismatches pay
  // zero wait; matches block until the bg compile finishes — it started
  // earlier, so waiting always beats recompiling the same SQL inline.
  if (actual_sql != pending_spec_->speculative_sql &&
      !NormalizedSqlEquals(actual_sql, pending_spec_->speculative_sql)) {
    spec_misses_++;
    g_spec_miss_history[spec_history_key_].insert(pending_spec_->target_iter);
    if (config_.enable_debug_print)
      std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                << " decision=MISS\n  spec_sql:   "
                << pending_spec_->speculative_sql << "\n  actual_sql: "
                << actual_sql << "\n";
    RetirePendingSpec();
    return false;
  }

  // Card guard: the speculative physical plan was frozen at Prepare time
  // using the temp's cardinality known at launch (compensate mode: the
  // Prepare ESTIMATE; legacy: the actual). If the actual cardinality
  // diverges, that plan (join sides, HT sizing, parallelism) may be much
  // slower than what a fresh Prepare would produce — reject. Returns false
  // on a CARD_MISS. retire_nonblocking: in compensate mode the guard runs
  // BEFORE the wait, so the bg compile may still be running and the
  // SpeculativeCompilation must be zombie-parked, not destroyed.
  uint64_t hit_est_card = 0, hit_actual_card = 0;
  auto card_guard_ok = [&](bool retire_nonblocking) -> bool {
    if (pending_spec_->assumed_temp_name.empty())
      return true;
    uint64_t actual_card = 0;
    bool found = false;
    for (auto it = temp_tables_.rbegin(); it != temp_tables_.rend(); ++it) {
      if (it->table_name == pending_spec_->assumed_temp_name) {
        actual_card = it->cardinality;
        found = true;
        break;
      }
    }
    if (!found)
      return true;
    double est = std::max<double>(1.0, (double)pending_spec_->assumed_card);
    double act = std::max<double>(1.0, (double)actual_card);
    double ratio = est > act ? est / act : act / est;
    if (ratio > 2.0) {
      spec_card_misses_++;
      g_spec_miss_history[spec_history_key_].insert(
          pending_spec_->target_iter);
      if (config_.enable_debug_print)
        std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                  << " decision=CARD_MISS (est="
                  << pending_spec_->assumed_card << " actual=" << actual_card
                  << ")\n";
      if (retire_nonblocking)
        RetirePendingSpec();
      else
        pending_spec_.reset();
      return false;
    }
    hit_est_card = pending_spec_->assumed_card;
    hit_actual_card = actual_card;
    return true;
  };

  // Usability is decided WITHOUT waiting for completion — the card guard
  // (actual card is known by now) runs before the wait, so an unusable spec
  // pays zero wait and goes to the inline miss policy.
  if (!card_guard_ok(/*retire_nonblocking=*/true))
    return false;

  if (pending_spec_->future.wait_for(std::chrono::seconds(0)) !=
      std::future_status::ready) {
    spec_not_ready_++; // matched but had to wait for the bg compile
    auto wait_start = std::chrono::high_resolution_clock::now();
    pending_spec_->future.wait();
    long wait_us = (long)std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - wait_start)
                       .count();
    spec_wait_us_ += wait_us;
    // Charge the wait to the adapter's next jit_compile CSV column: it
    // happens after the gen-sub-SQL toc and before the adapter's timer, an
    // otherwise untimed gap (CSV column sums would undercount wall time).
    if (auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_))
      duck->AddSpecWaitTime(wait_us);
    if (config_.enable_debug_print)
      std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                << " match, waited " << std::fixed << std::setprecision(3)
                << (wait_us / 1000.0) << " ms for bg compile\n";
  }
  bool bg_ok = pending_spec_->future.get();
  if (!bg_ok) {
    spec_bg_errors_++;
    g_spec_miss_history[spec_history_key_].insert(pending_spec_->target_iter);
    if (config_.enable_debug_print)
      std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                << " decision=BG_ERROR\n";
    pending_spec_.reset();
    return false;
  }

  spec_hits_++;
  if (config_.enable_debug_print) {
    std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_ << " decision=HIT";
    if (hit_actual_card || hit_est_card)
      std::cerr << " (est=" << hit_est_card << " actual=" << hit_actual_card
                << ")";
    std::cerr << "\n";
  }
  return true;
}
#endif // HAVE_DUCKDB && HAVE_LLVM

bool IRQuerySplitter::ExecuteOneIteration(
    std::unique_ptr<ir_sql_converter::AQPStmt> &remaining_ir) {

  // === Step 1: Extract Next Subquery ===
  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Step 1: Extracting next subquery" << std::endl;
  }

  std::chrono::high_resolution_clock::time_point timer;
  if (config_.enable_timing)
    timer = chrono_tic();
  std::unique_ptr<SubqueryExtraction> extraction;
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  if (precomputed_extraction_) {
    extraction = std::move(precomputed_extraction_);
  } else
#endif
  {
    extraction = splitter_->SplitIR(remaining_ir.get());
  }
  if (config_.enable_timing) {
    auto extract_next_sub_sql_time =
        chrono_toc(&timer, "Extract next sub-SQL time is\n", false);
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
    // Include the previous iteration's PrecomputeNextExtraction cost: that
    // call did THIS iteration's extraction (and SQL gen) early.
    extract_next_sub_sql_time += pending_extract_us_;
    pending_extract_us_ = 0.0;
#endif
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extract_next_sub_sql_time / 1000.0) << ", ";
    log_file.close();
  }

  if (!extraction) {
    if (config_.enable_debug_print) {
      std::cout << "[Iteration " << iteration_count_
                << "] No more subqueries to extract" << std::endl;
    }
    return false;
  }

  // Terminal extraction: sub_ir holds the final plan IR; no SQL execution.
  // Set remaining_ir so ExecuteSplitLoop can run the final query normally.
  if (extraction->is_final) {
    remaining_ir = std::move(extraction->sub_ir);
    return true;
  }

  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Extracted subquery with "
              << extraction->executed_table_indices.size() << " table(s)"
              << std::endl;
  }

  // === Step 2: Execute Sub-IR ===
  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Step 2: Executing sub-IR" << std::endl;
  }

  ir_sql_converter::AQPStmt *executable_ir = extraction->GetExecutableIR();

  if (!executable_ir) {
    std::cerr << "[Iteration " << iteration_count_
              << "] Error: No executable IR" << std::endl;
    return false;
  }

#ifndef NDEBUG
  // Assert no CROSS_PRODUCT in sub-IR — cross products are extremely slow
  // and indicate a bug in split-point selection or IR re-optimization.
  std::function<bool(const ir_sql_converter::AQPStmt *)> has_cross_product =
      [&](const ir_sql_converter::AQPStmt *node) -> bool {
    if (!node) return false;
    if (node->GetNodeType() ==
        ir_sql_converter::SimplestNodeType::CrossProductNode)
      return true;
    for (const auto &child : node->children)
      if (has_cross_product(child.get())) return true;
    return false;
  };
  if (has_cross_product(executable_ir)) {
    std::cerr << "[Iteration " << iteration_count_
              << "] FATAL: Sub-IR contains CROSS_PRODUCT node!\n";
    executable_ir->Print();
    throw std::runtime_error(
        "Sub-IR contains CROSS_PRODUCT node — aborting to prevent "
        "catastrophic performance. Check split-point selection or "
        "ReOptimizeIR.");
  }
#endif

  if (config_.enable_debug_print) {
    std::cout << "\n=== Sub-IR to Execute ===" << std::endl;
    executable_ir->Print();
  }

  bool kernel_executed = false;
  uint64_t cardinality = 0;
  std::string temp_table_name;

  // Kernel decision log variables
  bool log_plan_valid = false;
  std::string log_scan_table;
  uint64_t log_scan_rows = 0;
  size_t log_num_joins = 0, log_num_filters = 0, log_num_output_cols = 0;
  double log_exe_ms = 0.0;

#ifdef HAVE_DUCKDB
  // Kernel execution routing: pipeline kernel → query kernel → DuckDB fallback
  // (AQP_JIT_QUERY_JIT masked out: query-jit sub-queries never use kernels)
  if (storage_plan_ && storage_plan_->IsLoaded() &&
      config_.kernel_path != KernelPath::NONE &&
      config_.engine == BackendEngine::DUCKDB) {

    // Lambda: register a kernel-produced FlatTable (shared by both paths)
    auto RegisterKernelResult = [&](std::unique_ptr<storage::FlatTable> result_flat,
                                    const std::string &tname, bool build_csr) {
      cardinality = result_flat->row_count;
      for (size_t c = 0; c < result_flat->columns.size(); c++) {
        auto &attr = executable_ir->target_list[c];
        result_flat->column_names[c] =
            ComputeColumnAlias(attr->GetTableIndex(), attr->GetColumnName());
      }
      auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
      if (duck) {
        duck->RegisterKernelTemp(tname, result_flat.get());
        duck->RegisterTempMetadata(*result_flat, tname);
      }
      kernel_temp_ptrs_[tname] = result_flat.get();
      kernel_temps_[tname] = std::move(result_flat);
      if (build_csr) {
        // Async CSR: kick off background builds only for FK/PK join key columns.
        // Avoids wasting the single bg thread on columns that will never be joined.
        static constexpr int32_t MAX_CSR_DOMAIN = 10'000'000;
        const auto *flat_ptr = kernel_temps_[tname].get();
        for (size_t c = 0; c < flat_ptr->columns.size(); c++) {
          if (flat_ptr->columns[c].type != storage::FlatColumnType::INT32)
            continue;
          // Extract base column name from alias "{table}_{idx}_{col}"
          // by finding "_\d+_" pattern and taking everything after it.
          const auto &alias = flat_ptr->column_names[c];
          std::string base_col;
          for (size_t p = 0; p < alias.size(); p++) {
            if (alias[p] == '_' && p + 1 < alias.size() &&
                std::isdigit(static_cast<unsigned char>(alias[p + 1]))) {
              size_t d = p + 1;
              while (d < alias.size() &&
                     std::isdigit(static_cast<unsigned char>(alias[d])))
                d++;
              if (d < alias.size() && alias[d] == '_') {
                base_col = alias.substr(d + 1);
                break;
              }
            }
          }
          if (!base_col.empty() && storage_plan_ &&
              !storage_plan_->IsJoinKeyColumn(base_col))
            continue;
          int32_t max_val = 0;
          const auto *int_data =
              reinterpret_cast<const int32_t *>(flat_ptr->columns[c].data.get());
          for (uint64_t r = 0; r < flat_ptr->row_count; r++) {
            if (int_data[r] > max_val) max_val = int_data[r];
          }
          if (max_val > MAX_CSR_DOMAIN) continue;
          std::string csr_key = tname + "." + alias;
          async_csrs_[csr_key] = bg_pool_->Submit(
              [flat_ptr, c_idx = static_cast<int>(c), max_val, tname, alias]() {
                return storage::BuildCSR(*flat_ptr, c_idx, max_val,
                                         tname, alias, "", "");
              });
        }
      }
    };

    // --- Pipeline kernel path (hash-based, no CSR needed) ---
    if (config_.kernel_path == KernelPath::PIPELINE && !config_.no_kernel) {
      EnsureReferencedTempsReadyNoCsr(executable_ir);
      auto pipeline_plan = storage::AnalyzePipelineKernel(
          executable_ir, storage_plan_,
          kernel_temp_ptrs_,
          storage_plan_->GetDimensionCache());

      if (pipeline_plan.valid) {
        double analyze_ms = 0.0;
        if (config_.enable_timing) {
          auto analyze_time = chrono_toc(&timer, "AnalyzePipelineKernel time\n", false);
          analyze_ms = analyze_time / 1000.0;
        }
        log_plan_valid = true;
        log_scan_table = pipeline_plan.scan_table_name;
        log_scan_rows = pipeline_plan.scan_table->row_count;
        log_num_joins = pipeline_plan.join_steps.size();
        log_num_filters = pipeline_plan.scan_filters.size();
        log_num_output_cols = pipeline_plan.output_cols.size();

        adapter_->subquery_index++;
        temp_table_name = GenerateTempTableName();

        if (config_.enable_debug_print) {
          std::cout << "[PIPELINE-KERNEL] Executing: "
                    << "scan=" << pipeline_plan.scan_table_name
                    << " rows=" << pipeline_plan.scan_table->row_count
                    << " joins=" << pipeline_plan.join_steps.size()
                    << " filters=" << pipeline_plan.scan_filters.size() << std::endl;
        }

        double compile_ms = 0.0;

        if (config_.enable_timing) {
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << analyze_ms << ", " << compile_ms << ", ";
          log_file.close();
          timer = chrono_tic();
        }

        std::chrono::high_resolution_clock::time_point kernel_start;
        if (config_.enable_tuning)
          kernel_start = std::chrono::high_resolution_clock::now();
        auto result_flat = storage::ExecutePipelineKernel(pipeline_plan, temp_table_name);
        if (config_.enable_tuning)
          log_exe_ms = std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - kernel_start).count();

        if (config_.enable_timing) {
          auto kernel_exec_time = chrono_toc(&timer, "ExecutePipelineKernel time\n", false);
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << (kernel_exec_time / 1000.0) << ", ";
          log_file.close();
        }

        if (config_.enable_debug_print) {
          std::cout << "[PIPELINE-KERNEL] Result: " << result_flat->row_count
                    << " rows, " << result_flat->columns.size() << " columns" << std::endl;
        }

        // Pipeline kernel: NO CSR build on output
        RegisterKernelResult(std::move(result_flat), temp_table_name, false);

        if (config_.enable_timing) {
          auto materialize_time = chrono_toc(&timer, "Kernel materialization time\n", false);
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << (materialize_time / 1000.0) << ", ";
          log_file.close();
        }

        kernel_executed = true;
      }
    }

    // --- Query kernel path (CSR-based, existing) ---
    if (!kernel_executed && config_.kernel_path == KernelPath::QUERY && !config_.no_kernel) {
      EnsureReferencedTempsReady(executable_ir);

      auto lazy_csr_build = [&](const std::string &key) -> const storage::CSRIndex * {
        auto fit = async_csrs_.find(key);
        if (fit == async_csrs_.end()) return nullptr;
        runtime_csrs_[key] = fit->second.get();
        async_csrs_.erase(fit);
        return &runtime_csrs_[key];
      };

      auto sub_plan = storage::AnalyzeSubIR(
          executable_ir, storage_plan_,
          kernel_temp_ptrs_, runtime_csrs_,
          storage_plan_->GetDimensionCache(),
          lazy_csr_build);

      if (sub_plan.valid) {
        double analyze_ms = 0.0;
        if (config_.enable_timing) {
          auto analyze_time = chrono_toc(&timer, "AnalyzeSubIR time\n", false);
          analyze_ms = analyze_time / 1000.0;
        }
        log_plan_valid = true;
        log_scan_table = sub_plan.scan_table_name;
        log_scan_rows = sub_plan.scan_table->row_count;
        log_num_joins = sub_plan.join_steps.size();
        log_num_filters = sub_plan.scan_filters.size();
        log_num_output_cols = sub_plan.output_cols.size();

        adapter_->subquery_index++;
        temp_table_name = GenerateTempTableName();

        if (config_.enable_debug_print) {
          std::cout << "[CSR-KERNEL] Executing: "
                    << "scan=" << sub_plan.scan_table_name
                    << " rows=" << sub_plan.scan_table->row_count
                    << " joins=" << sub_plan.join_steps.size()
                    << " filters=" << sub_plan.scan_filters.size() << std::endl;
        }

        if (config_.enable_timing) {
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << analyze_ms << ", 0.000, ";
          log_file.close();
        }

        std::chrono::high_resolution_clock::time_point kernel_start;
        if (config_.enable_tuning)
          kernel_start = std::chrono::high_resolution_clock::now();
        auto result_flat = storage::ExecuteSubQueryPlan(sub_plan, temp_table_name);
        if (config_.enable_tuning)
          log_exe_ms = std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - kernel_start).count();

        if (config_.enable_timing) {
          auto kernel_exec_time = chrono_toc(&timer, "ExecuteSubQueryPlan time\n", false);
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << (kernel_exec_time / 1000.0) << ", ";
          log_file.close();
        }

        if (config_.enable_debug_print) {
          std::cout << "[CSR-KERNEL] Result: " << result_flat->row_count
                    << " rows, " << result_flat->columns.size() << " columns" << std::endl;
        }

        RegisterKernelResult(std::move(result_flat), temp_table_name, true);

        if (config_.enable_timing) {
          auto materialize_time = chrono_toc(&timer, "Kernel materialization time\n", false);
          std::ofstream log_file;
          log_file.open(g_timing_log_name, std::ios_base::app);
          log_file << std::fixed << std::setprecision(3)
                   << (materialize_time / 1000.0) << ", ";
          log_file.close();
        }

        kernel_executed = true;
      }
    }
  }
#endif

  if (!kernel_executed) {
    // Standard DuckDB execution path
    std::string sub_sql =
        adapter_->GenerateSQL(*executable_ir, adapter_->subquery_index++);
    temp_table_name = GenerateTempTableName();

    if (config_.enable_timing) {
      auto generate_sub_sql_time =
          chrono_toc(&timer, "Generate sub-SQL time is\n", false);
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (generate_sub_sql_time / 1000.0) << ", ";
      log_file.close();
    }

    if (config_.enable_sub_plan_combiner) {
      sub_plan_sqls_.emplace_back(temp_table_name, sub_sql);
    }

    if (config_.print_sql || config_.enable_debug_print) {
      std::cout << "\n=== Sub-Query SQL ===" << std::endl;
      std::cout << sub_sql << std::endl;
    }

    if (config_.enable_debug_print) {
      std::cout << "Executing sub-query and creating temp table: "
                << temp_table_name << std::endl;
    }

    bool spec_hit = false;
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
    // Compensate-jit miss policy: applies to this iteration when its spec
    // was launched but unusable (MISS/CARD_MISS/BG_ERROR) or was skipped /
    // invalidated via the learned-miss history (no pending_spec_). Never
    // fires without a spec launch attempt — none-jit and non-node-based
    // runs can't reach it (LaunchSpeculativeCompile's gates).
    bool compensate_miss = false;
    const bool learned_missed_iter =
        config_.spec_jit != 0 && spec_learned_miss_iter_ == iteration_count_;
    if (config_.spec_jit != 0)
      spec_learned_miss_iter_ = -1; // one-shot
    if (pending_spec_ && config_.engine == BackendEngine::DUCKDB) {
      ApplyCrossSubPlanOptimizations(sub_sql, /*inject_range_preds=*/true,
                                     /*build_bloom_filters=*/false);
      spec_hit = CheckSpeculativeResult(sub_sql, temp_table_name);
      if (!spec_hit) {
        ApplyCrossSubPlanOptimizations(sub_sql, /*inject_range_preds=*/false,
                                       /*build_bloom_filters=*/true);
        compensate_miss = true;
      }
    } else
#endif
    {
      ApplyCrossSubPlanOptimizations(sub_sql);
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
      compensate_miss =
          learned_missed_iter && config_.engine == BackendEngine::DUCKDB;
#endif
    }

    // Phase A speculation for the NEXT iteration is launched via the
    // adapter's hook inside the execute calls below — legacy mode at
    // post-execute (actual temp cardinality; overlaps only the
    // inter-iteration window), compensate mode at post-Prepare (estimated
    // cardinality; overlaps Execute(i)).

    std::chrono::high_resolution_clock::time_point duckdb_exe_start;
    if (config_.enable_tuning)
      duckdb_exe_start = std::chrono::high_resolution_clock::now();

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
    if (spec_hit) {
      if (!tune_entries_.empty())
        ApplyTuneOverride(iteration_count_ - 1);
      auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
      // Move out of pending_spec_ first: the spec hook fires inside
      // ExecuteSpeculativeAndCreateTempTable and its RetirePendingSpec()
      // would otherwise destroy the prepared statement while it executes.
      auto hit_spec = std::move(pending_spec_);
      if (hit_spec->qjit) {
        // Query-jit HIT: hand the bg-compiled payload to the adapter; the
        // normal execute call resolves sources and runs the compiled fn
        // (jit_compile column ≈ 0 — no Prepare, no codegen on this thread).
        duck->SetQjitSpecHit(std::move(hit_spec->qjit));
        adapter_->ExecuteSQLandCreateTempTable(
            sub_sql, temp_table_name, config_.enable_update_temp_card);
      } else {
        duck->ExecuteSpeculativeAndCreateTempTable(
            *hit_spec->spec_prepared, *hit_spec->spec_conn,
            temp_table_name, config_.enable_update_temp_card);
      }
      // hit_spec (and the compiler holding its JIT code) outlives execution;
      // the hook's bg compile used the other ping-pong compiler.
    } else
#endif
    {
      if (!tune_entries_.empty())
        ApplyTuneOverride(iteration_count_ - 1);
#ifdef HAVE_DUCKDB
#ifdef HAVE_LLVM
      {
        uint32_t duckdb_flags = config_.jit_flags & AQP_JIT_LEVEL_MASK;
        uint32_t adapter_flags = config_.jit_flags & (AQP_JIT_LEVEL_MASK | AQP_JIT_SIMD_MASK | AQP_JIT_SIMD);
        bool query_jit_level = (config_.jit_flags & AQP_JIT_QUERY_JIT) != 0;
        bool comp_fast = compensate_miss && config_.spec_jit == 1;
        bool comp_interp = compensate_miss && config_.spec_jit == 2;
        if (compensate_miss && config_.enable_debug_print)
          std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
                    << (comp_fast ? " action=COMPENSATE_FAST\n"
                                  : " action=COMPENSATE_INTERP\n");
        if (comp_fast)
          spec_compensate_fast_++;
        else if (comp_interp)
          spec_compensate_interp_++;
        if (duckdb_flags && config_.engine == BackendEngine::DUCKDB) {
          auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
          if (duck) {
            duck->SetTempColRanges({});
            if (!HasCrossProduct(executable_ir) && !comp_interp) {
              duck->SetJITPendingIR(executable_ir, adapter_flags);
              if (comp_fast)
                duck->SetCompensateMissAction(
                    DuckDBAdapter::CompensateMissAction::FAST_ONCE);
            } else {
              duck->SetJITFlags(adapter_flags);
            }
          }
        } else if (query_jit_level && compensate_miss &&
                   config_.engine == BackendEngine::DUCKDB) {
          auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
          if (duck)
            duck->SetCompensateMissAction(
                comp_fast
                    ? DuckDBAdapter::CompensateMissAction::FAST_ONCE
                    : DuckDBAdapter::CompensateMissAction::SKIP_QJIT_ONCE);
        }
      }
#endif
#endif

      if (SubPlanReferencesEmptyTemp(sub_sql)) {
        std::string short_sql = sub_sql;
        size_t semi = short_sql.rfind(';');
        if (semi != std::string::npos)
          short_sql.insert(semi, " LIMIT 0");
        else
          short_sql += " LIMIT 0";
        if (config_.enable_debug_print)
          std::cerr << "[EARLY-TERM] sub-plan references empty temp, appending LIMIT 0\n";
        adapter_->ExecuteSQLandCreateTempTable(short_sql, temp_table_name,
                                               config_.enable_update_temp_card);
      } else {
        adapter_->ExecuteSQLandCreateTempTable(sub_sql, temp_table_name,
                                               config_.enable_update_temp_card);
      }
    }
    if (config_.enable_tuning) {
      log_exe_ms = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - duckdb_exe_start).count();
    }

    // 7.3b: Lazy CSR — removed eager FlatTable+CSR build here.
    // EnsureReferencedTempsReady() will build on demand before AnalyzeSubIR.

    if (config_.enable_explain) {
      std::cout << "\n=== Sub-Query Plan ===\n"
                << adapter_->ExplainAnalyze(sub_sql)
                << "\n=== End Sub-Query Plan ===" << std::endl;
    }
  }

  if (config_.enable_tuning)
    LogKernelDecision(tuning_log_file_.c_str(),
                      current_repeat_, iteration_count_, "sub",
                      log_plan_valid, kernel_executed,
                      log_scan_table, log_scan_rows,
                      log_num_joins, log_num_filters,
                      log_num_output_cols, log_exe_ms);

  if (config_.enable_timing)
    timer = chrono_tic();

  // Generate temp table index
  unsigned int temp_table_index =
      splitter_->GetMaxTableIndex() + iteration_count_;
  if (!kernel_executed) {
    if (config_.enable_update_temp_card || extraction->estimated_rows <= 0) {
      cardinality = adapter_->GetTempTableCardinality(temp_table_name);
    } else {
      cardinality = static_cast<uint64_t>(extraction->estimated_rows);
      adapter_->SetTempTableCardinality(temp_table_name, cardinality);
    }
  }

  if (cardinality == 0) {
    empty_temp_tables_.insert(temp_table_name);
  }

  TempTableInfo temp_table =
      TempTableInfo(temp_table_name, temp_table_index, cardinality);

  // min/max for integer columns is computed lazily by the range injection
  // code (only when a join partner is found). No upfront scan needed.

  // Store column mappings with correct names (SQL generator uses:
  // {table}_{col})
  std::vector<std::pair<unsigned int, unsigned int>> col_mappings;
  std::vector<std::string> col_names;
  for (const auto &attr : executable_ir->target_list) {
    std::string col_alias =
        ComputeColumnAlias(attr->GetTableIndex(), attr->GetColumnName());
    temp_table.column_names.push_back(col_alias);
    temp_table.column_mappings.emplace_back(attr->GetTableIndex(),
                                            attr->GetColumnIndex(), col_alias);
    col_mappings.emplace_back(attr->GetTableIndex(), attr->GetColumnIndex());
    col_names.push_back(col_alias);
  }

  // Add temp table to the mapping for future iterations
  splitter_->AddTableMapping(temp_table_index, temp_table_name);

  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Created temp table: " << temp_table.table_name
              << " (index=" << temp_table.table_index
              << ", cardinality=" << temp_table.cardinality << ")" << std::endl;
  }

  // === Step 3: Update Remaining IR ===
  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Step 3: Updating remaining IR" << std::endl;
  }

  // Call strategy-specific UpdateRemainingIR (takes ownership of old IR)
  remaining_ir = splitter_->UpdateRemainingIR(
      std::move(remaining_ir), extraction->executed_table_indices,
      temp_table.table_index, temp_table.table_name, temp_table.cardinality,
      col_mappings, col_names);

  if (config_.enable_debug_print) {
    if (remaining_ir) {
      std::cout << "[Iteration " << iteration_count_
                << "] Successfully updated remaining IR" << std::endl;
    } else if (!splitter_->SkipUpdateIndices()) {
      std::cerr << "[Iteration " << iteration_count_
                << "] Warning: Failed to update remaining IR" << std::endl;
    }
  }

  // === Step 4: Update Indices (shared) ===
  // Skipped for NODE_BASED: DuckDB's UpdateSubqueriesIndex / UpdateTableExpr
  // keep all bindings consistent internally.
  if (!splitter_->SkipUpdateIndices()) {
    if (config_.enable_debug_print) {
      std::cout << "[Iteration " << iteration_count_
                << "] Step 4: Updating indices in remaining IR" << std::endl;
    }
    UpdateRemainingIRIndices(remaining_ir.get(), temp_table,
                             extraction->executed_table_indices);
    if (config_.enable_debug_print) {
      std::cout << "\n=== Updated Remaining IR ===" << std::endl;
      remaining_ir->Print();
    }
  }

  temp_tables_.push_back(temp_table);

  if (config_.enable_timing) {
    auto update_ir_time = chrono_toc(&timer, "Update IR time is\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (update_ir_time / 1000.0)
             << ", ";
    log_file.close();
  }

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
  // Phase B: run real SplitIR(i+1) to produce precomputed_extraction_ for
  // the next iteration.  bg compile was already launched in Phase A (before
  // Execute). Timed into pending_extract_us_, which the next iteration adds
  // to its extract_next_sub-IR column.
  if (!kernel_executed) {
    if (config_.enable_timing) {
      auto pre_timer = chrono_tic();
      PrecomputeNextExtraction(remaining_ir);
      pending_extract_us_ +=
          chrono_toc(&pre_timer, "PrecomputeNextExtraction time\n", false);
    } else {
      PrecomputeNextExtraction(remaining_ir);
    }
  }

  // `extraction` (and the IR behind executable_ir) dies at this return. A
  // Phase B bg compile launched in the PREVIOUS iteration may still be
  // reading it: kernel-path iterations never call CheckSpeculativeResult,
  // and a MISS retires the spec to zombie_specs_ without waiting.
  WaitSpecsBorrowingIR(executable_ir);
#endif

  return true;
}

TempTableInfo IRQuerySplitter::ExecuteSubIR(
    std::unique_ptr<ir_sql_converter::AQPStmt> sub_ir,
    const std::set<unsigned int> &executed_table_indices) {

  std::string temp_table_name = GenerateTempTableName();
  std::string sub_sql =
      adapter_->GenerateSQL(*sub_ir, adapter_->subquery_index++);

  if (config_.print_sql || config_.enable_debug_print) {
    std::cout << "\n=== Sub-Query SQL (" << temp_table_name
              << ") ===" << std::endl;
    std::cout << sub_sql << std::endl;
  }

  adapter_->ExecuteSQLandCreateTempTable(sub_sql, temp_table_name,
                                         config_.enable_update_temp_card);

  unsigned int temp_table_index = adapter_->subquery_index - 1;
  // TODO: support estimated_rows for enable_update_temp_card=false path
  uint64_t cardinality = adapter_->GetTempTableCardinality(temp_table_name);

  return TempTableInfo(temp_table_name, temp_table_index, cardinality);
}

std::string IRQuerySplitter::GenerateTempTableName() {
  return "temp" + std::to_string(adapter_->subquery_index);
}

std::string
IRQuerySplitter::GetTrivialTempTable(ir_sql_converter::AQPStmt *ir) const {
  if (!ir) {
    return "";
  }

  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = dynamic_cast<ir_sql_converter::SimplestChunk *>(ir);
    if (chunk && !chunk->GetContents().empty()) {
      return chunk->GetContents()[0];
    }
  }

  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ProjectionNode) {
    if (ir->children.size() == 1 && ir->children[0]) {
      auto *child = ir->children[0].get();
      if (child->GetNodeType() ==
          ir_sql_converter::SimplestNodeType::ChunkNode) {
        auto *chunk = dynamic_cast<ir_sql_converter::SimplestChunk *>(child);
        if (chunk && !chunk->GetContents().empty()) {
          return chunk->GetContents()[0];
        }
      }
    }
  }

  if (ir->children.size() == 1 && ir->children[0]) {
    return GetTrivialTempTable(ir->children[0].get());
  }

  return "";
}

bool IRQuerySplitter::SubPlanReferencesEmptyTemp(const std::string &sql) const {
  for (const auto &name : empty_temp_tables_) {
    if (sql.find(name) != std::string::npos)
      return true;
  }
  return false;
}

// Outlined from ExecuteOneIteration to keep the hot path compact.
// __attribute__((noinline)) prevents the compiler from inlining this
// ~250-line block back into the caller, preserving icache locality for
// the common no-optimization path.
__attribute__((noinline))
void IRQuerySplitter::ApplyCrossSubPlanOptimizations(
    std::string &sub_sql, bool inject_range_preds, bool build_bloom_filters) {
#ifdef HAVE_DUCKDB
  if (config_.engine != BackendEngine::DUCKDB) return;

  auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
  if (!duck) return;

  std::string extra_where;

  auto find_join_partner = [](const std::string &sql,
                              const std::string &temp_col_name) -> std::string {
    std::string needle = "." + temp_col_name;
    size_t pos = 0;
    while ((pos = sql.find(needle, pos)) != std::string::npos) {
      size_t after_needle = pos + needle.size();
      if (after_needle < sql.size() && (std::isalnum(sql[after_needle]) ||
          sql[after_needle] == '_')) {
        pos = after_needle;
        continue;
      }
      size_t temp_start = pos;
      while (temp_start > 0 && (std::isalnum(sql[temp_start - 1]) ||
             sql[temp_start - 1] == '_'))
        temp_start--;
      {
        size_t p = after_needle;
        while (p < sql.size() && sql[p] == ' ') p++;
        if (p < sql.size() && sql[p] == '=') {
          p++;
          while (p < sql.size() && sql[p] == ' ') p++;
          size_t start_base = p;
          while (p < sql.size() && (std::isalnum(sql[p]) ||
                 sql[p] == '_' || sql[p] == '.'))
            p++;
          std::string base_part = sql.substr(start_base, p - start_base);
          if (base_part.find('.') != std::string::npos &&
              base_part.find("temp") == std::string::npos)
            return base_part;
        }
      }
      {
        size_t p = temp_start;
        while (p > 0 && sql[p - 1] == ' ') p--;
        if (p > 0 && sql[p - 1] == '=') {
          p--;
          while (p > 0 && sql[p - 1] == ' ') p--;
          size_t end_base = p;
          size_t start_base = end_base;
          while (start_base > 0 && (std::isalnum(sql[start_base - 1]) ||
                 sql[start_base - 1] == '_' || sql[start_base - 1] == '.'))
            start_base--;
          std::string base_part = sql.substr(start_base, end_base - start_base);
          if (base_part.find('.') != std::string::npos &&
              base_part.find("temp") == std::string::npos)
            return base_part;
        }
      }
      pos = after_needle;
    }
    return "";
  };

  auto extract_base_table = [](const std::string &base_col) -> std::string {
    size_t dot = base_col.find('.');
    if (dot == std::string::npos) return "";
    std::string alias = base_col.substr(0, dot);
    size_t last_us = alias.rfind('_');
    if (last_us == std::string::npos) return alias;
    bool suffix_is_num = true;
    for (size_t i = last_us + 1; i < alias.size(); i++) {
      if (!std::isdigit(alias[i])) { suffix_is_num = false; break; }
    }
    return suffix_is_num ? alias.substr(0, last_us) : alias;
  };

  constexpr uint64_t kMaxHTCapacity = 4096;
  constexpr uint64_t kMinTempCard = 50;
  constexpr double kMaxSelectivity = 0.40;

  if (inject_range_preds)
  for (const auto &tt : temp_tables_) {
    uint64_t temp_card = tt.cardinality;
    if (temp_card < kMinTempCard) continue;
    bool is_medium = temp_card > kMaxHTCapacity;

    std::vector<std::pair<size_t, std::string>> join_matches;
    for (size_t col_pos = 0; col_pos < tt.column_names.size(); col_pos++) {
      std::string base_col = find_join_partner(sub_sql, tt.column_names[col_pos]);
      if (!base_col.empty())
        join_matches.emplace_back(col_pos, std::move(base_col));
    }
    if (join_matches.empty()) continue;

    // Temps are immutable once stored; cache the (full-scan) min/max so
    // repeated calls — inline per iteration plus the speculative launch —
    // never re-scan the same collection.
    auto mit = temp_min_max_cache_.find(tt.table_name);
    if (mit == temp_min_max_cache_.end())
      mit = temp_min_max_cache_
                .emplace(tt.table_name, duck->GetTempTableMinMax(tt.table_name))
                .first;
    const auto &col_min_max = mit->second;

    for (auto &[col_pos, base_col] : join_matches) {
      auto it = col_min_max.find(col_pos);
      if (it == col_min_max.end()) continue;
      int64_t range_min = it->second.first;
      int64_t range_max = it->second.second;
      double selectivity = (range_max > 0)
          ? static_cast<double>(range_max - range_min) / range_max
          : 1.0;

      if (is_medium) {
        if (selectivity >= 0.25) continue;
        std::string base_table = extract_base_table(base_col);
        uint64_t base_card = duck->GetBaseTableCardinality(base_table);
        double match_rate = (base_card > 0)
            ? static_cast<double>(temp_card) / base_card : 1.0;
        if (match_rate >= 0.25) continue;
      } else {
        if (selectivity >= kMaxSelectivity) continue;
      }

      if (config_.enable_debug_print) {
        std::cerr << "[RANGE-DIAG] base=" << extract_base_table(base_col)
                  << " sel=" << std::fixed << std::setprecision(4) << selectivity
                  << " card=" << temp_card << " medium=" << is_medium
                  << " inject=1\n";
      }
      extra_where += " AND " + base_col + " >= " +
                     std::to_string(range_min) +
                     " AND " + base_col + " <= " +
                     std::to_string(range_max);
    }
  }

  if (!extra_where.empty()) {
    size_t semi = sub_sql.rfind(';');
    if (semi != std::string::npos)
      sub_sql.insert(semi, extra_where);
    else
      sub_sql += extra_where;
    if (config_.enable_debug_print)
      std::cerr << "[RANGE-SQL] injected: " << extra_where << "\n";
  }

  if (!build_bloom_filters) return;

  constexpr uint64_t kBFMaxTempCard = 100000;

  std::vector<DuckDBAdapter::BloomFilterInfo> bloom_filters;
  std::set<std::pair<std::string, std::string>> bf_targets;
  for (const auto &tt : temp_tables_) {
    uint64_t temp_card = tt.cardinality;
    if (temp_card < kMinTempCard || temp_card > kBFMaxTempCard) continue;

    for (size_t col_pos = 0; col_pos < tt.column_names.size(); col_pos++) {
      std::string base_col = find_join_partner(sub_sql, tt.column_names[col_pos]);
      if (base_col.empty()) continue;

      std::string base_table = extract_base_table(base_col);

      uint64_t base_card = duck->GetBaseTableCardinality(base_table);
      if (base_card == 0) continue;
      double match_rate = static_cast<double>(temp_card) / base_card;
      if (match_rate >= 0.25) continue;
      std::string base_col_name = base_col.substr(base_col.find('.') + 1);

      auto key = std::make_pair(base_table, base_col_name);
      if (bf_targets.count(key)) continue;

      {
        std::set<std::string> aliases;
        size_t pos = 0;
        std::string pat = base_table + "_";
        while ((pos = sub_sql.find(pat, pos)) != std::string::npos) {
          size_t end = pos + pat.size();
          if (end < sub_sql.size() && std::isdigit(sub_sql[end])) {
            size_t alias_end = end;
            while (alias_end < sub_sql.size() && std::isdigit(sub_sql[alias_end]))
              alias_end++;
            if (alias_end >= sub_sql.size() || sub_sql[alias_end] == '.' ||
                sub_sql[alias_end] == ' ' || sub_sql[alias_end] == ',')
              aliases.insert(sub_sql.substr(pos, alias_end - pos));
          }
          pos = end;
        }
        if (aliases.size() > 1) continue;
      }

      auto bf_info = duck->BuildBloomFilter(tt.table_name, col_pos, temp_card);
      if (bf_info.bf_data.empty()) continue;

      bf_info.base_table_name = base_table;
      bf_info.base_col_name = base_col_name;
      bf_targets.insert(key);
      bloom_filters.push_back(std::move(bf_info));

      if (config_.enable_debug_print) {
        std::cerr << "[BF-DIAG] temp=" << tt.table_name
                  << " col=" << tt.column_names[col_pos]
                  << " card=" << temp_card
                  << " -> " << base_table << "." << base_col_name << "\n";
      }
    }
  }
  duck->SetPendingBloomFilters(std::move(bloom_filters));
#endif
}

// ===== Shared Index Update Functions =====

void IRQuerySplitter::UpdateExprIndices(
    ir_sql_converter::AQPExpr *expr, const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!expr) {
    return;
  }

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *var_const =
        dynamic_cast<ir_sql_converter::SimplestVarConstComparison *>(expr);
    if (var_const && var_const->attr) {
      auto updated = UpdateAttrIndices(var_const->attr.get(), temp_table,
                                       old_table_indices);
      if (updated) {
        var_const->attr = std::move(updated);
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *var_cmp =
        dynamic_cast<ir_sql_converter::SimplestVarComparison *>(expr);
    if (var_cmp) {
      if (var_cmp->left_attr) {
        auto updated = UpdateAttrIndices(var_cmp->left_attr.get(), temp_table,
                                         old_table_indices);
        if (updated) {
          var_cmp->left_attr = std::move(updated);
        }
      }
      if (var_cmp->right_attr) {
        auto updated = UpdateAttrIndices(var_cmp->right_attr.get(), temp_table,
                                         old_table_indices);
        if (updated) {
          var_cmp->right_attr = std::move(updated);
        }
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *is_null = dynamic_cast<ir_sql_converter::SimplestIsNullExpr *>(expr);
    if (is_null && is_null->attr) {
      auto updated =
          UpdateAttrIndices(is_null->attr.get(), temp_table, old_table_indices);
      if (updated) {
        is_null->attr = std::move(updated);
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *var_param =
        dynamic_cast<ir_sql_converter::SimplestVarParamComparison *>(expr);
    if (var_param && var_param->attr) {
      auto updated = UpdateAttrIndices(var_param->attr.get(), temp_table,
                                       old_table_indices);
      if (updated) {
        var_param->attr = std::move(updated);
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *logical = dynamic_cast<ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (logical) {
      UpdateExprIndices(logical->left_expr.get(), temp_table,
                        old_table_indices);
      UpdateExprIndices(logical->right_expr.get(), temp_table,
                        old_table_indices);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *single_attr =
        dynamic_cast<ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    if (single_attr && single_attr->attr) {
      auto updated = UpdateAttrIndices(single_attr->attr.get(), temp_table,
                                       old_table_indices);
      if (updated) {
        single_attr->attr = std::move(updated);
      }
    }
    return;
  }
}

std::unique_ptr<ir_sql_converter::SimplestAttr>
IRQuerySplitter::UpdateAttrIndices(
    const ir_sql_converter::SimplestAttr *attr, const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!attr) {
    return nullptr;
  }

  unsigned int old_table_idx = attr->GetTableIndex();
  unsigned int old_col_idx = attr->GetColumnIndex();

  if (old_table_indices.find(old_table_idx) == old_table_indices.end()) {
    return nullptr;
  }

  int new_col_idx = temp_table.FindNewColumnIndex(old_table_idx, old_col_idx);

  if (new_col_idx < 0) {
    std::cerr << "[UpdateAttrIndices] Warning: Column [" << old_table_idx << "."
              << old_col_idx << "] (" << attr->GetColumnName()
              << ") not found in temp table mapping" << std::endl;
    return nullptr;
  }

  // Use the column name from column_mappings which matches SQL generator's
  // convention Format: {chunk_name}_{column_name}
  std::string new_col_name =
      temp_table.column_mappings[new_col_idx].column_name;

  auto new_attr = std::make_unique<ir_sql_converter::SimplestAttr>(
      attr->GetType(), temp_table.table_index,
      static_cast<unsigned int>(new_col_idx), new_col_name);

  if (config_.enable_debug_print) {
    std::cout << "[UpdateAttrIndices] Updated [" << old_table_idx << "."
              << old_col_idx << "] (" << attr->GetColumnName() << ") -> ["
              << temp_table.table_index << "." << new_col_idx << "] ("
              << new_col_name << ")" << std::endl;
  }

  return new_attr;
}

void IRQuerySplitter::UpdateNodeIndices(
    ir_sql_converter::AQPStmt *node, const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!node) {
    return;
  }

  // Update target_list
  for (size_t i = 0; i < node->target_list.size(); i++) {
    auto updated = UpdateAttrIndices(node->target_list[i].get(), temp_table,
                                     old_table_indices);
    if (updated) {
      node->target_list[i] = std::move(updated);
    }
  }

  // Update qual_vec
  for (auto &qual : node->qual_vec) {
    if (qual) {
      UpdateExprIndices(qual.get(), temp_table, old_table_indices);
    }
  }

  // Update join conditions
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(node);
    if (join) {
      for (auto &cond : join->join_conditions) {
        if (cond->left_attr) {
          auto updated = UpdateAttrIndices(cond->left_attr.get(), temp_table,
                                           old_table_indices);
          if (updated) {
            cond->left_attr = std::move(updated);
          }
        }
        if (cond->right_attr) {
          auto updated = UpdateAttrIndices(cond->right_attr.get(), temp_table,
                                           old_table_indices);
          if (updated) {
            cond->right_attr = std::move(updated);
          }
        }
      }
    }
  }

  // Update hash_keys
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::HashNode) {
    auto *hash = dynamic_cast<ir_sql_converter::SimplestHash *>(node);
    if (hash) {
      for (size_t i = 0; i < hash->hash_keys.size(); i++) {
        auto updated = UpdateAttrIndices(hash->hash_keys[i].get(), temp_table,
                                         old_table_indices);
        if (updated) {
          hash->hash_keys[i] = std::move(updated);
        }
      }
    }
  }

  // Update aggregate groups and functions
  if (node->GetNodeType() ==
      ir_sql_converter::SimplestNodeType::AggregateNode) {
    auto *agg = dynamic_cast<ir_sql_converter::SimplestAggregate *>(node);
    if (agg) {
      for (size_t i = 0; i < agg->groups.size(); i++) {
        auto updated = UpdateAttrIndices(agg->groups[i].get(), temp_table,
                                         old_table_indices);
        if (updated) {
          agg->groups[i] = std::move(updated);
        }
      }
      for (auto &fn_pair : agg->agg_fns) {
        auto updated = UpdateAttrIndices(fn_pair.first.get(), temp_table,
                                         old_table_indices);
        if (updated) {
          fn_pair.first = std::move(updated);
        }
      }
    }
  }

  // Update order by
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::OrderNode) {
    auto *order = dynamic_cast<ir_sql_converter::SimplestOrderBy *>(node);
    if (order) {
      for (auto &ord : order->orders) {
        auto updated =
            UpdateAttrIndices(ord.attr.get(), temp_table, old_table_indices);
        if (updated) {
          ord.attr = std::move(updated);
        }
      }
    }
  }

  // Recursively update children
  for (auto &child : node->children) {
    UpdateNodeIndices(child.get(), temp_table, old_table_indices);
  }
}

void IRQuerySplitter::UpdateRemainingIRIndices(
    ir_sql_converter::AQPStmt *remaining_ir, const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!remaining_ir) {
    return;
  }

  if (config_.enable_debug_print) {
    std::cout
        << "[UpdateRemainingIRIndices] Updating indices for executed tables: ";
    for (unsigned int idx : old_table_indices) {
      std::cout << idx << " ";
    }
    std::cout << "-> temp table index " << temp_table.table_index << std::endl;
  }

  UpdateNodeIndices(remaining_ir, temp_table, old_table_indices);
}

std::string
IRQuerySplitter::ComputeColumnAlias(unsigned int table_idx,
                                    const std::string &col_name) const {
  // SQL generator convention: {chunk_name}_{table_index}_{column_name}
  // Index included to disambiguate when same table appears multiple times
  std::string table_name = splitter_->GetTableName(table_idx);
  if (!table_name.empty()) {
    return table_name + "_" + std::to_string(table_idx) + "_" + col_name;
  }
  // Fallback: prefix with "t" so the alias never starts with a digit
  // (digits at the start cause SQL parse errors, e.g. table.5_col → ".5")
  return "t" + std::to_string(table_idx) + "_" + col_name;
}

// Strip trailing whitespace and semicolons from a SQL string
static std::string StripTrailingSemicolon(const std::string &sql) {
  size_t end = sql.size();
  while (end > 0 &&
         (sql[end - 1] == ';' || sql[end - 1] == ' ' || sql[end - 1] == '\n' ||
          sql[end - 1] == '\r' || sql[end - 1] == '\t')) {
    --end;
  }
  return sql.substr(0, end);
}

std::string IRQuerySplitter::BuildCombinedSQL(
    const std::vector<std::pair<std::string, std::string>> &sub_plans,
    const std::string &final_sql) const {
  std::string result;
  for (const auto &plan : sub_plans) {
    result += "CREATE TEMP TABLE " + plan.first + " AS\n";
    result += StripTrailingSemicolon(plan.second) + ";\n\n";
  }
  result += StripTrailingSemicolon(final_sql) + ";";
  return result;
}

#ifdef HAVE_DUCKDB
void IRQuerySplitter::CollectChunkNames(
    const ir_sql_converter::AQPStmt *node, std::set<std::string> &names) {
  if (!node)
    return;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk =
        static_cast<const ir_sql_converter::SimplestChunk *>(node);
    names.insert(chunk->GetChunkName());
    return;
  }
  for (const auto &child : node->children)
    CollectChunkNames(child.get(), names);
}

void IRQuerySplitter::EnsureKernelTempReady(const std::string &temp_name) {
  if (kernel_temp_ptrs_.count(temp_name))
    return;
  auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
  if (!duck)
    return;
  const auto *stored = duck->GetStoredTempResult(temp_name);
  if (!stored || !stored->collection)
    return;

  auto flat = std::make_unique<storage::FlatTable>();
  flat->table_name = temp_name;
  auto &coll = *stored->collection;
  flat->row_count = coll.Count();
  flat->column_names = stored->column_names;
  auto types = coll.Types();
  flat->columns.resize(types.size());

  for (size_t c = 0; c < types.size(); c++) {
    auto &col = flat->columns[c];
    col.row_count = flat->row_count;
    col.nullable = false;
    if (types[c] == duckdb::LogicalType::INTEGER ||
        types[c] == duckdb::LogicalType::BIGINT) {
      col.type = storage::FlatColumnType::INT32;
      col.data = std::unique_ptr<char[]>(
          new char[flat->row_count * sizeof(int32_t)]);
    } else {
      col.type = storage::FlatColumnType::VARCHAR;
    }
  }

  duckdb::ColumnDataScanState scan_state;
  coll.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  coll.InitializeScanChunk(chunk);
  uint64_t offset = 0;
  std::vector<std::vector<std::string>> varchar_data(types.size());

  while (coll.Scan(scan_state, chunk)) {
    auto count = chunk.size();
    chunk.Flatten();
    for (size_t c = 0; c < types.size(); c++) {
      auto &vec = chunk.data[c];
      auto &validity = duckdb::FlatVector::Validity(vec);
      if (flat->columns[c].type == storage::FlatColumnType::INT32) {
        auto *dst =
            reinterpret_cast<int32_t *>(flat->columns[c].data.get()) + offset;
        if (types[c] == duckdb::LogicalType::INTEGER) {
          auto *src = duckdb::FlatVector::GetData<int32_t>(vec);
          for (duckdb::idx_t r = 0; r < count; r++)
            dst[r] = validity.RowIsValid(r) ? src[r] : -1;
        } else {
          auto *src = duckdb::FlatVector::GetData<int64_t>(vec);
          for (duckdb::idx_t r = 0; r < count; r++)
            dst[r] = validity.RowIsValid(r) ? static_cast<int32_t>(src[r]) : -1;
        }
      } else {
        auto *src = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        for (duckdb::idx_t r = 0; r < count; r++) {
          if (validity.RowIsValid(r))
            varchar_data[c].emplace_back(src[r].GetData(), src[r].GetSize());
          else
            varchar_data[c].emplace_back();
        }
      }
    }
    offset += count;
  }

  for (size_t c = 0; c < types.size(); c++) {
    if (flat->columns[c].type != storage::FlatColumnType::VARCHAR)
      continue;
    auto &col = flat->columns[c];
    auto &strs = varchar_data[c];
    uint64_t total_len = 0;
    for (const auto &s : strs)
      total_len += s.size();
    col.data = std::unique_ptr<char[]>(
        new char[(flat->row_count + 1) * sizeof(uint32_t)]);
    col.string_pool = std::unique_ptr<char[]>(new char[total_len]);
    col.string_pool_size = total_len;
    auto *offsets = reinterpret_cast<uint32_t *>(col.data.get());
    uint32_t off = 0;
    for (uint64_t r = 0; r < flat->row_count; r++) {
      offsets[r] = off;
      std::memcpy(col.string_pool.get() + off, strs[r].data(), strs[r].size());
      off += static_cast<uint32_t>(strs[r].size());
    }
    offsets[flat->row_count] = off;
  }

  kernel_temp_ptrs_[temp_name] = flat.get();
  kernel_temps_[temp_name] = std::move(flat);

  // Async CSR: kick off background builds only for FK/PK join key columns
  static constexpr int32_t MAX_CSR_DOMAIN = 10'000'000;
  const auto *flat_ptr = kernel_temps_[temp_name].get();
  for (size_t c = 0; c < flat_ptr->columns.size(); c++) {
    if (flat_ptr->columns[c].type != storage::FlatColumnType::INT32)
      continue;
    const auto &alias = flat_ptr->column_names[c];
    std::string base_col;
    for (size_t p = 0; p < alias.size(); p++) {
      if (alias[p] == '_' && p + 1 < alias.size() &&
          std::isdigit(static_cast<unsigned char>(alias[p + 1]))) {
        size_t d = p + 1;
        while (d < alias.size() &&
               std::isdigit(static_cast<unsigned char>(alias[d])))
          d++;
        if (d < alias.size() && alias[d] == '_') {
          base_col = alias.substr(d + 1);
          break;
        }
      }
    }
    if (!base_col.empty() && storage_plan_ &&
        !storage_plan_->IsJoinKeyColumn(base_col))
      continue;
    int32_t max_val = 0;
    const auto *int_data =
        reinterpret_cast<const int32_t *>(flat_ptr->columns[c].data.get());
    for (uint64_t r = 0; r < flat_ptr->row_count; r++) {
      if (int_data[r] > max_val) max_val = int_data[r];
    }
    if (max_val > MAX_CSR_DOMAIN) continue;
    std::string csr_key = temp_name + "." + alias;
    std::string col_name = alias;
    int col_idx = static_cast<int>(c);
    async_csrs_[csr_key] = bg_pool_->Submit(
        [flat_ptr, col_idx, max_val, temp_name, col_name]() {
          return storage::BuildCSR(*flat_ptr, col_idx, max_val,
                                   temp_name, col_name, "", "");
        });
  }
}

void IRQuerySplitter::EnsureReferencedTempsReady(
    const ir_sql_converter::AQPStmt *ir) {
  std::set<std::string> chunk_names;
  CollectChunkNames(ir, chunk_names);
  for (const auto &name : chunk_names)
    EnsureKernelTempReady(name);
}

void IRQuerySplitter::EnsureKernelTempReadyNoCsr(const std::string &temp_name) {
  if (kernel_temp_ptrs_.count(temp_name))
    return;
  auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
  if (!duck)
    return;
  const auto *stored = duck->GetStoredTempResult(temp_name);
  if (!stored || !stored->collection)
    return;

  auto flat = std::make_unique<storage::FlatTable>();
  flat->table_name = temp_name;
  auto &coll = *stored->collection;
  flat->row_count = coll.Count();
  flat->column_names = stored->column_names;
  auto types = coll.Types();
  flat->columns.resize(types.size());

  for (size_t c = 0; c < types.size(); c++) {
    auto &col = flat->columns[c];
    col.row_count = flat->row_count;
    col.nullable = false;
    if (types[c] == duckdb::LogicalType::INTEGER ||
        types[c] == duckdb::LogicalType::BIGINT) {
      col.type = storage::FlatColumnType::INT32;
      col.data = std::unique_ptr<char[]>(
          new char[flat->row_count * sizeof(int32_t)]);
    } else {
      col.type = storage::FlatColumnType::VARCHAR;
    }
  }

  duckdb::ColumnDataScanState scan_state;
  coll.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  coll.InitializeScanChunk(chunk);
  uint64_t offset = 0;
  std::vector<std::vector<std::string>> varchar_data(types.size());

  while (coll.Scan(scan_state, chunk)) {
    auto count = chunk.size();
    chunk.Flatten();
    for (size_t c = 0; c < types.size(); c++) {
      auto &vec = chunk.data[c];
      auto &validity = duckdb::FlatVector::Validity(vec);
      if (flat->columns[c].type == storage::FlatColumnType::INT32) {
        auto *dst =
            reinterpret_cast<int32_t *>(flat->columns[c].data.get()) + offset;
        if (types[c] == duckdb::LogicalType::INTEGER) {
          auto *src = duckdb::FlatVector::GetData<int32_t>(vec);
          for (duckdb::idx_t r = 0; r < count; r++)
            dst[r] = validity.RowIsValid(r) ? src[r] : -1;
        } else {
          auto *src = duckdb::FlatVector::GetData<int64_t>(vec);
          for (duckdb::idx_t r = 0; r < count; r++)
            dst[r] = validity.RowIsValid(r) ? static_cast<int32_t>(src[r]) : -1;
        }
      } else {
        auto *src = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        for (duckdb::idx_t r = 0; r < count; r++) {
          if (validity.RowIsValid(r))
            varchar_data[c].emplace_back(src[r].GetData(), src[r].GetSize());
          else
            varchar_data[c].emplace_back();
        }
      }
    }
    offset += count;
  }

  for (size_t c = 0; c < types.size(); c++) {
    if (flat->columns[c].type != storage::FlatColumnType::VARCHAR)
      continue;
    auto &col = flat->columns[c];
    auto &strs = varchar_data[c];
    uint64_t total_len = 0;
    for (const auto &s : strs)
      total_len += s.size();
    col.data = std::unique_ptr<char[]>(
        new char[(flat->row_count + 1) * sizeof(uint32_t)]);
    col.string_pool = std::unique_ptr<char[]>(new char[total_len]);
    col.string_pool_size = total_len;
    auto *offsets = reinterpret_cast<uint32_t *>(col.data.get());
    uint32_t off = 0;
    for (uint64_t r = 0; r < flat->row_count; r++) {
      offsets[r] = off;
      std::memcpy(col.string_pool.get() + off, strs[r].data(), strs[r].size());
      off += static_cast<uint32_t>(strs[r].size());
    }
    offsets[flat->row_count] = off;
  }

  // No CSR build — pipeline kernel uses hash join tables instead
  kernel_temp_ptrs_[temp_name] = flat.get();
  kernel_temps_[temp_name] = std::move(flat);
}

void IRQuerySplitter::EnsureReferencedTempsReadyNoCsr(
    const ir_sql_converter::AQPStmt *ir) {
  std::set<std::string> chunk_names;
  CollectChunkNames(ir, chunk_names);
  for (const auto &name : chunk_names)
    EnsureKernelTempReadyNoCsr(name);
}
#endif

} // namespace middleware
