/*
 * IR-level query splitter main pipeline
 */

#include "split/ir_query_splitter.h"
#include "kernel/pipeline_kernel.h"
#include "jit/aqp_jit_abi.h"
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
} // namespace

IRQuerySplitter::IRQuerySplitter(EngineAdapter *adapter,
                                 const ParamConfig &config,
                                 storage::StoragePlan *storage_plan)
    : adapter_(adapter), storage_plan_(storage_plan), config_(config),
      bg_pool_(std::make_unique<ThreadPool>(1)) {

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
  bg_pool_.reset();
}

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
#ifdef HAVE_DUCKDB
  if (config_.engine == BackendEngine::DUCKDB) {
    auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
    if (duck)
      duck->ClearKernelTemps();
  }
#endif

  if (!config_.NeedsSplit() || !splitter_) {
    std::cout << "[IRQuerySplitter] No splitting needed, executing directly"
              << std::endl;
    return adapter_->ExecuteSQL(sql);
  }

  // === Phase 1: Parse SQL ===
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Phase 1: Parsing SQL" << std::endl;
  }
  std::chrono::high_resolution_clock::time_point timer;
  if (config_.enable_timing)
    timer = chrono_tic();
  // For NODE_BASED: parse SQL with the DuckDB helper adapter so it builds a
  // DuckDB logical plan.  All other strategies parse on the execution adapter.
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
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (parse_sql_time / 1000.0)
             << ", ";
    log_file.close();
  }

  // === Phase 2: Pre-Optimize (ONLY for DuckDB, or node-based split) ===
#ifdef HAVE_DUCKDB
  {
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
      log_file.open("time_log.csv", std::ios_base::app);
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
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Strategy Preprocessing" << std::endl;
  }
  std::chrono::high_resolution_clock::time_point timer;
  if (config_.enable_timing)
    timer = chrono_tic();
  splitter_->Preprocess(remaining_ir);
  if (config_.enable_timing) {
    auto preprocess_time = chrono_toc(&timer, "Preprocess time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (preprocess_time / 1000.0)
             << ", ";
    log_file.close();
  }

  // Main loop: while (graph has edges) { extract → execute → merge }
  while (!splitter_->IsComplete(remaining_ir.get())) {
    iteration_count_++;

    if (config_.enable_debug_print) {
      std::cout << "\n========== Iteration " << iteration_count_
                << " ==========" << std::endl;
    }

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
  if (storage_plan_ && storage_plan_->IsLoaded() &&
      (config_.jit_flags != 0 || config_.kernel_path != KernelPath::NONE) &&
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
        log_file.open("time_log.csv", std::ios_base::app);
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
        log_file.open("time_log.csv", std::ios_base::app);
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
      log_file.open("time_log.csv", std::ios_base::app);
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
        log_file.open("time_log.csv", std::ios_base::app);
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
        if (config_.jit_flags & AQP_JIT_OPT3)
          std::cerr << "OPT3 ";
        std::cerr << ") engine=" << (int)config_.engine << "\n";
      }
#ifdef HAVE_DUCKDB
#ifdef HAVE_LLVM
      {
        uint32_t duckdb_flags = config_.jit_flags & AQP_JIT_LEVEL_MASK;
        if (duckdb_flags && config_.engine == BackendEngine::DUCKDB) {
          auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
          if (config_.enable_debug_print) {
            std::cerr << "[AQP-JIT-TRACE] duck=" << (void *)duck
                      << " remaining_ir=" << (void *)remaining_ir.get() << "\n";
          }
          if (duck && remaining_ir)
            duck->SetJITPendingIR(remaining_ir.get(), duckdb_flags);
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

  return query_result;
}

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
  // todo: potential optimization - Push Partial Aggregation into Sub-IR
  auto extraction = splitter_->SplitIR(remaining_ir.get());
  if (config_.enable_timing) {
    auto extract_next_sub_sql_time =
        chrono_toc(&timer, "Extract next sub-SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
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
  if (storage_plan_ && storage_plan_->IsLoaded() &&
      (config_.jit_flags != 0 || config_.kernel_path != KernelPath::NONE) &&
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
          log_file.open("time_log.csv", std::ios_base::app);
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
          log_file.open("time_log.csv", std::ios_base::app);
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
          log_file.open("time_log.csv", std::ios_base::app);
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
          log_file.open("time_log.csv", std::ios_base::app);
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
          log_file.open("time_log.csv", std::ios_base::app);
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
          log_file.open("time_log.csv", std::ios_base::app);
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
      log_file.open("time_log.csv", std::ios_base::app);
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

    ApplyCrossSubPlanOptimizations(sub_sql);

#ifdef HAVE_DUCKDB
#ifdef HAVE_LLVM
    {
      uint32_t duckdb_flags = config_.jit_flags & AQP_JIT_LEVEL_MASK;
      if (duckdb_flags && config_.engine == BackendEngine::DUCKDB) {
        auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
        if (duck) {
          duck->SetTempColRanges({});
          duck->SetJITPendingIR(executable_ir, duckdb_flags);
        }
      }
    }
#endif
#endif

    std::chrono::high_resolution_clock::time_point duckdb_exe_start;
    if (config_.enable_tuning)
      duckdb_exe_start = std::chrono::high_resolution_clock::now();
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
    if (config_.enable_tuning) {
      log_exe_ms = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - duckdb_exe_start).count();
    }

    // 7.3b: Lazy CSR — removed eager FlatTable+CSR build here.
    // EnsureReferencedTempsReady() will build on demand before AnalyzeSubIR.
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
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (update_ir_time / 1000.0)
             << ", ";
    log_file.close();
  }

  return true;
}

TempTableInfo IRQuerySplitter::ExecuteSubIR(
    std::unique_ptr<ir_sql_converter::AQPStmt> sub_ir,
    const std::set<unsigned int> &executed_table_indices) {

  std::string temp_table_name = GenerateTempTableName();
  std::string sub_sql =
      adapter_->GenerateSQL(*sub_ir, adapter_->subquery_index++);

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
void IRQuerySplitter::ApplyCrossSubPlanOptimizations(std::string &sub_sql) {
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

    std::unordered_map<size_t, std::pair<int64_t, int64_t>> col_min_max;
    if (!tt.col_min_max.empty()) {
      col_min_max = tt.col_min_max;
    } else {
      col_min_max = duck->GetTempTableMinMax(tt.table_name);
    }

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
