/*
 * Unified AQP Middleware Entry Point
 * Supports both DuckDB and PostgreSQL backends with configurable split
 * strategies
 */

#include <unistd.h>

#include "adapters/db_adapter.h"
#include "split/ir_query_splitter.h"
#include "storage/storage_plan.h"
#include "util/param_config.h"
#include "util/util.h"

// Include both adapters (conditionally compiled based on availability)
#ifdef HAVE_DUCKDB
#include "adapters/duckdb_adapter.h"
#ifdef HAVE_LLVM
#include "jit/ir_to_llvm.h"
#endif
#endif

#ifdef HAVE_LLVM
#include <llvm/Support/ManagedStatic.h>
#endif

#ifdef HAVE_POSTGRES
#include "adapters/postgres_adapter.h"
#endif

#ifdef HAVE_UMBRA
#include "adapters/umbra_adapter.h"
#endif

#ifdef HAVE_MARIADB
#include "adapters/mariadb_adapter.h"
#endif

#ifdef HAVE_OPENGAUSS
#include "adapters/opengauss_adapter.h"
#endif

#ifdef HAVE_LINGODB
#include "adapters/lingodb_adapter.h"
#include "adapters/lingodb_runtime_adapter.h"
#endif

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

using namespace middleware;

// Factory function to create the appropriate adapter based on config
std::unique_ptr<EngineAdapter> CreateAdapter(const ParamConfig &config) {
  switch (config.engine) {
#if defined(HAVE_DUCKDB)
  case BackendEngine::DUCKDB: {
    std::string db_path = config.in_memory ? ":memory:" : config.db_path_or_connection;
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating DuckDB adapter: "
                << db_path << std::endl;
    }
    auto adapter = std::make_unique<DuckDBAdapter>(db_path);
    if (config.in_memory) {
      adapter->LoadTablesFromCSV(config.schema_path, config.csv_dir);
    }
    return adapter;
  }
#endif

#if defined(HAVE_POSTGRES)
  case BackendEngine::POSTGRESQL: {
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating PostgreSQL adapter: "
                << config.db_path_or_connection << std::endl;
    }
    return std::make_unique<PostgreSQLAdapter>(config.db_path_or_connection);
  }
#endif

#if defined(HAVE_UMBRA)
  case BackendEngine::UMBRA: {
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating Umbra adapter: "
                << config.db_path_or_connection << std::endl;
    }
    return std::make_unique<UmbraAdapter>(config.db_path_or_connection);
  }
#endif

#if defined(HAVE_OPENGAUSS)
  case BackendEngine::OPENGAUSS: {
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating OpenGauss adapter: "
                << config.db_path_or_connection << std::endl;
    }
    return std::make_unique<OpenGaussAdapter>(config.db_path_or_connection);
  }
#endif

#if defined(HAVE_MARIADB)
  case BackendEngine::MARIADB: {
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating MariaDB adapter: "
                << config.db_path_or_connection << std::endl;
      if (config.UseCustomEstimator()) {
        std::cout << "[AQP Middleware] MariaDB estimator: "
                  << config.GetEstimatorName() << " (" << config.helper_db
                  << ")" << std::endl;
      }
    }
    if (config.strategy == SplitStrategy::NODE_BASED) {
      return std::make_unique<MariaDBAdapter>(config.db_path_or_connection, "");
    } else {
      return std::make_unique<MariaDBAdapter>(config.db_path_or_connection,
                                              config.helper_db);
    }
  }
#endif

#if defined(HAVE_LINGODB)
  case BackendEngine::LINGODB: {
    std::string db_path = config.in_memory ? ":memory:" : config.db_path_or_connection;
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating LingoDB adapter: "
                << db_path << std::endl;
    }
    auto adapter = std::make_unique<LingoDBAdapter>(db_path);
    adapter->SetExecutionMode(config.lingodb_mode);
    if (config.in_memory) {
      adapter->LoadTablesFromCSV(config.schema_path, config.csv_dir);
    }
    return adapter;
  }
  case BackendEngine::LINGODB_RUNTIME: {
    std::string db_path = config.in_memory ? ":memory:" : config.db_path_or_connection;
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating LingoDB-Runtime adapter: "
                << db_path << std::endl;
    }
    auto adapter = std::make_unique<LingoDBRuntimeAdapter>(db_path);
    adapter->SetExecutionMode(config.lingodb_mode);
    if (config.in_memory) {
      adapter->LoadTablesFromCSV(config.schema_path, config.csv_dir);
    }
    return adapter;
  }
#endif

  default:
    throw std::runtime_error("Backend engine not available. "
                             "Rebuild with support for " +
                             config.GetEngineName());
  }
}

std::string ReadSQLFile(const std::string &file_path) {
  std::ifstream sql_file(file_path);
  if (!sql_file.is_open()) {
    throw std::runtime_error("Failed to open SQL file: " + file_path);
  }

  std::stringstream buffer;
  buffer << sql_file.rdbuf();
  std::string sql = buffer.str();
  size_t start = sql.find_first_not_of(" \t\n\r");
  if (start != std::string::npos && sql.size() - start >= 16) {
    std::string head = sql.substr(start, 16);
    std::transform(head.begin(), head.end(), head.begin(), ::tolower);
    if (head == "explain analyze ")
      sql.erase(start, 16);
  }
  return sql;
}

// Execute single query with timing and result collection
void ExecuteSingleQuery(
    EngineAdapter *adapter, const std::string &sql_file_path,
    const ParamConfig &config, TestResult &result,
    middleware::storage::StoragePlan *storage_plan = nullptr,
    std::unique_ptr<CrossQueryPrepResult> cross_prep = nullptr) {
  result.query_file = get_filename(sql_file_path);
  result.success = false;
  result.num_rows = 0;

  try {
    if (config.enable_debug_print) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Testing: " << result.query_file << std::endl;
    }
    adapter->enable_timing_ = config.enable_timing;

    std::chrono::high_resolution_clock::time_point timer;
    std::string sql;
    if (cross_prep && cross_prep->success) {
      sql = cross_prep->sql;
      if (config.enable_timing) {
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << "0.000, "; // read_sql placeholder (done by bg thread)
        log_file.close();
      }
    } else {
      if (config.enable_timing)
        timer = chrono_tic();
      sql = ReadSQLFile(sql_file_path);
      if (config.enable_timing) {
        auto read_sql_time = chrono_toc(&timer, "Read SQL time is\n", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (read_sql_time / 1000.0) << ", ";
        log_file.close();
      }
    }

    if (config.print_sql || config.enable_debug_print) {
      std::cout << "========================================" << std::endl;
      std::cout << "Original SQL:\n" << sql << std::endl;
    }

    QueryResult query_result;

    // Extract query name (used by both split and AUTO dispatch)
    std::string qname = get_filename(sql_file_path);
    {
      auto dot = qname.rfind('.');
      if (dot != std::string::npos)
        qname = qname.substr(0, dot);
    }

    // Per-query split strategy resolution (AUTO mode)
    ParamConfig local_config = config;
    if (config.strategy == SplitStrategy::AUTO) {
      local_config.strategy = IRQuerySplitter::ResolveTuneSplit(
          config.tune_config_path, qname, SplitStrategy::NODE_BASED);
      if (config.enable_debug_print)
        std::cerr << "[AUTO] " << qname << ": split="
                  << local_config.GetStrategyName() << "\n";
    }

    if (local_config.NeedsSplit()) {
      if (config.enable_debug_print) {
        std::cout << "\n=== Execution with Split Strategy: "
                  << local_config.GetStrategyName() << " ===" << std::endl;
      }

      IRQuerySplitter splitter(adapter, local_config, storage_plan);
      splitter.SetQueryName(qname);

      if (cross_prep)
        splitter.SetCrossQueryPrep(std::move(cross_prep));

      query_result = splitter.ExecuteWithSplit(sql);

    } else {
      // Direct execution (no split)
      if (config.enable_debug_print) {
        std::cout << "\n=== Direct Execution (No Splitting) ===" << std::endl;
      }

#ifdef HAVE_LLVM
      // Pass JIT flags to the adapter so ExecuteSQL can trigger the JIT path
      // even without a splitter.
      auto *duckdb_adp = dynamic_cast<DuckDBAdapter *>(adapter);
      if (duckdb_adp) {
        duckdb_adp->SetJITFlags(config.jit_flags);
        duckdb_adp->SetKernelPath(config.kernel_path);
        duckdb_adp->SetJITDebug(config.enable_debug_print);
        duckdb_adp->SetJITOptFlags(
            config.jit_payload_prune,
            config.jit_prefetch, config.jit_prefetch_distance,
            config.jit_batch_probe, config.jit_skip_hash_cmp,
            config.single_col_int_join_mode);
        duckdb_adp->SetBenchmarkMode(config.benchmark_mode);
        duckdb_adp->SetJITCache(config.jit_cache);
        duckdb_adp->SetJITCacheDir(config.jit_cache_dir);
        duckdb_adp->SetCompileMode(config.compile_mode);
        duckdb_adp->SetJITProbePrefetchDistances(
            config.jit_prefetch_entry_distance,
            config.jit_prefetch_row_distance);
        duckdb_adp->SetQueryJit((config.jit_flags & AQP_JIT_QUERY_JIT) != 0,
                                config.query_jit_threads,
                                config.query_jit_morsel);
        duckdb_adp->SetQueryJitStoragePlan(storage_plan);
        duckdb_adp->SetRangeGuard(config.range_guard);
        duckdb_adp->SetBlockSkip(config.block_skip);
        duckdb_adp->SetMembershipPreprobe(config.membership_preprobe);
        duckdb_adp->SetDisableBidirectionalStorage(config.disable_bidirectional_storage);
        duckdb_adp->SetDisableEngineOptimizer(config.disable_engine_optimizer);
      }
#endif
#ifdef HAVE_POSTGRES
      auto *pg_adp = dynamic_cast<PostgreSQLAdapter *>(adapter);
      if (pg_adp) {
#ifdef HAVE_LLVM
        pg_adp->SetQueryJit((config.jit_flags & AQP_JIT_QUERY_JIT) != 0,
                            config.query_jit_threads, config.query_jit_morsel);
        pg_adp->SetQueryJitStoragePlan(storage_plan);
        pg_adp->SetCompileMode(config.compile_mode);
        pg_adp->SetSkipHashCmp(config.jit_skip_hash_cmp);
        pg_adp->SetJitFlags(config.jit_flags);
        pg_adp->SetJITCache(config.jit_cache);
        pg_adp->SetJITCacheDir(config.jit_cache_dir);
        pg_adp->SetJITDebug(config.enable_debug_print);
        pg_adp->SetJITPrefetch(config.jit_prefetch,
                               config.jit_prefetch_distance);
#endif
      }
#endif

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
      bool replay_hit = false;
      bool was_recording = false;
      bool is_query_jit = duckdb_adp &&
                          (config.jit_flags & AQP_JIT_QUERY_JIT) != 0;
      if (config.jit_cache >= 3 && is_query_jit &&
          config.engine == BackendEngine::DUCKDB) {
        std::string qname = get_filename(sql_file_path);
        auto dot = qname.rfind('.');
        if (dot != std::string::npos)
          qname = qname.substr(0, dot);
        auto &cache = DuckDBAdapter::QueryPlanCache();
        auto it = cache.find(qname);
        if (it != cache.end() && it->second.final_query.is_query_jit) {
          auto &sub = it->second.final_query;
          if (sub.is_interpreter_fallback) {
            // No compiled fn — fall through to normal ExecuteSQL
          } else {
            auto replay_timer = chrono_tic();
            query_result = duckdb_adp->ReplayQjitFinal(sub);
            replay_hit = true;
            if (config.enable_timing) {
              auto replay_us =
                  chrono_toc(&replay_timer, "split=none replay\n", false);
              std::ofstream log_file;
              log_file.open(g_timing_log_name, std::ios_base::app);
              // jit_compile=0, execute=measured (replay does no compilation)
              log_file << std::fixed << std::setprecision(3)
                       << "0.000, " << (replay_us / 1000.0) << ", ";
              log_file.close();
            }
          }
        } else {
          was_recording = true;
          duckdb_adp->BeginPlanRecording();
        }
      }
      if (!replay_hit) {
#endif
#if defined(HAVE_LINGODB) && defined(HAVE_DUCKDB)
        if (config.engine == BackendEngine::LINGODB_RUNTIME) {
          auto *rt_adapter =
              dynamic_cast<LingoDBRuntimeAdapter *>(adapter);
          if (!rt_adapter)
            throw std::runtime_error(
                "LINGODB_RUNTIME engine requires LingoDBRuntimeAdapter");

          DuckDBAdapter helper(config.helper_db);
          helper.ParseSQL(sql);
          helper.FilterOptimize();
          auto ir = helper.ConvertPlanToIR();
          if (!ir)
            throw std::runtime_error(
                "[lingo-db-runtime] Failed to convert DuckDB plan to IR");

          if (config.enable_debug_print) {
            std::string ir_sql =
                ir_sql_converter::ConvertIRToSQL(*ir, 0);
            std::cout << "DuckDB-optimized IR SQL:\n"
                      << ir_sql << std::endl;
          }

          query_result = rt_adapter->ExecuteIRQuery(*ir);
        } else
#endif
        {
          query_result = adapter->ExecuteSQL(sql);
        }
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
      }
      if (was_recording && duckdb_adp) {
        auto &recording = duckdb_adp->GetPlanRecording();
        std::string qname = get_filename(sql_file_path);
        auto dot = qname.rfind('.');
        if (dot != std::string::npos)
          qname = qname.substr(0, dot);
        CachedQueryPlan plan;
        if (!recording.empty()) {
          plan.final_query = std::move(recording[0]);
        } else {
          plan.final_query.sql = sql;
          plan.final_query.is_interpreter_fallback = true;
        }
        DuckDBAdapter::QueryPlanCache()[qname] = std::move(plan);
        duckdb_adp->EndPlanRecording();
      }
#endif
    }
    result.num_rows = query_result.num_rows;

    if (config.enable_timing)
      timer = chrono_tic();
    std::cout << "\n=== Query Results ===" << std::endl;
    std::cout << "Rows: " << query_result.num_rows
              << ", Columns: " << query_result.num_columns << std::endl;
    for (const auto &row : query_result.rows) {
      for (const auto &val : row) {
        std::cout << val << "|";
      }
      std::cout << std::endl;
    }

    result.success = true;

    if (config.enable_timing) {
      auto show_output_time =
          chrono_toc(&timer, "Show output time is\n", false);
      // save time to a file
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (show_output_time / 1000.0) << "\n";
      log_file.close();
    }

  } catch (const std::exception &e) {
    result.error_message = e.what();
    result.success = false;
    std::cerr << "\nTest FAILED for " << result.query_file << std::endl;
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

// Run benchmark on all queries in a directory
static void MergeIterationLogs(const std::vector<std::string> &iter_files,
                               const std::string &out_file) {
  // Per-iteration files have: "Running benchmark for .../Xa.sql...\n" header
  // then data rows. Merge them so all iterations of the same query are grouped:
  //   Running benchmark for .../10a.sql...
  //   <iter0 row>
  //   <iter1 row>
  //   ...
  //   Running benchmark for .../10b.sql...
  //   ...
  using Rows = std::vector<std::string>;
  std::map<std::string, Rows> per_query;       // header -> data rows (ordered)
  std::vector<std::string> query_order;        // preserve first-seen order

  for (const auto &path : iter_files) {
    std::ifstream in(path);
    if (!in.is_open()) continue;
    std::string cur_header;
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("Running", 0) == 0) {
        cur_header = line;
        if (per_query.find(cur_header) == per_query.end())
          query_order.push_back(cur_header);
        continue;
      }
      if (!cur_header.empty() && !line.empty())
        per_query[cur_header].push_back(line);
    }
  }

  std::ofstream out(out_file);
  for (const auto &hdr : query_order) {
    out << hdr << "\n";
    for (const auto &row : per_query[hdr])
      out << row << "\n";
  }

  for (const auto &path : iter_files)
    std::remove(path.c_str());
}

int RunBenchmark(EngineAdapter *adapter, const ParamConfig &config,
                 middleware::storage::StoragePlan *storage_plan = nullptr) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Running Benchmark: " << config.query_path << std::endl;
  std::cout << "========================================" << std::endl;

  // Get all .sql files
  std::vector<std::string> sql_files;
  try {
    sql_files = get_sql_files(config.query_path);
  } catch (const std::exception &e) {
    std::cerr << "Error reading directory: " << e.what() << std::endl;
    return 1;
  }

  if (sql_files.empty()) {
    std::cerr << "No .sql files found in: " << config.query_path << std::endl;
    return 1;
  }

  std::cout << "Found " << sql_files.size() << " SQL file(s)" << std::endl;

  std::vector<TestResult> results;
  results.reserve(sql_files.size() * config.repeat_count);

  int passed = 0;
  int failed = 0;
  bool first_run = true;

  std::vector<std::string> iter_log_files;
  std::vector<std::string> iter_ldb_files;
  bool is_lingodb = (config.engine == BackendEngine::LINGODB ||
                     config.engine == BackendEngine::LINGODB_RUNTIME);

  for (int iter = 0; iter < config.repeat_count; iter++) {
    std::string iter_log = "time_log_iter" + std::to_string(iter) + ".csv";
    iter_log_files.push_back(iter_log);

    if (config.enable_timing) {
      g_timing_log_name = iter_log;
      if (is_lingodb) {
        std::string ldb_log = "lingodb_compile_iter" + std::to_string(iter) + ".csv";
        iter_ldb_files.push_back(ldb_log);
        g_lingodb_compile_log_name = ldb_log;
      }
    }

    std::cout << "\n--- Iteration " << iter << " ---" << std::endl;

#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
    if (iter > 0 && config.jit_cache && config.jit_cache < 3)
      aqp_jit::IrToLlvmCompiler::ClearObjCache();
#endif

    // §7.2: Cross-query latency hiding — bg thread pool + pending future
    std::unique_ptr<ThreadPool> cross_query_pool;
    std::future<std::unique_ptr<CrossQueryPrepResult>> pending_cross_prep;
#ifdef HAVE_DUCKDB
    DuckDBAdapter *duck_for_cross =
        !config.no_cross_query_prep &&
                config.jit_cache >= 1 &&
                (config.strategy == SplitStrategy::NODE_BASED ||
                 config.strategy == SplitStrategy::TOP_DOWN ||
                 config.strategy == SplitStrategy::AUTO) &&
                config.engine == BackendEngine::DUCKDB &&
                !(config.jit_cache >= 3 && iter > 0)
            ? dynamic_cast<DuckDBAdapter *>(adapter)
            : nullptr;
    if (duck_for_cross)
      cross_query_pool = std::make_unique<ThreadPool>(1);
#if defined(HAVE_LLVM)
    // Ping-pong compilers for Phase 2: bg_prep(N+1) resets one compiler
    // while ExecuteSingleQuery(N) uses code from the other.
    std::unique_ptr<aqp_jit::IrToLlvmCompiler> cross_compilers[2];
    int cross_compiler_idx = 0;
#endif
#endif
#ifdef HAVE_POSTGRES
    bool pg_for_cross =
        !config.no_cross_query_prep &&
        !cross_query_pool &&
        config.jit_cache >= 1 &&
        (config.strategy == SplitStrategy::TOP_DOWN ||
         config.strategy == SplitStrategy::AUTO) &&
        config.engine == BackendEngine::POSTGRESQL &&
        !(config.jit_cache >= 3 && iter > 0);
    if (pg_for_cross)
      cross_query_pool = std::make_unique<ThreadPool>(1);
#endif

    for (size_t qi = 0; qi < sql_files.size(); qi++) {
      const auto &sql_file = sql_files[qi];
      std::cout << "Run " + sql_file << std::endl;
      if (config.enable_timing) {
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << "Running benchmark for " << sql_file << "...\n";
        log_file.close();
        if (is_lingodb) {
          std::ofstream ldb_log;
          ldb_log.open(g_lingodb_compile_log_name, std::ios_base::app);
          ldb_log << "Running benchmark for " << sql_file << "...\n";
          ldb_log << "frontend, QOpt, lowerRelAlg, lowerSubOp, lowerDB, "
                     "lowerArrow, lowerToLLVM, toLLVMIR, llvmOptimize, "
                     "llvmCodeGen, baselineLowering, baselineCodeGen, "
                     "baselineEmit, executionTime, total\n";
          ldb_log.close();
        }
      }

      std::chrono::high_resolution_clock::time_point timer;
      if (config.enable_timing)
        timer = chrono_tic();

      // Consume bg prep from previous query (if any)
      std::unique_ptr<CrossQueryPrepResult> cross_prep;
      if (pending_cross_prep.valid()) {
        cross_prep = pending_cross_prep.get();
        if (!cross_prep || !cross_prep->success) {
          if (config.enable_debug_print && cross_prep)
            std::cerr << "[CROSS-QUERY] prep failed: " << cross_prep->error
                      << "\n";
          cross_prep.reset();
        }
      }

      if (!first_run)
        adapter->ResetQueryState();
      first_run = false;
      if (config.enable_timing) {
        auto reset_time = chrono_toc(&timer, "", false);
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (reset_time / 1000.0) << ", ";
        log_file.close();
      }

      // Launch bg prep for next query (before executing current)
#ifdef HAVE_DUCKDB
      if (duck_for_cross && qi + 1 < sql_files.size()) {
        auto &db_ref = duck_for_cross->GetDB();
        const std::string &next_file = sql_files[qi + 1];
        bool debug = config.enable_debug_print;
        std::string next_qname;
        {
          auto sl = next_file.rfind('/');
          next_qname = (sl != std::string::npos)
                           ? next_file.substr(sl + 1) : next_file;
          auto dt = next_qname.rfind('.');
          if (dt != std::string::npos) next_qname = next_qname.substr(0, dt);
        }
#if defined(HAVE_LLVM)
        auto &bg_comp_slot = cross_compilers[cross_compiler_idx];
        cross_compiler_idx ^= 1;
        auto [eff_flags, eff_cm] =
            IRQuerySplitter::ResolveTuneFlags(config, next_qname, 0);
        SplitStrategy next_strategy = config.strategy;
        if (config.strategy == SplitStrategy::AUTO)
          next_strategy = IRQuerySplitter::ResolveTuneSplit(
              config.tune_config_path, next_qname, SplitStrategy::NODE_BASED);
        if (next_strategy == SplitStrategy::NONE) {
          // Next query uses direct execution; no cross-query prep needed.
        } else if (next_strategy == SplitStrategy::TOP_DOWN) {
          pending_cross_prep = cross_query_pool->Submit(
              [&next_file, &db_ref, duck_for_cross, &config, &bg_comp_slot,
               eff_flags, eff_cm]() {
                return IRQuerySplitter::PrepareNextQueryTopDown(
                    next_file, db_ref, duck_for_cross, config, bg_comp_slot,
                    eff_flags, eff_cm);
              });
        } else {
          pending_cross_prep = cross_query_pool->Submit(
              [&next_file, &db_ref, duck_for_cross, &config, &bg_comp_slot,
               eff_flags, eff_cm]() {
                return IRQuerySplitter::PrepareNextQuery(
                    next_file, db_ref, duck_for_cross, config, bg_comp_slot,
                    eff_flags, eff_cm);
              });
        }
#else
        SplitStrategy next_strategy = config.strategy;
        if (config.strategy == SplitStrategy::AUTO)
          next_strategy = IRQuerySplitter::ResolveTuneSplit(
              config.tune_config_path, next_qname, SplitStrategy::NODE_BASED);
        if (next_strategy == SplitStrategy::NONE) {
          // Next query uses direct execution; no cross-query prep needed.
        } else if (next_strategy == SplitStrategy::TOP_DOWN) {
          pending_cross_prep = cross_query_pool->Submit(
              [&next_file, &db_ref, duck_for_cross, &config]() {
                return IRQuerySplitter::PrepareNextQueryTopDown(
                    next_file, db_ref, duck_for_cross, config);
              });
        } else {
          pending_cross_prep = cross_query_pool->Submit(
              [&next_file, &db_ref, duck_for_cross, &config]() {
                return IRQuerySplitter::PrepareNextQuery(
                    next_file, db_ref, duck_for_cross, config);
              });
        }
#endif
      }
#endif
#ifdef HAVE_POSTGRES
      if (pg_for_cross && qi + 1 < sql_files.size()) {
        const std::string &next_file = sql_files[qi + 1];
        bool skip_pg_prep = false;
        if (config.strategy == SplitStrategy::AUTO) {
          std::string nq;
          {
            auto sl = next_file.rfind('/');
            nq = (sl != std::string::npos) ? next_file.substr(sl + 1)
                                           : next_file;
            auto dt = nq.rfind('.');
            if (dt != std::string::npos) nq = nq.substr(0, dt);
          }
          auto ns = IRQuerySplitter::ResolveTuneSplit(
              config.tune_config_path, nq, SplitStrategy::TOP_DOWN);
          skip_pg_prep = (ns == SplitStrategy::NONE);
        }
        if (!skip_pg_prep) {
          pending_cross_prep = cross_query_pool->Submit(
              [&next_file, adapter, &config]() {
                return IRQuerySplitter::PrepareNextQueryTopDownPG(
                    next_file, adapter, config);
              });
        }
      }
#endif

      TestResult result;
      ExecuteSingleQuery(adapter, sql_file, config, result, storage_plan,
                         std::move(cross_prep));
      results.push_back(result);

      if (result.success) {
        passed++;
      } else {
        failed++;
      }
    }
  }

  if (config.enable_timing) {
    g_timing_log_name = "time_log.csv";
    MergeIterationLogs(iter_log_files, g_timing_log_name);
    if (is_lingodb) {
      g_lingodb_compile_log_name = "lingodb_compile_time.csv";
      MergeIterationLogs(iter_ldb_files, g_lingodb_compile_log_name);
    }
  }

  // Print summary
  std::cout << "\n========================================" << std::endl;
  std::cout << "Benchmark Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Total queries: " << sql_files.size() << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;

  if (config.enable_timing) {
    double total_time = 0.0;
    for (const auto &result : results) {
      if (result.success) {
        total_time += result.execution_time_ms;
      }
    }
    std::cout << "Total execution time: " << total_time << " ms" << std::endl;
    std::cout << "Average time per query: "
              << (passed > 0 ? total_time / passed : 0.0) << " ms" << std::endl;
  }

  std::cout << "========================================" << std::endl;

  return failed > 0 ? 1 : 0;
}

int main(int argc, char **argv) {
  try {
    std::chrono::high_resolution_clock::time_point timer = chrono_tic();

    // Parse configuration
    auto config = ParamConfig::ParseFromArgs(argc, argv);

    if (config.query_path.empty()) {
      std::cerr << "Error: No query file or directory specified" << std::endl;
      ParamConfig::PrintUsage();
      return 1;
    }

    if (config.enable_debug_print) {
      // Print configuration
      std::cout << "========================================" << std::endl;
      std::cout << "AQP Middleware" << std::endl;
      std::cout << "========================================" << std::endl;
      config.Print();
    }

#if defined(HAVE_POSTGRES) || defined(HAVE_MARIADB)
    // Initialize schema parser (needed for correct column indices)
    if ((config.engine == BackendEngine::POSTGRESQL ||
         config.engine == BackendEngine::UMBRA ||
         config.engine == BackendEngine::MARIADB ||
         config.engine == BackendEngine::OPENGAUSS ||
         config.engine == BackendEngine::LINGODB ||
         config.engine == BackendEngine::LINGODB_RUNTIME) &&
        !config.schema_path.empty()) {
      if (!ir_sql_converter::InitSchemaParser(config.schema_path)) {
        std::cerr << "Warning: Failed to load schema, column indices will be 0"
                  << std::endl;
      }
    }
#endif

    // Create adapter
    auto adapter = CreateAdapter(config);

    // Load storage plan (flat column arrays + CSR indexes) if requested
    std::unique_ptr<middleware::storage::StoragePlan> storage_plan_ptr;
    middleware::storage::StoragePlan *storage_plan = nullptr;
    if (config.enable_storage_plan) {
      storage_plan_ptr = std::make_unique<middleware::storage::StoragePlan>();
      // CSR/sorted/inverted indexes are kernel-path-only; query-jit (the only
      // other storage-plan consumer) reads FlatTable columns exclusively.
      bool need_indexes = config.kernel_path != KernelPath::NONE;
      bool loaded_from_cache = false;
      if (!config.storage_cache_path.empty()) {
        loaded_from_cache = storage_plan_ptr->LoadFromFile(
            config.storage_cache_path, /*skip_indexes=*/!need_indexes);
      }
      if (!loaded_from_cache) {
#ifdef HAVE_POSTGRES
        if (config.engine == BackendEngine::POSTGRESQL) {
          auto *pg = dynamic_cast<PostgreSQLAdapter *>(adapter.get());
          storage_plan_ptr->LoadFromPostgreSQL(pg->GetConnection());
        } else
#endif
#ifdef HAVE_DUCKDB
        if (config.engine == BackendEngine::DUCKDB) {
          auto *duck = dynamic_cast<DuckDBAdapter *>(adapter.get());
          storage_plan_ptr->LoadFromDuckDB(duck->GetConnection());
        } else
#endif
        {
          throw std::runtime_error(
              "--storage-plan without a pre-built cache requires "
              "--engine=duckdb or --engine=postgresql. "
              "Build the cache first with: --storage-cache=<path>");
        }
        if (need_indexes) {
          if (!config.fkeys_path.empty()) {
            storage_plan_ptr->BuildCSRIndexes(config.fkeys_path);
          }
          storage_plan_ptr->BuildSortedIndices();
          storage_plan_ptr->BuildInvertedIndices();
        }
        // Without indexes SaveToFile writes empty index sections; a later
        // kernel-path run against this cache rebuilds them (below). Keep
        // kernel-path and query-jit on separate --storage-cache paths to
        // avoid repeated rebuilds.
        if (!config.storage_cache_path.empty()) {
          storage_plan_ptr->SaveToFile(config.storage_cache_path);
        }
      } else if (need_indexes) {
        // Trimmed cache (built by a query-jit run) detection: rebuild the
        // index sections kernel-path needs.
        if (storage_plan_ptr->GetCSRMap().empty() &&
            !config.fkeys_path.empty()) {
          storage_plan_ptr->BuildCSRIndexes(config.fkeys_path);
        }
        if (storage_plan_ptr->GetSortedIndicesMap().empty()) {
          storage_plan_ptr->BuildSortedIndices();
        }
        if (storage_plan_ptr->GetInvertedIndicesMap().empty()) {
          storage_plan_ptr->BuildInvertedIndices();
        }
      }
      if (config.enable_debug_print) {
        storage_plan_ptr->PrintSummary();
      }
      storage_plan = storage_plan_ptr.get();
    }

    if (config.enable_timing) {
      auto prepare_middleware_time =
          chrono_toc(&timer, "Prepare Middleware time is\n", false);
      if (!config.benchmark_mode) {
        std::ofstream log_file;
        log_file.open(g_timing_log_name, std::ios_base::app);
        log_file << std::fixed << std::setprecision(3)
                 << (prepare_middleware_time / 1000.0) << ", ";
        log_file.close();
      }
    }

    // Execute based on mode
    int return_code = 0;
    if (config.benchmark_mode) {
      return_code = RunBenchmark(adapter.get(), config, storage_plan);
    } else {
      for (int iter = 0; iter < config.repeat_count; iter++) {
        if (iter > 0) {
#if defined(HAVE_DUCKDB) && defined(HAVE_LLVM)
          if (config.jit_cache && config.jit_cache < 3)
            aqp_jit::IrToLlvmCompiler::ClearObjCache();
#endif
          if (config.enable_timing)
            timer = chrono_tic();
          adapter->ResetQueryState();
          if (config.enable_timing) {
            auto reset_time = chrono_toc(&timer, "", false);
            std::ofstream log_file;
            log_file.open(g_timing_log_name, std::ios_base::app);
            log_file << std::fixed << std::setprecision(3)
                     << (reset_time / 1000.0) << ", ";
            log_file.close();
          }
        }

        TestResult result;
        ExecuteSingleQuery(adapter.get(), config.query_path, config, result,
                           storage_plan);
        return_code = result.success ? 0 : 1;
      }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Execution completed" << std::endl;
    std::cout << "========================================" << std::endl;

#if defined(HAVE_POSTGRES) || defined(HAVE_MARIADB)
    // Cleanup schema parser
    ir_sql_converter::CleanupSchemaParser();
#endif

    adapter.reset();

#ifdef HAVE_LLVM
    llvm::llvm_shutdown();
#endif

    // _exit() skips __cxa_finalize which double-frees inside libLLVM.so.20
    _exit(return_code);

  } catch (const std::exception &e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    _exit(1);
  }
}
