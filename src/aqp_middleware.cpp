/*
 * Unified AQP Middleware Entry Point
 * Supports both DuckDB and PostgreSQL backends with configurable split
 * strategies
 */

#include "adapters/db_adapter.h"
#include "split/ir_query_splitter.h"
#include "storage/storage_plan.h"
#include "util/param_config.h"
#include "util/util.h"

// Include both adapters (conditionally compiled based on availability)
#ifdef HAVE_DUCKDB
#include "adapters/duckdb_adapter.h"
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
  return buffer.str();
}

// Execute single query with timing and result collection
void ExecuteSingleQuery(EngineAdapter *adapter, const std::string &sql_file_path,
                        const ParamConfig &config, TestResult &result,
                        middleware::storage::StoragePlan *storage_plan = nullptr) {
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
    if (config.enable_timing)
      timer = chrono_tic();
    // Read SQL file
    std::string sql = ReadSQLFile(sql_file_path);
    if (config.enable_timing) {
      auto read_sql_time = chrono_toc(&timer, "Read SQL time is\n", false);
      // save time to a file
      std::ofstream log_file;
      log_file.open(g_timing_log_name, std::ios_base::app);
      log_file << std::fixed << std::setprecision(3) << (read_sql_time / 1000.0)
               << ", ";
      log_file.close();
    }

    if (config.print_sql || config.enable_debug_print) {
      std::cout << "========================================" << std::endl;
      std::cout << "Original SQL:\n" << sql << std::endl;
    }

    QueryResult query_result;

    if (config.NeedsSplit()) {
      // Execute with split strategy using IRQuerySplitter
      if (config.enable_debug_print) {
        std::cout << "\n=== Execution with Split Strategy: "
                  << config.GetStrategyName() << " ===" << std::endl;
      }

      // Create IRQuerySplitter with the selected strategy
      IRQuerySplitter splitter(adapter, config, storage_plan);

      // Pass query name for per-subquery tune config lookup
      {
        std::string qname = get_filename(sql_file_path);
        auto dot = qname.rfind('.');
        if (dot != std::string::npos)
          qname = qname.substr(0, dot);
        splitter.SetQueryName(qname);
      }

      // Execute with split
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
        duckdb_adp->SetCompileMode(config.compile_mode);
        duckdb_adp->SetJITProbePrefetchDistances(
            config.jit_prefetch_entry_distance,
            config.jit_prefetch_row_distance);
        duckdb_adp->SetQueryJit((config.jit_flags & AQP_JIT_QUERY_JIT) != 0,
                                config.query_jit_threads,
                                config.query_jit_morsel);
        duckdb_adp->SetQueryJitStoragePlan(storage_plan);
      }
#endif

      query_result = adapter->ExecuteSQL(sql);
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
  bool is_lingodb = (config.engine == BackendEngine::LINGODB);

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

      TestResult result;
      ExecuteSingleQuery(adapter, sql_file, config, result, storage_plan);
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
         config.engine == BackendEngine::LINGODB) &&
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
#ifdef HAVE_DUCKDB
        if (config.engine != BackendEngine::DUCKDB) {
          throw std::runtime_error(
              "--storage-plan without a pre-built cache requires --engine=duckdb. "
              "Build the cache first with: --engine=duckdb --storage-cache=<path>");
        }
        auto *duck = dynamic_cast<DuckDBAdapter *>(adapter.get());
        storage_plan_ptr->LoadFromDuckDB(duck->GetConnection());
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
#else
        throw std::runtime_error(
            "--storage-plan requires a pre-built cache (--storage-cache=<path>)");
#endif
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

    return return_code;

  } catch (const std::exception &e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    return 1;
  }
}
