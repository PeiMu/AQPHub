/*
 * Unified AQP Middleware Entry Point
 * Supports both DuckDB and PostgreSQL backends with configurable split
 * strategies
 */

#include "adapters/db_adapter.h"
#include "split/ir_query_splitter.h"
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

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace middleware;

// Factory function to create the appropriate adapter based on config
std::unique_ptr<DBAdapter> CreateAdapter(const ParamConfig &config) {
  switch (config.engine) {
#if defined(HAVE_DUCKDB)
  case BackendEngine::DUCKDB: {
    if (config.enable_debug_print) {
      std::cout << "[AQP Middleware] Creating DuckDB adapter: "
                << config.db_path_or_connection << std::endl;
    }
    return std::make_unique<DuckDBAdapter>(config.db_path_or_connection);
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
void ExecuteSingleQuery(DBAdapter *adapter, const std::string &sql_file_path,
                        const ParamConfig &config, TestResult &result) {
  result.query_file = get_filename(sql_file_path);
  result.success = false;
  result.num_rows = 0;

  try {
    if (config.enable_debug_print) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Testing: " << result.query_file << std::endl;
    }

    // Read SQL file
    std::string sql = ReadSQLFile(sql_file_path);

    if (config.enable_debug_print) {
      std::cout << "========================================" << std::endl;
      std::cout << "Original SQL:\n" << sql << std::endl;
    }

    QueryResult query_result;
    auto exec_start = std::chrono::high_resolution_clock::now();

    if (config.NeedsSplit()) {
      // Execute with split strategy using IRQuerySplitter
      if (config.enable_debug_print) {
        std::cout << "\n=== Execution with Split Strategy: "
                  << config.GetStrategyName() << " ===" << std::endl;
      }

      // Create IRQuerySplitter with the selected strategy
      IRQuerySplitter splitter(adapter, config);

      // Execute with split
      query_result = splitter.ExecuteWithSplit(sql);

    } else {
      // Direct execution (no split)
      if (config.enable_debug_print) {
        std::cout << "\n=== Direct Execution (No Splitting) ===" << std::endl;
      }

      query_result = adapter->ExecuteSQL(sql);
    }

    auto exec_end = std::chrono::high_resolution_clock::now();

    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(exec_end - exec_start)
            .count();
    result.num_rows = query_result.num_rows;

    std::cout << "\n=== Query Results ===" << std::endl;
    std::cout << "Rows: " << query_result.num_rows
              << ", Columns: " << query_result.num_columns << std::endl;
    for (const auto &row : query_result.rows) {
      for (const auto &val : row) {
        std::cout << val << "|";
      }
      std::cout << std::endl;
    }

    if (config.enable_timing) {
      std::cout << "Execution time: " << result.execution_time_ms << " ms"
                << std::endl;
    }

    result.success = true;

  } catch (const std::exception &e) {
    result.error_message = e.what();
    result.success = false;
    std::cerr << "\nTest FAILED for " << result.query_file << std::endl;
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

// Run benchmark on all queries in a directory
int RunBenchmark(DBAdapter *adapter, const ParamConfig &config) {
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

  // Run tests on all queries
  std::vector<TestResult> results;
  results.reserve(sql_files.size());

  int passed = 0;
  int failed = 0;

  for (const auto &sql_file : sql_files) {
    std::cout << "Run " + sql_file << std::endl;
    TestResult result;
    ExecuteSingleQuery(adapter, sql_file, config, result);
    results.push_back(result);

    if (result.success) {
      passed++;
    } else {
      failed++;
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

#ifdef HAVE_POSTGRES
    // Initialize schema parser for PostgreSQL (needed for correct column
    // indices)
    if ((config.engine == BackendEngine::POSTGRESQL ||
         config.engine == BackendEngine::UMBRA) &&
        !config.schema_path.empty()) {
      if (!ir_sql_converter::InitSchemaParser(config.schema_path)) {
        std::cerr << "Warning: Failed to load schema, column indices will be 0"
                  << std::endl;
      }
    }
#endif

    // Create adapter
    auto adapter = CreateAdapter(config);

    // Execute based on mode
    int return_code = 0;

    if (config.benchmark_mode) {
      return_code = RunBenchmark(adapter.get(), config);
    } else {
      TestResult result;
      ExecuteSingleQuery(adapter.get(), config.query_path, config, result);
      return_code = result.success ? 0 : 1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Execution completed" << std::endl;
    std::cout << "========================================" << std::endl;

#ifdef HAVE_POSTGRES
    // Cleanup schema parser
    ir_sql_converter::CleanupSchemaParser();
#endif

    return return_code;

  } catch (const std::exception &e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    return 1;
  }
}
