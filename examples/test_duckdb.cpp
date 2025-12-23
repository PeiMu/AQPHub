/*
 * Entrance for testing DuckDB Adapter
 * */

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb_adapter.h"
#include "util.h"
#include <iostream>

void test_single_query(middleware::DuckDBAdapter *adapter,
                       const std::string &sql_file_path,
                       middleware::TestResult &result) {
  result.query_file = middleware::get_filename(sql_file_path);
  result.success = false;
  result.num_rows = 0;

  try {
    // Read SQL file
    std::ifstream sql_file(sql_file_path);
    if (!sql_file.is_open()) {
      result.error_message = "Failed to open file";
      return;
    }

    std::stringstream buffer;
    buffer << sql_file.rdbuf();
    std::string sql = buffer.str();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing: " << result.query_file << std::endl;
#if DEBUG_MIDDLEWARE
    std::cout << "========================================" << std::endl;
    std::cout << "Original SQL:\n" << sql << std::endl;
#endif

    // Parse SQL
    auto parse_start = std::chrono::high_resolution_clock::now();
    adapter->ParseSQL(sql);
    auto parse_end = std::chrono::high_resolution_clock::now();
    result.parse_time_ms =
        std::chrono::duration<double, std::milli>(parse_end - parse_start)
            .count();

#if DEBUG_MIDDLEWARE
    std::cout << "\n=== SQL Parsed Successfully ===" << std::endl;
    std::cout << "Parse time: " << result.parse_time_ms << " ms" << std::endl;
#endif

    // Cast to LogicalOperator
    auto *logical_plan =
        static_cast<duckdb::LogicalOperator *>(adapter->GetLogicalPlan());
#if DEBUG_MIDDLEWARE
    std::cout << "\n=== Init Logical Plan ===" << std::endl;
    logical_plan->Print();
    std::cout << "===================\n" << std::endl;
#endif

    // Pre optimizer
    adapter->PreOptimizePlan();
    logical_plan =
        static_cast<duckdb::LogicalOperator *>(adapter->GetLogicalPlan());
#if DEBUG_MIDDLEWARE
    std::cout << "\n=== Logical Plan After Pre Optimizer ===" << std::endl;
    logical_plan->Print();
    std::cout << "===================\n" << std::endl;
#endif

    // Convert logical plan to IR
    auto ir_start = std::chrono::high_resolution_clock::now();
    auto simplest_ir = adapter->ConvertPlanToIR();
    auto ir_end = std::chrono::high_resolution_clock::now();
    result.ir_convert_time_ms =
        std::chrono::duration<double, std::milli>(ir_end - ir_start).count();

#if DEBUG_MIDDLEWARE
    std::cout << "\n=== Simplest IR ===" << std::endl;
    simplest_ir->Print();
    std::cout << "IR conversion time: " << result.ir_convert_time_ms << " ms"
              << std::endl;
#endif

    // Convert IR to SQL
    auto sql_gen_start = std::chrono::high_resolution_clock::now();
    std::string generated_sql = adapter->GenerateSQL(*simplest_ir, 1);
    auto sql_gen_end = std::chrono::high_resolution_clock::now();
    result.sql_gen_time_ms =
        std::chrono::duration<double, std::milli>(sql_gen_end - sql_gen_start)
            .count();

#if DEBUG_MIDDLEWARE
    std::cout << "\n=== Generated SQL ===" << std::endl;
    std::cout << generated_sql << std::endl;
    std::cout << "SQL generation time: " << result.sql_gen_time_ms << " ms"
              << std::endl;
#endif

    // Execute and verify
    auto exec_start = std::chrono::high_resolution_clock::now();
    auto query_result = adapter->ExecuteSQL(generated_sql);
    auto exec_end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(exec_end - exec_start)
            .count();

    result.num_rows = query_result.num_rows;

    std::cout << "\n=== Query Results ===" << std::endl;
    for (const auto &row : query_result.rows) {
      for (const auto &val : row) {
        std::cout << val << " ";
      }
      std::cout << std::endl;
    }

#if DEBUG_MIDDLEWARE
    std::cout << "Execution time: " << result.execution_time_ms << " ms"
              << std::endl;
#endif

    result.success = true;

  } catch (const std::exception &e) {
    result.error_message = e.what();
    result.success = false;
    std::cerr << "\nTest FAILED for " << result.query_file << std::endl;
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

int test_benchmark(middleware::DuckDBAdapter *adapter,
                   const std::string &benchmark_path) {
  // Get all .sql files
  std::vector<std::string> sql_files;
  try {
    sql_files = middleware::get_sql_files(benchmark_path);
  } catch (const std::exception &e) {
    std::cerr << "Error reading directory: " << e.what() << std::endl;
    return 1;
  }

  if (sql_files.empty()) {
    std::cerr << "No .sql files found in: " << benchmark_path << std::endl;
    return 1;
  }

  try {
    // Run tests on all queries
    std::vector<middleware::TestResult> results;
    results.reserve(sql_files.size());

    for (const auto &sql_file : sql_files) {
      middleware::TestResult result;
      test_single_query(adapter, sql_file, result);
      results.push_back(result);
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;

    return 1;
  }
}

int main() {
  // Create adapter
  auto adapter = std::make_unique<middleware::DuckDBAdapter>(
      "/home/pei/Project/duckdb/measure/imdb.db");

#if MEASURE_SINGLE_QUERY
  // test single query - example
  std::string single_query_path =
      "/home/pei/Project/benchmarks/imdb_job-postgres/queries/6d.sql";
  middleware::TestResult result;
  test_single_query(adapter.get(), single_query_path, result);
  std::cout << "\n========================================" << std::endl;
#else
  // test JOB benchmark
  std::string queries_dir =
      "/home/pei/Project/benchmarks/imdb_job-postgres/queries/";
  test_benchmark(adapter.get(), queries_dir);
#endif

  return 0;
}