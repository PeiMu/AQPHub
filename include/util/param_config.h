/*
 * Configuration for query splitting strategies and backends
 */

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace middleware {

enum class BackendEngine { DUCKDB, POSTGRESQL, UMBRA, MARIADB };

enum class SplitStrategy {
  NONE,                // No splitting - execute whole query directly
  TOP_DOWN,            // Top-down traversal split (pipeline breakers)
  MIN_SUBQUERY,        // FK-based: minimize subquery size
  RELATIONSHIP_CENTER, // FK-based: relationship-centric
  ENTITY_CENTER,       // FK-based: entity-centric
  NODE_BASED           // DuckDB MiddleOptimize-driven, works with any execution adapter
};

struct ParamConfig {
  BackendEngine engine = BackendEngine::DUCKDB;
  SplitStrategy strategy = SplitStrategy::NONE;

  std::string db_path_or_connection; // Database path or connection string

  // Cost estimator (can differ from execution engine)
  // Default: same as engine. MariaDB default: POSTGRESQL.
  BackendEngine estimator_engine =
      BackendEngine::DUCKDB; // overridden in ParseFromArgs
  std::string estimator_db;  // Connection string for the estimator engine
                             // (only needed when estimator_engine != engine)

  // Mode selection
  bool benchmark_mode = false; // false = single query, true = benchmark

  // Paths
  std::string query_path;  // Either single .sql file or directory for benchmark
  std::string schema_path; // Path to schema.sql for column index lookup
                           // (PostgreSQL only)
  std::string fkeys_path;  // Path to fkeys.sql for FK extraction from file

  // Options
  bool enable_reorder_get = true; // Only applies when strategy=TOP_DOWN
  bool enable_update_temp_card = true;
  bool enable_analyze = true;
  bool enable_correctness_check = false;
  bool enable_timing = false;
  bool enable_debug_print = false;

  // Parse configuration from command-line arguments
  static ParamConfig ParseFromArgs(int argc, char **argv);

  static void PrintUsage();

  // Utility functions
  std::string GetEngineName() const {
    switch (engine) {
    case BackendEngine::DUCKDB:
      return "DuckDB";
    case BackendEngine::POSTGRESQL:
      return "PostgreSQL";
    case BackendEngine::UMBRA:
      return "Umbra";
    case BackendEngine::MARIADB:
      return "MariaDB";
    default:
      return "Unknown";
    }
  }

  std::string GetStrategyName() const {
    switch (strategy) {
    case SplitStrategy::NONE:
      return "None";
    case SplitStrategy::TOP_DOWN:
      return "TopDown";
    case SplitStrategy::MIN_SUBQUERY:
      return "MinSubquery";
    case SplitStrategy::RELATIONSHIP_CENTER:
      return "RelationshipCenter";
    case SplitStrategy::ENTITY_CENTER:
      return "EntityCenter";
    case SplitStrategy::NODE_BASED:
      return "NodeBased";
    default:
      return "Unknown";
    }
  }

  bool UseCustomEstimator() const { return estimator_engine != engine; }

  std::string GetEstimatorName() const {
    if (!UseCustomEstimator())
      return GetEngineName();
    switch (estimator_engine) {
    case BackendEngine::DUCKDB:
      return "DuckDB";
    case BackendEngine::POSTGRESQL:
      return "PostgreSQL";
    case BackendEngine::UMBRA:
      return "Umbra";
    case BackendEngine::MARIADB:
      return "MariaDB";
    default:
      return "Unknown";
    }
  }

  bool NeedsSplit() const { return strategy != SplitStrategy::NONE; }

  bool NeedsReorderGet() const {
    return strategy == SplitStrategy::TOP_DOWN && enable_reorder_get;
  }

  bool NeedsForeignKeys() const {
    return strategy == SplitStrategy::MIN_SUBQUERY ||
           strategy == SplitStrategy::RELATIONSHIP_CENTER ||
           strategy == SplitStrategy::ENTITY_CENTER;
  }

  void Print() const {
    std::cout << "=== Split Configuration ===" << std::endl;
    std::cout << "  Engine: " << GetEngineName() << std::endl;
    std::cout << "  Estimator: " << GetEstimatorName() << std::endl;
    std::cout << "  Strategy: " << GetStrategyName() << std::endl;
    std::cout << "  ReorderGet: "
              << (NeedsReorderGet() ? "enabled" : "disabled") << std::endl;
    std::cout << "  Correctness Check: "
              << (enable_correctness_check ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  Timing: " << (enable_timing ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  Debug Print: "
              << (enable_debug_print ? "enabled" : "disabled") << std::endl;
    std::cout << "===========================" << std::endl;
  }
};

} // namespace middleware
