/*
 * Configuration for query splitting strategies and backends
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace middleware {

enum class BackendEngine { DUCKDB, POSTGRESQL, UMBRA, MARIADB, OPENGAUSS, LINGODB, LINGODB_RUNTIME };

enum class SplitStrategy {
  NONE,                // No splitting - execute whole query directly
  TOP_DOWN,            // Top-down traversal split (pipeline breakers)
  MIN_SUBQUERY,        // FK-based: minimize subquery size
  RELATIONSHIP_CENTER, // FK-based: relationship-centric
  ENTITY_CENTER,       // FK-based: entity-centric
  NODE_BASED // DuckDB MiddleOptimize-driven, works with any execution adapter
};

enum class KernelPath {
  NONE,     // No kernel path — fall through to DuckDB
  PIPELINE, // Pipeline kernel (hash-based, no CSR)
  QUERY     // Query kernel (CSR-based)
};

struct ParamConfig {
  BackendEngine engine = BackendEngine::DUCKDB;
  SplitStrategy strategy = SplitStrategy::NONE;

  std::string db_path_or_connection; // Database path or connection string

  // Cost estimator (can differ from execution engine)
  // Default: same as engine. MariaDB default: POSTGRESQL.
  BackendEngine estimator_engine =
      BackendEngine::DUCKDB; // overridden in ParseFromArgs
  std::string helper_db;     // Connection string for the estimator engine
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
  bool print_sql = false;
  bool enable_correctness_check = false;
  bool enable_timing = false;
  bool enable_debug_print = false;
  bool enable_sub_plan_combiner = false;
  bool enable_explain = false; // Print EXPLAIN ANALYZE plan per sub-SQL

  // Measurement infrastructure
  int repeat_count = 1;      // --repeat=N: run query N times in-process
  bool in_memory = false;     // --in-memory: use :memory: + CSV loading
  std::string csv_dir;        // --csv-dir=<path>: CSV file directory

  // LingoDB execution mode: llvm (LLVM-optimized JIT) or tpde (TPDE fast codegen).
  // Set via: --lingodb-mode=llvm|tpde
  std::string lingodb_mode = "SPEED";

  // Storage plan: load flat column arrays + CSR indexes at startup
  bool enable_storage_plan = false;
  std::string storage_cache_path; // --storage-cache=<path>: binary cache file

  // Kernel threshold tuning
  bool enable_tuning = false;  // --tuning: log per-sub-query features + timing
  bool no_kernel = false;      // --no-kernel: force DuckDB path (for comparison)

  // Kernel path: which kernel implementation to use for sub-query execution.
  // NONE = no kernel (DuckDB only), PIPELINE = hash-based, QUERY = CSR-based.
  // Set via: --kernel-path=none|pipeline|query
  KernelPath kernel_path = KernelPath::NONE;

  // JIT compilation flags — bitmask of AQPJIT_* constants from aqp_jit_abi.h.
  // 0 = no JIT. Each level implies all lower levels for fallback.
  // Set via: --jit (=AQPJIT_EXPR), --jit-level=operator, --jit-simd
  uint32_t jit_flags = 0;

  // Per-optimization toggles (pipeline-jit and kernel-path).
  bool jit_payload_prune = true;
  bool jit_prefetch = true;
  int jit_prefetch_distance = 8;
  // Phase 6: ROF probe-side look-ahead distances (CompileFilterProbeProjectFusion).
  // entry = ht_entry_t cache-line look-ahead; row = build-row look-ahead.
  // 0 disables that level. Plan §10.4 defaults.
  int jit_prefetch_entry_distance = 24;
  int jit_prefetch_row_distance = 12;
  bool jit_batch_probe = true;
  int jit_skip_hash_cmp = 0; // 0=off, 1=single-int-key, 2=all-int-keys
  // JIT object cache mode:
  //   0 = off (no caching)
  //   1 = single-run-strict (in-memory, exact plan match, cleared between
  //       --repeat iterations so each iteration sees cold-start + within-run hits)
  //   2 = single-run-template (parameterized compilation: constants loaded from
  //       runtime params array, cache key strips constant values + column names,
  //       cleared between iterations)
  //   3 = full (persistent disk cache, survives process restart)
  // Default OFF so every repeat measures real compile time.
  int jit_cache = 0;
  // Compile mode: 0=llvm (full quality LLVM O2), 1=fastisel (LLVM O0+FastISel),
  // 2=tpde (TPDE fast codegen). Set via: --compile-mode=llvm|fastisel|tpde.
  // Spec-jit miss recompile always uses TPDE regardless of this setting.
  int compile_mode = 0;
  // Force the regular hash-table path (bypass perfect/array hash join) for
  // HJs that are members of a fused multi-probe chain, so the fused probe
  // function can dispatch. Mirrors lingo-db, which has no perfect-HJ path.
  bool single_col_int_join_mode = true;
  // Speculative JIT: 0=off, 1=recompile (TPDE on miss), 2=interpret (no
  // JIT on miss). The bg spec compile (always full quality LLVM O2) launches
  // early at post-Prepare and overlaps execute(i); miss recompile always uses
  // TPDE regardless of --compile-mode. Only takes effect for node-based +
  // DuckDB + non-zero jit level.
  int spec_jit = 0;
  // Cross-query latency hiding: prepare query N+1's first sub-query on a
  // background thread while query N executes. Only NODE_BASED + DuckDB.
  bool hide_latency_across_queries = false;
  // Query-jit (--jit-level=query) worker count; 0 = hardware_concurrency.
  // 1 runs the same outlined-morsel code path serially (debug/bisect knob).
  int query_jit_threads = 0;
  // Query-jit morsel size in rows (lingo-db uses 20000).
  int query_jit_morsel = 20000;

  // Per-subquery JIT config tuning: JSON file mapping (query, sub_idx)
  // to the best config label. Generated by tune_per_subquery.py.
  std::string tune_config_path;

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
    case BackendEngine::OPENGAUSS:
      return "OpenGauss";
    case BackendEngine::LINGODB:
      return "LingoDB";
    case BackendEngine::LINGODB_RUNTIME:
      return "LingoDB-Runtime";
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
    case BackendEngine::OPENGAUSS:
      return "OpenGauss";
    case BackendEngine::LINGODB:
      return "LingoDB";
    case BackendEngine::LINGODB_RUNTIME:
      return "LingoDB-Runtime";
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
    std::cout << "  Sub-Plan Combiner: "
              << (enable_sub_plan_combiner ? "enabled" : "disabled")
              << std::endl;
    if (repeat_count > 1)
      std::cout << "  Repeat: " << repeat_count << std::endl;
    if (in_memory)
      std::cout << "  In-Memory: enabled (csv_dir=" << csv_dir << ")" << std::endl;
    std::cout << "===========================" << std::endl;
  }
};

} // namespace middleware
