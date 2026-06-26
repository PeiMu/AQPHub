/*
 * Implementation of split configuration parsing and backends
 */

#include "util/param_config.h"
#include "jit/aqp_jit_abi.h"
#include <algorithm>
#include <cctype>

namespace middleware {

// Helper to convert string to lowercase
static std::string to_lower(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

ParamConfig ParamConfig::ParseFromArgs(int argc, char **argv) {
  ParamConfig config;

  for (int i = 1; i < argc - 1; i++) {
    std::string arg = argv[i];

    // Parse --engine=<value>
    if (arg.find("--engine=") == 0) {
      std::string engine_str = to_lower(arg.substr(9));
      if (engine_str == "duckdb") {
        config.engine = BackendEngine::DUCKDB;
      } else if (engine_str == "postgres" || engine_str == "postgresql") {
        config.engine = BackendEngine::POSTGRESQL;
      } else if (engine_str == "umbra") {
        config.engine = BackendEngine::UMBRA;
      } else if (engine_str == "mariadb") {
        config.engine = BackendEngine::MARIADB;
      } else if (engine_str == "opengauss") {
        config.engine = BackendEngine::OPENGAUSS;
      } else if (engine_str == "lingodb" || engine_str == "lingo-db") {
        config.engine = BackendEngine::LINGODB;
      } else if (engine_str == "lingo-db-runtime" || engine_str == "lingodb-runtime") {
        config.engine = BackendEngine::LINGODB_RUNTIME;
      } else {
        throw std::runtime_error(
            "Unknown engine: " + arg.substr(9) +
            " (valid: duckdb, postgres, umbra, mariadb, opengauss, lingodb)");
      }
    }
    // Parse --db=<value>
    else if (arg.find("--db=") == 0) {
      config.db_path_or_connection = arg.substr(5);
    }
    // Parse --schema=<value> (for PostgreSQL column index lookup)
    else if (arg.find("--schema=") == 0) {
      config.schema_path = arg.substr(9);
    }
    // Parse --fkeys=<value> (for FK extraction from file)
    else if (arg.find("--fkeys=") == 0) {
      config.fkeys_path = arg.substr(8);
    }
    // Parse --estimator=<engine> (which engine's optimizer to use for EXPLAIN)
    else if (arg.find("--estimator=") == 0) {
      std::string est_str = to_lower(arg.substr(12));
      if (est_str == "duckdb") {
        config.estimator_engine = BackendEngine::DUCKDB;
      } else if (est_str == "postgres" || est_str == "postgresql") {
        config.estimator_engine = BackendEngine::POSTGRESQL;
      } else if (est_str == "umbra") {
        config.estimator_engine = BackendEngine::UMBRA;
      } else if (est_str == "mariadb") {
        config.estimator_engine = BackendEngine::MARIADB;
      } else if (est_str == "opengauss") {
        config.estimator_engine = BackendEngine::OPENGAUSS;
      } else {
        throw std::runtime_error(
            "Unknown estimator engine: " + arg.substr(12) +
            " (valid: duckdb, postgres, umbra, mariadb, opengauss)");
      }
    }
    // Parse --helper-db-path=<connection> (connection string for estimator
    // engine, or duckdb database path when using node-based split)
    else if (arg.find("--helper-db-path=") == 0) {
      config.helper_db = arg.substr(17);
    }
    // Parse --split=<value>
    else if (arg.find("--split=") == 0) {
      std::string strategy_str = to_lower(arg.substr(8));
      if (strategy_str == "none") {
        config.strategy = SplitStrategy::NONE;
      } else if (strategy_str == "topdown" || strategy_str == "top_down") {
        config.strategy = SplitStrategy::TOP_DOWN;
      } else if (strategy_str == "minsubquery" ||
                 strategy_str == "min-subquery") {
        config.strategy = SplitStrategy::MIN_SUBQUERY;
      } else if (strategy_str == "relationshipcenter" ||
                 strategy_str == "relationship-center") {
        config.strategy = SplitStrategy::RELATIONSHIP_CENTER;
      } else if (strategy_str == "entitycenter" ||
                 strategy_str == "entity-center") {
        config.strategy = SplitStrategy::ENTITY_CENTER;
      } else if (strategy_str == "nodebased" || strategy_str == "node_based" ||
                 strategy_str == "node-based") {
        config.strategy = SplitStrategy::NODE_BASED;
      } else {
        throw std::runtime_error(
            "Unknown split strategy: " + arg.substr(8) +
            " (valid: none, topdown, minsubquery, "
            "relationship-center, entity-center, node-based)");
      }
    }
    // Parse boolean flags
    else if (arg == "--benchmark") {
      config.benchmark_mode = true;
    } else if (arg == "--no-reorder-get") {
      config.enable_reorder_get = false;
    } else if (arg == "--no-update-temp-card") {
      config.enable_update_temp_card = false;
    } else if (arg == "--no-analyze") {
      config.enable_analyze = false;
    } else if (arg == "--print-sql") {
      config.print_sql = true;
    } else if (arg == "--check-correctness") {
      config.enable_correctness_check = true;
    } else if (arg == "--timing") {
      config.enable_timing = true;
    } else if (arg == "--tuning") {
      config.enable_tuning = true;
    } else if (arg == "--no-kernel") {
      config.no_kernel = true;
    } else if (arg == "--debug") {
      config.enable_debug_print = true;
    } else if (arg == "--combine-sub-plans") {
      config.enable_sub_plan_combiner = true;
    } else if (arg == "--jit") {
      // Backward-compatible: --jit alone enables expression-level JIT
      config.jit_flags |= AQP_JIT_EXPR;
    } else if (arg.find("--jit-level=") == 0) {
      std::string level = to_lower(arg.substr(12));
      if (level == "none") {
        config.jit_flags |= AQP_JIT_NONE;
      } else if (level == "expr") {
        config.jit_flags |= AQP_JIT_EXPR;
      } else if (level == "operator") {
        config.jit_flags |= AQP_JIT_EXPR | AQP_JIT_OPERATOR;
      } else if (level == "pipeline") {
        config.jit_flags |= AQP_JIT_EXPR | AQP_JIT_OPERATOR | AQP_JIT_PIPELINE_JIT;
      } else if (level == "query") {
        config.jit_flags |= AQP_JIT_QUERY_JIT;
      } else {
        throw std::runtime_error(
            "Unknown JIT level: " + arg.substr(12) +
            " (valid: none, expr, operator, pipeline, query)");
      }
    } else if (arg.find("--kernel-path=") == 0) {
      std::string kp = to_lower(arg.substr(14));
      if (kp == "none") {
        config.kernel_path = KernelPath::NONE;
      } else if (kp == "pipeline") {
        config.kernel_path = KernelPath::PIPELINE;
      } else if (kp == "query") {
        config.kernel_path = KernelPath::QUERY;
      } else {
        throw std::runtime_error(
            "Unknown kernel path: " + arg.substr(14) +
            " (valid: none, pipeline, query)");
      }
    } else if (arg.find("--jit-simd=") == 0) {
      std::string simd = to_lower(arg.substr(11));
      config.jit_flags &= ~AQP_JIT_SIMD_MASK;
      if (simd == "off" || simd == "none")
        config.jit_flags |= AQP_JIT_SIMD_OFF;
      else if (simd == "sse2")
        config.jit_flags |= AQP_JIT_SIMD_SSE2;
      else if (simd == "avx")
        config.jit_flags |= AQP_JIT_SIMD_AVX;
      else if (simd == "avx2")
        config.jit_flags |= AQP_JIT_SIMD_AVX2;
      else if (simd == "avx512")
        config.jit_flags |= AQP_JIT_SIMD_AVX512;
      else if (simd == "auto")
        config.jit_flags |= AQP_JIT_SIMD_AUTO;
      else
        throw std::runtime_error(
            "Unknown SIMD level: " + simd +
            " (valid: off, sse2, avx, avx2, avx512, auto)");
    } else if (arg == "--jit-simd") {
      config.jit_flags = (config.jit_flags & ~AQP_JIT_SIMD_MASK) | AQP_JIT_SIMD_AUTO;
    }
    // Per-optimization toggles
    else if (arg == "--jit-payload-prune") {
      config.jit_payload_prune = true;
    } else if (arg == "--no-jit-payload-prune") {
      config.jit_payload_prune = false;
    } else if (arg == "--jit-prefetch") {
      config.jit_prefetch = true;
    } else if (arg.find("--jit-prefetch=") == 0) {
      config.jit_prefetch = true;
      config.jit_prefetch_distance = std::stoi(arg.substr(15));
    } else if (arg == "--no-jit-prefetch") {
      config.jit_prefetch = false;
    } else if (arg.find("--jit-prefetch-entry-dist=") == 0) {
      config.jit_prefetch_entry_distance = std::stoi(arg.substr(26));
    } else if (arg.find("--jit-prefetch-row-dist=") == 0) {
      config.jit_prefetch_row_distance = std::stoi(arg.substr(24));
    } else if (arg == "--jit-batch-probe") {
      config.jit_batch_probe = true;
    } else if (arg == "--no-jit-batch-probe") {
      config.jit_batch_probe = false;
    } else if (arg == "--jit-skip-hash-cmp=off") {
      config.jit_skip_hash_cmp = 0;
    } else if (arg == "--jit-skip-hash-cmp=single") {
      config.jit_skip_hash_cmp = 1;
    } else if (arg == "--jit-skip-hash-cmp=all") {
      config.jit_skip_hash_cmp = 2;
    } else if (arg == "--jit-skip-hash-cmp") {
      config.jit_skip_hash_cmp = 2; // bare flag = all (legacy compat)
    } else if (arg == "--jit-cache" || arg == "--jit-cache=single-run" ||
               arg == "--jit-cache=single-run-strict") {
      config.jit_cache = 1;
    } else if (arg == "--jit-cache=single-run-template") {
      config.jit_cache = 2;
    } else if (arg == "--jit-cache=full") {
      config.jit_cache = 3;
    } else if (arg == "--no-jit-cache") {
      config.jit_cache = 0;
    } else if (arg.substr(0, 16) == "--jit-cache-dir=") {
      config.jit_cache_dir = arg.substr(16);
    } else if (arg == "--compile-mode=fastisel") {
      config.compile_mode = 1;
    } else if (arg == "--compile-mode=tpde") {
      config.compile_mode = 2;
    } else if (arg == "--compile-mode=llvm" || arg == "--compile-mode=off") {
      config.compile_mode = 0;
    } else if (arg == "--single-column-int-join-mode") {
      config.single_col_int_join_mode = true;
    } else if (arg == "--no-single-column-int-join-mode") {
      config.single_col_int_join_mode = false;
    } else if (arg == "--spec-jit=recompile") {
      config.spec_jit = 1;
    } else if (arg == "--spec-jit=interpret") {
      config.spec_jit = 2;
    } else if (arg == "--spec-jit=off" || arg == "--no-spec-jit") {
      config.spec_jit = 0;
    } else if (arg.find("--query-jit-threads=") == 0) {
      config.query_jit_threads = std::stoi(arg.substr(20));
      if (config.query_jit_threads < 0)
        throw std::runtime_error("--query-jit-threads must be >= 0");
    } else if (arg.find("--query-jit-morsel=") == 0) {
      config.query_jit_morsel = std::stoi(arg.substr(19));
      if (config.query_jit_morsel < 1)
        throw std::runtime_error("--query-jit-morsel must be >= 1");
    } else if (arg.find("--repeat=") == 0) {
      config.repeat_count = std::stoi(arg.substr(9));
      if (config.repeat_count < 1)
        throw std::runtime_error("--repeat must be >= 1");
    } else if (arg == "--storage-plan") {
      config.enable_storage_plan = true;
    } else if (arg.find("--storage-cache=") == 0) {
      config.storage_cache_path = arg.substr(16);
    } else if (arg.find("--tune-config=") == 0) {
      config.tune_config_path = arg.substr(14);
    } else if (arg == "--in-memory") {
      config.in_memory = true;
    } else if (arg.find("--lingodb-mode=") == 0) {
      std::string mode = to_lower(arg.substr(15));
      if (mode == "llvm") {
        config.lingodb_mode = "SPEED";
      } else if (mode == "tpde") {
        config.lingodb_mode = "BASELINE_SPEED";
      } else {
        throw std::runtime_error(
            "Unknown lingodb mode: " + arg.substr(15) +
            " (valid: llvm, tpde)");
      }
    } else if (arg.find("--csv-dir=") == 0) {
      config.csv_dir = arg.substr(10);
    } else if (arg == "--explain") {
      config.enable_explain = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      exit(0);
    }
    // Unknown argument - could be non-split related, just warn
    else if (arg.find("--") == 0) {
      std::cerr << "Warning: Unknown argument: " << arg << std::endl;
    }
  }

  config.query_path = argv[argc - 1];

  if (config.kernel_path != KernelPath::NONE)
    config.enable_storage_plan = true;

  if ((config.jit_flags & AQP_JIT_PIPELINE_JIT) &&
      config.kernel_path != KernelPath::NONE)
    throw std::runtime_error(
        "--jit-level=pipeline and --kernel-path are mutually exclusive: "
        "pipeline-jit compiles probe chains in the DuckDB execution path, "
        "while kernel-path routes through PipelineKernel");

  if (config.jit_flags & AQP_JIT_QUERY_JIT) {
    if (config.kernel_path != KernelPath::NONE)
      throw std::runtime_error(
          "--jit-level=query and --kernel-path are mutually exclusive: "
          "query-jit owns scan-to-sink execution, kernel-path routes through "
          "PipelineKernel");
    if (config.jit_flags & AQP_JIT_LEVEL_MASK)
      throw std::runtime_error(
          "--jit-level=query cannot be combined with other --jit-level "
          "values: query-jit bypasses the DuckDB-embedded JIT path");
    // Query-jit scans base tables through the storage plan's flat tables.
    config.enable_storage_plan = true;
  }

  if (config.engine == BackendEngine::LINGODB_RUNTIME &&
      config.helper_db.empty())
    throw std::runtime_error(
        "--engine=lingo-db-runtime requires --helper-db-path=<path> "
        "(DuckDB database for query optimization)");

  if (config.in_memory) {
    if (config.engine != BackendEngine::DUCKDB && config.engine != BackendEngine::LINGODB &&
        config.engine != BackendEngine::LINGODB_RUNTIME)
      throw std::runtime_error("--in-memory is only supported with --engine=duckdb, --engine=lingodb, or --engine=lingo-db-runtime");
    if (config.csv_dir.empty() && !config.schema_path.empty()) {
      auto pos = config.schema_path.find_last_of('/');
      config.csv_dir = (pos != std::string::npos ? config.schema_path.substr(0, pos) : ".") + "/csv";
    }
    if (config.csv_dir.empty())
      throw std::runtime_error("--in-memory requires --csv-dir=<path> or --schema=<path>");
  }

  if (config.jit_cache >= 3 && config.jit_cache_dir.empty())
    config.jit_cache_dir = "/dev/shm/aqp_jit_cache/v1";

  return config;
}
void ParamConfig::PrintUsage() {
  std::cout << "Usage: AQP_middleware [options]" << std::endl;
  std::cout << "\nOptions:" << std::endl;
  std::cout
      << "  --engine=<duckdb|postgres|umbra|mariadb|opengauss>  Backend engine "
         "(default: duckdb)"
      << std::endl;
  std::cout
      << "  --split=<strategy>               Split strategy (default: none)"
      << std::endl;
  std::cout << "  --schema=<path>                  Schema SQL file for column "
               "index lookup (PostgreSQL/Umbra)"
            << std::endl;
  std::cout << "  --fkeys=<path>                   FK constraints SQL file "
               "(for engines without information_schema)"
            << std::endl;
  std::cout << "  --estimator=<engine>             Engine to use for cost "
               "estimation (default: own engine)"
            << std::endl;
  std::cout << "  --helper-db-path=<conn>            Connection string for the "
               "helper engine"
            << std::endl;
  std::cout << "    Strategies: none, topdown, minsubquery, "
               "relationship-center, entity-center"
            << std::endl;
  std::cout << "  --no-reorder-get                 Disable ReorderGet for "
               "TopDown (default: enabled)"
            << std::endl;
  std::cout << "  --no-update-temp-card            Disable updating "
               "cardinality for temp table (default: enabled)"
            << std::endl;
  std::cout << "  --no-analyze                     Disable ANALYZE in "
               "PostgreSQL adapter"
            << std::endl;
  std::cout << "  --print-sql                      Print vanilla SQL and "
               "generated sub-SQLs"
            << std::endl;
  std::cout << "  --check-correctness              Enable correctness "
               "checking (default: disabled)"
            << std::endl;
  std::cout << "  --no-timing                      Disable timing "
               "measurements (default: enabled)"
            << std::endl;
  std::cout << "  --debug                          Enable debug output "
               "(default: disabled)"
            << std::endl;
  std::cout << "  --combine-sub-plans              Collect all sub-SQLs and "
               "print combined CTE at end (default: disabled)"
            << std::endl;
  std::cout << "  --jit                            JIT-compile filter expressions "
               "(same as --jit-level=expr)"
            << std::endl;
  std::cout << "  --jit-level=<level>              JIT granularity: none, "
               "expr, operator, pipeline, query"
            << std::endl;
  std::cout << "  --kernel-path=<path>             Kernel execution path: "
               "none, pipeline, query (default: none)"
            << std::endl;
  std::cout << "  --jit-simd[=<level>]             Enable SIMD: sse2, avx, "
               "avx2, avx512, auto (default: auto)"
            << std::endl;
  std::cout << "  --no-jit                         Disable JIT compilation"
            << std::endl;
  std::cout << "\n  Pipeline-JIT optimization toggles (enabled by "
               "default at pipeline+ level):"
            << std::endl;
  std::cout << "  --[no-]jit-payload-prune         Hash build payload "
               "pruning"
            << std::endl;
  std::cout << "  --[no-]jit-prefetch[=N]          Software prefetch "
               "for hash probe (default N=8)"
            << std::endl;
  std::cout << "  --jit-prefetch-entry-dist=N      ROF stage-2 entry "
               "prefetch distance (default 24)"
            << std::endl;
  std::cout << "  --jit-prefetch-row-dist=N        ROF stage-2 row "
               "prefetch distance (default 12)"
            << std::endl;
  std::cout << "  --[no-]jit-batch-probe           Batch/vectorized "
               "hash probe"
            << std::endl;
  std::cout << "  --jit-skip-hash-cmp=off|single|all  Skip hash cmp "
               "for int keys (single=1-key, all=any)"
            << std::endl;
  std::cout << "  --jit-cache[=MODE]               JIT object cache mode "
               "(default: off)\n"
               "                                     single-run-strict: exact plan "
               "match, cleared between iterations\n"
               "                                     single-run-template: "
               "parameterized constants, relaxed key (PLANNED)\n"
               "                                     full: persistent disk cache "
               "(PLANNED)\n"
               "                                     bare --jit-cache = "
               "single-run-strict"
            << std::endl;
  std::cout << "  --compile-mode=llvm|fastisel|tpde JIT backend: llvm (LLVM O2, "
               "default), fastisel (LLVM O0+FastISel), or tpde (TPDE fast codegen)"
            << std::endl;
  std::cout << "  --spec-jit=off|recompile|interpret  Speculative JIT: off = "
               "disabled (default), recompile = TPDE on miss, interpret = no "
               "JIT on miss"
            << std::endl;
  std::cout << "  --query-jit-threads=N            Query-jit worker threads "
               "(default 0 = hardware concurrency; 1 = serial debug)"
            << std::endl;
  std::cout << "  --query-jit-morsel=N             Query-jit morsel size in "
               "rows (default 20000)"
            << std::endl;
  std::cout << "\n  Measurement:" << std::endl;
  std::cout << "  --repeat=N                       Run query N times in-process "
               "(default: 1)"
            << std::endl;
  std::cout << "  --in-memory                      Use :memory: DB + load from "
               "CSV (DuckDB only)"
            << std::endl;
  std::cout << "  --csv-dir=<path>                 CSV directory for --in-memory "
               "(default: derived from --schema)"
            << std::endl;
  std::cout << "  --lingodb-mode=<mode>            LingoDB backend: llvm "
               "(LLVM JIT) or tpde (TPDE fast codegen) (default: llvm)"
            << std::endl;
  std::cout << "  --storage-plan                   Load flat column arrays + CSR "
               "indexes at startup (DuckDB only)"
            << std::endl;
  std::cout << "  --storage-cache=<path>           Binary cache file for "
               "--storage-plan (auto-creates if missing)"
            << std::endl;
  std::cout << "  --tune-config=<path>             Per-subquery JIT config "
               "JSON (from tune_per_subquery.py)"
	    << std::endl;
  std::cout << "  --explain                        Print EXPLAIN ANALYZE plan "
               "for each sub-SQL (default: disabled)"
            << std::endl;
  std::cout << "  --help, -h                       Show this help message"
            << std::endl;
  std::cout << "\nExamples:" << std::endl;
  std::cout << "  AQP_middleware --engine=duckdb --split=topdown" << std::endl;
  std::cout << "  AQP_middleware --engine=postgres --split=minsubquery "
               "--check-correctness"
            << std::endl;
}

} // namespace middleware
