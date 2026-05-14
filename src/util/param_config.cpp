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
      } else {
        throw std::runtime_error(
            "Unknown engine: " + arg.substr(9) +
            " (valid: duckdb, postgres, umbra, mariadb, opengauss)");
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
        config.jit_flags |= AQP_JIT_EXPR | AQP_JIT_OPERATOR | AQP_JIT_PIPELINE;
      } else if (level == "sql" || level == "subplan") {
        config.jit_flags |= AQP_JIT_EXPR | AQP_JIT_OPERATOR | AQP_JIT_PIPELINE | AQP_JIT_SQL;
      } else {
        throw std::runtime_error(
            "Unknown JIT level: " + arg.substr(12) +
            " (valid: expr, operator, pipeline, sql, subplan)");
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
    else if (arg == "--jit-fusion-probe") {
      config.jit_fusion_probe = true;
    } else if (arg == "--no-jit-fusion-probe") {
      config.jit_fusion_probe = false;
    } else if (arg == "--jit-inline-hash") {
      config.jit_inline_hash = true;
    } else if (arg == "--no-jit-inline-hash") {
      config.jit_inline_hash = false;
    } else if (arg == "--jit-payload-prune") {
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
    } else if (arg == "--jit-cache") {
      config.jit_cache = true;
    } else if (arg.find("--jit-cache=") == 0) {
      config.jit_cache = true;
      config.jit_cache_dir = arg.substr(12);
    } else if (arg == "--no-jit-cache") {
      config.jit_cache = false;
    } else if (arg.find("--jit-opt=") == 0) {
      std::string opt = to_lower(arg.substr(10));
      config.jit_flags &= ~AQP_JIT_OPT_MASK;
      if (opt == "o0" || opt == "0")
        config.jit_flags |= AQP_JIT_OPT_O0;
      else if (opt == "o1" || opt == "1")
        config.jit_flags |= AQP_JIT_OPT_O1;
      else if (opt == "o2" || opt == "2")
        config.jit_flags |= AQP_JIT_OPT_O2;
      else if (opt == "o3" || opt == "3")
        config.jit_flags |= AQP_JIT_OPT_O3;
      else
        throw std::runtime_error(
            "Unknown JIT opt level: " + opt +
            " (valid: O0, O1, O2, O3)");
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
  std::cout << "  --jit-level=<level>              JIT granularity: expr, "
               "operator, pipeline, sql (alias: subplan)"
            << std::endl;
  std::cout << "  --jit-simd[=<level>]             Enable SIMD: sse2, avx, "
               "avx2, avx512, auto (default: auto)"
            << std::endl;
  std::cout << "  --jit-opt=<O0|O1|O2|O3>         LLVM optimization level "
               "(default: O1)"
            << std::endl;
  std::cout << "  --no-jit                         Disable JIT compilation"
            << std::endl;
  std::cout << "\n  Pipeline-JIT optimization toggles (enabled by "
               "default at pipeline+ level):"
            << std::endl;
  std::cout << "  --[no-]jit-fusion-probe          Filter+Probe+"
               "Projection fusion"
            << std::endl;
  std::cout << "  --[no-]jit-inline-hash           Inline FNV-1a hash "
               "as LLVM IR"
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
  std::cout << "  --[no-]jit-cache[=path]          Cross-process "
               "compiled binary cache"
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
