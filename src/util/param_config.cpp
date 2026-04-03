/*
 * Implementation of split configuration parsing and backends
 */

#include "util/param_config.h"
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
      config.enable_jit = true;
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
               "with LLVM (DuckDB only, default: disabled)"
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
