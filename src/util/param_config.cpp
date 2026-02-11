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
      } else {
        throw std::runtime_error("Unknown engine: " + arg.substr(9) +
                                 " (valid: duckdb, postgres)");
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
      } else {
        throw std::runtime_error("Unknown split strategy: " + arg.substr(8) +
                                 " (valid: none, topdown, minsubquery, "
                                 "relationship-center, entity-center)");
      }
    }
    // Parse boolean flags
    else if (arg == "--benchmark") {
      config.benchmark_mode = true;
    } else if (arg == "--no-reorder-get") {
      config.enable_reorder_get = false;
    } else if (arg == "--no-update-temp-card") {
      config.enable_update_temp_card = false;
    } else if (arg == "--no-postgres-analyze") {
      config.enable_postgres_analyze = false;
    } else if (arg == "--check-correctness") {
      config.enable_correctness_check = true;
    } else if (arg == "--timing") {
      config.enable_timing = true;
    } else if (arg == "--debug") {
      config.enable_debug_print = true;
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
  std::cout << "  --engine=<duckdb|postgres>       Backend engine "
               "(default: duckdb)"
            << std::endl;
  std::cout
      << "  --split=<strategy>               Split strategy (default: none)"
      << std::endl;
  std::cout << "  --schema=<path>                  Schema SQL file for column "
               "index lookup (PostgreSQL)"
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
  std::cout << "  --no-postgres-analyze            Disable ANALYZE in "
               "PostgreSQL adapter"
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
  std::cout << "  --help, -h                       Show this help message"
            << std::endl;
  std::cout << "\nExamples:" << std::endl;
  std::cout << "  AQP_middleware --engine=duckdb --split=topdown" << std::endl;
  std::cout << "  AQP_middleware --engine=postgres --split=minsubquery "
               "--check-correctness"
            << std::endl;
}

} // namespace middleware
