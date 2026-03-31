#include <iostream>
#include <string>
#include <vector>

#include "adapters/duckdb_adapter.h"
#include "transql/transql_runner.h"

// Usage: test_transql <weights.duckdb> [token_id] [--topology topology.json]
//
// Examples:
//   ./test_transql /path/to/weights.duckdb 1234
//   ./test_transql /path/to/weights.duckdb 1234 --topology topology.json
//
// The weights.duckdb file must be prepared by:
//   python transql/python/extract_weights.py --output-dir /tmp/npy
//   python transql/python/preprocess_weights.py --npy-dir /tmp/npy --csv-dir /tmp/csv
//   python transql/python/load_weights_duckdb.py --csv-dir /tmp/csv --db-path weights.duckdb
//
// For ONNX-sourced topology:
//   python transql/python/extract_weights.py --source onnx --onnx-path model.onnx \
//       --output-dir /tmp/npy
//   (produces /tmp/npy/topology.json alongside weights)
//   ./test_transql weights.duckdb 1234 --topology /tmp/npy/topology.json

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_transql <weights.duckdb> [token_id] [--topology topology.json]\n";
        return 1;
    }
    std::string db_path = argv[1];
    int token_id = 1;
    std::string topology_path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--topology" && i + 1 < argc) {
            topology_path = argv[++i];
        } else {
            try { token_id = std::stoi(arg); } catch (...) {}
        }
    }

    std::cout << "Opening DuckDB: " << db_path << "\n";
    middleware::DuckDBAdapter adapter(db_path);

    std::cout << "Initialising TranSQL runner";
    if (!topology_path.empty())
        std::cout << " (topology: " << topology_path << ")";
    std::cout << "...\n";
    transql::TranSQLRunner runner(&adapter);
    runner.Init(32, topology_path);

    std::cout << "Running inference for token_id = " << token_id << "\n";
    auto result = runner.RunInference({token_id});

    std::cout << "Logits shape: " << result.num_rows << " rows x "
              << result.num_columns << " cols\n";

    // Print top-1 logit position (greedy decode).
    if (!result.rows.empty()) {
        int best_col = 0;
        double best_val = std::stod(result.rows[0][0]);
        for (int c = 1; c < result.num_columns; ++c) {
            double v = std::stod(result.rows[0][c]);
            if (v > best_val) { best_val = v; best_col = c; }
        }
        std::cout << "Top-1 predicted token: " << best_col
                  << " (logit = " << best_val << ")\n";
    }

    runner.CleanUp();
    return 0;
}
