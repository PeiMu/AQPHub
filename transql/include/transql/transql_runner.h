#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "adapters/db_adapter.h"
#include "dag_to_tree.h"
#include "tensor_dag.h"

namespace transql {

// Orchestrates Llama3-8B inference via SQL executed through an EngineAdapter.
//
// Usage:
//   DuckDBAdapter adapter("weights.duckdb");
//   TranSQLRunner runner(&adapter);
//   runner.Init();                              // build DAG, expand to SQL steps
//   auto result = runner.RunInference({token}); // execute pipeline
//   runner.CleanUp();                           // drop temp tables
class TranSQLRunner {
public:
    explicit TranSQLRunner(middleware::EngineAdapter* adapter,
                           int chunk_size = 32);

    // Build the DAG and expand to SQL steps (idempotent).
    // If json_path is non-empty, load topology from a topology.json produced
    // by extract_weights.py --source onnx; otherwise build Llama3-8B DAG
    // from the hardcoded C++ definition.
    void Init(int num_layers = 32, const std::string& json_path = "");

    // Load token IDs, execute the full inference SQL pipeline, return logits.
    // tokens: list of token IDs (current implementation: single token).
    middleware::QueryResult RunInference(const std::vector<int>& tokens);

    // Drop all temporary tables created during the last RunInference call.
    void CleanUp();

private:
    // Insert token IDs into a temp table used by EmbedLookup.
    void LoadTokens(const std::vector<int>& tokens);

    // Execute each step as CREATE TEMP TABLE name AS (sql).
    void ExecuteSteps();

    middleware::EngineAdapter* adapter_;
    int chunk_size_;
    std::vector<TensorDagSplitter::Step> steps_;
    std::vector<std::string> created_tables_;
    std::string output_table_;
    bool initialised_ = false;
};

} // namespace transql
