#include "transql/transql_runner.h"

#include <iostream>
#include <stdexcept>

#include "cpp_interface.h"

namespace transql {

TranSQLRunner::TranSQLRunner(middleware::EngineAdapter* adapter, int chunk_size)
    : adapter_(adapter), chunk_size_(chunk_size) {}

void TranSQLRunner::Init(int num_layers, const std::string& json_path) {
    if (initialised_)
        return;

    TensorComputeDAG dag = json_path.empty()
        ? TensorComputeDAG::BuildLlama3_8B(num_layers, chunk_size_)
        : TensorComputeDAG::BuildFromJSON(json_path);
    output_table_ = dag.GetNode(dag.OutputNodeId()).output_table;

    TensorDagSplitter splitter;
    steps_ = splitter.Convert(dag);

    // Record all temp table names that will be created during execution.
    created_tables_.push_back("input_tokens");
    for (const auto& step : steps_)
        created_tables_.push_back(step.second);

    initialised_ = true;
}

void TranSQLRunner::LoadTokens(const std::vector<int>& tokens) {
    adapter_->ExecuteSQL("DROP TABLE IF EXISTS input_tokens");
    adapter_->ExecuteSQL(
        "CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)");

    for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
        adapter_->ExecuteSQL(
            "INSERT INTO input_tokens VALUES (" +
            std::to_string(i) + ", " + std::to_string(tokens[i]) + ")");
    }
}

void TranSQLRunner::ExecuteSteps() {
    for (const auto& step : steps_) {
        const std::string& table_name = step.second;
        std::string sql = ir_sql_converter::ConvertIRToSQL(*step.first, 0);
        std::string create_sql =
            "CREATE TEMP TABLE " + table_name + " AS (" + sql + ")";
        adapter_->ExecuteSQL(create_sql);
    }
}

middleware::QueryResult TranSQLRunner::RunInference(
        const std::vector<int>& tokens) {
    if (!initialised_)
        throw std::runtime_error("TranSQLRunner: call Init() before RunInference()");

    LoadTokens(tokens);
    ExecuteSteps();
    return adapter_->ExecuteSQL("SELECT * FROM " + output_table_);
}

void TranSQLRunner::CleanUp() {
    // Drop in reverse order so downstream tables go first.
    for (int i = static_cast<int>(created_tables_.size()) - 1; i >= 0; --i)
        adapter_->ExecuteSQL("DROP TABLE IF EXISTS " + created_tables_[i]);
}

} // namespace transql
