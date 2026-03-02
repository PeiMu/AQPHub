/*
 * NodeBasedSplitter: stub implementation.
 * Real logic lives in IRQuerySplitter::ExecuteSplitLoopNodeBased which
 * bypasses the SplitAlgorithm loop entirely and drives the DuckDB plan
 * directly. This class only exists so splitter_ is non-null and carries
 * the strategy name.
 */

#ifdef HAVE_DUCKDB

#include "split/node_based_splitter.h"

namespace middleware {

NodeBasedSplitter::NodeBasedSplitter(DBAdapter *exec_adapter,
                                     DuckDBAdapter * /*plan_adapter*/)
    : SplitAlgorithm(exec_adapter) {}

void NodeBasedSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::SimplestStmt> & /*ir*/) {}

std::unique_ptr<SubqueryExtraction> NodeBasedSplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt * /*remaining_ir*/) {
  return nullptr;
}

bool NodeBasedSplitter::IsComplete(
    const ir_sql_converter::SimplestStmt * /*remaining_ir*/) {
  return true;
}

std::unique_ptr<ir_sql_converter::SimplestStmt>
NodeBasedSplitter::UpdateRemainingIR(
    std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
    const std::set<unsigned int> &, unsigned int, const std::string &,
    uint64_t, const std::vector<std::pair<unsigned int, unsigned int>> &,
    const std::vector<std::string> &) {
  return remaining_ir;
}

} // namespace middleware

#endif // HAVE_DUCKDB
