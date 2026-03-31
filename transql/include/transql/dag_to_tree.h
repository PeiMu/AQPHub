#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "simplest_ir.h"
#include "tensor_dag.h"

namespace transql {

// Converts a TensorComputeDAG into a flat, topologically-ordered sequence of
// (SimplestRawSQL, temp_table_name) pairs.
//
// The runner executes each pair as:
//   CREATE TEMP TABLE <name> AS (<raw_sql>)
//
// Multi-step ops (RMSNorm: 2 steps, Softmax: 4 steps, AttnVMul: 3 steps)
// expand to multiple pairs; the final pair in each sequence carries the
// logical op's output_table name recorded in the DAG node.
class TensorDagSplitter {
public:
    using Step = std::pair<std::unique_ptr<ir_sql_converter::AQPStmt>,
                           std::string>;

    // Convert the full DAG to execution steps in topological order.
    std::vector<Step> Convert(const TensorComputeDAG& dag);

private:
    // Expand one DAG node into one or more SQL steps.
    std::vector<Step> ExpandNode(const TensorDagNode& node);
};

} // namespace transql
