#include "transql/dag_to_tree.h"

#include <stdexcept>

#include "transql/sql_template.h"

namespace transql {

static int GetIntParam(const TensorDagNode& node, const std::string& key) {
    auto it = node.params.find(key);
    if (it == node.params.end())
        throw std::runtime_error("TensorDagNode missing param: " + key);
    return std::stoi(it->second);
}

static float GetFloatParam(const TensorDagNode& node, const std::string& key) {
    auto it = node.params.find(key);
    if (it == node.params.end())
        throw std::runtime_error("TensorDagNode missing param: " + key);
    return std::stof(it->second);
}

// Wrap a (sql_select_body, table_name) pair into a Step.
static TensorDagSplitter::Step MakeStep(const SqlStep& s) {
    return {std::unique_ptr<ir_sql_converter::AQPStmt>(
                new ir_sql_converter::SimplestRawSQL(s.first)),
            s.second};
}

std::vector<TensorDagSplitter::Step>
TensorDagSplitter::ExpandNode(const TensorDagNode& node) {
    SqlSteps sql_steps;
    const auto& in  = node.input_tables;
    const auto& out = node.output_table;

    switch (node.op_type) {
    case TensorOpType::EmbedLookup:
        sql_steps = EmbedLookupSQL(in[0], in[1], out);
        break;

    case TensorOpType::MatMul:
        sql_steps = MatMulSQL(in[0], in[1], out);
        break;

    case TensorOpType::RMSNorm:
        sql_steps = RMSNormSQL(in[0], in[1], out,
                               GetIntParam(node, "hidden_dim"),
                               GetFloatParam(node, "eps"));
        break;

    case TensorOpType::RoPE:
        sql_steps = RoPESQL(in[0], in[1], out,
                            GetIntParam(node, "chunk_size"));
        break;

    case TensorOpType::QKAttn:
        sql_steps = QKAttnSQL(in[0], in[1], out,
                              GetIntParam(node, "num_q_heads"),
                              GetIntParam(node, "num_kv_heads"),
                              GetIntParam(node, "head_dim"),
                              GetIntParam(node, "chunk_size"));
        break;

    case TensorOpType::Softmax:
        sql_steps = SoftmaxSQL(in[0], out);
        break;

    case TensorOpType::AttnVMul:
        sql_steps = AttnVMulSQL(in[0], in[1], out,
                                GetIntParam(node, "num_q_heads"),
                                GetIntParam(node, "num_kv_heads"),
                                GetIntParam(node, "head_dim"),
                                GetIntParam(node, "chunk_size"));
        break;

    case TensorOpType::SwiGLU:
        sql_steps = SwiGLUSQL(in[0], in[1], out);
        break;

    case TensorOpType::ResidualAdd:
        sql_steps = ResidualAddSQL(in[0], in[1], out);
        break;

    default:
        throw std::runtime_error("TensorDagSplitter: unknown TensorOpType");
    }

    std::vector<Step> steps;
    steps.reserve(sql_steps.size());
    for (const auto& s : sql_steps)
        steps.push_back(MakeStep(s));
    return steps;
}

std::vector<TensorDagSplitter::Step>
TensorDagSplitter::Convert(const TensorComputeDAG& dag) {
    std::vector<Step> all_steps;
    for (const auto& node : dag.Nodes()) {
        auto node_steps = ExpandNode(node);
        for (auto& s : node_steps)
            all_steps.push_back(std::move(s));
    }
    return all_steps;
}

} // namespace transql
