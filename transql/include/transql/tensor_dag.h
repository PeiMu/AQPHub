#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace transql {

enum class TensorOpType {
    EmbedLookup,
    RMSNorm,
    MatMul,
    RoPE,
    QKAttn,
    Softmax,
    AttnVMul,
    SwiGLU,
    ResidualAdd
};

// A single logical operation in the LLM computation graph.
// input_tables: ordered list of input temp/weight table names.
// output_table: name of the temp table this op materialises to.
// params: op-specific string parameters (e.g. "hidden_dim", "eps").
// is_shared: true when multiple downstream ops consume this output;
//            used by future AQP strategies to decide materialisation.
struct TensorDagNode {
    int id;
    TensorOpType op_type;
    std::string output_table;
    std::vector<std::string> input_tables;
    bool is_shared;
    std::unordered_map<std::string, std::string> params;
};

// Directed acyclic graph of the full LLM forward pass.
// Nodes are stored in topological order (BuildLlama3_8B guarantees this).
class TensorComputeDAG {
public:
    // Construct the Llama3-8B forward-pass DAG for single-token inference.
    // num_layers: number of transformer layers (32 for Llama3-8B).
    // chunk_size: floats per REAL[] chunk (32 matches the paper).
    static TensorComputeDAG BuildLlama3_8B(int num_layers = 32,
                                           int chunk_size = 32);

    // Build DAG from a topology.json file produced by extract_weights.py
    // --source onnx.  json_path must be a valid file path.
    static TensorComputeDAG BuildFromJSON(const std::string& json_path);

    const TensorDagNode& GetNode(int id) const { return nodes_[id]; }
    const std::vector<TensorDagNode>& Nodes() const { return nodes_; }

    // ID of the final output node (logits).
    int OutputNodeId() const { return output_node_id_; }

private:
    int AddNode(TensorOpType op,
                const std::string& output_table,
                const std::vector<std::string>& input_tables,
                bool is_shared,
                const std::unordered_map<std::string, std::string>& params = {});

    std::vector<TensorDagNode> nodes_;
    int output_node_id_ = -1;
};

} // namespace transql
