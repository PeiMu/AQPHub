#include "transql/tensor_dag.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace transql {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static std::string LayerPrefix(int l) {
    return "l" + std::to_string(l) + "_";
}

// Weight table names produced by load_weights_duckdb.py.
static std::string WeightTable(int l, const std::string& name) {
    return "layer_" + std::to_string(l) + "_" + name;
}

// ---------------------------------------------------------------------------
// TensorComputeDAG
// ---------------------------------------------------------------------------

int TensorComputeDAG::AddNode(
        TensorOpType op,
        const std::string& output_table,
        const std::vector<std::string>& input_tables,
        bool is_shared,
        const std::unordered_map<std::string, std::string>& params) {
    int id = static_cast<int>(nodes_.size());
    TensorDagNode node;
    node.id           = id;
    node.op_type      = op;
    node.output_table = output_table;
    node.input_tables = input_tables;
    node.is_shared    = is_shared;
    node.params       = params;
    nodes_.push_back(std::move(node));
    return id;
}

TensorComputeDAG TensorComputeDAG::BuildLlama3_8B(int num_layers,
                                                   int chunk_size) {
    TensorComputeDAG dag;

    const int hidden_dim  = 4096;
    const int num_q_heads = 32;
    const int num_kv_heads = 8;
    const int head_dim    = 128;  // hidden_dim / num_q_heads
    const float eps       = 1e-5f;

    const std::string tokens_table = "input_tokens";  // (pos, token_id)
    const std::string embed_table  = "embed_tokens";  // weight

    // -----------------------------------------------------------------------
    // Embedding lookup
    // -----------------------------------------------------------------------
    // Output: x_0 (row_id=pos, chunk_id 0..127, v FLOAT[32])
    dag.AddNode(TensorOpType::EmbedLookup, "x_0",
                {tokens_table, embed_table},
                /*is_shared=*/true,  // consumed by layer 0 RMSNorm1 AND ResidualAdd
                {});

    std::string x_in = "x_0";  // rolling residual stream table name

    for (int l = 0; l < num_layers; ++l) {
        std::string pfx = LayerPrefix(l);

        // -----------------------------------------------------------------
        // Pre-attention RMSNorm
        // -----------------------------------------------------------------
        std::string norm1_out = pfx + "norm1_out";
        dag.AddNode(TensorOpType::RMSNorm, norm1_out,
                    {x_in, WeightTable(l, "norm1")},
                    /*is_shared=*/true,   // Q, K, V all consume norm1_out
                    {{"hidden_dim", std::to_string(hidden_dim)},
                     {"eps",        std::to_string(eps)}});

        // -----------------------------------------------------------------
        // Q / K / V projections
        // -----------------------------------------------------------------
        std::string q_proj = pfx + "q";
        dag.AddNode(TensorOpType::MatMul, q_proj,
                    {norm1_out, WeightTable(l, "q_proj")},
                    /*is_shared=*/false,
                    {});

        std::string k_proj = pfx + "k";
        dag.AddNode(TensorOpType::MatMul, k_proj,
                    {norm1_out, WeightTable(l, "k_proj")},
                    /*is_shared=*/false,
                    {});

        std::string v_proj = pfx + "v";
        dag.AddNode(TensorOpType::MatMul, v_proj,
                    {norm1_out, WeightTable(l, "v_proj")},
                    /*is_shared=*/true,   // KV cache: materialized for decoding reuse
                    {});

        // -----------------------------------------------------------------
        // RoPE on Q and K
        // -----------------------------------------------------------------
        // rope table: (row_id=pos, chunk_id, cos FLOAT[], sin FLOAT[])
        // precomputed by preprocess_weights.py; joined on chunk_id AND row_id
        // (position) inside RoPESQL. Shared for Q and K.
        std::string rope_q = pfx + "q_rope";
        dag.AddNode(TensorOpType::RoPE, rope_q,
                    {q_proj, "rope"},
                    /*is_shared=*/true,   // QKAttn needs it from a separate CTE group
                    {{"chunk_size", std::to_string(chunk_size)}});

        std::string rope_k = pfx + "k_rope";
        dag.AddNode(TensorOpType::RoPE, rope_k,
                    {k_proj, "rope"},
                    /*is_shared=*/true,   // KV cache: materialized for decoding reuse
                    {{"chunk_size", std::to_string(chunk_size)}});

        // -----------------------------------------------------------------
        // QK attention scores
        // -----------------------------------------------------------------
        std::string qk_scores = pfx + "qk_scores";
        dag.AddNode(TensorOpType::QKAttn, qk_scores,
                    {rope_q, rope_k},
                    /*is_shared=*/false,
                    {{"num_q_heads",  std::to_string(num_q_heads)},
                     {"num_kv_heads", std::to_string(num_kv_heads)},
                     {"head_dim",     std::to_string(head_dim)},
                     {"chunk_size",   std::to_string(chunk_size)}});

        // -----------------------------------------------------------------
        // Softmax
        // -----------------------------------------------------------------
        std::string attn_weights = pfx + "attn_weights";
        dag.AddNode(TensorOpType::Softmax, attn_weights,
                    {qk_scores},
                    /*is_shared=*/false,
                    {});

        // -----------------------------------------------------------------
        // Attention × V
        // -----------------------------------------------------------------
        std::string attn_out = pfx + "attn_out";
        dag.AddNode(TensorOpType::AttnVMul, attn_out,
                    {attn_weights, v_proj},
                    /*is_shared=*/false,
                    {{"num_q_heads",  std::to_string(num_q_heads)},
                     {"num_kv_heads", std::to_string(num_kv_heads)},
                     {"head_dim",     std::to_string(head_dim)},
                     {"chunk_size",   std::to_string(chunk_size)}});

        // -----------------------------------------------------------------
        // O projection
        // -----------------------------------------------------------------
        std::string o_proj = pfx + "o_proj";
        dag.AddNode(TensorOpType::MatMul, o_proj,
                    {attn_out, WeightTable(l, "o_proj")},
                    /*is_shared=*/false,
                    {});

        // -----------------------------------------------------------------
        // First residual add: x_after_attn = x_in + o_proj
        // -----------------------------------------------------------------
        std::string x_after_attn = pfx + "x_after_attn";
        dag.AddNode(TensorOpType::ResidualAdd, x_after_attn,
                    {x_in, o_proj},
                    /*is_shared=*/true,   // consumed by RMSNorm2 AND ResidualAdd2
                    {});

        // -----------------------------------------------------------------
        // Pre-FFN RMSNorm
        // -----------------------------------------------------------------
        std::string norm2_out = pfx + "norm2_out";
        dag.AddNode(TensorOpType::RMSNorm, norm2_out,
                    {x_after_attn, WeightTable(l, "norm2")},
                    /*is_shared=*/true,   // gate AND up both read norm2_out
                    {{"hidden_dim", std::to_string(hidden_dim)},
                     {"eps",        std::to_string(eps)}});

        // -----------------------------------------------------------------
        // Gate / Up projections
        // -----------------------------------------------------------------
        std::string gate = pfx + "gate";
        dag.AddNode(TensorOpType::MatMul, gate,
                    {norm2_out, WeightTable(l, "gate_proj")},
                    /*is_shared=*/false,
                    {});

        std::string up = pfx + "up";
        dag.AddNode(TensorOpType::MatMul, up,
                    {norm2_out, WeightTable(l, "up_proj")},
                    /*is_shared=*/false,
                    {});

        // -----------------------------------------------------------------
        // SwiGLU activation
        // -----------------------------------------------------------------
        std::string ffn_act = pfx + "ffn_act";
        dag.AddNode(TensorOpType::SwiGLU, ffn_act,
                    {gate, up},
                    /*is_shared=*/false,
                    {});

        // -----------------------------------------------------------------
        // Down projection
        // -----------------------------------------------------------------
        std::string down = pfx + "down";
        dag.AddNode(TensorOpType::MatMul, down,
                    {ffn_act, WeightTable(l, "down_proj")},
                    /*is_shared=*/false,
                    {});

        // -----------------------------------------------------------------
        // Second residual add: x_out = x_after_attn + down
        // -----------------------------------------------------------------
        std::string x_out = pfx + "x_out";
        dag.AddNode(TensorOpType::ResidualAdd, x_out,
                    {x_after_attn, down},
                    /*is_shared=*/(l < num_layers - 1),  // shared if not last layer
                    {});

        x_in = x_out;  // feed into next layer
    }

    // -----------------------------------------------------------------------
    // Final RMSNorm
    // -----------------------------------------------------------------------
    std::string final_norm_out = "final_norm_out";
    dag.AddNode(TensorOpType::RMSNorm, final_norm_out,
                {x_in, "final_norm"},
                /*is_shared=*/false,
                {{"hidden_dim", std::to_string(hidden_dim)},
                 {"eps",        std::to_string(eps)}});

    // -----------------------------------------------------------------------
    // LM head (MatMul → logits)
    // -----------------------------------------------------------------------
    std::string logits = "logits";
    int logits_id = dag.AddNode(TensorOpType::MatMul, logits,
                                {final_norm_out, "lm_head"},
                                /*is_shared=*/false,
                                {});

    dag.output_node_id_ = logits_id;
    return dag;
}

// ---------------------------------------------------------------------------
// BuildFromJSON
// ---------------------------------------------------------------------------

static TensorOpType OpTypeFromString(const std::string& s) {
    if (s == "EmbedLookup")  return TensorOpType::EmbedLookup;
    if (s == "RMSNorm")      return TensorOpType::RMSNorm;
    if (s == "MatMul")       return TensorOpType::MatMul;
    if (s == "RoPE")         return TensorOpType::RoPE;
    if (s == "QKAttn")       return TensorOpType::QKAttn;
    if (s == "Softmax")      return TensorOpType::Softmax;
    if (s == "AttnVMul")     return TensorOpType::AttnVMul;
    if (s == "SwiGLU")       return TensorOpType::SwiGLU;
    if (s == "ResidualAdd")  return TensorOpType::ResidualAdd;
    if (s == "TopKRouting")  return TensorOpType::TopKRouting;
    if (s == "ExpertFFN")    return TensorOpType::ExpertFFN;
    if (s == "MoeAggregate") return TensorOpType::MoeAggregate;
    throw std::runtime_error("Unknown TensorOpType: " + s);
}

TensorComputeDAG TensorComputeDAG::BuildFromJSON(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("BuildFromJSON: cannot open " + json_path);

    json j;
    f >> j;

    TensorComputeDAG dag;

    for (const auto& n : j.at("nodes")) {
        TensorOpType op = OpTypeFromString(n.at("op_type").get<std::string>());
        std::string output_table = n.at("output_table").get<std::string>();
        bool is_shared = n.at("is_shared").get<bool>();

        std::vector<std::string> input_tables;
        for (const auto& t : n.at("input_tables"))
            input_tables.push_back(t.get<std::string>());

        std::unordered_map<std::string, std::string> params;
        if (n.contains("params")) {
            for (const auto& [k, v] : n.at("params").items())
                params[k] = v.get<std::string>();
        }

        dag.AddNode(op, output_table, input_tables, is_shared, params);
    }

    dag.output_node_id_ = j.at("output_node_id").get<int>();
    return dag;
}

} // namespace transql
