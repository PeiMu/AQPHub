#pragma once

#include <string>
#include <utility>
#include <vector>

namespace transql {

// A single SQL step: (SELECT body, destination temp table name).
// The runner wraps it as: CREATE TEMP TABLE name AS (sql).
using SqlStep  = std::pair<std::string, std::string>;
using SqlSteps = std::vector<SqlStep>;

// Embedding lookup: tokens_table(pos, token_id), embed_table(token_id, chunk_id, v).
// Output: (row_id=pos, chunk_id, v FLOAT[]).
SqlSteps EmbedLookupSQL(const std::string& tokens_table,
                        const std::string& embed_table,
                        const std::string& out_table);

// Chunked matrix multiply: act(row_id, chunk_id, v) x weight(row_id, chunk_id, v).
// Produces (row_id, chunk_id, v FLOAT[]) in two steps:
//   step 1: dot-product intermediates  → out_table + "_dp"
//   step 2: re-chunk scalars into FLOAT[] → out_table
SqlSteps MatMulSQL(const std::string& act_table,
                   const std::string& weight_table,
                   const std::string& out_table,
                   int chunk_size = 32);

// RMSNorm with learnable gamma.
// hidden_dim: total feature dimension (e.g. 4096).
// eps: numerical stability constant (1e-5 for Llama3).
// Steps: sum-of-squares → normalise+scale.
SqlSteps RMSNormSQL(const std::string& input_table,
                    const std::string& gamma_table,
                    const std::string& out_table,
                    int hidden_dim,
                    float eps,
                    int chunk_size = 32);

// Rotary positional encoding (RoPE).
// rope_table: (row_id=pos, chunk_id, cos FLOAT[chunk/2], sin FLOAT[chunk/2]).
// Output uses split even/odd layout: (row_id, chunk_id, v_even FLOAT[], v_odd FLOAT[]).
SqlSteps RoPESQL(const std::string& q_table,
                 const std::string& rope_table,
                 const std::string& out_table,
                 int chunk_size = 32);

// QK attention scores with GQA.
// Inputs use the RoPE split layout (v_even, v_odd columns).
// Output: (q_tok, k_tok, head_id, score FLOAT) — scalar layout.
SqlSteps QKAttnSQL(const std::string& q_rope_table,
                   const std::string& k_rope_table,
                   const std::string& out_table,
                   int num_q_heads,
                   int num_kv_heads,
                   int head_dim,
                   int chunk_size = 32);

// Softmax over attention scores per (q_tok, head_id).
// Input/output: (q_tok, k_tok, head_id, score/attn_weight) scalar layout.
// Steps: max → exp → sum → divide.
SqlSteps SoftmaxSQL(const std::string& input_table,
                    const std::string& out_table);

// Attention weighted sum: attn_weights @ V.
// attn_table: (q_tok, k_tok, head_id, attn_weight) scalar.
// v_table: standard chunked layout (tok, chunk_id, v FLOAT[]).
// Output: standard chunked layout (row_id, chunk_id, v FLOAT[]) sized [seq, hidden].
// Steps: expand V to scalar → weighted sum → re-chunk.
SqlSteps AttnVMulSQL(const std::string& attn_table,
                     const std::string& v_table,
                     const std::string& out_table,
                     int num_q_heads,
                     int num_kv_heads,
                     int head_dim,
                     int chunk_size = 32);

// SwiGLU activation: gate * SiLU(up), element-wise over chunks.
SqlSteps SwiGLUSQL(const std::string& gate_table,
                   const std::string& up_table,
                   const std::string& out_table);

// Element-wise residual addition.
SqlSteps ResidualAddSQL(const std::string& table_a,
                        const std::string& table_b,
                        const std::string& out_table);

} // namespace transql
