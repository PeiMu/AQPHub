"""
Diagnostic: for each layer, inject PyTorch's exact hidden state as SQL input,
run that single SQL layer, and compare output to PyTorch.
Uses explicit sub-module calls to avoid transformers API version issues.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'transql', 'python'))
import duckdb, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from run_prefill import (
    build_layer_steps, run_steps,
    CHUNK_SIZE, HIDDEN_DIM, N_CHUNKS_HIDDEN, NUM_LAYERS,
    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
)

DB = 'weights.duckdb'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B'
SEQ_LEN = 3
tokens = [128000, 1, 2]
TEST_LAYERS = [0, 1, 2, 8, 16, 24, 31]

ROPE_BASE = 500000.0
EPS = 1e-5


def apply_rope_pt(x, positions):
    """Apply RoPE to x [num_heads, seq_len, head_dim] at given positions."""
    half = HEAD_DIM // 2
    dim_idx = np.arange(0, HEAD_DIM, 2, dtype=np.float32)
    theta = 1.0 / (ROPE_BASE ** (dim_idx / HEAD_DIM))  # [64]
    x_np = x.detach().numpy()
    out = np.zeros_like(x_np)
    for i, pos in enumerate(positions):
        angles = pos * theta  # [64]
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        for h in range(x_np.shape[0]):
            x_even = x_np[h, i, 0::2]  # [64]
            x_odd  = x_np[h, i, 1::2]  # [64]
            out[h, i, 0::2] = x_even * cos_a - x_odd * sin_a
            out[h, i, 1::2] = x_odd  * cos_a + x_even * sin_a
    return torch.tensor(out)


def layer_forward_pt(model, layer_idx, x_np, positions=None):
    """Run one transformer layer using explicit sub-module calls.
    x_np: [SEQ_LEN, HIDDEN_DIM] numpy array.
    Returns: [SEQ_LEN, HIDDEN_DIM] numpy array.
    """
    if positions is None:
        positions = list(range(SEQ_LEN))

    layer = model.model.layers[layer_idx]
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # [1, seq, hidden]

    # 1. RMSNorm
    norm1_out = layer.input_layernorm(x)  # [1, seq, hidden]

    # 2. Q, K, V projections
    q = layer.self_attn.q_proj(norm1_out)  # [1, seq, 4096]
    k = layer.self_attn.k_proj(norm1_out)  # [1, seq, 1024]
    v = layer.self_attn.v_proj(norm1_out)  # [1, seq, 1024]

    # 3. Reshape to heads
    q = q.view(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)  # [1, 32, seq, 128]
    k = k.view(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [1, 8, seq, 128]
    v = v.view(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [1, 8, seq, 128]

    # Scale Q
    scale = 1.0 / (HEAD_DIM ** 0.5)
    q = q * scale

    # 4. RoPE
    q_rot = torch.zeros_like(q)
    k_rot = torch.zeros_like(k)
    for h in range(NUM_Q_HEADS):
        q_rot[0, h] = apply_rope_pt(q[0, h:h+1], positions)[0]
    for h in range(NUM_KV_HEADS):
        k_rot[0, h] = apply_rope_pt(k[0, h:h+1], positions)[0]

    # 5. GQA attention
    gs = NUM_Q_HEADS // NUM_KV_HEADS  # 4
    attn_out = torch.zeros(1, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    for h in range(NUM_Q_HEADS):
        kv_h = h // gs
        # Scores: [seq, seq]
        scores = torch.matmul(q_rot[0, h], k_rot[0, kv_h].transpose(-1, -2))  # [seq, seq]
        # Causal mask
        causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
        scores = scores.masked_fill(causal == 0, float('-inf'))
        attn_w = F.softmax(scores, dim=-1)
        # Output: [seq, head_dim]
        attn_out[0, h] = torch.matmul(attn_w, v[0, kv_h])

    # 6. Merge heads and o_proj
    attn_out = attn_out.transpose(1, 2).reshape(1, SEQ_LEN, HIDDEN_DIM)
    o = layer.self_attn.o_proj(attn_out)

    # 7. Residual
    x_after_attn = x + o

    # 8. FFN
    norm2_out = layer.post_attention_layernorm(x_after_attn)
    gate = layer.mlp.gate_proj(norm2_out)
    up_out = layer.mlp.up_proj(norm2_out)
    ffn_act = F.silu(gate) * up_out
    down = layer.mlp.down_proj(ffn_act)

    x_out = x_after_attn + down
    return x_out.squeeze(0).detach().numpy()


def hidden_to_rows(arr):
    """Convert [seq_len, hidden_dim] array to (pos, chunk_id, v) rows."""
    rows = []
    seq_len, hidden_dim = arr.shape
    n_chunks = hidden_dim // CHUNK_SIZE
    for pos in range(seq_len):
        for c in range(n_chunks):
            v = arr[pos, c*CHUNK_SIZE:(c+1)*CHUNK_SIZE].tolist()
            rows.append((pos, c, v))
    return rows


def read_hidden_sql(con, name, seq_len=SEQ_LEN):
    rows = con.execute(
        f'SELECT row_id, chunk_id, v FROM {name} ORDER BY row_id, chunk_id'
    ).fetchall()
    arr = np.zeros((seq_len, HIDDEN_DIM), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        arr[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = v
    return arr


# ---- Load model ----
print('Loading PyTorch model...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map='cpu'
)

# ---- Get PyTorch hidden states at each layer ----
print('Running PyTorch forward pass to get hidden states at each layer...')
emb = model.model.embed_tokens(torch.tensor([tokens]))  # [1,3,4096]
pt_layer_inputs = {}
pt_layer_outputs = {}
x_pt_np = emb.squeeze(0).detach().numpy()  # [3, 4096]

for l in range(max(TEST_LAYERS) + 1):
    pt_layer_inputs[l] = x_pt_np.copy()
    pt_layer_outputs[l] = layer_forward_pt(model, l, x_pt_np)
    x_pt_np = pt_layer_outputs[l]
    if (l + 1) % 8 == 0:
        print(f'  PyTorch: layer {l+1}/{max(TEST_LAYERS)+1}')

print('PyTorch done.\n')

# ---- For each test layer: inject PT input, run SQL, compare ----
print(f"{'Layer':>6}  {'pos=0':>10}  {'pos=1':>10}  {'pos=2':>10}")
print("-" * 46)

for l in TEST_LAYERS:
    pt_in = pt_layer_inputs[l]
    pt_out = pt_layer_outputs[l]

    con = duckdb.connect(DB, read_only=True)
    # Create temp table with PT's exact input
    con.execute(f'CREATE TEMP TABLE x_in_test (row_id INTEGER, chunk_id INTEGER, v FLOAT[{CHUNK_SIZE}])')
    con.executemany('INSERT INTO x_in_test VALUES (?,?,?)', hidden_to_rows(pt_in))

    steps = []
    layer_steps, x_out_name = build_layer_steps(l, 'x_in_test', cached_wt=False)
    steps += layer_steps
    run_steps(con, steps)

    sql_out = read_hidden_sql(con, x_out_name)
    con.close()

    out_diffs = [np.abs(sql_out[pos] - pt_out[pos]).max() for pos in range(SEQ_LEN)]
    print(f"{l:>6}  {out_diffs[0]:>10.4e}  {out_diffs[1]:>10.4e}  {out_diffs[2]:>10.4e}")
