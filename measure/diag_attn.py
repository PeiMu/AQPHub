"""
Diagnostic: isolate where layer-0 multi-token attention diverges from PyTorch.
Focuses on the attention computation for a 3-token sequence.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'transql', 'python'))
import duckdb, numpy as np, torch
from transformers import AutoModelForCausalLM
from run_prefill import (
    embed_lookup_sql, rmsnorm_sql, pivoted_matmul_sql, rope_sql,
    qk_attn_sql, softmax_sql, attn_vmul_sql, residual_add_sql,
    run_steps, CHUNK_SIZE, HIDDEN_DIM, N_CHUNKS_HIDDEN,
    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
)

DB = 'weights.duckdb'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B'
SEQ_LEN = 3
tokens = [128000, 1, 2]
VOCAB = 128256

pfx = 'l0_'
wt = lambda name: f'layer_0_{name}'

# ---- SQL ----
con = duckdb.connect(DB, read_only=True)
con.execute('CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)')
con.executemany('INSERT INTO input_tokens VALUES (?,?)', list(enumerate(tokens)))

steps = []
steps += embed_lookup_sql('input_tokens', 'embed_tokens', 'x_0')
steps += rmsnorm_sql('x_0', wt('norm1'), pfx+'norm1_out')
steps += pivoted_matmul_sql(pfx+'norm1_out', wt('q_proj'), pfx+'q', N_CHUNKS_HIDDEN)
steps += pivoted_matmul_sql(pfx+'norm1_out', wt('k_proj'), pfx+'k', N_CHUNKS_HIDDEN)
steps += pivoted_matmul_sql(pfx+'norm1_out', wt('v_proj'), pfx+'v', N_CHUNKS_HIDDEN)
steps += rope_sql(pfx+'q', 'rope', pfx+'q_rope')
steps += rope_sql(pfx+'k', 'rope', pfx+'k_rope')
steps += qk_attn_sql(pfx+'q_rope', pfx+'k_rope', pfx+'qk_scores')
steps += softmax_sql(pfx+'qk_scores', pfx+'attn_weights')
steps += attn_vmul_sql(pfx+'attn_weights', pfx+'v', pfx+'attn_out')
steps += pivoted_matmul_sql(pfx+'attn_out', wt('o_proj'), pfx+'o_proj', N_CHUNKS_HIDDEN)
steps += residual_add_sql('x_0', pfx+'o_proj', pfx+'x_after_attn')

run_steps(con, steps)

def read_2d(con, name, nrows, ndim):
    rows = con.execute(f'SELECT row_id, chunk_id, v FROM {name} ORDER BY row_id, chunk_id').fetchall()
    arr = np.zeros((nrows, ndim), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        arr[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = v
    return arr

# Key tables to compare
sql_norm1  = read_2d(con, pfx+'norm1_out', SEQ_LEN, HIDDEN_DIM)
sql_q      = read_2d(con, pfx+'q', SEQ_LEN, HIDDEN_DIM)
N_KV_DIM   = NUM_KV_HEADS * HEAD_DIM  # 1024
sql_k      = read_2d(con, pfx+'k', SEQ_LEN, N_KV_DIM)
sql_v      = read_2d(con, pfx+'v', SEQ_LEN, N_KV_DIM)
sql_attn   = read_2d(con, pfx+'attn_out', SEQ_LEN, HIDDEN_DIM)

# SQL attention scores
sql_scores = {}  # (q_tok, head_id) -> [k_tok -> score]
score_rows = con.execute(
    f'SELECT q_tok, k_tok, head_id, score FROM {pfx}qk_scores ORDER BY q_tok, head_id, k_tok'
).fetchall()
for q_tok, k_tok, head_id, score in score_rows:
    sql_scores[(q_tok, head_id)] = sql_scores.get((q_tok, head_id), {})
    sql_scores[(q_tok, head_id)][k_tok] = score

# SQL attention weights
sql_weights = {}
w_rows = con.execute(
    f'SELECT q_tok, k_tok, head_id, attn_weight FROM {pfx}attn_weights ORDER BY q_tok, head_id, k_tok'
).fetchall()
for q_tok, k_tok, head_id, w in w_rows:
    sql_weights[(q_tok, head_id)] = sql_weights.get((q_tok, head_id), {})
    sql_weights[(q_tok, head_id)][k_tok] = w

con.close()

# ---- PyTorch ----
print('Loading PyTorch...')
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map='cpu')
layer = model.model.layers[0]
attn = layer.self_attn
emb = model.model.embed_tokens(torch.tensor([tokens]))  # [1,3,4096]

scale = 1.0 / np.sqrt(HEAD_DIM)
norm1_pt = layer.input_layernorm(emb).squeeze(0)  # [3,4096]
q_pt = attn.q_proj(norm1_pt)  # [3,4096]
k_pt = attn.k_proj(norm1_pt)  # [3,1024]
v_pt = attn.v_proj(norm1_pt)  # [3,1024]

norm1_np = norm1_pt.detach().numpy()
q_np = (q_pt * scale).detach().numpy()   # apply scaling as in preprocess_weights.py
k_np = k_pt.detach().numpy()
v_np = v_pt.detach().numpy()

print('\n=== Intermediate comparison at layer 0 ===')
for name, sql_arr, pt_arr in [
    ('norm1', sql_norm1, norm1_np),
    ('q_proj (scaled)', sql_q, q_np),
    ('k_proj', sql_k, k_np),
    ('v_proj', sql_v, v_np),
]:
    for pos in range(SEQ_LEN):
        diff = np.abs(sql_arr[pos] - pt_arr[pos]).max()
        print(f'  {name} pos={pos}: max_diff={diff:.4e}')

# Compute PyTorch attention scores manually
# Q: [3, num_q_heads, head_dim] = [3, 32, 128]
# K: [3, num_kv_heads, head_dim] = [3, 8, 128]
nq, nkv = NUM_Q_HEADS, NUM_KV_HEADS
gs = nq // nkv  # group size = 4
q_heads = q_np.reshape(SEQ_LEN, nq, HEAD_DIM)   # [3, 32, 128]
k_heads = k_np.reshape(SEQ_LEN, nkv, HEAD_DIM)  # [3, 8, 128]
v_heads = v_np.reshape(SEQ_LEN, nkv, HEAD_DIM)  # [3, 8, 128]

# Apply RoPE manually (using extract_weights.py formula, base=500000)
rope_base = 500000.0
head_dim_half = HEAD_DIM // 2
dims = np.arange(0, HEAD_DIM, 2, dtype=np.float32)
theta = 1.0 / (rope_base ** (dims / HEAD_DIM))  # [64]

def apply_rope(x, pos):
    """x: [num_heads, head_dim]"""
    angle = pos * theta  # [64]
    cos_a = np.cos(angle)  # [64]
    sin_a = np.sin(angle)  # [64]
    x_even = x[:, 0::2]  # [heads, 64]
    x_odd  = x[:, 1::2]  # [heads, 64]
    x_rot_even = x_even * cos_a - x_odd * sin_a
    x_rot_odd  = x_odd  * cos_a + x_even * sin_a
    out = np.zeros_like(x)
    out[:, 0::2] = x_rot_even
    out[:, 1::2] = x_rot_odd
    return out

q_rot = np.stack([apply_rope(q_heads[t], t) for t in range(SEQ_LEN)])  # [3, 32, 128]
k_rot = np.stack([apply_rope(k_heads[t], t) for t in range(SEQ_LEN)])  # [3, 8, 128]

# Compute attention scores for pos=2, head=0
# head=0 uses k_head=0 (group 0)
print('\n=== Attention scores comparison (pos=2, head=0) ===')
# SQL scores for (q_tok=2, head_id=0)
sql_s = sql_scores.get((2, 0), {})
print(f'SQL scores: {sql_s}')

# PyTorch scores: Q[2, head=0] dot K[0..2, kv_head=0]
pt_scores = {k_t: float(np.dot(q_rot[2, 0], k_rot[k_t, 0])) for k_t in range(SEQ_LEN)}
print(f'PT scores:  {pt_scores}')

# Compare all heads for pos=2
print('\n=== Attention scores max diff by (q_tok, head_id) ===')
all_diffs = []
for head in range(NUM_Q_HEADS):
    kv_head = head // gs
    sql_s = sql_scores.get((2, head), {})
    pt_s = {k_t: float(np.dot(q_rot[2, head], k_rot[k_t, kv_head])) for k_t in range(SEQ_LEN)}
    diffs = [abs(sql_s.get(k_t, 0) - pt_s[k_t]) for k_t in range(SEQ_LEN)]
    all_diffs.append(max(diffs))
print(f'  max diff across all heads: {max(all_diffs):.4e}')
print(f'  mean diff across all heads: {np.mean(all_diffs):.4e}')

# Compare attention weights for pos=2
print('\n=== Attention weights comparison (pos=2, head=0) ===')
sql_w = sql_weights.get((2, 0), {})
# PyTorch softmax of scores
pt_score_arr = np.array([pt_scores[k_t] for k_t in range(SEQ_LEN)])
pt_score_arr -= pt_score_arr.max()
pt_w_arr = np.exp(pt_score_arr) / np.sum(np.exp(pt_score_arr))
pt_w = {k_t: float(pt_w_arr[k_t]) for k_t in range(SEQ_LEN)}
print(f'SQL weights: {sql_w}')
print(f'PT weights:  {pt_w}')

# Compare attn_out for all positions
print('\n=== Attention output (before o_proj) ===')
# Compute PT attn_out manually
pt_attn_out = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=np.float32)
for q_t in range(SEQ_LEN):
    for head in range(NUM_Q_HEADS):
        kv_head = head // gs
        # Compute scores (q_t attends to 0..q_t)
        scores = np.array([np.dot(q_rot[q_t, head], k_rot[k_t, kv_head])
                           for k_t in range(q_t+1)], dtype=np.float64)
        scores -= scores.max()
        w = np.exp(scores) / np.sum(np.exp(scores))
        # Weighted V
        weighted_v = np.sum(
            w[:, None] * v_heads[:q_t+1, kv_head, :].astype(np.float64), axis=0
        ).astype(np.float32)
        # Place in output at head's position
        pt_attn_out[q_t, head*HEAD_DIM:(head+1)*HEAD_DIM] = weighted_v

for pos in range(SEQ_LEN):
    diff = np.abs(sql_attn[pos] - pt_attn_out[pos]).max()
    print(f'  pos={pos}: max_diff={diff:.4e}  sql_sum={sql_attn[pos].sum():.4f}  pt_sum={pt_attn_out[pos].sum():.4f}')
