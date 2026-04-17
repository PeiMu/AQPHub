"""
Diagnostic: compare SQL vs PyTorch for layer-0 output with 3 tokens.
This isolates where the error first appears in multi-token inference.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'transql', 'python'))
import duckdb, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoConfig
from run_prefill import (
    build_layer_steps, embed_lookup_sql, run_steps,
    CHUNK_SIZE, HIDDEN_DIM, rmsnorm_sql, rope_sql, qk_attn_sql,
    softmax_sql, attn_vmul_sql, pivoted_matmul_sql, residual_add_sql,
    N_CHUNKS_HIDDEN,
)

DB = 'weights.duckdb'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B'
SEQ_LEN = 3
tokens = [128000, 1, 2]


def read_table(con, name, seq_len, dim):
    rows = con.execute(
        f'SELECT row_id, chunk_id, v FROM {name} ORDER BY row_id, chunk_id'
    ).fetchall()
    arr = np.zeros((seq_len, dim), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        arr[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = v
    return arr


# ---- SQL: run layer 0, capturing intermediate tables ----
con = duckdb.connect(DB, read_only=True)
con.execute('CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)')
con.executemany('INSERT INTO input_tokens VALUES (?,?)', list(enumerate(tokens)))

steps = []
steps += embed_lookup_sql('input_tokens', 'embed_tokens', 'x_0')

# Layer 0 steps manually
pfx = 'l0_'
wt = lambda name: f'layer_0_{name}'

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

# Read intermediate SQL tables
sql_norm1 = read_table(con, pfx+'norm1_out', SEQ_LEN, HIDDEN_DIM)
sql_attn_out = read_table(con, pfx+'attn_out', SEQ_LEN, HIDDEN_DIM)
sql_o_proj = read_table(con, pfx+'o_proj', SEQ_LEN, HIDDEN_DIM)
sql_x_after_attn = read_table(con, pfx+'x_after_attn', SEQ_LEN, HIDDEN_DIM)

# Read Q and K (not rope-rotated)
sql_q = read_table(con, pfx+'q', SEQ_LEN, HIDDEN_DIM)
sql_k_rows = con.execute(f'SELECT row_id, chunk_id, v FROM {pfx}k ORDER BY row_id, chunk_id').fetchall()
sql_k = np.zeros((SEQ_LEN, 1024), dtype=np.float32)  # K is 1024-dim
for row_id, chunk_id, v in sql_k_rows:
    sql_k[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = v

# Read attention weights (q_tok, k_tok, head_id, attn_weight)
attn_rows = con.execute(
    f'SELECT q_tok, k_tok, head_id, attn_weight FROM {pfx}attn_weights ORDER BY q_tok, head_id, k_tok'
).fetchall()
print(f'\nAttention weights sample (q_tok=2):')
for r in attn_rows:
    if r[0] == 2:  # only pos=2
        print(f'  q_tok={r[0]} k_tok={r[1]} head={r[2]} w={r[3]:.6f}')

con.close()

# ---- PyTorch: run layer 0 ----
print('\nLoading PyTorch model...')
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map='cpu')

# Get intermediates using hooks
import torch.nn as nn

layer = model.model.layers[0]
emb = model.model.embed_tokens(torch.tensor([tokens]))  # [1, 3, 4096]
emb_np = emb.squeeze(0).detach().numpy()

# RMSNorm output
pos_ids = torch.arange(SEQ_LEN).unsqueeze(0)
cache_pos = torch.arange(SEQ_LEN)

try:
    pt_out, _, cache = layer(
        emb, position_ids=pos_ids, cache_position=cache_pos,
        use_cache=True
    )
except TypeError:
    pt_out, _, cache = layer(emb, position_ids=pos_ids, use_cache=True)

pt_out_np = pt_out.squeeze(0).detach().numpy()

# Get norm1 output
norm1_out_pt = layer.input_layernorm(emb).squeeze(0).detach().numpy()

# Compare norms
print('\n--- norm1 output comparison ---')
for pos in range(SEQ_LEN):
    diff = np.abs(sql_norm1[pos] - norm1_out_pt[pos]).max()
    print(f'  pos={pos}: max_diff={diff:.4e}')

# Compare Q proj (with 1/sqrt(128) scaling already in weights)
scale = 1.0 / np.sqrt(128)
q_proj = layer.self_attn.q_proj
k_proj = layer.self_attn.k_proj
v_proj = layer.self_attn.v_proj

pt_q = q_proj(torch.tensor(norm1_out_pt)).detach().numpy() * scale  # [3, 4096]
pt_k = k_proj(torch.tensor(norm1_out_pt)).detach().numpy()           # [3, 1024]

print('\n--- Q proj comparison (includes 1/sqrt(128) scale) ---')
for pos in range(SEQ_LEN):
    diff = np.abs(sql_q[pos] - pt_q[pos]).max()
    print(f'  pos={pos}: max_diff={diff:.4e}')

print('\n--- K proj comparison ---')
for pos in range(SEQ_LEN):
    diff = np.abs(sql_k[pos] - pt_k[pos]).max()
    print(f'  pos={pos}: max_diff={diff:.4e}')

print('\n--- attn_out comparison ---')
# Need PyTorch attn_out. Let's extract it via hook.
pt_attn_out_hook = {}
def hook(mod, inp, out):
    pt_attn_out_hook['out'] = out

handle = layer.self_attn.o_proj.register_forward_pre_hook(hook)
try:
    pt_out2, _, _ = layer(emb, position_ids=pos_ids, cache_position=cache_pos, use_cache=True)
except TypeError:
    pt_out2, _, _ = layer(emb, position_ids=pos_ids, use_cache=True)
handle.remove()

pt_attn_out_np = pt_attn_out_hook['out'][0].squeeze(0).detach().numpy()
for pos in range(SEQ_LEN):
    diff = np.abs(sql_attn_out[pos] - pt_attn_out_np[pos]).max()
    print(f'  pos={pos}: max_diff={diff:.4e}  sql_mean={sql_attn_out[pos].mean():.4f}  pt_mean={pt_attn_out_np[pos].mean():.4f}')

print('\n--- x_after_attn comparison ---')
for pos in range(SEQ_LEN):
    diff = np.abs(sql_x_after_attn[pos] - pt_out_np[pos]).max()
    # Actually pt_out_np is x_out (after FFN). Need x_after_attn specifically.

# Actually let's just compare final layer-0 output
print('\n--- layer-0 x_out (after FFN) comparison ---')
for pos in range(SEQ_LEN):
    diff = np.abs(sql_x_after_attn[pos] - pt_out_np[pos]).max()
    print(f'  pos={pos}: sql_x_after_attn vs pt_x_out: max_diff={diff:.4e}')
