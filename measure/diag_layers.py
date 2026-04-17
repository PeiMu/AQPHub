"""
Diagnostic: compare SQL x_out at each layer checkpoint (1, 2, 4, 8, 16, 32)
against PyTorch for a 3-token sequence. Identifies which layer the error
first appears and how quickly it grows.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'transql', 'python'))
import duckdb, numpy as np, torch
from transformers import AutoModelForCausalLM
from run_prefill import (
    build_full_pipeline, build_layer_steps, embed_lookup_sql, run_steps,
    CHUNK_SIZE, HIDDEN_DIM, N_CHUNKS_HIDDEN, NUM_LAYERS,
)

DB = 'weights.duckdb'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B'
SEQ_LEN = 3
tokens = [128000, 1, 2]
CHECK_LAYERS = [0, 1, 2, 3, 7, 15, 31]  # layers to compare


def read_hidden(con, name, seq_len=SEQ_LEN):
    rows = con.execute(
        f'SELECT row_id, chunk_id, v FROM {name} ORDER BY row_id, chunk_id'
    ).fetchall()
    arr = np.zeros((seq_len, HIDDEN_DIM), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        arr[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = v
    return arr


# ---- SQL: run full pipeline, save intermediate x_out per layer ----
print("Running SQL pipeline...")
con = duckdb.connect(DB, read_only=True)
con.execute('CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)')
con.executemany('INSERT INTO input_tokens VALUES (?,?)', list(enumerate(tokens)))

steps = []
steps += embed_lookup_sql('input_tokens', 'embed_tokens', 'x_0')

x_in = 'x_0'
layer_outs = {}
for l in range(NUM_LAYERS):
    layer_steps, x_out = build_layer_steps(l, x_in, cached_wt=False)
    steps += layer_steps
    x_in = x_out
    if l in CHECK_LAYERS:
        layer_outs[l] = x_out

run_steps(con, steps)

sql_hidden = {}
for l, name in layer_outs.items():
    sql_hidden[l] = read_hidden(con, name)
con.close()
print("SQL done.")

# ---- PyTorch: run incrementally, save x_out per layer ----
print("Loading PyTorch...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map='cpu'
)
emb = model.model.embed_tokens(torch.tensor([tokens]))  # [1, 3, 4096]
pt_hidden = {}
x_pt = emb
pos_ids = torch.arange(SEQ_LEN).unsqueeze(0)
cache_pos = torch.arange(SEQ_LEN)

# Causal mask: lower-triangular (Llama uses -inf for masked positions)
causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).unsqueeze(0).unsqueeze(0)
causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float32).min

for l in range(NUM_LAYERS):
    layer = model.model.layers[l]
    try:
        out = layer(x_pt, attention_mask=causal_mask, position_ids=pos_ids,
                    cache_position=cache_pos)
    except TypeError:
        out = layer(x_pt, attention_mask=causal_mask, position_ids=pos_ids)
    x_pt = out[0]
    if l in CHECK_LAYERS:
        pt_hidden[l] = x_pt.squeeze(0).detach().numpy()

print("PyTorch done.\n")

# ---- Compare ----
print(f"{'Layer':>6}  {'pos=0':>10}  {'pos=1':>10}  {'pos=2':>10}")
print("-" * 46)
for l in CHECK_LAYERS:
    diffs = []
    for pos in range(SEQ_LEN):
        diff = np.abs(sql_hidden[l][pos] - pt_hidden[l][pos]).max()
        diffs.append(diff)
    print(f"{l:>6}  {diffs[0]:>10.4e}  {diffs[1]:>10.4e}  {diffs[2]:>10.4e}")
