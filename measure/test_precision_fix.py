"""
Quick test: after the pivoted_matmul DOUBLE precision fix,
compare SQL logits vs PyTorch for 5 tokens.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'transql', 'python'))
sys.path.insert(0, os.path.dirname(__file__))

import duckdb, numpy as np, torch
from transformers import AutoModelForCausalLM
from run_prefill import build_full_pipeline, run_steps, CHUNK_SIZE, HIDDEN_DIM, NUM_LAYERS

DB = 'weights.duckdb'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B'
TOKEN_IDS = [128000, 1, 2, 3, 4]
SEQ_LEN = 5
VOCAB = 128256

# ---- SQL ----
con = duckdb.connect(DB, read_only=True)
con.execute('CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)')
con.executemany('INSERT INTO input_tokens VALUES (?,?)', list(enumerate(TOKEN_IDS)))
pipeline = build_full_pipeline(NUM_LAYERS)
run_steps(con, pipeline)
rows = con.execute('SELECT act_row, out_col, val FROM logits_dp ORDER BY act_row, out_col').fetchall()
sql_logits = np.zeros((SEQ_LEN, VOCAB), dtype=np.float64)
for act_row, out_col, val in rows:
    sql_logits[act_row, out_col] = val
con.close()
print('SQL done')

# ---- PyTorch ----
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, device_map='cpu')
with torch.no_grad():
    out = model(torch.tensor([TOKEN_IDS]))
pt_logits = out.logits[0].numpy()
print('PT done\n')

print('=== Top-5 match per position ===')
for pos in range(SEQ_LEN - 1):
    sql_top5 = np.argsort(sql_logits[pos])[-5:][::-1].tolist()
    pt_top5  = np.argsort(pt_logits[pos])[-5:][::-1].tolist()
    match = '✓' if sql_top5 == pt_top5 else '✗'
    print(f'pos={pos}: {match}  SQL={sql_top5}  PT={pt_top5}')

print()
print('=== Logit diff at next-token position ===')
for pos in range(SEQ_LEN - 1):
    target = TOKEN_IDS[pos + 1]
    diff = abs(float(sql_logits[pos, target]) - float(pt_logits[pos, target]))
    print(f'pos={pos} -> target={target}: SQL={sql_logits[pos,target]:.6f}  '
          f'PT={pt_logits[pos,target]:.6f}  diff={diff:.2e}')
