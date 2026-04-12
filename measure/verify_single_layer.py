#!/usr/bin/env python3
"""
verify_single_layer.py -- Run 1-layer inference with REAL weights from weights.duckdb
and compare against a NumPy reference computation.

This gradually verifies both weight correctness AND SQL correctness together,
without needing to regenerate the weight file.

Steps:
  1. Extract layer 0 weights from weights.duckdb into NumPy arrays
  2. Create a known input (embed token 1)
  3. Run 1-layer SQL pipeline using run_prefill.py
  4. Run the same computation in pure NumPy
  5. Compare results

Usage:
    python measure/verify_single_layer.py --db-path weights.duckdb
    python measure/verify_single_layer.py --db-path weights.duckdb --token-id 42
    python measure/verify_single_layer.py --db-path weights.duckdb --check-pytorch

Levels of verification:
  Level 1 (default): SQL pipeline produces valid output (no NULLs/NaNs, correct shape)
  Level 2 (--check-numpy): SQL output matches NumPy reference using extracted weights
  Level 3 (--check-pytorch): SQL output matches PyTorch HuggingFace model (requires transformers)
"""

import argparse
import os
import sys
import time

import numpy as np
import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "transql", "python"))


# ---------------------------------------------------------------------------
# Model constants (Llama3-8B)
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 32
HIDDEN_DIM   = 4096
NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
FFN_DIM      = 14336
EPS          = 1e-5
KV_DIM       = NUM_KV_HEADS * HEAD_DIM  # 1024


# ---------------------------------------------------------------------------
# Weight extraction from DuckDB
# ---------------------------------------------------------------------------

def extract_2d_weight(conn, table_name, out_dim, in_dim):
    """Extract a 2D weight table [out_dim, in_dim] from DuckDB."""
    rows = conn.execute(
        f'SELECT row_id, chunk_id, v FROM "{table_name}" '
        f'ORDER BY row_id, chunk_id'
    ).fetchall()
    w = np.zeros((out_dim, in_dim), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        w[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = np.array(v, dtype=np.float32)
    return w


def extract_1d_weight(conn, table_name, dim):
    """Extract a 1D norm weight [dim] from DuckDB."""
    rows = conn.execute(
        f'SELECT chunk_id, v FROM "{table_name}" ORDER BY chunk_id'
    ).fetchall()
    w = np.zeros(dim, dtype=np.float32)
    for chunk_id, v in rows:
        w[chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = np.array(v, dtype=np.float32)
    return w


def extract_rope(conn, seq_len):
    """Extract RoPE cos/sin tables.
    Returns cos, sin as [seq_len, dim//2] flat arrays."""
    rows = conn.execute(
        "SELECT row_id, chunk_id, cos, sin FROM rope "
        "WHERE row_id < ? ORDER BY row_id, chunk_id",
        [seq_len]
    ).fetchall()
    half = CHUNK_SIZE // 2
    n_chunks = HIDDEN_DIM // CHUNK_SIZE
    cos = np.zeros((seq_len, n_chunks, half), dtype=np.float32)
    sin = np.zeros((seq_len, n_chunks, half), dtype=np.float32)
    for row_id, chunk_id, c, s in rows:
        if chunk_id < n_chunks:
            cos[row_id, chunk_id] = np.array(c, dtype=np.float32)
            sin[row_id, chunk_id] = np.array(s, dtype=np.float32)
    return cos, sin


def extract_embedding(conn, token_id):
    """Extract a single token's embedding vector."""
    rows = conn.execute(
        "SELECT chunk_id, v FROM embed_tokens WHERE row_id = ? ORDER BY chunk_id",
        [token_id]
    ).fetchall()
    emb = np.zeros(HIDDEN_DIM, dtype=np.float32)
    for chunk_id, v in rows:
        emb[chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = np.array(v, dtype=np.float32)
    return emb


# ---------------------------------------------------------------------------
# NumPy reference forward pass
# ---------------------------------------------------------------------------

def numpy_rmsnorm(x, gamma, eps=EPS):
    ss = np.sum(x ** 2, axis=-1, keepdims=True)
    rms = np.sqrt(ss / x.shape[-1] + eps)
    return (x / rms * gamma).astype(np.float32)


def numpy_rope(q, cos, sin):
    """q: [seq, dim], cos/sin: [seq, dim//2]."""
    q_even = q[:, 0::2]
    q_odd  = q[:, 1::2]
    v_even = (q_even * cos - q_odd * sin).astype(np.float32)
    v_odd  = (q_odd * cos + q_even * sin).astype(np.float32)
    return v_even, v_odd


def numpy_qk_attn(q_even, q_odd, k_even, k_odd, num_q, num_kv, head_dim):
    gs = num_q // num_kv
    seq = q_even.shape[0]
    half_hd = head_dim // 2
    scores = np.zeros((seq, seq, num_q), dtype=np.float64)
    for h in range(num_q):
        kv_h = h // gs
        qe = q_even[:, h*half_hd:(h+1)*half_hd]
        qo = q_odd[:, h*half_hd:(h+1)*half_hd]
        ke = k_even[:, kv_h*half_hd:(kv_h+1)*half_hd]
        ko = k_odd[:, kv_h*half_hd:(kv_h+1)*half_hd]
        scores[:, :, h] = qe @ ke.T + qo @ ko.T
    return scores.astype(np.float32)


def numpy_softmax_causal(scores):
    seq = scores.shape[0]
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    scores_masked = scores.copy()
    scores_masked[mask] = -1e30
    max_s = np.max(scores_masked, axis=1, keepdims=True)
    exp_s = np.exp(scores_masked - max_s)
    exp_s[mask] = 0
    sum_s = np.sum(exp_s, axis=1, keepdims=True)
    return (exp_s / sum_s).astype(np.float32)


def numpy_attn_vmul(attn, v, num_q, num_kv, head_dim):
    gs = num_q // num_kv
    seq = attn.shape[0]
    out = np.zeros((seq, num_q * head_dim), dtype=np.float32)
    for h in range(num_q):
        kv_h = h // gs
        v_h = v[:, kv_h*head_dim:(kv_h+1)*head_dim]
        out[:, h*head_dim:(h+1)*head_dim] = (attn[:, :, h] @ v_h).astype(np.float32)
    return out


def numpy_forward_layer(x, norm1_w, norm2_w, wq, wk, wv, wo,
                         wg, wu, wd, cos, sin):
    """Full single-layer forward pass in NumPy."""
    seq = x.shape[0]

    # Attention
    x_norm1 = numpy_rmsnorm(x, norm1_w)
    q = (x_norm1 @ wq.T).astype(np.float32)
    k = (x_norm1 @ wk.T).astype(np.float32)
    v = (x_norm1 @ wv.T).astype(np.float32)

    q_cos = cos[:seq].reshape(seq, -1)
    q_sin = sin[:seq].reshape(seq, -1)
    q_even, q_odd = numpy_rope(q, q_cos, q_sin)

    kv_chunks = KV_DIM // CHUNK_SIZE
    half = CHUNK_SIZE // 2
    k_cos = cos[:seq, :kv_chunks, :].reshape(seq, -1)
    k_sin = sin[:seq, :kv_chunks, :].reshape(seq, -1)
    k_even, k_odd = numpy_rope(k, k_cos, k_sin)

    scores = numpy_qk_attn(q_even, q_odd, k_even, k_odd,
                            NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
    attn = numpy_softmax_causal(scores)
    attn_out = numpy_attn_vmul(attn, v, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
    o = (attn_out @ wo.T).astype(np.float32)
    x_after_attn = (x + o).astype(np.float32)

    # FFN
    x_norm2 = numpy_rmsnorm(x_after_attn, norm2_w)
    gate = (x_norm2 @ wg.T).astype(np.float32)
    up = (x_norm2 @ wu.T).astype(np.float32)
    # SwiGLU: SiLU(gate) * up (NOT gate * SiLU(up))
    silu_gate = gate / (1.0 + np.exp(-gate.astype(np.float64)))
    ffn_act = (silu_gate * up).astype(np.float32)
    down = (ffn_act @ wd.T).astype(np.float32)
    x_out = (x_after_attn + down).astype(np.float32)

    return x_out


# ---------------------------------------------------------------------------
# SQL pipeline execution
# ---------------------------------------------------------------------------

def sql_forward_layer(conn, token_ids):
    """Run 1-layer forward pass using SQL from run_prefill.py."""
    from run_prefill import (build_full_pipeline, embed_lookup_sql,
                             run_steps as prefill_run_steps)

    # Load tokens
    conn.execute("CREATE TEMP TABLE input_tokens (pos INTEGER, token_id INTEGER)")
    conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                     [(i, t) for i, t in enumerate(token_ids)])

    # Build 1-layer pipeline (no pivot for simplicity)
    pipeline = build_full_pipeline(num_layers=1, cached_wt=False)

    print(f"  Executing {len(pipeline)} SQL steps...")
    t0 = time.perf_counter()
    prefill_run_steps(conn, pipeline)
    dt = time.perf_counter() - t0
    print(f"  SQL pipeline: {dt:.3f}s")

    # Read output (l0_x_out is the 1-layer output)
    rows = conn.execute(
        "SELECT row_id, chunk_id, v FROM l0_x_out ORDER BY row_id, chunk_id"
    ).fetchall()
    seq_len = max(r[0] for r in rows) + 1
    result = np.zeros((seq_len, HIDDEN_DIM), dtype=np.float32)
    for row_id, chunk_id, v in rows:
        result[row_id, chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE] = np.array(v, dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify single-layer inference with real weights")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--token-id", type=int, default=1,
                        help="Token ID for single-token test")
    parser.add_argument("--seq-len", type=int, default=3,
                        help="Number of tokens (repeats token-id)")
    parser.add_argument("--check-numpy", action="store_true", default=True,
                        help="Compare SQL output vs NumPy reference (default)")
    parser.add_argument("--check-pytorch", action="store_true",
                        help="Compare SQL output vs PyTorch HuggingFace model")
    args = parser.parse_args()

    print(f"Verifying single-layer inference: {args.db_path}")
    print(f"  token_id={args.token_id}, seq_len={args.seq_len}\n")

    conn = duckdb.connect(args.db_path, read_only=True)

    # --- Level 1: SQL produces valid output ---
    print("=== Level 1: SQL pipeline validity ===")
    token_ids = [args.token_id] * args.seq_len

    try:
        sql_out = sql_forward_layer(conn, token_ids)
    except Exception as e:
        print(f"  FAIL: SQL pipeline error: {e}")
        sys.exit(1)

    # Check shape
    if sql_out.shape != (args.seq_len, HIDDEN_DIM):
        print(f"  FAIL: wrong shape {sql_out.shape}, expected ({args.seq_len}, {HIDDEN_DIM})")
        sys.exit(1)
    print(f"  Shape: OK ({sql_out.shape})")

    # Check no NaN/Inf
    if np.any(np.isnan(sql_out)):
        nan_count = np.sum(np.isnan(sql_out))
        print(f"  FAIL: {nan_count} NaN values in output")
        sys.exit(1)
    if np.any(np.isinf(sql_out)):
        inf_count = np.sum(np.isinf(sql_out))
        print(f"  FAIL: {inf_count} Inf values in output")
        sys.exit(1)
    print(f"  No NaN/Inf: OK")

    # Check reasonable value range
    out_std = np.std(sql_out)
    out_mean = np.mean(sql_out)
    out_max = np.max(np.abs(sql_out))
    print(f"  Stats: mean={out_mean:.6f}, std={out_std:.6f}, max_abs={out_max:.4f}")
    if out_max > 10000:
        print(f"  WARNING: very large values, possible numerical issue")
    if out_std < 1e-8:
        print(f"  WARNING: near-zero std, possible all-zeros output")
    print(f"  Level 1: PASS\n")

    # --- Level 2: SQL matches NumPy reference ---
    if args.check_numpy:
        print("=== Level 2: SQL vs NumPy reference ===")
        print("  Extracting weights from DuckDB...")

        t0 = time.perf_counter()
        norm1_w = extract_1d_weight(conn, "layer_0_norm1", HIDDEN_DIM)
        norm2_w = extract_1d_weight(conn, "layer_0_norm2", HIDDEN_DIM)
        wq = extract_2d_weight(conn, "layer_0_q_proj", HIDDEN_DIM, HIDDEN_DIM)
        wk = extract_2d_weight(conn, "layer_0_k_proj", KV_DIM, HIDDEN_DIM)
        wv = extract_2d_weight(conn, "layer_0_v_proj", KV_DIM, HIDDEN_DIM)
        wo = extract_2d_weight(conn, "layer_0_o_proj", HIDDEN_DIM, HIDDEN_DIM)
        wg = extract_2d_weight(conn, "layer_0_gate_proj", FFN_DIM, HIDDEN_DIM)
        wu = extract_2d_weight(conn, "layer_0_up_proj", FFN_DIM, HIDDEN_DIM)
        wd = extract_2d_weight(conn, "layer_0_down_proj", HIDDEN_DIM, FFN_DIM)
        cos, sin = extract_rope(conn, max(args.seq_len, 128))
        dt = time.perf_counter() - t0
        print(f"  Weight extraction: {dt:.1f}s")

        # Get embedding for input tokens
        x = np.zeros((args.seq_len, HIDDEN_DIM), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            x[i] = extract_embedding(conn, tid)

        print("  Running NumPy forward pass...")
        t0 = time.perf_counter()
        numpy_out = numpy_forward_layer(x, norm1_w, norm2_w, wq, wk, wv, wo,
                                         wg, wu, wd, cos, sin)
        dt = time.perf_counter() - t0
        print(f"  NumPy forward: {dt:.3f}s")

        # Compare
        max_diff = np.max(np.abs(sql_out - numpy_out))
        mean_diff = np.mean(np.abs(sql_out - numpy_out))
        rel_err = max_diff / (np.max(np.abs(numpy_out)) + 1e-10)
        print(f"  Max abs diff:  {max_diff:.8f}")
        print(f"  Mean abs diff: {mean_diff:.8f}")
        print(f"  Relative err:  {rel_err:.8f}")

        if max_diff < 0.01:
            print(f"  Level 2: PASS (max diff < 0.01)")
        elif max_diff < 0.1:
            print(f"  Level 2: PASS (max diff < 0.1, acceptable for float32)")
        elif max_diff < 1.0:
            print(f"  Level 2: WARNING (max diff < 1.0, may indicate precision issue)")
        else:
            print(f"  Level 2: FAIL (max diff >= 1.0, likely a bug)")
            # Print which positions diverge most
            diffs = np.abs(sql_out - numpy_out)
            worst_idx = np.unravel_index(np.argmax(diffs), diffs.shape)
            print(f"    Worst at [{worst_idx}]: "
                  f"sql={sql_out[worst_idx]:.6f} vs numpy={numpy_out[worst_idx]:.6f}")
        print()

    # --- Level 3: SQL matches PyTorch ---
    if args.check_pytorch:
        print("=== Level 3: SQL vs PyTorch HuggingFace ===")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("  Loading Llama3-8B from HuggingFace...")
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            model.eval()

            # Run single layer
            with torch.no_grad():
                emb = model.model.embed_tokens(
                    torch.tensor([token_ids]))  # [1, seq, hidden]
                layer = model.model.layers[0]
                pos_ids = torch.arange(args.seq_len).unsqueeze(0)
                pt_out = layer(emb, position_ids=pos_ids)[0]
                pt_out = pt_out.squeeze(0).numpy()

            max_diff = np.max(np.abs(sql_out - pt_out))
            print(f"  Max abs diff: {max_diff:.8f}")
            if max_diff < 0.1:
                print(f"  Level 3: PASS")
            else:
                print(f"  Level 3: FAIL")

        except ImportError:
            print("  SKIP: transformers/torch not installed")
        except Exception as e:
            print(f"  ERROR: {e}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
