"""
run_decode.py -- Measure TranSQL+ decoding latency with KV cache on DuckDB.

Usage:
    python measure/run_decode.py \
        --db-path /path/to/weights.duckdb \
        --prompts-dir measure/prompts \
        --output results/decode.json \
        [--num-layers 32] \
        [--decode-steps 49] \
        [--breakdown]

After prefill, K_rope and V_proj tables persist as the KV cache.
Each decoding step processes a single new token, appends its K/V to the cache,
and runs attention + FFN for that token.
"""

import argparse
import json
import os
import resource
import sys
import time

import duckdb
import numpy as np

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
NUM_LAYERS   = 32

N_CHUNKS_HIDDEN = HIDDEN_DIM // CHUNK_SIZE   # 128
N_CHUNKS_FFN    = FFN_DIM // CHUNK_SIZE      # 448
N_CHUNKS_KV     = (NUM_KV_HEADS * HEAD_DIM) // CHUNK_SIZE  # 32


# ---------------------------------------------------------------------------
# SQL helpers (single-token decoding step)
# ---------------------------------------------------------------------------

def _wt(l, name):
    return f"layer_{l}_{name}"


def _decode_embed_sql(token_id, pos, out):
    """Look up embedding for a single token."""
    return [(
        f"SELECT {pos} AS row_id, e.chunk_id, e.v "
        f"FROM embed_tokens e WHERE e.row_id = {token_id}", out)]


def _decode_matmul_sql(act, weight, out, cs=CHUNK_SIZE):
    """MatMul for a single-token activation (row_id is always the current pos)."""
    dp = out + "_dp"
    return [
        (f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
         f"SUM(list_dot_product(a.v, w.v)) AS val "
         f"FROM {act} a JOIN {weight} w ON a.chunk_id = w.chunk_id "
         f"GROUP BY a.row_id, w.row_id", dp),
        (f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
         f"array_agg(val ORDER BY out_col) AS v "
         f"FROM {dp} GROUP BY act_row, out_col // {cs}", out),
    ]


# ---------------------------------------------------------------------------
# ROW2COL pivoted matmul for decode (single-token)
# ---------------------------------------------------------------------------

def _pivot_sql(table_name, n_chunks, chunk_start=0):
    """Pivot chunked table to wide format."""
    cols = ", ".join(
        f"MAX(CASE WHEN chunk_id = {chunk_start + i} "
        f"THEN CAST(v AS FLOAT[]) END) AS chunk{i}"
        for i in range(n_chunks))
    return f"SELECT row_id, {cols} FROM {table_name} GROUP BY row_id"


def _decode_pivoted_matmul_sql(act, weight, out, n_chunks, cs=CHUNK_SIZE,
                                cached_wt_pivot=False, wt_piv_override=None):
    """ROW2COL pivoted matmul for decode. Explicit pivot tables + CROSS JOIN.
    wt_piv_override: use this name for the weight pivot table instead of out+'_wt_piv'."""
    act_piv_name = out + "_act_piv"
    wt_piv_name = wt_piv_override if wt_piv_override else out + "_wt_piv"
    dp = out + "_dp"

    dot_expr = " + ".join(
        f"list_dot_product(a.chunk{c}, w.chunk{c})"
        for c in range(n_chunks))

    cross = (
        f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
        f"{dot_expr} AS val "
        f"FROM {act_piv_name} a CROSS JOIN {wt_piv_name} w")

    rechunk = (
        f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {dp} GROUP BY act_row, out_col // {cs}")

    steps = [(_pivot_sql(act, n_chunks), act_piv_name)]
    if not cached_wt_pivot:
        steps.append((_pivot_sql(weight, n_chunks), wt_piv_name))
    steps.append((cross, dp))
    steps.append((rechunk, out))
    return steps


def _decode_weight_pivot_steps(num_layers):
    """Generate all weight pivot steps for decode (run once, cache across steps)."""
    steps = []
    for l in range(num_layers):
        pfx = f"dec_l{l}_"
        wt = lambda name, layer=l: f"layer_{layer}_{name}"
        for proj, out_name, n_chunks in [
            ("q_proj", "q", N_CHUNKS_HIDDEN),
            ("k_proj", "k", N_CHUNKS_HIDDEN),
            ("v_proj", "v", N_CHUNKS_HIDDEN),
            ("o_proj", "o_proj", N_CHUNKS_HIDDEN),
            ("gate_proj", "gate", N_CHUNKS_HIDDEN),
            ("up_proj", "up", N_CHUNKS_HIDDEN),
            ("down_proj", "down", N_CHUNKS_FFN),
        ]:
            steps.append((
                _pivot_sql(wt(proj), n_chunks),
                pfx + out_name + "_wt_piv"))
    # lm_head + final_norm (decode uses dec_ prefix for logits)
    steps.append((
        _pivot_sql("lm_head", N_CHUNKS_HIDDEN),
        "dec_logits_wt_piv"))
    return steps


# ---------------------------------------------------------------------------
# Other decode SQL operators
# ---------------------------------------------------------------------------

def _decode_rmsnorm_sql(inp, gamma, out, hidden_dim=HIDDEN_DIM, eps=EPS):
    sq = out + "_sq"
    return [
        (f"SELECT a.row_id, SUM(list_dot_product(a.v, a.v)) AS ss "
         f"FROM {inp} a GROUP BY a.row_id", sq),
        (f"SELECT a.row_id, a.chunk_id, "
         f"list_transform(list_zip(a.v, g.v), x -> "
         f"CAST(x[1] * x[2] / sqrt({sq}.ss / {hidden_dim}.0 + {eps}) AS FLOAT)) AS v "
         f"FROM {inp} a "
         f"JOIN {gamma} g ON a.chunk_id = g.chunk_id "
         f"JOIN {sq} ON a.row_id = {sq}.row_id", out),
    ]


def _decode_rope_sql(q_table, rope_table, pos, out, cs=CHUNK_SIZE):
    """RoPE for a single token at given position."""
    half = cs // 2
    # DuckDB lists are 1-based: generate_series(1, half)
    return [(
        f"SELECT q.row_id, q.chunk_id, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] AS FLOAT)) AS v_even, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd "
        f"FROM {q_table} q "
        f"JOIN {rope_table} r ON r.chunk_id = q.chunk_id "
        f"WHERE r.row_id = {pos}", out)]


def _decode_qk_attn_sql(q_rope, k_cache, out,
                         num_q=NUM_Q_HEADS, num_kv=NUM_KV_HEADS,
                         head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    """QK attention: single query token against full K cache."""
    cph = head_dim // cs
    cphg = cph * (num_q // num_kv)
    return [(
        f"SELECT q.row_id AS q_tok, k.row_id AS k_tok, "
        f"q.chunk_id // {cph} AS head_id, "
        f"SUM(list_dot_product(q.v_even, k.v_even) + "
        f"list_dot_product(q.v_odd, k.v_odd)) AS score "
        f"FROM {q_rope} q JOIN {k_cache} k "
        f"ON q.chunk_id % {cph} = k.chunk_id % {cph} "
        f"AND q.chunk_id // {cphg} = k.chunk_id // {cph} "
        f"GROUP BY q.row_id, k.row_id, q.chunk_id // {cph}", out)]


def _decode_softmax_sql(inp, out):
    max_t, exp_t, sum_t = out+"_max", out+"_exp", out+"_sum"
    return [
        (f"SELECT q_tok, head_id, MAX(score) AS max_score "
         f"FROM {inp} GROUP BY q_tok, head_id", max_t),
        (f"SELECT s.q_tok, s.k_tok, s.head_id, "
         f"EXP(s.score - m.max_score) AS exp_val "
         f"FROM {inp} s "
         f"JOIN {max_t} m ON s.q_tok = m.q_tok AND s.head_id = m.head_id",
         exp_t),
        (f"SELECT q_tok, head_id, SUM(exp_val) AS sum_exp "
         f"FROM {exp_t} GROUP BY q_tok, head_id", sum_t),
        (f"SELECT e.q_tok, e.k_tok, e.head_id, "
         f"CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight "
         f"FROM {exp_t} e "
         f"JOIN {sum_t} s ON e.q_tok = s.q_tok AND e.head_id = s.head_id",
         out),
    ]


def _decode_attn_vmul_sql(attn, v_cache, out,
                           num_q=NUM_Q_HEADS, num_kv=NUM_KV_HEADS,
                           head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    """Attention x V: single query against full V cache."""
    gs = num_q // num_kv
    cph = head_dim // cs
    scalar_t = out + "_scalar"
    weighted_t = out + "_weighted"
    return [
        (f"SELECT v.row_id AS tok, v.chunk_id, "
         f"UNNEST(generate_series(0, {cs}-1)) AS dim_idx, "
         f"UNNEST(v.v) AS val "
         f"FROM {v_cache} v", scalar_t),
        (f"SELECT a.q_tok AS row_id, "
         f"a.head_id * {cph} + s.chunk_id % {cph} AS chunk_id, "
         f"s.dim_idx, SUM(a.attn_weight * s.val) AS val "
         f"FROM {attn} a JOIN {scalar_t} s ON a.k_tok = s.tok "
         f"WHERE s.chunk_id >= (a.head_id // {gs}) * {cph} "
         f"AND s.chunk_id < (a.head_id // {gs} + 1) * {cph} "
         f"GROUP BY a.q_tok, a.head_id * {cph} + s.chunk_id % {cph}, s.dim_idx",
         weighted_t),
        (f"SELECT row_id, chunk_id, "
         f"array_agg(CAST(val AS FLOAT) ORDER BY dim_idx) AS v "
         f"FROM {weighted_t} GROUP BY row_id, chunk_id", out),
    ]


def _swiglu_sql(gate, up, out):
    return [(
        f"SELECT g.row_id, g.chunk_id, "
        f"list_transform(list_zip(g.v, u.v), x -> "
        f"CAST((x[1] / (1.0 + EXP(-CAST(x[1] AS DOUBLE)))) * x[2] AS FLOAT)) AS v "
        f"FROM {gate} g JOIN {up} u "
        f"ON g.row_id = u.row_id AND g.chunk_id = u.chunk_id", out)]


def _residual_add_sql(a, b, out):
    return [(
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(list_zip(a.v, b.v), x -> "
        f"CAST(x[1] + x[2] AS FLOAT)) AS v "
        f"FROM {a} a JOIN {b} b "
        f"ON a.row_id = b.row_id AND a.chunk_id = b.chunk_id", out)]


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def run_steps(conn, steps):
    """Execute SQL steps, creating temp tables. Returns per-step timings."""
    timings = []
    for i, (sql, table_name) in enumerate(steps):
        t0 = time.perf_counter()
        try:
            conn.execute(f"CREATE TEMP TABLE {table_name} AS ({sql})")
        except Exception as e:
            print(f"  FAILED at step {i}: {table_name}")
            print(f"  SQL (first 500 chars): {sql[:500]}")
            raise
        dt = time.perf_counter() - t0
        timings.append((table_name, dt))
    return timings


def drop_temp_tables(conn, table_names):
    """Drop temp tables to free memory."""
    for t in table_names:
        conn.execute(f"DROP TABLE IF EXISTS {t}")


# ---------------------------------------------------------------------------
# Decoding pipeline
# ---------------------------------------------------------------------------

def run_prefill(conn, token_ids, num_layers, use_pivot=True):
    """Run prefill and return list of KV cache table names."""
    from run_prefill import build_full_pipeline, embed_lookup_sql, \
        weight_pivot_steps, run_steps as prefill_run_steps

    # Load tokens
    conn.execute("CREATE TEMP TABLE input_tokens "
                 "(pos INTEGER, token_id INTEGER)")
    conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                     [(i, tid) for i, tid in enumerate(token_ids)])

    # Pre-pivot weights for prefill
    if use_pivot:
        wt_steps = weight_pivot_steps(num_layers)
        prefill_run_steps(conn, wt_steps)

    pipeline = build_full_pipeline(num_layers, cached_wt=use_pivot)
    prefill_run_steps(conn, pipeline)


def run_decode_step(conn, token_id, pos, num_layers, use_pivot=True):
    """Run one decoding step for a single new token at position `pos`.
    Appends new K/V to the KV cache tables from prefill."""

    tables_to_drop = []
    c = use_pivot

    # Embed the new token
    x_name = f"dec_x_{pos}"
    steps = _decode_embed_sql(token_id, pos, x_name)
    run_steps(conn, steps)
    tables_to_drop.append(x_name)

    x_in = x_name

    for l in range(num_layers):
        pfx = f"dec_l{l}_"

        # RMSNorm
        norm1 = pfx + "norm1"
        steps = _decode_rmsnorm_sql(x_in, _wt(l, "norm1"), norm1)
        run_steps(conn, steps)
        tables_to_drop.extend([norm1 + "_sq", norm1])

        # Q/K/V projections
        q = pfx + "q"
        k = pfx + "k"
        v = pfx + "v"
        if use_pivot:
            for name, n_chunks in [(q, N_CHUNKS_HIDDEN), (k, N_CHUNKS_HIDDEN),
                                    (v, N_CHUNKS_HIDDEN)]:
                steps = _decode_pivoted_matmul_sql(
                    norm1, None, name, n_chunks, cached_wt_pivot=True)
                run_steps(conn, steps)
                tables_to_drop.extend([name + "_act_piv", name + "_dp", name])
        else:
            for name, weight in [(q, "q_proj"), (k, "k_proj"), (v, "v_proj")]:
                steps = _decode_matmul_sql(norm1, _wt(l, weight), name)
                run_steps(conn, steps)
                tables_to_drop.extend([name + "_dp", name])

        # RoPE
        q_rope = pfx + "q_rope"
        k_rope_new = pfx + "k_rope_new"
        steps = _decode_rope_sql(q, "rope", pos, q_rope)
        run_steps(conn, steps)
        tables_to_drop.append(q_rope)

        steps = _decode_rope_sql(k, "rope", pos, k_rope_new)
        run_steps(conn, steps)
        tables_to_drop.append(k_rope_new)

        # Append new K to KV cache
        k_cache = f"l{l}_k_rope"
        conn.execute(
            f"INSERT INTO {k_cache} "
            f"SELECT row_id, chunk_id, v_even, v_odd "
            f"FROM {k_rope_new}")

        # Append new V to KV cache
        v_cache = f"l{l}_v"
        conn.execute(
            f"INSERT INTO {v_cache} "
            f"SELECT row_id, chunk_id, v FROM {pfx}v")

        # QK attention (query against full K cache)
        qk = pfx + "qk"
        steps = _decode_qk_attn_sql(q_rope, k_cache, qk)
        run_steps(conn, steps)
        tables_to_drop.append(qk)

        # Softmax
        attn_w = pfx + "attn_w"
        steps = _decode_softmax_sql(qk, attn_w)
        run_steps(conn, steps)
        tables_to_drop.extend([attn_w + "_max", attn_w + "_exp",
                               attn_w + "_sum", attn_w])

        # Attention x V
        attn_out = pfx + "attn_out"
        steps = _decode_attn_vmul_sql(attn_w, v_cache, attn_out)
        run_steps(conn, steps)
        tables_to_drop.extend([attn_out + "_scalar", attn_out + "_weighted",
                               attn_out])

        # O projection
        o = pfx + "o_proj"
        if use_pivot:
            steps = _decode_pivoted_matmul_sql(
                attn_out, None, o, N_CHUNKS_HIDDEN, cached_wt_pivot=True)
            run_steps(conn, steps)
            tables_to_drop.extend([o + "_act_piv", o + "_dp", o])
        else:
            steps = _decode_matmul_sql(attn_out, _wt(l, "o_proj"), o)
            run_steps(conn, steps)
            tables_to_drop.extend([o + "_dp", o])

        # Residual add
        x_after = pfx + "x_after"
        steps = _residual_add_sql(x_in, o, x_after)
        run_steps(conn, steps)
        tables_to_drop.append(x_after)

        # RMSNorm 2
        norm2 = pfx + "norm2"
        steps = _decode_rmsnorm_sql(x_after, _wt(l, "norm2"), norm2)
        run_steps(conn, steps)
        tables_to_drop.extend([norm2 + "_sq", norm2])

        # FFN
        gate = pfx + "gate"
        up = pfx + "up"
        ffn_act = pfx + "ffn_act"
        down = pfx + "down"

        if use_pivot:
            for name, n_chunks, src in [(gate, N_CHUNKS_HIDDEN, norm2),
                                         (up, N_CHUNKS_HIDDEN, norm2),
                                         (down, N_CHUNKS_FFN, ffn_act)]:
                if name == down:
                    # SwiGLU first
                    steps = _swiglu_sql(gate, up, ffn_act)
                    run_steps(conn, steps)
                    tables_to_drop.append(ffn_act)
                steps = _decode_pivoted_matmul_sql(
                    src, None, name, n_chunks, cached_wt_pivot=True)
                run_steps(conn, steps)
                tables_to_drop.extend([name + "_act_piv", name + "_dp", name])
        else:
            steps = _decode_matmul_sql(norm2, _wt(l, "gate_proj"), gate)
            run_steps(conn, steps)
            tables_to_drop.extend([gate + "_dp", gate])

            steps = _decode_matmul_sql(norm2, _wt(l, "up_proj"), up)
            run_steps(conn, steps)
            tables_to_drop.extend([up + "_dp", up])

            steps = _swiglu_sql(gate, up, ffn_act)
            run_steps(conn, steps)
            tables_to_drop.append(ffn_act)

            steps = _decode_matmul_sql(ffn_act, _wt(l, "down_proj"), down)
            run_steps(conn, steps)
            tables_to_drop.extend([down + "_dp", down])

        # Residual add 2
        x_out = pfx + "x_out"
        steps = _residual_add_sql(x_after, down, x_out)
        run_steps(conn, steps)
        tables_to_drop.append(x_out)

        # Clean up previous x_in (not needed anymore)
        if l > 0:
            drop_temp_tables(conn, [x_in])

        x_in = x_out

    # Final norm + lm_head
    final_norm = f"dec_final_norm_{pos}"
    steps = _decode_rmsnorm_sql(x_in, "final_norm", final_norm)
    run_steps(conn, steps)
    tables_to_drop.extend([final_norm + "_sq", final_norm])

    logits = f"dec_logits_{pos}"
    if use_pivot:
        steps = _decode_pivoted_matmul_sql(
            final_norm, None, logits, N_CHUNKS_HIDDEN, cached_wt_pivot=True,
            wt_piv_override="dec_logits_wt_piv")
        run_steps(conn, steps)
        tables_to_drop.extend([logits + "_act_piv", logits + "_dp", logits])
    else:
        steps = _decode_matmul_sql(final_norm, "lm_head", logits)
        run_steps(conn, steps)
        tables_to_drop.extend([logits + "_dp", logits])

    # Get argmax token
    result = conn.execute(
        f"SELECT out_col FROM {logits}_dp "
        f"ORDER BY val DESC LIMIT 1").fetchone()
    next_token = result[0] if result else 0

    # Clean up all temp tables from this step
    drop_temp_tables(conn, tables_to_drop)

    return next_token


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def get_peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure_decode(db_path, prompt_path, num_layers, decode_steps,
                   use_pivot=True):
    with open(prompt_path) as f:
        prompt = json.load(f)
    token_ids = prompt["token_ids"]
    seq_len = len(token_ids)

    conn = duckdb.connect(db_path, read_only=False)

    # Pre-pivot decode weight tables
    if use_pivot:
        print(f"  Pre-pivoting decode weight tables...")
        t0 = time.perf_counter()
        wt_steps = _decode_weight_pivot_steps(num_layers)
        run_steps(conn, wt_steps)
        pivot_time = time.perf_counter() - t0
        print(f"  Decode weight pivoting: {pivot_time:.3f}s ({len(wt_steps)} tables)")

    # Prefill
    print(f"  Prefill (seq_len={seq_len})...")
    t0 = time.perf_counter()
    run_prefill(conn, token_ids, num_layers, use_pivot=use_pivot)
    prefill_time = time.perf_counter() - t0
    print(f"  Prefill: {prefill_time:.3f}s  "
          f"({seq_len / prefill_time:.2f} tok/s)")

    # Decoding steps
    decode_latencies = []
    current_token = 0  # dummy first token (would be argmax of logits)

    for step in range(decode_steps):
        pos = seq_len + step
        t0 = time.perf_counter()
        next_token = run_decode_step(conn, current_token, pos, num_layers,
                                     use_pivot=use_pivot)
        dt = time.perf_counter() - t0
        decode_latencies.append(dt)
        current_token = next_token
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Decode step {step+1}/{decode_steps}: {dt:.3f}s  "
                  f"({1.0/dt:.2f} tok/s)")

    peak_rss_mb = get_peak_rss_mb()
    conn.close()

    mean_decode = float(np.mean(decode_latencies))
    e2e = prefill_time + sum(decode_latencies)
    total_tokens = seq_len + decode_steps

    return {
        "seq_len": seq_len,
        "prefill_latency_s": prefill_time,
        "prefill_throughput_tok_per_s": seq_len / prefill_time,
        "decode_latencies_s": decode_latencies,
        "mean_decode_latency_s": mean_decode,
        "decode_throughput_tok_per_s": 1.0 / mean_decode,
        "end_to_end_latency_s": e2e,
        "end_to_end_throughput_tok_per_s": total_tokens / e2e,
        "peak_rss_mb": peak_rss_mb,
        "decode_steps": decode_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--prompts-dir", default="measure/prompts")
    parser.add_argument("--output", default="measure/results/decode.json")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--decode-steps", type=int, default=49)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--no-pivot", action="store_true",
                        help="Use basic matmul instead of pivoted")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []

    for length in args.lengths:
        prompt_path = os.path.join(args.prompts_dir, f"prompt_{length}.json")
        if not os.path.exists(prompt_path):
            print(f"SKIP: {prompt_path} not found")
            continue

        print(f"\nPrompt length {length}:")
        result = measure_decode(args.db_path, prompt_path,
                                args.num_layers, args.decode_steps,
                                use_pivot=not args.no_pivot)
        results.append({
            "prompt_length": length,
            **result,
        })
        print(f"  Mean decode: {result['mean_decode_latency_s']:.3f}s/step  "
              f"({result['decode_throughput_tok_per_s']:.2f} tok/s)  "
              f"E2E: {result['end_to_end_latency_s']:.3f}s  "
              f"Peak RSS: {result['peak_rss_mb']:.0f} MB")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
