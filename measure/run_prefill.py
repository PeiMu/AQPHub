"""
run_prefill.py -- Measure TranSQL+ prefill latency on DuckDB.

Usage:
    python measure/run_prefill.py \
        --db-path /path/to/weights.duckdb \
        --prompts-dir measure/prompts \
        --output results/prefill.json \
        [--num-layers 32] \
        [--breakdown]

Measures wall-clock time from loading tokens to completing all SQL steps
(i.e., time to produce the logits for the first output token).

Reports: latency, throughput (tokens/sec), peak RAM (RSS), DB file size.
Connection is reused across runs; weight pivots are cached.
"""

import argparse
import json
import os
import resource
import sys
import time

import duckdb
import numpy as np

# Add transql python path for imports
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
# SQL template mirrors (matches sql_template.cpp)
# ---------------------------------------------------------------------------

def embed_lookup_sql(tok_table, embed_table, out):
    return [(
        f"SELECT t.pos AS row_id, e.chunk_id, e.v "
        f"FROM {tok_table} t "
        f"JOIN {embed_table} e ON e.row_id = t.token_id "
        f"ORDER BY t.pos, e.chunk_id", out)]


def matmul_sql(act, weight, out, cs=CHUNK_SIZE):
    dp = out + "_dp"
    return [
        (f"SELECT a.row_id AS act_row, w.row_id AS out_col, "
         f"SUM(list_dot_product(a.v, w.v)) AS val "
         f"FROM {act} a JOIN {weight} w ON a.chunk_id = w.chunk_id "
         f"GROUP BY a.row_id, w.row_id", dp),
        (f"SELECT act_row AS row_id, out_col // {cs} AS chunk_id, "
         f"array_agg(CAST(val AS FLOAT) ORDER BY out_col) AS v "
         f"FROM {dp} GROUP BY act_row, out_col // {cs}", out),
    ]


# ---------------------------------------------------------------------------
# ROW2COL pivoted matmul (Section 4.3)
# ---------------------------------------------------------------------------

def pivot_sql(table_name, n_chunks, chunk_start=0):
    """Pivot chunked table to wide format using DuckDB native PIVOT syntax.
    Produces (row_id, chunk0, ..., chunk{n_chunks-1}) with FLOAT[32] columns."""
    in_cols = ", ".join(
        f"{chunk_start + i} AS chunk{i}"
        for i in range(n_chunks))
    return (f"SELECT * FROM (PIVOT {table_name} "
            f"ON chunk_id IN ({in_cols}) USING FIRST(v) GROUP BY row_id)")


def pivoted_matmul_sql(act, weight, out, n_chunks, cs=CHUNK_SIZE,
                       cached_wt_pivot=False):
    """ROW2COL pivoted matmul (Section 4.3). Explicit pivot tables + CROSS JOIN.
    If cached_wt_pivot=True, assumes weight pivot table already exists."""
    act_piv_name = out + "_act_piv"
    wt_piv_name = out + "_wt_piv"
    dp = out + "_dp"

    # list_dot_product(FLOAT[], FLOAT[]) → FLOAT in DuckDB; FLOAT+FLOAT=FLOAT.
    # Per-layer float32 error is ~1.68e-4, within acceptable range (~5e-3 over
    # 32 layers) given RMSNorm normalizes between layers. Matches llama.cpp
    # semantics (float32 throughout). Use DOUBLE[] inputs only if PPL degrades.
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

    steps = [(pivot_sql(act, n_chunks), act_piv_name)]
    if not cached_wt_pivot:
        steps.append((pivot_sql(weight, n_chunks), wt_piv_name))
    steps.append((cross, dp))
    steps.append((rechunk, out))
    return steps


def weight_pivot_steps(num_layers):
    """Generate all weight pivot steps (run once, cache across measurements)."""
    steps = []
    for l in range(num_layers):
        pfx = f"l{l}_"
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
                pivot_sql(wt(proj), n_chunks),
                pfx + out_name + "_wt_piv"))
    # lm_head
    steps.append((
        pivot_sql("lm_head", N_CHUNKS_HIDDEN),
        "logits_wt_piv"))
    return steps


# ---------------------------------------------------------------------------
# Other SQL operators
# ---------------------------------------------------------------------------

def rmsnorm_sql(inp, gamma, out, hidden_dim=HIDDEN_DIM, eps=EPS,
                cs=CHUNK_SIZE):
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


def rope_sql(q_table, rope_table, out, cs=CHUNK_SIZE):
    half = cs // 2
    # DuckDB lists are 1-based: generate_series(1, half), v[2*i-1] = even, v[2*i] = odd
    return [(
        f"SELECT q.row_id, q.chunk_id, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i]) AS v_even, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i]) AS v_odd "
        f"FROM {q_table} q "
        f"JOIN {rope_table} r ON r.chunk_id = q.chunk_id AND r.row_id = q.row_id",
        out)]


def qk_attn_sql(q_rope, k_rope, out, num_q=NUM_Q_HEADS,
                num_kv=NUM_KV_HEADS, head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    cph = head_dim // cs
    cphg = cph * (num_q // num_kv)
    return [(
        f"SELECT q.row_id AS q_tok, k.row_id AS k_tok, "
        f"q.chunk_id // {cph} AS head_id, "
        f"SUM(list_dot_product(q.v_even, k.v_even) + "
        f"list_dot_product(q.v_odd, k.v_odd)) AS score "
        f"FROM {q_rope} q JOIN {k_rope} k "
        f"ON q.chunk_id % {cph} = k.chunk_id % {cph} "
        f"AND q.chunk_id // {cphg} = k.chunk_id // {cph} "
        f"AND k.row_id <= q.row_id "
        f"GROUP BY q.row_id, k.row_id, q.chunk_id // {cph}", out)]


def softmax_sql(inp, out):
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


def attn_vmul_sql(attn, v_table, out, num_q=NUM_Q_HEADS,
                  num_kv=NUM_KV_HEADS, head_dim=HEAD_DIM, cs=CHUNK_SIZE):
    gs = num_q // num_kv
    cph = head_dim // cs
    scalar_t = out + "_scalar"
    weighted_t = out + "_weighted"
    return [
        (f"SELECT v.row_id AS tok, v.chunk_id, "
         f"UNNEST(generate_series(0, {cs}-1)) AS dim_idx, "
         f"UNNEST(v.v) AS val "
         f"FROM {v_table} v", scalar_t),
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


def swiglu_sql(gate, up, out):
    return [(
        f"SELECT g.row_id, g.chunk_id, "
        f"list_transform(list_zip(g.v, u.v), x -> "
        f"CAST((x[1] / (1.0 + EXP(-x[1]))) * x[2] AS FLOAT)) AS v "
        f"FROM {gate} g JOIN {up} u "
        f"ON g.row_id = u.row_id AND g.chunk_id = u.chunk_id", out)]


def residual_add_sql(a, b, out):
    return [(
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(list_zip(a.v, b.v), x -> x[1] + x[2]) AS v "
        f"FROM {a} a JOIN {b} b "
        f"ON a.row_id = b.row_id AND a.chunk_id = b.chunk_id", out)]


# ---------------------------------------------------------------------------
# Build SQL pipeline for full model
# ---------------------------------------------------------------------------

def build_layer_steps(l, x_in, cached_wt=False):
    """Build SQL steps for one transformer layer. Returns (steps, x_out_name)."""
    pfx = f"l{l}_"
    wt = lambda name: f"layer_{l}_{name}"
    c = cached_wt

    steps = []
    norm1 = pfx + "norm1_out"
    steps += rmsnorm_sql(x_in, wt("norm1"), norm1)

    q = pfx + "q"
    steps += pivoted_matmul_sql(norm1, wt("q_proj"), q, N_CHUNKS_HIDDEN, cached_wt_pivot=c)
    k = pfx + "k"
    steps += pivoted_matmul_sql(norm1, wt("k_proj"), k, N_CHUNKS_HIDDEN, cached_wt_pivot=c)
    v = pfx + "v"
    steps += pivoted_matmul_sql(norm1, wt("v_proj"), v, N_CHUNKS_HIDDEN, cached_wt_pivot=c)

    q_rope = pfx + "q_rope"
    steps += rope_sql(q, "rope", q_rope)
    k_rope = pfx + "k_rope"
    steps += rope_sql(k, "rope", k_rope)

    qk = pfx + "qk_scores"
    steps += qk_attn_sql(q_rope, k_rope, qk)
    attn_w = pfx + "attn_weights"
    steps += softmax_sql(qk, attn_w)

    attn_out = pfx + "attn_out"
    steps += attn_vmul_sql(attn_w, v, attn_out)

    o = pfx + "o_proj"
    steps += pivoted_matmul_sql(attn_out, wt("o_proj"), o, N_CHUNKS_HIDDEN, cached_wt_pivot=c)

    x_after_attn = pfx + "x_after_attn"
    steps += residual_add_sql(x_in, o, x_after_attn)

    norm2 = pfx + "norm2_out"
    steps += rmsnorm_sql(x_after_attn, wt("norm2"), norm2)

    gate = pfx + "gate"
    steps += pivoted_matmul_sql(norm2, wt("gate_proj"), gate, N_CHUNKS_HIDDEN, cached_wt_pivot=c)
    up = pfx + "up"
    steps += pivoted_matmul_sql(norm2, wt("up_proj"), up, N_CHUNKS_HIDDEN, cached_wt_pivot=c)
    ffn_act = pfx + "ffn_act"
    steps += swiglu_sql(gate, up, ffn_act)
    down = pfx + "down"
    steps += pivoted_matmul_sql(ffn_act, wt("down_proj"), down, N_CHUNKS_FFN, cached_wt_pivot=c)

    x_out = pfx + "x_out"
    steps += residual_add_sql(x_after_attn, down, x_out)

    return steps, x_out


def build_full_pipeline(num_layers, cached_wt=False):
    """Build the complete SQL pipeline for prefill."""
    steps = []
    steps += embed_lookup_sql("input_tokens", "embed_tokens", "x_0")

    x_in = "x_0"
    for l in range(num_layers):
        layer_steps, x_out = build_layer_steps(l, x_in, cached_wt)
        steps += layer_steps
        x_in = x_out

    steps += rmsnorm_sql(x_in, "final_norm", "final_norm_out")
    steps += pivoted_matmul_sql("final_norm_out", "lm_head", "logits",
                                N_CHUNKS_HIDDEN, cached_wt_pivot=cached_wt)
    return steps


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def run_steps(conn, steps):
    """Execute SQL steps, creating temp tables. Returns per-step timings."""
    timings = []
    for i, (sql, table_name) in enumerate(steps):
        t0 = time.perf_counter()
        try:
            conn.execute(
                f"CREATE TEMP TABLE {table_name} AS ({sql})")
        except Exception as e:
            print(f"  FAILED at step {i}: {table_name}")
            print(f"  SQL (first 500 chars): {sql[:500]}")
            raise
        dt = time.perf_counter() - t0
        timings.append((table_name, dt))
    return timings


def drop_temp_tables(conn, keep_wt_pivots=False):
    """Drop temp tables to prepare for next run.
    If keep_wt_pivots=True, preserves cached weight pivot tables."""
    tables = conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE temporary = true"
    ).fetchall()
    for (t,) in tables:
        if keep_wt_pivots and t.endswith('_wt_piv'):
            continue
        conn.execute(f'DROP TABLE IF EXISTS "{t}"')


def print_breakdown(timings):
    """Print timing breakdown: top slowest steps + category summary."""
    total = sum(dt for _, dt in timings)

    # Top 10 slowest steps
    by_time = sorted(timings, key=lambda x: -x[1])
    print(f"\n  Top 10 slowest steps (of {len(timings)}, total {total:.3f}s):")
    for name, dt in by_time[:10]:
        print(f"    {name:40s} {dt:8.3f}s  ({dt/total*100:5.1f}%)")

    # Category summary
    cats = {}
    for name, dt in timings:
        if '_act_piv' in name:
            cat = 'act_pivot'
        elif '_wt_piv' in name:
            cat = 'wt_pivot'
        elif name.endswith('_dp'):
            cat = 'cross_join(matmul)'
        elif 'rope' in name:
            cat = 'rope'
        elif 'qk_scores' in name or 'qk' in name:
            cat = 'qk_attn'
        elif 'attn_weights' in name:
            cat = 'softmax'
        elif 'attn_out' in name:
            cat = 'attn_vmul'
        elif 'ffn_act' in name:
            cat = 'swiglu'
        elif 'x_after' in name or 'x_out' in name:
            cat = 'residual_add'
        elif '_sq' in name or 'norm' in name:
            cat = 'rmsnorm'
        elif name == 'x_0':
            cat = 'embed_lookup'
        else:
            cat = 'rechunk/other'
        cats.setdefault(cat, [0, 0.0])
        cats[cat][0] += 1
        cats[cat][1] += dt

    print(f"\n  By category:")
    for cat in sorted(cats, key=lambda c: -cats[c][1]):
        n, t = cats[cat]
        print(f"    {cat:20s}: {t:8.3f}s  ({n:3d} steps, {t/total*100:5.1f}%)")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def get_peak_rss_mb():
    """Get peak RSS in MB (Linux: maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_db_size_gb(db_path):
    """Get DuckDB file size in GB."""
    return os.path.getsize(db_path) / (1024 ** 3)


def measure_prefill(conn, prompt_path, num_layers, breakdown=False,
                    cached_wt=False):
    """Measure prefill latency for a single prompt. Reuses connection."""
    # Clean up previous run's temp tables (keep weight pivots if cached)
    drop_temp_tables(conn, keep_wt_pivots=cached_wt)

    # Load prompt
    with open(prompt_path) as f:
        prompt = json.load(f)
    token_ids = prompt["token_ids"]
    seq_len = len(token_ids)

    # Load tokens into input table
    conn.execute("CREATE TEMP TABLE input_tokens "
                 "(pos INTEGER, token_id INTEGER)")
    conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                     [(i, tid) for i, tid in enumerate(token_ids)])

    # Build pipeline (skip weight pivots if cached)
    pipeline = build_full_pipeline(num_layers, cached_wt=cached_wt)

    # Measure
    print(f"  Running prefill (seq_len={seq_len}, "
          f"{len(pipeline)} SQL steps, {num_layers} layers)...")
    t0 = time.perf_counter()
    timings = run_steps(conn, pipeline)
    t1 = time.perf_counter()

    latency = t1 - t0
    peak_rss_mb = get_peak_rss_mb()
    throughput = seq_len / latency  # tokens/sec

    if breakdown:
        print_breakdown(timings)

    return {"seq_len": seq_len,
            "prefill_latency_s": latency,
            "prefill_throughput_tok_per_s": throughput,
            "peak_rss_mb": peak_rss_mb,
            "steps": len(pipeline)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--prompts-dir", default="measure/prompts")
    parser.add_argument("--output", default="measure/results/prefill.json")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of repetitions per prompt length")
    parser.add_argument("--breakdown", action="store_true",
                        help="Print per-step timing breakdown (first run)")
    parser.add_argument("--no-pivot", action="store_true",
                        help="Use basic matmul instead of pivoted (for comparison)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    db_size_gb = get_db_size_gb(args.db_path)
    print(f"DuckDB file size: {db_size_gb:.2f} GB")

    # Open connection once, reuse across all runs
    conn = duckdb.connect(args.db_path, read_only=True,
                          config={"threads": os.cpu_count()})
    print("Connection opened (weights loaded once, reused across runs)")

    # Pre-pivot all weight tables once
    if not args.no_pivot:
        print(f"Pre-pivoting weight tables ({args.num_layers} layers)...")
        t0 = time.perf_counter()
        wt_steps = weight_pivot_steps(args.num_layers)
        run_steps(conn, wt_steps)
        pivot_time = time.perf_counter() - t0
        print(f"  Weight pivoting: {pivot_time:.3f}s ({len(wt_steps)} tables)")

    results = []

    for length in args.lengths:
        prompt_path = os.path.join(args.prompts_dir, f"prompt_{length}.json")
        if not os.path.exists(prompt_path):
            print(f"SKIP: {prompt_path} not found")
            continue

        print(f"\nPrompt length {length}:")
        latencies = []
        throughputs = []
        peak_rss = 0.0
        for r in range(args.repeat):
            show_breakdown = args.breakdown and r == 0
            result = measure_prefill(conn, prompt_path,
                                     args.num_layers, show_breakdown,
                                     cached_wt=not args.no_pivot)
            latencies.append(result["prefill_latency_s"])
            throughputs.append(result["prefill_throughput_tok_per_s"])
            peak_rss = max(peak_rss, result["peak_rss_mb"])
            print(f"  Run {r+1}: {result['prefill_latency_s']:.3f}s  "
                  f"({result['prefill_throughput_tok_per_s']:.2f} tok/s)  "
                  f"RSS: {result['peak_rss_mb']:.0f} MB")

        results.append({
            "prompt_length": length,
            "prefill_latencies_s": latencies,
            "prefill_latency_s": float(np.mean(latencies)),
            "prefill_latency_std_s": float(np.std(latencies)),
            "prefill_throughput_tok_per_s": float(np.mean(throughputs)),
            "peak_rss_mb": peak_rss,
            "db_size_gb": db_size_gb,
            "num_layers": args.num_layers,
            "num_steps": result["steps"],
        })
        print(f"  Mean: {results[-1]['prefill_latency_s']:.3f}s "
              f"(+/- {results[-1]['prefill_latency_std_s']:.3f}s)  "
              f"Throughput: {results[-1]['prefill_throughput_tok_per_s']:.2f} tok/s  "
              f"Peak RSS: {peak_rss:.0f} MB")

    conn.close()
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
