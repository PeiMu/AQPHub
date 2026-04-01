"""
run_prefill.py -- Measure TranSQL+ prefill latency on DuckDB.

Usage:
    python measure/run_prefill.py \
        --db-path /path/to/weights.duckdb \
        --prompts-dir measure/prompts \
        --output results/prefill.json \
        [--num-layers 32]

Measures wall-clock time from loading tokens to completing all SQL steps
(i.e., time to produce the logits for the first output token).

Reports: latency, throughput (tokens/sec), peak RAM (RSS), DB file size.
The database is freshly opened before each run to clear caches.
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


# ---------------------------------------------------------------------------
# SQL template mirrors (same as in test files)
# ---------------------------------------------------------------------------

def embed_lookup_sql(tok_table, embed_table, out):
    cs = CHUNK_SIZE
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
         f"array_agg(val ORDER BY out_col) AS v "
         f"FROM {dp} GROUP BY act_row, out_col // {cs}", out),
    ]


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
    return [(
        f"SELECT q.row_id, q.chunk_id, "
        f"list_transform(generate_series(0, {half}-1), "
        f"i -> CAST(q.v[2*i] * r.cos[i] - q.v[2*i+1] * r.sin[i] AS FLOAT)) AS v_even, "
        f"list_transform(generate_series(0, {half}-1), "
        f"i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd "
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
         f"a.head_id * {cph} + s.chunk_id AS chunk_id, "
         f"s.dim_idx, SUM(a.attn_weight * s.val) AS val "
         f"FROM {attn} a JOIN {scalar_t} s ON a.k_tok = s.tok "
         f"AND s.chunk_id = (a.head_id // {gs}) * {cph} + "
         f"(a.head_id * {cph} + s.chunk_id) % {cph} "
         f"WHERE s.chunk_id >= (a.head_id // {gs}) * {cph} "
         f"AND s.chunk_id < (a.head_id // {gs} + 1) * {cph} "
         f"GROUP BY a.q_tok, a.head_id * {cph} + s.chunk_id, s.dim_idx",
         weighted_t),
        (f"SELECT row_id, chunk_id, "
         f"array_agg(CAST(val AS FLOAT) ORDER BY dim_idx) AS v "
         f"FROM {weighted_t} GROUP BY row_id, chunk_id", out),
    ]


def swiglu_sql(gate, up, out):
    return [(
        f"SELECT g.row_id, g.chunk_id, "
        f"list_transform(list_zip(g.v, u.v), x -> "
        f"CAST(x[1] * (x[2] / (1.0 + EXP(-CAST(x[2] AS DOUBLE)))) AS FLOAT)) AS v "
        f"FROM {gate} g JOIN {up} u "
        f"ON g.row_id = u.row_id AND g.chunk_id = u.chunk_id", out)]


def residual_add_sql(a, b, out):
    return [(
        f"SELECT a.row_id, a.chunk_id, "
        f"list_transform(list_zip(a.v, b.v), x -> "
        f"CAST(x[1] + x[2] AS FLOAT)) AS v "
        f"FROM {a} a JOIN {b} b "
        f"ON a.row_id = b.row_id AND a.chunk_id = b.chunk_id", out)]


# ---------------------------------------------------------------------------
# Build SQL pipeline for full model
# ---------------------------------------------------------------------------

def build_layer_steps(l, x_in):
    """Build SQL steps for one transformer layer. Returns (steps, x_out_name)."""
    pfx = f"l{l}_"
    wt = lambda name: f"layer_{l}_{name}"

    steps = []
    norm1 = pfx + "norm1_out"
    steps += rmsnorm_sql(x_in, wt("norm1"), norm1)

    q = pfx + "q"
    steps += matmul_sql(norm1, wt("q_proj"), q)
    k = pfx + "k"
    steps += matmul_sql(norm1, wt("k_proj"), k)
    v = pfx + "v"
    steps += matmul_sql(norm1, wt("v_proj"), v)

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
    steps += matmul_sql(attn_out, wt("o_proj"), o)

    x_after_attn = pfx + "x_after_attn"
    steps += residual_add_sql(x_in, o, x_after_attn)

    norm2 = pfx + "norm2_out"
    steps += rmsnorm_sql(x_after_attn, wt("norm2"), norm2)

    gate = pfx + "gate"
    steps += matmul_sql(norm2, wt("gate_proj"), gate)
    up = pfx + "up"
    steps += matmul_sql(norm2, wt("up_proj"), up)
    ffn_act = pfx + "ffn_act"
    steps += swiglu_sql(gate, up, ffn_act)
    down = pfx + "down"
    steps += matmul_sql(ffn_act, wt("down_proj"), down)

    x_out = pfx + "x_out"
    steps += residual_add_sql(x_after_attn, down, x_out)

    return steps, x_out


def build_full_pipeline(num_layers):
    """Build the complete SQL pipeline for prefill."""
    steps = []
    steps += embed_lookup_sql("input_tokens", "embed_tokens", "x_0")

    x_in = "x_0"
    for l in range(num_layers):
        layer_steps, x_out = build_layer_steps(l, x_in)
        steps += layer_steps
        x_in = x_out

    steps += rmsnorm_sql(x_in, "final_norm", "final_norm_out")
    steps += matmul_sql("final_norm_out", "lm_head", "logits")
    return steps


def run_steps(conn, steps):
    """Execute SQL steps, creating temp tables."""
    for sql, table_name in steps:
        conn.execute(
            f"CREATE TEMP TABLE {table_name} AS ({sql})")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def get_peak_rss_mb():
    """Get peak RSS in MB (Linux: maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_db_size_gb(db_path):
    """Get DuckDB file size in GB."""
    return os.path.getsize(db_path) / (1024 ** 3)


def measure_prefill(db_path, prompt_path, num_layers):
    """Measure prefill latency for a single prompt."""
    # Load prompt
    with open(prompt_path) as f:
        prompt = json.load(f)
    token_ids = prompt["token_ids"]
    seq_len = len(token_ids)

    # Fresh connection (clears all caches)
    conn = duckdb.connect(db_path, read_only=True)

    # Load tokens into input table
    conn.execute("CREATE TEMP TABLE input_tokens "
                 "(pos INTEGER, token_id INTEGER)")
    conn.executemany("INSERT INTO input_tokens VALUES (?, ?)",
                     [(i, tid) for i, tid in enumerate(token_ids)])

    # Build pipeline
    pipeline = build_full_pipeline(num_layers)

    # Measure
    print(f"  Running prefill (seq_len={seq_len}, "
          f"{len(pipeline)} SQL steps, {num_layers} layers)...")
    t0 = time.perf_counter()
    run_steps(conn, pipeline)
    t1 = time.perf_counter()

    latency = t1 - t0
    peak_rss_mb = get_peak_rss_mb()
    throughput = seq_len / latency  # tokens/sec

    conn.close()
    return {"seq_len": seq_len, "latency_s": latency,
            "throughput_tok_per_s": throughput,
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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    db_size_gb = get_db_size_gb(args.db_path)
    print(f"DuckDB file size: {db_size_gb:.2f} GB")

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
            result = measure_prefill(args.db_path, prompt_path,
                                     args.num_layers)
            latencies.append(result["latency_s"])
            throughputs.append(result["throughput_tok_per_s"])
            peak_rss = max(peak_rss, result["peak_rss_mb"])
            print(f"  Run {r+1}: {result['latency_s']:.3f}s  "
                  f"({result['throughput_tok_per_s']:.2f} tok/s)  "
                  f"RSS: {result['peak_rss_mb']:.0f} MB")

        results.append({
            "prompt_length": length,
            "latencies_s": latencies,
            "mean_latency_s": float(np.mean(latencies)),
            "std_latency_s": float(np.std(latencies)),
            "mean_throughput_tok_per_s": float(np.mean(throughputs)),
            "peak_rss_mb": peak_rss,
            "db_size_gb": db_size_gb,
            "num_layers": args.num_layers,
            "num_steps": result["steps"],
        })
        print(f"  Mean: {results[-1]['mean_latency_s']:.3f}s "
              f"(+/- {results[-1]['std_latency_s']:.3f}s)  "
              f"Throughput: {results[-1]['mean_throughput_tok_per_s']:.2f} tok/s  "
              f"Peak RSS: {peak_rss:.0f} MB")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
