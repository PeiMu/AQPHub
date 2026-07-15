#!/usr/bin/env python3
"""Per-subquery winner tuning across multiple JIT configs.

Reads breakdown CSV files from job_result/ for each candidate config,
extracts per-subquery (per-iteration-group) warm-mean total time
(jit_compile + execute), and picks the best config for each
(query, subquery_index) pair.  Optimizes END-TO-END time (compile
cost included).

All candidate CSVs must be measured with --spec-jit=off so per-subquery
times are self-contained (no cross-iteration speculative compile
overlap).

Each candidate config is defined by:
  - jit level (none/expr/operator/pipeline/query)
  - simd (off/auto)
  - compile_mode (llvm/tpde, via --compile-mode)
  - per-optimization flags (payload_prune, prefetch, batch_probe,
    skip_hash_cmp) — future sweeps; currently all use global defaults.

Usage:
  python3 tune_per_subquery.py [split] [--engine=ENGINE]

  split:   node-based (default) | none | topdown | relationship-center
  engine:  duckdb (default) | postgresql

Candidate configs are defined in CONFIGS below.  Each entry specifies
the CSV filename, whether it has a jit_compile column, and the flag
values to emit in the output JSON.  Add new configs (e.g. after
implementing prefetch for query-jit) by adding entries.  The script
auto-skips missing files.

Output:
  1. Per-subquery winner table (stdout)
  2. JSON file with per-subquery optimal config:
     job_result/tuned_per_subquery_<split>.json

The JSON has structure:
  { "query": { "sub_idx": { "config": "...", "total_ms": ...,
      "compile_mode": 0, "simd": false, ... }, ... }, ... }

Fields in each sub-query entry:
  - config: label string matching ParseTuneLabel (e.g. "query_tpde",
    "operator_simd", "interp")
  - total_ms: predicted jit+exe time from the source CSV
  - compile_mode: 0=llvm, 2=tpde
  - simd: true/false
  - payload_prune, prefetch, batch_probe, skip_hash_cmp: true/false
    (only emitted when they differ from the default)

Warm-row selection: drops first 5 runs (warmup), uses last 10 runs.
Aggregation: arithmetic mean per column (matches plot_middleware_jit.py).

CSV layout (verified):
  - head columns: 4 for node-based/none, 5 for topdown/relationship-center
  - per iteration group: 6 columns (jit present) or 5 (no jit)
    head + N*group_size + tail = total columns per row
  - warm rows: rows[5:] if len > 6, else rows[1:]
"""
import json, os, re, sys


def mean(vals):
    """Arithmetic mean of an iterable."""
    xs = list(vals)
    return sum(xs) / len(xs)


def parse_csv(path, hasjit=True, head=4):
    """Parse a breakdown CSV into per-query, per-subquery means."""
    raw = {}
    cur = None
    for line in open(path):
        if line.startswith("Running"):
            cur = re.search(r"/([0-9a-z]+)\.sql", line).group(1)
            raw[cur] = []
            continue
        if cur is None:
            continue
        parts = line.strip().rstrip(",").split(",")
        if not parts or not parts[0]:
            continue
        try:
            raw[cur].append([float(x) for x in parts])
        except ValueError:
            continue

    gsz = 6 if hasjit else 5
    tail = 4 if hasjit else 3
    out = {}
    for q, rows in raw.items():
        warm = rows[5:] if len(rows) > 6 else rows[1:]
        lens = [len(r) for r in warm]
        if not lens:
            continue
        mode_len = max(set(lens), key=lens.count)
        warm = [r for r in warm if len(r) == mode_len]
        if not warm:
            continue

        body = mode_len - head - tail
        extra = body % gsz
        n = (body - extra) // gsz

        subs = []
        for i in range(n):
            base = head + i * gsz
            if hasjit:
                jit_avg = mean(r[base + 2] for r in warm)
                exe_avg = mean(r[base + 3] for r in warm)
            else:
                jit_avg = 0.0
                exe_avg = mean(r[base + 2] for r in warm)
            total_avg = jit_avg + exe_avg
            subs.append(dict(jit=jit_avg, exe=exe_avg, total=total_avg))

        # tail (final sub-query)
        if hasjit:
            t_jit = mean(r[-3] for r in warm)
            t_exe = mean(r[-2] for r in warm)
        else:
            t_jit = 0.0
            t_exe = mean(r[-2] for r in warm)
        subs.append(dict(jit=t_jit, exe=t_exe, total=t_jit + t_exe))

        query_total = mean(sum(r) for r in warm)
        out[q] = dict(subs=subs, query_total=query_total)
    return out


# ---- Config definition ----
# Each config has:
#   label:    short name (used in JSON "config" field, matched by ParseTuneLabel)
#   filename: CSV filename template (with {split} placeholder)
#   hasjit:   whether the CSV has a jit_compile column
#   flags:    dict of flag values to emit in JSON (only non-default values)
#
# Flag defaults (matching the global --jit-* defaults when not overridden):
#   compile_mode=0, simd=False, payload_prune=True, prefetch=True,
#   batch_probe=True, skip_hash_cmp="all" (or "off")
#
# The "config" label must match ParseTuneLabel in ir_query_splitter.cpp.
# Additional flags (compile_mode, simd, payload_prune, etc.) are read
# from the JSON by LoadTuneEntry and override the label's defaults.

DEFAULTS = dict(compile_mode=0, simd=False)

def make_configs(split, engine="duckdb"):
    """Build the candidate config list for a given split and engine."""
    e = engine

    # PostgreSQL only supports query-jit (no expr/operator/pipeline)
    if engine == "postgresql":
        return [
            dict(label="interp",
                 filename=f"{e}_{split}_none_off_breakdown_time_log.csv",
                 hasjit=False, flags=dict()),
            dict(label="query_full",
                 filename=f"{e}_{split}_query_none_breakdown_time_log.csv",
                 hasjit=True, flags=dict()),
            dict(label="query_fastisel",
                 filename=f"{e}_{split}_query_none_fcfastisel_breakdown_time_log.csv",
                 hasjit=True, flags=dict(compile_mode=1)),
            dict(label="query_tpde",
                 filename=f"{e}_{split}_query_none_fctpde_breakdown_time_log.csv",
                 hasjit=True, flags=dict(compile_mode=2)),
        ]

    return [
        # ---- Interpreter (no JIT) ----
        dict(label="interp",
             filename=f"{e}_{split}_none_off_breakdown_time_log.csv",
             hasjit=False, flags=dict()),

        # ---- expr-jit ----
        dict(label="expr",
             filename=f"{e}_{split}_expr_none_llvm_breakdown_time_log.csv",
             hasjit=True, flags=dict()),
        # dict(label="expr_simd",
        #      filename=f"{e}_{split}_expr_auto_llvm_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(simd=True)),
        dict(label="expr_fastisel",
             filename=f"{e}_{split}_expr_none_fastisel_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=1)),
        # dict(label="expr_fastisel_simd",
        #      filename=f"{e}_{split}_expr_auto_fastisel_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(compile_mode=1, simd=True)),
        dict(label="expr_tpde",
             filename=f"{e}_{split}_expr_none_tpde_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=2)),

        # ---- operator-jit ----
        dict(label="operator",
             filename=f"{e}_{split}_operator_none_llvm_breakdown_time_log.csv",
             hasjit=True, flags=dict()),
        # dict(label="operator_simd",
        #      filename=f"{e}_{split}_operator_auto_llvm_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(simd=True)),
        dict(label="operator_fastisel",
             filename=f"{e}_{split}_operator_none_fastisel_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=1)),
        # dict(label="operator_fastisel_simd",
        #      filename=f"{e}_{split}_operator_auto_fastisel_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(compile_mode=1, simd=True)),
        dict(label="operator_tpde",
             filename=f"{e}_{split}_operator_none_tpde_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=2)),

        # ---- pipeline-jit ----
        dict(label="pipeline",
             filename=f"{e}_{split}_pipeline_none_llvm_breakdown_time_log.csv",
             hasjit=True, flags=dict()),
        # dict(label="pipeline_simd",
        #      filename=f"{e}_{split}_pipeline_auto_llvm_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(simd=True)),
        dict(label="pipeline_fastisel",
             filename=f"{e}_{split}_pipeline_none_fastisel_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=1)),
        # dict(label="pipeline_fastisel_simd",
        #      filename=f"{e}_{split}_pipeline_auto_fastisel_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(compile_mode=1, simd=True)),
        dict(label="pipeline_tpde",
             filename=f"{e}_{split}_pipeline_none_tpde_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=2)),

        # ---- query-jit (full LLVM backend) ----
        dict(label="query_full",
             filename=f"{e}_{split}_query_none_llvm_breakdown_time_log.csv",
             hasjit=True, flags=dict()),

        # ---- query-jit (FastISel backend) ----
        dict(label="query_fastisel",
             filename=f"{e}_{split}_query_none_fastisel_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=1)),

        # ---- query-jit (TPDE fast backend) ----
        dict(label="query_tpde",
             filename=f"{e}_{split}_query_none_tpde_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=2)),

        # ---- query-jit with skip_hash_cmp=off ----
        dict(label="query_full",
             filename=f"duckdb_{split}_query_none_noskiphashcmp_llvm_breakdown_time_log.csv",
             hasjit=True, flags=dict(skip_hash_cmp="off")),
        dict(label="query_fastisel",
             filename=f"duckdb_{split}_query_none_noskiphashcmp_fastisel_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=1, skip_hash_cmp="off")),
        dict(label="query_tpde",
             filename=f"duckdb_{split}_query_none_noskiphashcmp_tpde_breakdown_time_log.csv",
             hasjit=True, flags=dict(compile_mode=2, skip_hash_cmp="off")),

        # ---- query-jit SIMD (ROF two-phase; TPDE excluded — ROF disabled) ----
        # dict(label="query_full_simd",
        #      filename=f"{e}_{split}_query_auto_llvm_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(simd=True)),
        # dict(label="query_fastisel_simd",
        #      filename=f"{e}_{split}_query_auto_fastisel_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(compile_mode=1, simd=True)),

        # ---- future configs (uncomment when CSVs exist) ----
        # dict(label="pipeline",
        #      filename=f"{e}_{split}_pipeline_none_nopayprune_llvm_breakdown_time_log.csv",
        #      hasjit=True, flags=dict(payload_prune=False)),
    ]


def main():
    split = "node-based"
    engine = "duckdb"
    for arg in sys.argv[1:]:
        if arg.startswith("--engine="):
            engine = arg.split("=", 1)[1]
        elif not arg.startswith("-"):
            split = arg
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_result")
    head = 5 if split in ("relationship-center", "topdown") else 4

    CONFIGS = make_configs(split, engine)

    data = {}
    config_flags = {}  # label → flags dict
    for cfg in CONFIGS:
        label = cfg["label"]
        p = os.path.join(base, cfg["filename"])
        if os.path.exists(p):
            parsed = parse_csv(p, hasjit=cfg["hasjit"], head=head)
            # Use (label, filename) as key to allow same label with different flags
            key = cfg["filename"]
            data[key] = dict(parsed=parsed, label=label, flags=cfg["flags"])
            print(f"loaded: {label} ({len(parsed)} queries) [{cfg['filename']}]")
        else:
            print(f"missing: {cfg['filename']} (skipped)")

    if not data:
        print("no config CSVs found")
        sys.exit(1)

    keys = list(data.keys())
    qs = sorted(set.intersection(*(set(data[k]["parsed"]) for k in keys)))
    labels = [data[k]["label"] for k in keys]
    print(f"\nsplit: {split}  queries: {len(qs)}  configs: {labels}")

    # ---- Per-subquery winner selection ----
    result = {}
    total_tuned = 0.0
    total_per_config = {k: 0.0 for k in keys}
    winner_count = {}

    for q in qs:
        nsubs = min(len(data[k]["parsed"][q]["subs"]) for k in keys)
        q_result = {}
        q_tuned_total = 0.0

        for si in range(nsubs):
            best_key = None
            best_t = float("inf")
            for k in keys:
                subs = data[k]["parsed"][q]["subs"]
                if si < len(subs):
                    t = subs[si]["total"]
                    if t < best_t:
                        best_t = t
                        best_key = k

            best_label = data[best_key]["label"]
            best_flags = data[best_key]["flags"]

            entry = dict(config=best_label, total_ms=round(best_t, 3))
            # Emit non-default flags
            for flag_name, default_val in DEFAULTS.items():
                val = best_flags.get(flag_name, default_val)
                if val != default_val:
                    entry[flag_name] = val
            # Emit per-optimization flags only when explicitly set
            for opt_flag in ("payload_prune", "prefetch", "batch_probe",
                             "skip_hash_cmp"):
                if opt_flag in best_flags:
                    entry[opt_flag] = best_flags[opt_flag]

            q_result[str(si)] = entry
            q_tuned_total += best_t
            winner_count[best_label] = winner_count.get(best_label, 0) + 1

        result[q] = q_result
        total_tuned += q_tuned_total
        for k in keys:
            total_per_config[k] += sum(
                s["total"] for s in data[k]["parsed"][q]["subs"][:nsubs])

    # ---- Print summary ----
    print(f"\n{'config':18} {'suite_total_s':>14}")
    for k in keys:
        print(f"{data[k]['label']:18} {total_per_config[k]/1000:14.2f}")
    print(f"{'TUNED':18} {total_tuned/1000:14.2f}")

    print(f"\nper-subquery winner counts:")
    for k in sorted(winner_count, key=winner_count.get, reverse=True):
        print(f"  {k:18} {winner_count[k]}")

    total_subs = sum(winner_count.values())
    print(f"  total sub-queries: {total_subs}")

    # ---- Save JSON ----
    out_path = os.path.join(base, f"tuned_per_subquery_{split}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
