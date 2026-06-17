#!/usr/bin/env python3
"""Detailed verification: compare tuned CSV per-subquery jit+exe
against the source config CSVs used by tune_per_subquery.py.

Shows per-subquery breakdown for a few queries to diagnose the
measured-vs-predicted gap.

Usage: python3 verify_tuned_detail.py [query] [split]
"""
import csv, json, os, re, sys


def mean(vals):
    xs = list(vals)
    return sum(xs) / len(xs)


def parse_per_subquery(path, hasjit=True, head=4):
    """Same parser as tune_per_subquery.py — returns per-query per-subquery means."""
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
            subs.append(dict(jit=jit_avg, exe=exe_avg, total=jit_avg + exe_avg))

        if hasjit:
            t_jit = mean(r[-3] for r in warm)
            t_exe = mean(r[-2] for r in warm)
        else:
            t_jit = 0.0
            t_exe = mean(r[-2] for r in warm)
        subs.append(dict(jit=t_jit, exe=t_exe, total=t_jit + t_exe))

        out[q] = subs
    return out


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "10a"
    split = sys.argv[2] if len(sys.argv) > 2 else "node-based"
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_result")
    head = 4

    tune_json = json.load(open(os.path.join(base, f"tuned_per_subquery_{split}.json")))
    tuned_csv = os.path.join(base,
        f"duckdb_{split}_query_none_nospecjit_tuned_breakdown_time_log.csv")
    tuned_data = parse_per_subquery(tuned_csv, hasjit=True, head=head)

    if query not in tune_json:
        print(f"query {query} not in tune JSON")
        sys.exit(1)

    subs_json = tune_json[query]
    subs_meas = tuned_data.get(query, [])

    # Also load the source config CSVs for this query's configs
    source_configs = {}
    CONFIG_FILES = {
        "interp":        (f"duckdb_{split}_none_off_breakdown_time_log.csv", False),
        "expr":          (f"duckdb_{split}_expr_none_nospecjit_breakdown_time_log.csv", True),
        "expr_simd":     (f"duckdb_{split}_expr_auto_nospecjit_breakdown_time_log.csv", True),
        "operator":      (f"duckdb_{split}_operator_none_nospecjit_breakdown_time_log.csv", True),
        "operator_simd": (f"duckdb_{split}_operator_auto_nospecjit_breakdown_time_log.csv", True),
        "pipeline":      (f"duckdb_{split}_pipeline_none_nospecjit_breakdown_time_log.csv", True),
        "pipeline_simd": (f"duckdb_{split}_pipeline_auto_nospecjit_breakdown_time_log.csv", True),
        "query_full":    (f"duckdb_{split}_query_none_breakdown_time_log.csv", True),
        "query_fastisel":(f"duckdb_{split}_query_none_fcfastisel_breakdown_time_log.csv", True),
        "query_tpde":    (f"duckdb_{split}_query_none_fctpde_breakdown_time_log.csv", True),
    }
    for label, (fname, hasjit) in CONFIG_FILES.items():
        p = os.path.join(base, fname)
        if os.path.exists(p):
            source_configs[label] = parse_per_subquery(p, hasjit=hasjit, head=head)

    nsubs = len(subs_json)
    print(f"query: {query}  sub-queries: {nsubs}\n")
    print(f"{'idx':>4} {'config':18} {'json_ms':>9} {'source_ms':>10} {'tuned_ms':>10} {'diff':>8}")
    print("-" * 63)

    total_json = 0
    total_source = 0
    total_tuned = 0
    for idx_str in sorted(subs_json.keys(), key=int):
        idx = int(idx_str)
        s = subs_json[idx_str]
        label = s["config"]
        json_total = s["total_ms"]

        source_total = 0
        if label in source_configs and query in source_configs[label]:
            src_subs = source_configs[label][query]
            if idx < len(src_subs):
                source_total = src_subs[idx]["total"]

        meas_total = 0
        if idx < len(subs_meas):
            meas_total = subs_meas[idx]["total"]

        total_json += json_total
        total_source += source_total
        total_tuned += meas_total
        diff = meas_total - json_total

        print(f"{idx:4} {label:18} {json_total:9.2f} {source_total:10.2f} "
              f"{meas_total:10.2f} {diff:+8.2f}")

    print("-" * 63)
    print(f"{'':4} {'TOTAL':18} {total_json:9.2f} {total_source:10.2f} "
          f"{total_tuned:10.2f} {total_tuned - total_json:+8.2f}")


if __name__ == "__main__":
    main()
