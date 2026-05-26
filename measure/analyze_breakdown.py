#!/usr/bin/env python3
"""Parse JOB benchmark breakdown_time_log.csv files and report per-query execution times.

Usage:
    python3 analyze_breakdown.py                    # use default job_result/ directory
    python3 analyze_breakdown.py /path/to/job_result

Outputs:
  - Total execution/JIT/wall time per config
  - Top 20 heaviest queries (node-based/pipeline-jit)
  - Biggest regressions vs none-split/none-jit baseline
  - JIT effect on execution (node-based: jit vs none-jit)
"""

import csv
import re
import os
import sys


def parse_none_split(csv_file, has_jit):
    results = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].startswith("Running benchmark for"):
                match = re.search(r'/([0-9a-z]+)\.sql', row[0])
                if not match:
                    continue
                sql_name = match.group(1)
                rows = []
                for _ in range(15):
                    try:
                        r = next(reader)
                        vals = [float(v.strip()) for v in r]
                        if has_jit:
                            exe = vals[3]
                            jit = vals[2]
                        else:
                            exe = vals[2]
                            jit = 0
                        rows.append({'exe': exe, 'jit': jit, 'wall': sum(vals), 'middleware': sum(vals) - exe - jit})
                    except StopIteration:
                        break
                if len(rows) >= 15:
                    valid = rows[5:]
                    results[sql_name] = {
                        'exe': sum(r['exe'] for r in valid) / len(valid),
                        'jit': sum(r['jit'] for r in valid) / len(valid),
                        'wall': sum(r['wall'] for r in valid) / len(valid),
                        'middleware': sum(r['middleware'] for r in valid) / len(valid),
                    }
    return results


def parse_node_based(csv_file, has_jit):
    if has_jit:
        group_cols = 6  # extract, gen_sql, jit_compile, execute, extra_mat, update_IR
        exe_idx = 3
        jit_idx = 2
    else:
        group_cols = 5  # extract, gen_sql, execute, extra_mat, update_IR
        exe_idx = 2
        jit_idx = -1

    results = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].startswith("Running benchmark for"):
                match = re.search(r'/([0-9a-z]+)\.sql', row[0])
                if not match:
                    continue
                sql_name = match.group(1)
                rows = []
                for _ in range(15):
                    try:
                        r = next(reader)
                        vals = [float(v.strip()) for v in r]
                        tail_size = 4 if has_jit else 3
                        group_values = vals[4:-tail_size]

                        if len(group_values) % group_cols != 0:
                            group_values = group_values[:-1]

                        num_groups = len(group_values) // group_cols

                        exe_sum = 0
                        jit_sum = 0
                        for i in range(num_groups):
                            exe_sum += group_values[i * group_cols + exe_idx]
                            if has_jit:
                                jit_sum += group_values[i * group_cols + jit_idx]

                        final_exe = vals[-2]
                        exe_sum += final_exe

                        if has_jit:
                            jit_compile_final = vals[-3]
                            jit_sum += jit_compile_final

                        wall = sum(vals)
                        middleware = wall - exe_sum - jit_sum

                        rows.append({'exe': exe_sum, 'jit': jit_sum, 'wall': wall, 'middleware': middleware})
                    except StopIteration:
                        break
                if len(rows) >= 15:
                    valid = rows[5:]
                    results[sql_name] = {
                        'exe': sum(r['exe'] for r in valid) / len(valid),
                        'jit': sum(r['jit'] for r in valid) / len(valid),
                        'wall': sum(r['wall'] for r in valid) / len(valid),
                        'middleware': sum(r['middleware'] for r in valid) / len(valid),
                    }
    return results


def find_config_file(base, pattern_parts):
    """Find a breakdown_time_log.csv matching pattern parts."""
    for f in os.listdir(base):
        if not f.endswith("_breakdown_time_log.csv"):
            continue
        if all(p in f for p in pattern_parts):
            return os.path.join(base, f)
    return None


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "job_result")

    configs = {
        'none-split/none-jit': {
            'patterns': ['duckdb_none_none'],
            'has_jit': False, 'is_node': False,
        },
        'none-split/pipeline-jit': {
            'patterns': ['duckdb_none_pipeline_o1_none'],
            'has_jit': True, 'is_node': False,
        },
        'none-split/pipeline-jit-auto': {
            'patterns': ['duckdb_none_pipeline_o1_auto'],
            'has_jit': True, 'is_node': False,
        },
        'node-based/none-jit': {
            'patterns': ['duckdb_node-based_none'],
            'has_jit': False, 'is_node': True,
        },
        'node-based/pipeline-jit': {
            'patterns': ['duckdb_node-based_pipeline_o1_none'],
            'has_jit': True, 'is_node': True,
        },
        'node-based/pipeline-jit-auto': {
            'patterns': ['duckdb_node-based_pipeline_o1_auto'],
            'has_jit': True, 'is_node': True,
        },
    }

    all_results = {}
    for name, cfg in configs.items():
        path = find_config_file(base, cfg['patterns'])
        if not path:
            print(f"MISSING: {name}")
            continue
        if cfg['is_node']:
            all_results[name] = parse_node_based(path, cfg['has_jit'])
        else:
            all_results[name] = parse_none_split(path, cfg['has_jit'])
        print(f"Parsed {name}: {len(all_results[name])} queries  ({os.path.basename(path)})")

    # Summary totals
    print("\n=== TOTAL TIME (ms, sum of all queries, avg of runs 6-15) ===")
    for name in configs:
        if name not in all_results:
            continue
        r = all_results[name]
        total_exe = sum(v['exe'] for v in r.values())
        total_jit = sum(v.get('jit', 0) for v in r.values())
        total_wall = sum(v['wall'] for v in r.values())
        total_mw = sum(v.get('middleware', 0) for v in r.values())
        print(f"  {name:30s}: exe={total_exe:9.1f}  jit={total_jit:8.1f}  mw={total_mw:8.1f}  wall={total_wall:9.1f}")

    # Top 20 heaviest queries (node-based/pipeline-jit)
    print("\n=== TOP 20 HEAVIEST QUERIES (node-based/pipeline-jit, by exe time) ===")
    if 'node-based/pipeline-jit' in all_results:
        r = all_results['node-based/pipeline-jit']
        sorted_q = sorted(r.items(), key=lambda x: x[1]['exe'], reverse=True)
        baseline = all_results.get('none-split/none-jit', {})
        nb_nojit = all_results.get('node-based/none-jit', {})
        print(f"{'Query':>8s} {'nb-jit exe':>12s} {'nb-nojit exe':>12s} {'ns-nojit exe':>12s} {'nb-jit mw':>10s} {'nb-jit jit':>10s} {'split delta':>11s}")
        for q, v in sorted_q[:20]:
            b = baseline.get(q, {})
            n = nb_nojit.get(q, {})
            split_delta = v['exe'] - b.get('exe', 0) if b else float('nan')
            print(f"  {q:>8s} {v['exe']:10.1f}ms {n.get('exe', 0):10.1f}ms {b.get('exe', 0):10.1f}ms {v.get('middleware', 0):8.1f}ms {v['jit']:8.1f}ms {split_delta:+9.1f}ms")

    # Biggest regressions
    print("\n=== BIGGEST REGRESSIONS (node-based/pipeline-jit exe vs none-split/none-jit exe) ===")
    if 'node-based/pipeline-jit' in all_results and 'none-split/none-jit' in all_results:
        r_jit = all_results['node-based/pipeline-jit']
        r_base = all_results['none-split/none-jit']
        deltas = []
        for q in r_jit:
            if q in r_base:
                delta = r_jit[q]['exe'] - r_base[q]['exe']
                deltas.append((q, delta, r_jit[q]['exe'], r_base[q]['exe']))
        deltas.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Query':>8s} {'delta':>10s} {'nb-jit':>10s} {'ns-nojit':>10s}")
        for q, d, jit_e, base_e in deltas[:20]:
            print(f"  {q:>8s} {d:+8.1f}ms {jit_e:8.1f}ms {base_e:8.1f}ms")

    # JIT effect (node-based only)
    print("\n=== JIT EFFECT ON EXECUTION (node-based: pipeline-jit vs none-jit) ===")
    if 'node-based/pipeline-jit' in all_results and 'node-based/none-jit' in all_results:
        r_jit = all_results['node-based/pipeline-jit']
        r_nojit = all_results['node-based/none-jit']
        deltas = []
        for q in r_jit:
            if q in r_nojit:
                delta = r_jit[q]['exe'] - r_nojit[q]['exe']
                deltas.append((q, delta, r_jit[q]['exe'], r_nojit[q]['exe']))
        deltas.sort(key=lambda x: x[1])
        print(f"{'Query':>8s} {'delta':>10s} {'jit exe':>10s} {'nojit exe':>10s}")
        print("--- JIT helps most ---")
        for q, d, je, ne in deltas[:10]:
            print(f"  {q:>8s} {d:+8.1f}ms {je:8.1f}ms {ne:8.1f}ms")
        print("--- JIT hurts most ---")
        for q, d, je, ne in deltas[-10:]:
            print(f"  {q:>8s} {d:+8.1f}ms {je:8.1f}ms {ne:8.1f}ms")

    # JIT effect (none-split: pipeline-jit vs none-jit)
    print("\n=== JIT EFFECT ON EXECUTION (none-split: pipeline-jit vs none-jit) ===")
    for jit_cfg in ['none-split/pipeline-jit', 'none-split/pipeline-jit-auto']:
        if jit_cfg in all_results and 'none-split/none-jit' in all_results:
            r_jit = all_results[jit_cfg]
            r_nojit = all_results['none-split/none-jit']
            deltas = []
            for q in r_jit:
                if q in r_nojit:
                    delta = r_jit[q]['exe'] - r_nojit[q]['exe']
                    deltas.append((q, delta, r_jit[q]['exe'], r_nojit[q]['exe']))
            deltas.sort(key=lambda x: x[1])
            total_jit = sum(d[2] for d in deltas)
            total_nojit = sum(d[3] for d in deltas)
            simd_label = "(auto SIMD)" if "auto" in jit_cfg else "(no SIMD)"
            print(f"\n--- {simd_label}: total exe {total_jit:.1f}ms vs {total_nojit:.1f}ms = {total_jit-total_nojit:+.1f}ms ({(total_jit-total_nojit)/total_nojit*100:+.1f}%) ---")
            print(f"{'Query':>8s} {'delta':>10s} {'jit exe':>10s} {'nojit exe':>10s}")
            print("  Top 10 improved:")
            for q, d, je, ne in deltas[:10]:
                print(f"  {q:>8s} {d:+8.1f}ms {je:8.1f}ms {ne:8.1f}ms")
            print("  Top 5 regressed:")
            for q, d, je, ne in deltas[-5:]:
                print(f"  {q:>8s} {d:+8.1f}ms {je:8.1f}ms {ne:8.1f}ms")

    # Biggest improvements from split
    print("\n=== BIGGEST IMPROVEMENTS (node-based/pipeline-jit exe vs none-split/none-jit exe) ===")
    if 'node-based/pipeline-jit' in all_results and 'none-split/none-jit' in all_results:
        r_jit = all_results['node-based/pipeline-jit']
        r_base = all_results['none-split/none-jit']
        deltas = []
        for q in r_jit:
            if q in r_base:
                delta = r_jit[q]['exe'] - r_base[q]['exe']
                deltas.append((q, delta, r_jit[q]['exe'], r_base[q]['exe']))
        deltas.sort(key=lambda x: x[1])
        print(f"{'Query':>8s} {'delta':>10s} {'nb-jit':>10s} {'ns-nojit':>10s}")
        for q, d, jit_e, base_e in deltas[:15]:
            print(f"  {q:>8s} {d:+8.1f}ms {jit_e:8.1f}ms {base_e:8.1f}ms")


if __name__ == "__main__":
    main()
