#!/usr/bin/env python3
"""Find the top-N heaviest queries from breakdown_time_log.csv files.

Usage:
    python3 find_top_queries.py                                    # scan all CSVs in job_result/
    python3 find_top_queries.py /path/to/specific_breakdown.csv    # single file
    python3 find_top_queries.py /path/to/job_result/               # scan directory
    python3 find_top_queries.py --top=10 /path/to/file.csv         # top 10 instead of 5

Each query block in the CSV starts with "Running benchmark for .../query.sql..."
followed by 15 data rows (5 warmup + 10 actual runs). Each data row is comma-separated
phase timings in ms; the row sum is total wall-clock time for that iteration.
"""

import re
import os
import sys
import statistics
import glob


def parse_breakdown_csv(path):
    """Returns dict: query_name -> list of total wall-clock times (ms), skipping 5 warmup."""
    results = {}
    current_query = None
    iterations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'Running benchmark for .*/([^/]+\.sql)', line)
            if m:
                if current_query and len(iterations) > 5:
                    results[current_query] = iterations[5:]
                current_query = m.group(1)
                iterations = []
                continue
            if current_query and line and not line.startswith('#'):
                try:
                    row_sum = sum(float(x) for x in line.split(',') if x.strip())
                    iterations.append(row_sum)
                except ValueError:
                    pass
        if current_query and len(iterations) > 5:
            results[current_query] = iterations[5:]
    return results


def print_top_queries(results, path, top_n):
    ranked = sorted(results.items(), key=lambda kv: statistics.median(kv[1]), reverse=True)
    basename = os.path.basename(path)
    print(f"\n--- {basename} ---")
    print(f"{'Rank':>4s}  {'Query':>10s}  {'Median(ms)':>10s}  {'Mean(ms)':>10s}  {'Stdev(ms)':>10s}")
    for i, (query, times) in enumerate(ranked[:top_n], 1):
        med = statistics.median(times)
        avg = statistics.mean(times)
        sd = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  {i:>2d}   {query:>10s}  {med:>10.1f}  {avg:>10.1f}  {sd:>10.1f}")
    return ranked[:top_n]


def main():
    top_n = 5
    args = []
    for a in sys.argv[1:]:
        if a.startswith('--top='):
            top_n = int(a.split('=')[1])
        else:
            args.append(a)

    if not args:
        args = [os.path.join(os.path.dirname(__file__), "job_result")]

    target = args[0]

    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*_breakdown_time_log.csv")))
        if not files:
            print(f"No breakdown_time_log.csv files found in {target}")
            sys.exit(1)
        for f in files:
            results = parse_breakdown_csv(f)
            if results:
                print_top_queries(results, f, top_n)
    elif os.path.isfile(target):
        results = parse_breakdown_csv(target)
        if results:
            print_top_queries(results, target, top_n)
        else:
            print(f"No query data found in {target}")
            sys.exit(1)
    else:
        print(f"Not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
