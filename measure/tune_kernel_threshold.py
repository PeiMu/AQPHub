#!/usr/bin/env python3
"""Analyze kernel vs DuckDB tuning data and recommend a threshold.

Usage:
    python3 tune_kernel_threshold.py [tuning_data_dir]

Reads CSV files produced by tune_kernel_threshold.sh:
    {strategy}_kernel.csv   — kernel-enabled runs
    {strategy}_no-kernel.csv — DuckDB-only runs

Matches rows by (query, repeat, iteration, type) and compares execution times.
Outputs: scatter plot, feature importance, recommended threshold formula.
"""

import os
import sys
import csv
from collections import defaultdict


def load_csv(path):
    """Load tuning CSV, return list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["repeat"] = int(r["repeat"])
            r["iteration"] = int(r["iteration"])
            r["kernel_valid"] = int(r["kernel_valid"])
            r["kernel_used"] = int(r["kernel_used"])
            r["scan_rows"] = int(r["scan_rows"])
            r["num_joins"] = int(r["num_joins"])
            r["num_filters"] = int(r["num_filters"])
            r["num_output_cols"] = int(r["num_output_cols"])
            r["exe_time_ms"] = float(r["exe_time_ms"])
            rows.append(r)
    return rows


def match_and_compare(kernel_rows, nokernel_rows, warmup=2):
    """Match kernel vs no-kernel rows by (query, repeat, iteration, type).

    Only uses rows where kernel_valid=1 and repeat >= warmup.
    Returns list of dicts with features + kernel_ms + duckdb_ms + speedup.
    """
    # Index no-kernel rows by key
    nk_index = {}
    for r in nokernel_rows:
        if r["repeat"] < warmup:
            continue
        key = (r["query"], r["repeat"], r["iteration"], r["type"])
        nk_index[key] = r

    pairs = []
    for r in kernel_rows:
        if r["repeat"] < warmup:
            continue
        if r["kernel_valid"] == 0:
            continue
        key = (r["query"], r["repeat"], r["iteration"], r["type"])
        nk = nk_index.get(key)
        if nk is None:
            continue
        pairs.append({
            "query": r["query"],
            "repeat": r["repeat"],
            "iteration": r["iteration"],
            "type": r["type"],
            "scan_table": r["scan_table"],
            "scan_rows": r["scan_rows"],
            "num_joins": r["num_joins"],
            "num_filters": r["num_filters"],
            "num_output_cols": r["num_output_cols"],
            "kernel_ms": r["exe_time_ms"],
            "duckdb_ms": nk["exe_time_ms"],
        })

    for p in pairs:
        if p["duckdb_ms"] > 0:
            p["speedup"] = p["duckdb_ms"] / p["kernel_ms"]
        else:
            p["speedup"] = 1.0
        p["kernel_wins"] = p["kernel_ms"] < p["duckdb_ms"]

    return pairs


def aggregate_pairs(pairs):
    """Average across repeats for each (query, iteration, type)."""
    groups = defaultdict(list)
    for p in pairs:
        key = (p["query"], p["iteration"], p["type"])
        groups[key].append(p)

    agg = []
    for key, group in groups.items():
        n = len(group)
        avg = {
            "query": key[0],
            "iteration": key[1],
            "type": key[2],
            "scan_table": group[0]["scan_table"],
            "scan_rows": group[0]["scan_rows"],
            "num_joins": group[0]["num_joins"],
            "num_filters": group[0]["num_filters"],
            "num_output_cols": group[0]["num_output_cols"],
            "kernel_ms": sum(p["kernel_ms"] for p in group) / n,
            "duckdb_ms": sum(p["duckdb_ms"] for p in group) / n,
            "n_repeats": n,
        }
        avg["speedup"] = avg["duckdb_ms"] / avg["kernel_ms"] if avg["kernel_ms"] > 0 else 1.0
        avg["kernel_wins"] = avg["kernel_ms"] < avg["duckdb_ms"]
        agg.append(avg)

    return sorted(agg, key=lambda x: abs(x["duckdb_ms"] - x["kernel_ms"]), reverse=True)


def print_summary(agg, strategy):
    """Print summary table and statistics."""
    kernel_wins = [p for p in agg if p["kernel_wins"]]
    duckdb_wins = [p for p in agg if not p["kernel_wins"]]

    total_kernel_ms = sum(p["kernel_ms"] for p in agg)
    total_duckdb_ms = sum(p["duckdb_ms"] for p in agg)
    total_saved = total_duckdb_ms - total_kernel_ms

    print(f"\n{'='*80}")
    print(f"Strategy: {strategy}")
    print(f"{'='*80}")
    print(f"Total sub-queries analyzed: {len(agg)}")
    print(f"  Kernel wins: {len(kernel_wins)}")
    print(f"  DuckDB wins: {len(duckdb_wins)}")
    print(f"Total kernel time:  {total_kernel_ms:.1f} ms")
    print(f"Total DuckDB time:  {total_duckdb_ms:.1f} ms")
    print(f"Net saved by kernel: {total_saved:.1f} ms ({total_saved/total_duckdb_ms*100:.1f}%)")

    # Top kernel wins
    print(f"\n--- Top 15 kernel wins (biggest time saved) ---")
    print(f"{'Query':<8} {'Iter':>4} {'Type':<5} {'ScanTable':<20} {'Rows':>10} "
          f"{'Joins':>5} {'Filt':>4} {'Cols':>4} {'Kernel':>8} {'DuckDB':>8} {'Speedup':>7}")
    for p in sorted(kernel_wins, key=lambda x: x["duckdb_ms"] - x["kernel_ms"], reverse=True)[:15]:
        print(f"{p['query']:<8} {p['iteration']:>4} {p['type']:<5} {p['scan_table']:<20} "
              f"{p['scan_rows']:>10} {p['num_joins']:>5} {p['num_filters']:>4} "
              f"{p['num_output_cols']:>4} {p['kernel_ms']:>7.1f}ms {p['duckdb_ms']:>7.1f}ms "
              f"{p['speedup']:>6.2f}x")

    # Top DuckDB wins (kernel regressions)
    if duckdb_wins:
        print(f"\n--- Top 15 DuckDB wins (kernel regressions) ---")
        print(f"{'Query':<8} {'Iter':>4} {'Type':<5} {'ScanTable':<20} {'Rows':>10} "
              f"{'Joins':>5} {'Filt':>4} {'Cols':>4} {'Kernel':>8} {'DuckDB':>8} {'Slowdown':>8}")
        for p in sorted(duckdb_wins, key=lambda x: x["kernel_ms"] - x["duckdb_ms"], reverse=True)[:15]:
            slowdown = p["kernel_ms"] / p["duckdb_ms"] if p["duckdb_ms"] > 0 else 999
            print(f"{p['query']:<8} {p['iteration']:>4} {p['type']:<5} {p['scan_table']:<20} "
                  f"{p['scan_rows']:>10} {p['num_joins']:>5} {p['num_filters']:>4} "
                  f"{p['num_output_cols']:>4} {p['kernel_ms']:>7.1f}ms {p['duckdb_ms']:>7.1f}ms "
                  f"{slowdown:>7.2f}x")


def analyze_thresholds(all_pairs):
    """Analyze which features predict kernel vs DuckDB winner.

    Try simple threshold rules on scan_rows, num_joins, etc.
    Report accuracy and net time saved for each rule.
    """
    print(f"\n{'='*80}")
    print("Threshold Analysis (across all strategies)")
    print(f"{'='*80}")

    total_duckdb_ms = sum(p["duckdb_ms"] for p in all_pairs)

    # Analyze by feature bins
    features = [
        ("scan_rows", [0, 100, 1000, 10000, 100000, 500000, 1000000, 5000000, 50000000]),
        ("num_joins", [0, 1, 2, 3, 5]),
        ("num_filters", [0, 1, 2, 3, 5]),
        ("num_output_cols", [0, 1, 2, 3, 4, 5, 10]),
    ]

    for feat_name, bins in features:
        print(f"\n--- By {feat_name} ---")
        print(f"{'Range':<25} {'Count':>6} {'KernWin':>8} {'DuckWin':>8} "
              f"{'AvgSpeedup':>10} {'NetSaved':>10}")
        for i in range(len(bins)):
            lo = bins[i]
            hi = bins[i + 1] if i + 1 < len(bins) else float("inf")
            bucket = [p for p in all_pairs if lo <= p[feat_name] < hi]
            if not bucket:
                continue
            k_wins = sum(1 for p in bucket if p["kernel_wins"])
            d_wins = len(bucket) - k_wins
            avg_speedup = sum(p["speedup"] for p in bucket) / len(bucket)
            net_saved = sum(p["duckdb_ms"] - p["kernel_ms"] for p in bucket)
            label = f"[{lo}, {hi})" if hi != float("inf") else f"[{lo}, inf)"
            print(f"{label:<25} {len(bucket):>6} {k_wins:>8} {d_wins:>8} "
                  f"{avg_speedup:>9.2f}x {net_saved:>9.1f}ms")

    # Try scan_rows thresholds: use kernel when scan_rows >= T
    print(f"\n--- scan_rows threshold sweep (use kernel when rows >= T) ---")
    print(f"{'Threshold':>12} {'KernelUsed':>10} {'Correct':>8} {'Wrong':>6} "
          f"{'NetSaved':>10} {'vs_allDuck':>10}")
    thresholds = [0, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    best_thresh = 0
    best_saved = float("-inf")
    for t in thresholds:
        used = [p for p in all_pairs if p["scan_rows"] >= t]
        skipped = [p for p in all_pairs if p["scan_rows"] < t]
        # For used: kernel runs, time = kernel_ms
        # For skipped: duckdb runs, time = duckdb_ms
        total_with_rule = sum(p["kernel_ms"] for p in used) + sum(p["duckdb_ms"] for p in skipped)
        net_saved = total_duckdb_ms - total_with_rule
        correct = sum(1 for p in used if p["kernel_wins"]) + sum(1 for p in skipped if not p["kernel_wins"])
        wrong = len(all_pairs) - correct
        pct = net_saved / total_duckdb_ms * 100 if total_duckdb_ms > 0 else 0
        print(f"{t:>12} {len(used):>10} {correct:>8} {wrong:>6} "
              f"{net_saved:>9.1f}ms {pct:>9.1f}%")
        if net_saved > best_saved:
            best_saved = net_saved
            best_thresh = t

    print(f"\nBest scan_rows threshold: {best_thresh} (saves {best_saved:.1f}ms, "
          f"{best_saved/total_duckdb_ms*100:.1f}% of total DuckDB time)")

    # Try combined: scan_rows >= T AND num_joins <= J
    print(f"\n--- Combined threshold: scan_rows >= T AND num_joins <= J ---")
    print(f"{'T':>10} {'J':>4} {'NetSaved':>10} {'vs_allDuck':>10}")
    best_combo = (0, 99)
    best_combo_saved = float("-inf")
    for t in [0, 100, 1000, 10000, 100000]:
        for j in [0, 1, 2, 3, 5, 99]:
            used = [p for p in all_pairs if p["scan_rows"] >= t and p["num_joins"] <= j]
            skipped = [p for p in all_pairs if not (p["scan_rows"] >= t and p["num_joins"] <= j)]
            total_with_rule = sum(p["kernel_ms"] for p in used) + sum(p["duckdb_ms"] for p in skipped)
            net_saved = total_duckdb_ms - total_with_rule
            if net_saved > best_combo_saved:
                best_combo_saved = net_saved
                best_combo = (t, j)
            pct = net_saved / total_duckdb_ms * 100 if total_duckdb_ms > 0 else 0
            print(f"{t:>10} {j:>4} {net_saved:>9.1f}ms {pct:>9.1f}%")

    print(f"\nBest combined: scan_rows >= {best_combo[0]} AND num_joins <= {best_combo[1]} "
          f"(saves {best_combo_saved:.1f}ms)")

    # Final recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print(f"Simple rule:    Use kernel when scan_rows >= {best_thresh}")
    print(f"Combined rule:  Use kernel when scan_rows >= {best_combo[0]} AND num_joins <= {best_combo[1]}")
    print(f"Expected saving: {best_combo_saved:.1f}ms ({best_combo_saved/total_duckdb_ms*100:.1f}% "
          f"of total DuckDB time)")

    return best_thresh, best_combo


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "tuning_data"
    if not os.path.isdir(data_dir):
        print(f"Error: directory '{data_dir}' not found")
        print(f"Usage: {sys.argv[0]} [tuning_data_dir]")
        sys.exit(1)

    strategies = ["node-based", "relationship-center"]
    all_pairs = []

    for strategy in strategies:
        kernel_path = os.path.join(data_dir, f"{strategy}_kernel.csv")
        nokernel_path = os.path.join(data_dir, f"{strategy}_no-kernel.csv")

        if not os.path.exists(kernel_path) or not os.path.exists(nokernel_path):
            print(f"Skipping {strategy}: missing {kernel_path} or {nokernel_path}")
            continue

        kernel_rows = load_csv(kernel_path)
        nokernel_rows = load_csv(nokernel_path)

        pairs = match_and_compare(kernel_rows, nokernel_rows, warmup=2)
        if not pairs:
            print(f"Skipping {strategy}: no matched pairs with kernel_valid=1")
            continue

        agg = aggregate_pairs(pairs)
        print_summary(agg, strategy)
        all_pairs.extend(agg)

    if not all_pairs:
        print("\nNo data to analyze. Check that tuning CSVs exist and contain kernel_valid=1 rows.")
        sys.exit(1)

    analyze_thresholds(all_pairs)


if __name__ == "__main__":
    main()
