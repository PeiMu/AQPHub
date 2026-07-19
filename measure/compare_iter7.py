#!/usr/bin/env python3
"""Compare Iter 7 split-governor sweep CSVs against the baseline config.

Usage: python3 compare_iter7.py [suffix ...]
Default suffixes: govE3 govE5 gate34m govE3gate34m
"""
import statistics
import sys

BASE = "job_result/duckdb_node-based_query_none_noprefetch_tpde"
TAIL = "_breakdown_time_log.csv"
WARMUP_ROWS = 5


def load(path):
    """Return {query: total_ms_median} plus per-query component sums."""
    blocks = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Running benchmark"):
                cur = line.split("/")[-1].replace(".sql...", "")
                blocks[cur] = []
            elif line and cur is not None:
                try:
                    blocks[cur].append([float(x) for x in line.split(",") if x.strip()])
                except ValueError:
                    pass
    totals = {}
    for q, rows in blocks.items():
        rows = [r for r in rows[WARMUP_ROWS:] if r]
        if not rows:
            continue
        per_run = [sum(r) for r in rows]
        totals[q] = statistics.median(per_run)
    return totals


def main():
    suffixes = sys.argv[1:] or ["govE3", "govE5", "gate34m", "govE3gate34m"]
    base = load(BASE + TAIL)
    base_total = sum(base.values())
    print(f"baseline: {base_total:,.1f} ms over {len(base)} queries\n")
    for sfx in suffixes:
        try:
            cfg = load(f"{BASE}_{sfx}{TAIL}")
        except FileNotFoundError:
            print(f"{sfx:>16}: (missing)")
            continue
        common = sorted(set(base) & set(cfg))
        delta = {q: cfg[q] - base[q] for q in common}
        total = sum(cfg[q] for q in common)
        base_c = sum(base[q] for q in common)
        wins = sorted(delta.items(), key=lambda kv: kv[1])[:5]
        losses = sorted(delta.items(), key=lambda kv: kv[1])[-5:]
        print(f"{sfx:>16}: {total:,.1f} ms  (Δ {total-base_c:+,.1f} ms, "
              f"{len(common)} queries)")
        print(f"{'':>18}best: " + ", ".join(f"{q} {d:+.1f}" for q, d in wins))
        print(f"{'':>18}worst: " + ", ".join(f"{q} {d:+.1f}" for q, d in losses))
    print()


if __name__ == "__main__":
    main()
