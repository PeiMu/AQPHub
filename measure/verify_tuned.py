#!/usr/bin/env python3
"""Verify tuned CSV performance against tune_per_subquery.py predictions.

Compares per-query (jit_compile + execute) totals from the measured tuned CSV
against the predicted totals in tuned_per_subquery_<split>.json.

Usage: python3 verify_tuned.py [--bench=dsb] [split]
"""
import csv, json, os, re, sys


def parse_per_query_totals(csv_file, has_jit, is_node_based):
    """Parse per-query jit+exe totals (mean of warm rows, same as tune script)."""
    results = {}
    if has_jit:
        group_columns = ["extract_next_sub-IR", "generate_sub-SQL",
                         "jit_compile", "execute_sub-SQL",
                         "extra_materialization", "update_IR"]
    else:
        group_columns = ["extract_next_sub-IR", "generate_sub-SQL",
                         "execute_sub-SQL", "extra_materialization",
                         "update_IR"]
    tail_size = 4 if has_jit else 3
    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].startswith("Running benchmark for"):
                match = re.search(r"/([0-9a-z_]+)\.sql", row[0])
                if not match:
                    continue
                sql_name = match.group(1)
                perf_rows = []
                for _ in range(15):
                    try:
                        perf_row = next(reader)
                        if is_node_based:
                            gv = perf_row[4:-tail_size]
                        else:
                            gv = perf_row[5:-tail_size]

                        extra = 0.0
                        if len(gv) % len(group_columns) != 0:
                            gv = gv[:-1]

                        jit_sum = 0.0
                        exe_sum = 0.0
                        n = len(gv) // len(group_columns)
                        for i in range(n):
                            if has_jit:
                                jit_sum += float(gv[i * len(group_columns) + 2])
                                exe_sum += float(gv[i * len(group_columns) + 3])
                            else:
                                exe_sum += float(gv[i * len(group_columns) + 2])

                        if has_jit:
                            jit_sum += float(perf_row[-3])
                            exe_sum += float(perf_row[-2])
                        else:
                            exe_sum += float(perf_row[-2])

                        perf_rows.append(jit_sum + exe_sum)
                    except (StopIteration, ValueError, IndexError):
                        break

                if len(perf_rows) < 6:
                    continue
                warm = perf_rows[5:]
                results[sql_name] = sum(warm) / len(warm)
    return results


def main():
    bench = "job"
    positional = []
    for a in sys.argv[1:]:
        if a.startswith("--bench="):
            bench = a.split("=", 1)[1]
        else:
            positional.append(a)
    split = positional[0] if positional else "node-based"
    result_dir = "dsb_result" if bench == "dsb" else "job_result"
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), result_dir)
    is_nb = split == "node-based"

    tuned_csv = os.path.join(base,
        f"duckdb_{split}_query_none_tuned_breakdown_time_log.csv")
    tune_json_path = os.path.join(base,
        f"tuned_per_subquery_{split}.json")

    if not os.path.exists(tuned_csv):
        print(f"Missing: {tuned_csv}")
        sys.exit(1)
    if not os.path.exists(tune_json_path):
        print(f"Missing: {tune_json_path}")
        sys.exit(1)

    measured = parse_per_query_totals(tuned_csv, True, is_nb)
    tune_json = json.load(open(tune_json_path))

    # Predicted total = sum of per-subquery total_ms from JSON
    predicted = {}
    for q, subs in tune_json.items():
        predicted[q] = sum(s["total_ms"] for s in subs.values())

    common = sorted(set(measured) & set(predicted))
    print(f"split: {split}  queries: {len(common)}\n")
    print(f"{'query':8} {'predicted_ms':>13} {'measured_ms':>12} {'diff_ms':>9} {'ratio':>7}")
    print("-" * 53)

    total_pred = 0.0
    total_meas = 0.0
    big_diffs = []
    for q in common:
        p = predicted[q]
        m = measured[q]
        d = m - p
        r = m / p if p > 0 else float("inf")
        total_pred += p
        total_meas += m
        if abs(d) > 5:
            big_diffs.append((q, p, m, d, r))
        print(f"{q:8} {p:13.2f} {m:12.2f} {d:9.2f} {r:7.2f}")

    print("-" * 53)
    print(f"{'TOTAL':8} {total_pred:13.2f} {total_meas:12.2f} "
          f"{total_meas - total_pred:9.2f} {total_meas / total_pred:7.2f}")
    print(f"\npredicted suite: {total_pred / 1000:.2f} s")
    print(f"measured suite:  {total_meas / 1000:.2f} s")
    print(f"difference:      {(total_meas - total_pred) / 1000:.2f} s "
          f"({(total_meas / total_pred - 1) * 100:+.1f}%)")

    if big_diffs:
        print(f"\n{len(big_diffs)} queries with |diff| > 5 ms:")
        for q, p, m, d, r in sorted(big_diffs, key=lambda x: -abs(x[3])):
            print(f"  {q:8} pred={p:.1f}  meas={m:.1f}  diff={d:+.1f} ms")


if __name__ == "__main__":
    main()
