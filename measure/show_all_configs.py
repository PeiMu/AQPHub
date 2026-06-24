#!/usr/bin/env python3
"""Show performance breakdown of all node-based configs in job_result/.

Uses the same parser as plot_middleware_jit.py (drop 5 warmup, mean of 10).
Columns: middleware overhead, jit compile, execute, end-to-end total (all in seconds).

Usage: python3 show_all_configs.py [split]
  split: node-based (default) | none
"""
import csv, json, os, re, sys


def analyze_middleware_breakdown(csv_file, has_jit, is_node_based):
    """Copied from plot_middleware_jit.py — canonical parser."""
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
                match = re.search(r"/([0-9a-z]+)\.sql", row[0])
                if not match:
                    continue
                sql_name = match.group(1)
                perf_rows = []
                for _ in range(15):
                    try:
                        perf_row = next(reader)
                        prepare_middleware_time = float(perf_row[0])
                        read_sql_time = float(perf_row[1])
                        parse_sql_time = float(perf_row[2])
                        preprocess_time = float(perf_row[3])
                        if is_node_based:
                            convert_plan_to_ir_time = 0.0
                            group_values = perf_row[4:-tail_size]
                        else:
                            convert_plan_to_ir_time = float(perf_row[4])
                            group_values = perf_row[5:-tail_size]

                        show_output_time = float(perf_row[-1])
                        final_exe_time = float(perf_row[-2])
                        if has_jit:
                            jit_compile_final_time = float(perf_row[-3])
                            generate_final_sub_sql_time = float(perf_row[-4])
                        else:
                            jit_compile_final_time = 0.0
                            generate_final_sub_sql_time = float(perf_row[-3])

                        extra_extract = 0.0
                        if len(group_values) % len(group_columns) != 0:
                            extra_extract = float(group_values[-1])
                            group_values = group_values[:-1]

                        groups = []
                        n = len(group_values) // len(group_columns)
                        for i in range(n):
                            g = {}
                            for j, col in enumerate(group_columns):
                                g[col] = float(group_values[i * len(group_columns) + j])
                            groups.append(g)

                        row_data = {
                            "prepare_middleware": prepare_middleware_time,
                            "read_sql": read_sql_time,
                            "parse_sql": parse_sql_time,
                            "preprocess": preprocess_time,
                            "convert_plan_to_ir": convert_plan_to_ir_time,
                            "groups": groups,
                            "num_executes": len(groups),
                            "extra_extract": extra_extract,
                            "generate_final_sub_sql": generate_final_sub_sql_time,
                            "jit_compile_final": jit_compile_final_time,
                            "final_exe": final_exe_time,
                            "show_output": show_output_time,
                        }
                        perf_rows.append(row_data)
                    except (StopIteration, ValueError, IndexError):
                        break

                if len(perf_rows) < 6:
                    continue
                valid = perf_rows[5:]

                overhead = 0.0
                jit_total = 0.0
                exe_total = 0.0
                for r in valid:
                    overhead += (r["prepare_middleware"] + r["read_sql"]
                                 + r["parse_sql"] + r["preprocess"]
                                 + r["convert_plan_to_ir"])
                    for g in r["groups"]:
                        overhead += (g["extract_next_sub-IR"]
                                     + g["generate_sub-SQL"]
                                     + g.get("extra_materialization", 0)
                                     + g["update_IR"])
                        if has_jit:
                            jit_total += g["jit_compile"]
                        exe_total += g["execute_sub-SQL"]
                    overhead += r["extra_extract"]
                    overhead += r["generate_final_sub_sql"]
                    if has_jit:
                        jit_total += r["jit_compile_final"]
                    exe_total += r["final_exe"]
                    overhead += r["show_output"]
                n = len(valid)
                results[sql_name] = {
                    "overhead_ms": overhead / n,
                    "jit_ms": jit_total / n,
                    "exe_ms": exe_total / n,
                    "total_ms": (overhead + jit_total + exe_total) / n,
                }
    return results


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "node-based"
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_result")
    is_nb = split == "node-based"

    CONFIGS = [
        ("interp",
         f"duckdb_{split}_none_off_breakdown_time_log.csv", False),
        ("expr",
         f"duckdb_{split}_expr_none_breakdown_time_log.csv", True),
        ("expr_simd",
         f"duckdb_{split}_expr_auto_breakdown_time_log.csv", True),
        ("operator",
         f"duckdb_{split}_operator_none_breakdown_time_log.csv", True),
        ("operator_simd",
         f"duckdb_{split}_operator_auto_breakdown_time_log.csv", True),
        ("pipeline",
         f"duckdb_{split}_pipeline_none_breakdown_time_log.csv", True),
        ("pipeline_simd",
         f"duckdb_{split}_pipeline_auto_breakdown_time_log.csv", True),
        ("query_full",
         f"duckdb_{split}_query_none_breakdown_time_log.csv", True),
        ("query_fastisel",
         f"duckdb_{split}_query_none_fcfastisel_breakdown_time_log.csv", True),
        ("query_tpde",
         f"duckdb_{split}_query_none_fctpde_breakdown_time_log.csv", True),
        ("query_full_simd",
         f"duckdb_{split}_query_auto_breakdown_time_log.csv", True),
        ("query_fastisel_simd",
         f"duckdb_{split}_query_auto_fcfastisel_breakdown_time_log.csv", True),
        ("tuned",
         f"duckdb_{split}_query_none_tuned_breakdown_time_log.csv", True),
        # --- query-jit spec-jit (no cache) ---
        ("query_full_specrecomp",
         f"duckdb_{split}_query_none_specrecompile_breakdown_time_log.csv", True),
        ("query_full_specinterp",
         f"duckdb_{split}_query_none_specinterpret_breakdown_time_log.csv", True),
        ("query_fasti_specrecomp",
         f"duckdb_{split}_query_none_specrecompile_fcfastisel_breakdown_time_log.csv", True),
        ("query_fasti_specinterp",
         f"duckdb_{split}_query_none_specinterpret_fcfastisel_breakdown_time_log.csv", True),
        ("query_tpde_specrecomp",
         f"duckdb_{split}_query_none_specrecompile_fctpde_breakdown_time_log.csv", True),
        ("query_tpde_specinterp",
         f"duckdb_{split}_query_none_specinterpret_fctpde_breakdown_time_log.csv", True),
        # --- query-jit SIMD (no cache) ---
        ("query_full_simd_specrecomp",
         f"duckdb_{split}_query_auto_specrecompile_breakdown_time_log.csv", True),
        ("query_full_simd_specinterp",
         f"duckdb_{split}_query_auto_specinterpret_breakdown_time_log.csv", True),
        ("query_fasti_simd_specrecomp",
         f"duckdb_{split}_query_auto_specrecompile_fcfastisel_breakdown_time_log.csv", True),
        ("query_fasti_simd_specinterp",
         f"duckdb_{split}_query_auto_specinterpret_fcfastisel_breakdown_time_log.csv", True),
        # --- jit-cache=single-run-strict ---
        ("interp_strict",
         f"duckdb_{split}_none_off_jitcache_single_run_strict_breakdown_time_log.csv", False),
        ("expr_strict",
         f"duckdb_{split}_expr_none_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("expr_simd_strict",
         f"duckdb_{split}_expr_auto_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("oper_strict",
         f"duckdb_{split}_operator_none_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("oper_simd_strict",
         f"duckdb_{split}_operator_auto_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("pipe_strict",
         f"duckdb_{split}_pipeline_none_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("pipe_simd_strict",
         f"duckdb_{split}_pipeline_auto_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("q_full_strict",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("q_fasti_strict",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_fcfastisel_breakdown_time_log.csv", True),
        ("q_tpde_strict",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_fctpde_breakdown_time_log.csv", True),
        ("q_strict_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_specrecompile_breakdown_time_log.csv", True),
        ("q_strict_sint",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_specinterpret_breakdown_time_log.csv", True),
        ("q_fasti_strict_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_specrecompile_fcfastisel_breakdown_time_log.csv", True),
        ("q_fasti_strict_sint",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_specinterpret_fcfastisel_breakdown_time_log.csv", True),
        ("q_tpde_strict_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_specrecompile_fctpde_breakdown_time_log.csv", True),
        ("q_tpde_strict_sint",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_specinterpret_fctpde_breakdown_time_log.csv", True),
        ("q_simd_strict",
         f"duckdb_{split}_query_auto_jitcache_single_run_strict_breakdown_time_log.csv", True),
        ("q_simd_fasti_strict",
         f"duckdb_{split}_query_auto_jitcache_single_run_strict_fcfastisel_breakdown_time_log.csv", True),
        ("q_simd_strict_srec",
         f"duckdb_{split}_query_auto_jitcache_single_run_strict_specrecompile_breakdown_time_log.csv", True),
        ("q_simd_strict_sint",
         f"duckdb_{split}_query_auto_jitcache_single_run_strict_specinterpret_breakdown_time_log.csv", True),
        ("q_simd_fasti_strict_srec",
         f"duckdb_{split}_query_auto_jitcache_single_run_strict_specrecompile_fcfastisel_breakdown_time_log.csv", True),
        ("q_simd_fasti_strict_sint",
         f"duckdb_{split}_query_auto_jitcache_single_run_strict_specinterpret_fcfastisel_breakdown_time_log.csv", True),
        ("tuned_strict",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_tuned_breakdown_time_log.csv", True),
        ("tuned_strict_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_strict_specrecompile_tuned_breakdown_time_log.csv", True),
        # --- jit-cache=single-run-template ---
        ("expr_tmpl",
         f"duckdb_{split}_expr_none_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("expr_simd_tmpl",
         f"duckdb_{split}_expr_auto_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("oper_tmpl",
         f"duckdb_{split}_operator_none_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("oper_simd_tmpl",
         f"duckdb_{split}_operator_auto_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("pipe_tmpl",
         f"duckdb_{split}_pipeline_none_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("pipe_simd_tmpl",
         f"duckdb_{split}_pipeline_auto_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("q_full_tmpl",
         f"duckdb_{split}_query_none_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("q_fasti_tmpl",
         f"duckdb_{split}_query_none_jitcache_single_run_template_fcfastisel_breakdown_time_log.csv", True),
        ("q_tpde_tmpl",
         f"duckdb_{split}_query_none_jitcache_single_run_template_fctpde_breakdown_time_log.csv", True),
        ("q_tmpl_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_template_specrecompile_breakdown_time_log.csv", True),
        ("q_tmpl_sint",
         f"duckdb_{split}_query_none_jitcache_single_run_template_specinterpret_breakdown_time_log.csv", True),
        ("q_fasti_tmpl_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_template_specrecompile_fcfastisel_breakdown_time_log.csv", True),
        ("q_fasti_tmpl_sint",
         f"duckdb_{split}_query_none_jitcache_single_run_template_specinterpret_fcfastisel_breakdown_time_log.csv", True),
        ("q_tpde_tmpl_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_template_specrecompile_fctpde_breakdown_time_log.csv", True),
        ("q_tpde_tmpl_sint",
         f"duckdb_{split}_query_none_jitcache_single_run_template_specinterpret_fctpde_breakdown_time_log.csv", True),
        ("q_simd_tmpl",
         f"duckdb_{split}_query_auto_jitcache_single_run_template_breakdown_time_log.csv", True),
        ("q_simd_fasti_tmpl",
         f"duckdb_{split}_query_auto_jitcache_single_run_template_fcfastisel_breakdown_time_log.csv", True),
        ("q_simd_tmpl_srec",
         f"duckdb_{split}_query_auto_jitcache_single_run_template_specrecompile_breakdown_time_log.csv", True),
        ("q_simd_tmpl_sint",
         f"duckdb_{split}_query_auto_jitcache_single_run_template_specinterpret_breakdown_time_log.csv", True),
        ("q_simd_fasti_tmpl_srec",
         f"duckdb_{split}_query_auto_jitcache_single_run_template_specrecompile_fcfastisel_breakdown_time_log.csv", True),
        ("q_simd_fasti_tmpl_sint",
         f"duckdb_{split}_query_auto_jitcache_single_run_template_specinterpret_fcfastisel_breakdown_time_log.csv", True),
        ("tuned_tmpl",
         f"duckdb_{split}_query_none_jitcache_single_run_template_tuned_breakdown_time_log.csv", True),
        ("tuned_tmpl_srec",
         f"duckdb_{split}_query_none_jitcache_single_run_template_specrecompile_tuned_breakdown_time_log.csv", True),
    ]

    data = {}
    for label, fname, has_jit in CONFIGS:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            data[label] = analyze_middleware_breakdown(p, has_jit, is_nb)

    if not data:
        print("No CSV files found")
        sys.exit(1)

    # Common queries across all loaded configs
    all_qs = None
    for label, d in data.items():
        qs = set(d.keys())
        all_qs = qs if all_qs is None else all_qs & qs
    all_qs = sorted(all_qs)

    print(f"split: {split}  queries: {len(all_qs)}  configs: {len(data)}\n")
    print(f"{'config':28} {'overhead_s':>11} {'jit_s':>10} {'exe_s':>10} {'total_s':>10}")
    print("-" * 73)
    for label in [l for l, _, _ in CONFIGS if l in data]:
        d = data[label]
        oh = sum(d[q]["overhead_ms"] for q in all_qs) / 1000
        jit = sum(d[q]["jit_ms"] for q in all_qs) / 1000
        exe = sum(d[q]["exe_ms"] for q in all_qs) / 1000
        total = sum(d[q]["total_ms"] for q in all_qs) / 1000
        print(f"{label:28} {oh:11.2f} {jit:10.2f} {exe:10.2f} {total:10.2f}")


if __name__ == "__main__":
    main()
