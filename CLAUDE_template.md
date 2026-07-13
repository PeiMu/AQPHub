You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

## Goal

## Analysis 

### Root cause

### Fix

## Implementation Plan

## Key Code Locations (Reference)

### AQPHub Core

| File | What |
|------|------|

### SimplestIR (AQPHub's intermediate representation)

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions (1633 lines) |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan → SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR → SQL string (reference for how IR is walked) |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

## Repositories

- **AQPHub**: `/home/pei/Project/AQPHub` (branch: `topdown_fix`)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`

Build commands:
```bash
# Middleware (debug / release)
cd /home/pei/Project/AQPHub && cmake --build build_debug -j$(nproc)
cd /home/pei/Project/AQPHub && cmake --build build_release -j$(nproc)
# DuckDB — the middleware links the PREBUILT libduckdb.so from
# ${DUCKDB_ROOT}/build/release/src (find_library + dynamic link).
cd /home/pei/Project/duckdb && cmake --build build/release -j 16
```

Build hazards:
- build_debug and build_release share `lib/*.a` outputs. If a debug build
  poisons the release archive (ASan), rebuild with
  `--target IR_SQL_Converter_C_static --clean-first`.

## Verification Workflow

### Step 2: Single query end-to-end

Take a simple JOB query that related to our changes (e.g., 1a — single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_job.sh). Compare result to DuckDB golden output.

### Step 4: Full JOB correctness

Run all 113 JOB queries with the correct flags, compare against golden files:
- `measure/duckdb_job_no-split_golden.txt`
- `measure/duckdb_job_node-based_golden.txt`

For more flags, check measure/correctness_test.sh.

### Analysis scripts (measure/*.py)
- `tune_per_subquery.py [split]` — pick best config per (query, sub-query)
- `show_all_configs.py [split]` — summary table across all configs
- `find_top_queries.py [path] [--top=N]` — rank queries by slowest median

---

## Implementation Status: 


### Verification

- Build: passes (release)
- Correctness: all 113 JOB queries pass for all related configs
- CSV format: unchanged (no new columns), parseable by /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
- No new print statements; existing LIKE debug trace guarded by `#ifndef NDEBUG`
- Check if need changes to measure/*.sh or measure/*.py
- Performance (top 15 worst LIKE queries): measure/measure_breakdown_time_job.sh with correct configs 
