You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

## Goal
Make PG adapter + node-based split + query-JIT faster than PG adapter + node-based split + no-JIT on DSB sf50.

## Analysis

### Root cause
1. JIT fails for ALL DSB queries (agg:grouped, CrossProduct, expr:cast, source:temp-missing cascade)
2. 53GB storage plan loaded into heap → only 9GB left for PG page cache → 186GB PG database reads from disk
3. PG temp tables from failed JIT sub-queries not loaded back into qjit_temps_ → cascade failure
4. shared_buffers=512MB is too low

### Fixes (priority order)
- [x] Fix 1: Increase shared_buffers 512MB→4GB (postgresql.conf) — **REVERTED: 7% slower** (4GB shared_buffers steals OS page cache)
- [x] Fix 2: Only load 13 needed DSB tables in storage plan (saves ~2GB — most volume is in fact tables that DSB uses) — **13.6% faster, kept**
- [ ] Fix 3: Cascade fix — load PG temp table results back into qjit_temps_ (postgres_adapter.cpp) — SEGFAULT in FetchPgTempIntoQjitTemps, needs debugging (disabled)
- [x] Fix 4: Support GROUP BY in JIT — IMPLEMENTED, correct results, but no perf improvement due to 52GB storage plan memory pressure
- [ ] Fix 5: Skip MaterializeQjitTempToPostgreSQL when next sub-query is JIT-eligible
- [x] Fix 6: mmap for storage plan instead of fread — **26.9x faster, RSS 220MB vs 52GB, kept**
- [ ] Fix 7: Bloom filter to reduce PG disk reads

### Performance log (DSB sf50, PG adapter, node-based split)
| Config | Compilation (s) | Middleware (s) | Execution (s) | Total (s) | Notes |
|--------|----------------|---------------|---------------|-----------|-------|
| Baseline no-JIT | 0.000 | 0.083 | 98.642 | 98.725 | Target to beat |
| Baseline query-JIT TPDE | 3.276 | 9.977 | 2704.381 | 2717.634 | 27.5x slower — all JIT fails, disk I/O |
| Fix1 shared_buffers=4GB | 3.609 | 11.428 | 2898.680 | 2913.716 | REVERTED: 7% slower |
| Fix2 table filtering only | 0.612 | 2.548 | 2345.570 | 2348.730 | 13.6% faster, still 23.8x slower than no-JIT |
| Fix2+4 GROUP BY | 0.554 | 2.671 | 2748.459 | 2751.684 | No improvement — 52GB storage plan memory pressure dominates |
| Fix2+4+6 mmap | 0.008 | 0.247 | 102.072 | 102.327 | **26.9x faster!** RSS 220MB, within 3.6% of no-JIT |

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

- **AQPHub**: `/home/pei/Project/AQP_middleware`
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **DSB queries**: `/home/pei/Project/benchmarks/dsb-postgres/code/tools/1_instance_out_aqp/1/`
- **TPC-H queries**: `/home/pei/Project/benchmarks/tpch-postgres/dbgen/out_50/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DSB schema**: `/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql`
- **TPC-H schema**: `/home/pei/Project/benchmarks/tpch-postgres/dbgen/dss.ddl`
- **DuckDB JOB database**: `/home/pei/Project/duckdb/measure/imdb.db`
- **DuckDB DSB database**: `/home/pei/Project/duckdb/measure/dsb_50.db`
- **DuckDB TPC-H database**: `/home/pei/Project/duckdb/measure/tpch_50.db`

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

### Step 1: Single query end-to-end

Take a simple JOB query that related to our changes (e.g., 1a — single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_aqp.sh job). Compare result to DuckDB golden output.

### Step 2: Full JOB and DSB correctness

check measure/correctness_test_job_duckdb.sh and measure/correctness_test_dsb_duckdb.sh.

### Analysis scripts (measure/*.py)
CSV parser is: /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
---

## Implementation Status: 


### Verification

- Build: passes (release)
- Correctness: all 113 JOB queries pass for all related configs
- CSV format: unchanged (no new columns), parseable by /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
- No new print statements; existing LIKE debug trace guarded by `#ifndef NDEBUG`
- Check if need changes to measure/*.sh or measure/*.py
- Performance (top 15 worst LIKE queries): measure/measure_breakdown_time_aqp.sh job with correct configs 
