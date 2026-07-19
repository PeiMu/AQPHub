## Goal

Enable the TopDown/SDS (Stats-Driven IR Splitter) optimizations -- currently DuckDB-only on
`topdown_fix` -- to work with PostgreSQL on the `jit` branch, without regressing any existing
DuckDB or PostgreSQL code paths.

## Branches

| Branch | Role | Key changes since last sync |
|--------|------|-----------------------------|
| `topdown_fix` | develop | TopDown/SDS rewrite, IRJoinOptimizer DP, DistinctCache, TryBuildBinaryPlanFromIR, early termination, cross-query prep for TOP_DOWN, projection remapping in query-jit |
| `jit` | release (collaboration) | PostgreSQL adapter + query-JIT, LaunchSpeculativeCompilePG, StripCollationDependentRangeQuals, StoragePlan for PG, PG measurement scripts |
| Last sync | `f36b16c` (jit) / `645a90a` (topdown_fix) | |

### Constraints

- No changes to the Postgres database codebase (`/home/pei/Project/Postgresql-18.3`) required
- All changes behind `#ifdef HAVE_POSTGRES` or strategy checks -- no DuckDB path regressions
- `V17Mode` (`AQP_TD_V17`) and `NoPlanCtor` (`AQP_TD_NO_PLANCTOR`) kill-switches remain functional

## Current Status

TODOs 1-4 and 6 are complete. Details and performance results in `pg_jit_improvement.log`.

Current performance (JOB, 110 queries, NB+tmpl best config):
- **2.42x** vs no-split+no-jit, **1.25x** vs NB+no-jit (64 winners, 46 losers)
- TopDown splitting regresses on PG (~0.90x) — extra PG round-trips dominate

## TODOs

### TODO 5: Query-JIT performance for PostgreSQL

Goal: achieve the best possible performance for query-JIT on PostgreSQL.

**Primary issue: full table scan vs PG indexes.** Query-JIT always reads from `StoragePlan`
flat column arrays (full scans + hash joins). PG's native path uses B-tree indexes for
selective queries. DuckDB also does full scans with query-JIT but doesn't regress because
DuckDB's native path is also full-scan (columnar engine, no B-tree indexes).

Measured impact (JOB, node-based, best config node-based/topdown+query-jit+template-cache+tpde+no-spec-jit):
[TODO]

Potential fixes to reach 2x:
1. Index support in StoragePlan (sorted/CSR indexes for selective filters)
2. Selective JIT bypass using EXPLAIN cost estimates (skip JIT when PG would use index scan)
3. Better block-skipping with min/max predicate pushdown

**Secondary issue: MaterializeQjitTempToPostgreSQL overhead.** Row-by-row `PQputCopyData`
with text serialization (`std::to_string` for ints, escape processing for strings). PG server
re-parses text back into internal format. Measured overhead: 5-430ms depending on result size
(small for selective queries, significant for large intermediates like 25c). Potential fix:
use binary COPY format or batch rows into larger buffers.

## Key Code Locations (Reference)

### Splitter Layer (adapter-agnostic)

| File | What |
|------|------|
| `include/split/topdown_splitter.h` | TopDown/SDS splitter header; `V17Mode()`, `NoPlanCtor()` static methods |
| `src/split/topdown_splitter.cpp` | SDS `PlanNext()`, `SplitIR()`, `EstimateFilteredCardinality()` |
| `include/split/ir_join_optimizer.h` | DP join-order optimizer (pure algorithm, no engine deps) |
| `src/split/ir_join_optimizer.cpp` | DPccp on JoinRel/JoinEdge with tdom cardinality model |
| `include/split/distinct_cache.h` | Per-(table,column) stats cache |
| `src/split/distinct_cache.cpp` | `Get()`, `GetCorrelation()`, `GetRowCount()` -- PG uses `pg_stats`/`pg_class`, DuckDB uses SQL |
| `src/split/fk_based_splitter.cpp` | Sub-IR construction, mark-join cloning, condition dedup |

### Orchestrator

| File | What |
|------|------|
| `include/split/ir_query_splitter.h` | `IRQuerySplitter` header; `SpeculativeCompilation` struct, `CrossQueryPrepResult` |
| `src/split/ir_query_splitter.cpp` | `ExecuteWithSplit`, `ExecuteSplitLoop`, `ExecuteOneIteration` -- engine+strategy dispatch |

### Adapters

| File | What |
|------|------|
| `include/adapters/db_adapter.h` | `EngineAdapter` base class; `ApplyEngineSetting` (no-op default), `GenerateSQL` |
| `include/adapters/duckdb_adapter.h` | DuckDB adapter; `SetQjitPendingIR(ir, use_engine_plan=true)`, `TryBuildBinaryPlanFromIR` |
| `src/adapters/duckdb_adapter.cpp` | `ExecuteSQLandCreateTempTable` with plan-ctor fast path + SQL fallback |
| `include/adapters/postgres_adapter.h` | PG adapter; `PgCachedSubquery`, `PgCachedQueryPlan`, `ReplayQjitSubquery`, `ReplayQjitFinal` |
| `src/adapters/postgres_adapter.cpp` | PG `ExecuteSQLandCreateTempTable`, `AnnotateBuildSidesByCard`, `SpeculativeQueryJitCompile`, `ResetQueryState` |

### SimplestIR (intermediate representation)

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan -> SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR -> SQL string |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

### Query JIT

| File | What |
|------|------|
| `src/qjit/query_jit_steps.cpp` | `PlanBuilder` with `proj_remap_` for DuckDB compressed-mat projections |
| `src/jit/ir_to_llvm.cpp` | IR-to-LLVM codegen |

## Repositories

- **AQP_middleware**: `/home/pei/Project/AQP_middleware` (branches: `topdown_fix`, `jit`)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **PostgreSQL**: `/home/pei/Project/Postgresql-18.3`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`

Build commands:
```bash
# Middleware (debug / release)
cd /home/pei/Project/AQP_middleware && cmake --build build_debug -j$(nproc)
cd /home/pei/Project/AQP_middleware && cmake --build build_release -j$(nproc)
# DuckDB
cd /home/pei/Project/duckdb && cmake --build build/release -j 16
```

Build hazards:
- build_debug and build_release share `lib/*.a` outputs. If a debug build
  poisons the release archive (ASan), rebuild with
  `--target IR_SQL_Converter_C_static --clean-first`.

## Verification Workflow

### Single query end-to-end

Take a simple JOB query (e.g., 1a -- single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_job.sh). Compare result
to golden output.

### Full JOB correctness

Run all 113 JOB queries with the correct flags, compare against golden files:
- DuckDB: `measure/duckdb_job_no-split_golden.txt`, `measure/duckdb_job_node-based_golden.txt`
- PostgreSQL: `measure/pg_job_no-split_golden.txt`, `measure/pg_job_node-based_golden.txt`, `measure/pg_job_topdown_golden.txt`

For more flags, check `measure/correctness_test.sh` (DuckDB) and `measure/correctness_test_pg.sh` (PG).

### Analysis scripts (measure/*.py)
- `tune_per_subquery.py [split]` -- pick best config per (query, sub-query)
- `show_all_configs.py [split]` -- summary table across all configs
- `find_top_queries.py [path] [--top=N]` -- rank queries by slowest median
- `verify_tuned_detail.py` -- verify tuned config results
