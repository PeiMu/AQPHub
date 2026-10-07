You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

## Goal

## Analysis: Scan Forwarding for PostgreSQL Adapter with Query-JIT

### Context

DuckDB adapter has a **replacement scan** mechanism: when a query references a temp
table, DuckDB's binder intercepts the lookup and serves data from middleware-owned
in-memory buffers (ColumnDataCollection, QjitTable, FlatTable) via registered table
functions (`scan_temp_collection`, `scan_qjit_temp`, `scan_kernel_temp`). No
materialization to the DuckDB catalog is needed (unless `--no-scan-forwarding` is set).

PostgreSQL adapter has **no equivalent hook**. All JIT-produced intermediate results
are always materialized to PG temp tables via `COPY FROM STDIN`, even when the next
sub-query also JIT-compiles and reads from `qjit_temps_` in memory. This is pure
overhead in the common case.

### Verified facts (config: `postgresql node-based query none on on on all single-run-strict recompile fastisel`)

1. **JIT→JIT cascade works via `qjit_temps_` alone** — confirmed by running 10 JOB
   queries. All sub-queries JIT-compiled; zero interpreter fallbacks. 32 temp tables
   were unnecessarily materialized to PG.
2. **PG optimizer (`ConvertPlanToIRFromPgOptimizer`) is skipped** in the JIT path
   because the splitter always sets `qjit_pending_ir_` (fast IR path at
   `postgres_adapter.cpp:430`).
3. **`AnnotateBuildSidesByCard` does NOT query PG** — reads only `temp_table_card_`
   (in-memory map) and `qjit_storage_plan_` (static metadata).
4. **Speculative compilation does NOT need PG temp tables** — runs on bg thread with
   no DB access, uses cached cardinality snapshot.
5. **`FetchPgTempIntoQjitTemps`** (interp→JIT cascade) exists at
   `postgres_adapter.cpp:1530-1592` but is disabled ("cascade fix temporarily disabled
   for debugging" in commit `6e0d625`).

### Root cause

`MaterializeQjitTempToPostgreSQL()` is called unconditionally after every JIT execution
at `postgres_adapter.cpp:382` (spec-hit) and `:490` (normal JIT), even though the data
is already available in `qjit_temps_` for subsequent JIT sub-queries.

### Fix: Three complementary TODOs

## Implementation Plan

### TODO 1: Defer materialization for JIT-produced temps

**Status: DONE**

Skip `MaterializeQjitTempToPostgreSQL` when JIT succeeds. Track which temps have been
materialized to PG via `pg_materialized_temps_` set. Materialize lazily only when the
PG interpreter fallback needs them (analysis rejected, compilation failed, or final
query PG fallback). `DropTempTable` skips PG DROP for non-materialized temps.

**Files changed:**
- `include/adapters/postgres_adapter.h` — added `pg_materialized_temps_` set and
  `EnsureTempsMaterializedForSQL` method
- `src/adapters/postgres_adapter.cpp` — removed unconditional materialization at
  spec-hit, normal JIT, and replay paths; added on-demand materialization before
  interpreter fallback (`ExecuteSQLandCreateTempTable`) and before final PG fallback
  (`ExecuteSQL`); `DropTempTable` skips non-materialized temps; word-boundary matching
  in `EnsureTempsMaterializedForSQL` to avoid `temp1` matching `temp10`

**Verification (10 JOB queries, config: pg node-based query none ... recompile fastisel):**
- Correctness: 10/10 pass
- Baseline: ~2100-2900 ms (3 runs: 2873, 2126, 2143)
- Deferred: ~1600-1960 ms (3 runs: 1962, 1664, 1618)
- Speedup: ~22-25% wall time reduction (eliminated 32→0 unnecessary COPY FROM STDIN
  materializations per 10 queries)
- `ConvertPlanToIRFromPgOptimizer` gracefully falls back to parse-tree IR when temps
  not in PG catalog

### TODO 2: PG replacement scan extension (server-side)

**Status: DONE**

Created `aqp_scan_forward` PostgreSQL extension at
`Postgresql-18.3/contrib/aqp_scan_forward/`. Uses `get_relation_info_hook` +
`set_rel_pathlist_hook` + CustomScan to intercept scans on registered temp tables
and serve data directly from `/dev/shm/` shared memory files written by the middleware.

**Extension files:**
- `aqp_scan_forward.c` — hooks, CustomScan callbacks, register/unregister functions
- Build: `cd build && make -C contrib/aqp_scan_forward && sudo make -C contrib/aqp_scan_forward install`
- Setup: `CREATE EXTENSION aqp_scan_forward` + `ALTER DATABASE imdb SET session_preload_libraries = 'aqp_scan_forward'`

**Middleware files changed:**
- `include/adapters/postgres_adapter.h` — added `shm_forwarded_temps_`,
  `scan_forward_available_`, `WriteQjitTempToShm`, `ForwardQjitTempViaShm`,
  `UnlinkShmFile`, `ShmPath`
- `include/qjit/query_jit_runtime.h` — added `FlatDataConst`, `FlatValidityConst`
- `src/adapters/postgres_adapter.cpp` — shared memory serialization format (header +
  column metadata + data + validity + string pool), `ForwardQjitTempViaShm` (schema-only
  CREATE + register), feature detection with EXPLAIN round-trip test, cleanup in
  DropTempTable and ResetQueryState

**Key design decisions:**
- Schema-only CREATE TEMP TABLE (~1ms) so PG parser can resolve table references
- `get_relation_info_hook` overrides `rel->tuples` for correct join cardinality
- VARCHAR strings >12 bytes: ptr rewritten to string pool offset in shm file
- DOUBLE/FLOAT columns mapped to PG `text` type (matching old COPY path)
- Feature detection: EXPLAIN round-trip test verifies hooks are actually loaded
- Graceful fallback: if extension not loaded, falls back to COPY materialization

**Verification (113 JOB queries, config: pg node-based query none ... recompile fastisel):**
- Correctness: 113/113 pass, zero diff against golden
- Baseline (no changes): ~93s
- TODO 1 only: ~75s
- TODO 1 + TODO 2: **~73s**

Does NOT affect the DuckDB adapter.

### TODO 3: Re-enable `FetchPgTempIntoQjitTemps` (interp→JIT cascade)

**Status: DONE**

Re-enabled the `FetchPgTempIntoQjitTemps` call at `postgres_adapter.cpp:638-642`.
When PG interpreter creates a temp table, this fetches the data back into
`qjit_temps_` so subsequent JIT sub-queries can resolve it without a PG round-trip.

**Bug fixed during implementation:**
- `QjitTable::ElemSize` doesn't handle FLOAT (dtype=5) or BOOL (dtype=0).
  DSB queries with NUMERIC columns map to FloatVar in the IR, and
  `IrTargetListToDtypes` maps FloatVar with default bit_width (0) to
  `AQP_DTYPE_FLOAT`. Fixed by promoting FLOAT→DOUBLE and BOOL→INT32
  in `FetchPgTempIntoQjitTemps` before creating the QjitTable.
- Added DOUBLE (dtype=6) parsing via `strtod` in the row conversion loop
  (previously fell through to VARCHAR/string branch).

**Files changed:**
- `src/adapters/postgres_adapter.cpp` — uncommented FetchPgTempIntoQjitTemps call;
  added FLOAT→DOUBLE/BOOL→INT32 promotion in FetchPgTempIntoQjitTemps;
  added DOUBLE parsing branch

**Verification:**
- JOB correctness: 113/113 pass (config: pg node-based query none ... strict recompile fastisel)
- DSB correctness: 22/23 pass (config: pg node-based query none ... parameterized recompile tpde)
  — same pass/fail as baseline (1 query fails due to CHAR(n) type, pre-existing)
- Performance: JOB and DSB CSVs generated with 5 iterations

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
