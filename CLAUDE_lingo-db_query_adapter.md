You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

## Goal

## Design: LinGo-DB Query-JIT Adapter (Implemented)

### Overview

The `lingodb` engine adapter (`LingoDBAdapter`) supports the middleware's own
query-JIT compilation and runtime, with full configuration parity with DuckDB
and PostgreSQL. Single engine — no separate `lingo-db-runtime`.

```
SQL → pg_query.h parse → SimplestIR → split strategy → sub-IR
    → AnnotateBuildSidesFromMLIR (IR→MLIR→OptimizeImplementationsPass→walk)
    → AnalyzeQueryJit → CompileQuerySteps (LLVM/FastISel/TPDE)
    → QjitExecutor::Run → FlatTable result → Arrow → LinGo-DB temp table
    → fallback: existing IR→MLIR→LinGo-DB JIT path (when query-JIT rejects)
```

### Supported Configuration Matrix

| Config | Values | Default |
|--------|--------|---------|
| `--engine` | `lingodb` | — |
| `--split` | `none`, `node-based`, `topdown`, `relationship-center`, `entity-center`, `min-subquery` | `none` |
| `--jit-level` | `none`, `query` | `none` |
| `--jit-simd` | `none`, `avx2`, etc. | `none` |
| `--compile-mode` | `llvm` (O2), `fastisel` (O0+FastISel), `tpde` (TPDE fast codegen) | `llvm` |
| `--jit-cache` | off, `single-run`, `single-run-template`, `full`, `structural` | off |
| `--payload-prune` | on/off | on |
| `--prefetch` | on/off/distance | on |
| `--batch-probe` | on/off | on |
| `--skip-hash-cmp` | off/all | all |
| `--spec-jit` | off (not wired for LinGo-DB) | off |
| `--plan-optimizer` | `duckdb`, `lingodb` (alias: `--estimator=`) | same as engine |
| `--lingodb-mode` | `llvm` (LLVM JIT), `tpde` (TPDE fast codegen) | `llvm` |

When `--jit-level=none`: `$4` is `lingodb_mode` (llvm/tpde), LinGo-DB's own JIT executes.
When `--jit-level=query`: middleware query-JIT compiles and executes sub-queries.

### Architecture Decisions

**AD-1: Parser — pg_query.h for IR generation.**
LinGo-DB's SQL frontend produces MLIR directly, not SimplestIR. We use pg_query.h
→ parse tree → `ConvertParseTreeToIRWithSchema()` for IR generation. Same as
PostgreSQL. LinGo-DB's own optimizer is used via MLIR for build/probe annotation.

**AD-2: Build/probe annotation — MLIR-based via `AnnotateBuildSidesFromMLIR`.**
Per sub-query (like DuckDB), each sub-IR is:
1. Converted to MLIR RelAlg via `IRToRelAlgConverter` (each join gets `irJoinId` attr)
2. Optimized by `PartialQueryOptimizer` (runs `OptimizeImplementationsPass` which
   sets `useHashJoin`, `leftHash`, `rightHash` based on cardinality + index info)
3. Walked: for each `InnerJoinOp`/`MarkJoinOp` with `irJoinId`, the corresponding
   IR `SimplestJoin::SetBuildChild()` is set based on LinGo-DB's convention
   (MLIR right = build = IR children[0])
4. Fallback: `AnnotateBuildSidesByCard()` for any unannotated joins

Key fact: `OptimizeImplementationsPass` does NOT swap InnerJoinOp MLIR operands
for hash joins — only swaps `leftHash`/`rightHash` attr contents. MLIR operand
order matches `convertJoin()`. For SemiJoin/MarkJoin, `reverseSides` attr is set.

**AD-3: ReOptimizeIR — not needed.** Zero call sites in entire codebase.

**AD-4: Node-based split — works via `helper_db`.**
`owned_duckdb_adapter_` from `config_.helper_db` handles planning.
`external_execution_` flag routes execution through LinGo-DB.

**AD-5: Speculative compilation — not wired for LinGo-DB.**

**AD-6: StoragePlan — load from `--storage-cache` or DuckDB helper.**
`aqp_middleware.cpp` loads from cache file or via `DuckDBAdapter(helper_db)`.

**AD-7: `--plan-optimizer` flag (alias: `--estimator`).**
CLI: `--plan-optimizer=duckdb|postgresql|lingodb|umbra`. When `duckdb` (default
for lingodb with `--helper-db-path`), the no-split path uses DuckDB helper for
IR generation. When not specified and no helper_db, LinGo-DB's own pipeline runs.

### Key Code Locations (LinGo-DB Query-JIT)

| File | What |
|------|------|
| `include/adapters/lingodb_adapter.h` | `LingoDBAdapter` class with JIT interface, `QjitCompiled` struct, `ConfigureQueryJit*` overrides |
| `src/adapters/lingodb_adapter.cpp:595-700` | `IRToRelAlgConverter` (IR→MLIR, `irJoinId` attrs in `convertJoin`) |
| `src/adapters/lingodb_adapter.cpp:2152-2225` | `PartialQueryOptimizer` (RelAlg passes, skips join reorder) |
| `src/adapters/lingodb_adapter.cpp:2558-2620` | `AnnotateBuildSidesByCard` + `AnnotateBuildSidesFromMLIR` |
| `src/adapters/lingodb_adapter.cpp:2290-2350` | `ExecuteIRandCreateTempTable` (JIT fast path + MLIR fallback) |
| `src/adapters/lingodb_adapter.cpp:2430-2490` | `ExecuteIRQuery` (JIT fast path for final query) |
| `src/adapters/lingodb_adapter.cpp:2630-2750` | `TryCompileQueryJit`, `ExecuteQueryJit`, `ExecuteQueryJitFinal` |
| `src/adapters/lingodb_adapter.cpp:2780-2850` | `MaterializeQjitTempToLingoDB` (QjitTable → Arrow → LinGo-DB) |
| `include/qjit/qjit_annotate.h` | Shared: `AnnotateUnannotatedJoinsByCard`, `EstimateSubtreeCard`, `CollectTableNames`, etc. |
| `src/qjit/qjit_annotate.cpp` | Implementation of shared annotation helpers |
| `include/adapters/db_adapter.h` | `ConfigureQueryJit*` virtual no-ops (generic JIT config interface) |
| `src/split/ir_query_splitter.cpp:1064` | JIT config wiring for `LINGODB` in split loop |
| `src/aqp_middleware.cpp:410` | JIT config wiring for `LINGODB` in no-split path |
| `src/aqp_middleware.cpp:470` | No-split DuckDB-helper IR path (when `--plan-optimizer=duckdb`) |

### Component Design (implemented)

#### 1. `--plan-optimizer` flag (alias: `--estimator`)

**File**: `include/util/param_config.h`, `src/util/param_config.cpp`

```
--plan-optimizer=duckdb|postgresql|lingodb|umbra  (default: same as --engine)
--helper-db-path=<connection>  (unchanged)
```

Rename `estimator_engine` → `plan_optimizer` throughout. Add `lingodb` to the
valid values in `ParseFromArgs()`. Backward-compatible: keep `--estimator=` as
an alias.

When `plan_optimizer=lingodb`:
- `ParseSQL()` uses LinGo-DB's SQL frontend → full MLIR module
- Full `createQueryOptPipeline()` runs (including `OptimizeJoinOrderPass`)
- The optimized MLIR module is walked to extract the plan structure
- This is a new `ConvertPlanToIR` path — future work, not part of the initial
  query-JIT implementation

For the initial implementation, `plan_optimizer=duckdb` (the default) is
sufficient — DuckDB's optimizer provides join ordering and the cardinality
estimates that drive build/probe annotation.

#### 2. Build/probe annotation (MLIR-based, implemented)

`AnnotateBuildSidesFromMLIR()` per sub-query: converts IR→MLIR, runs
`PartialQueryOptimizer`, walks the optimized module to set `SetBuildChild()`.
Falls back to `AnnotateBuildSidesByCard()` for unannotated joins. See AD-2.

#### 3. Query-JIT compiler integration (implemented)

All JIT state, config setters, and methods are in `LingoDBAdapter` (same
interface as PostgreSQL). Config is applied via `ConfigureQueryJit*` virtual
overrides from `EngineAdapter` base class, avoiding C++20 header dependencies
in the splitter (which compiles with C++17).

#### 4. Execution paths (implemented)

`ExecuteIRandCreateTempTable()` and `ExecuteIRQuery()` have JIT fast paths:
`AnnotateBuildSidesFromMLIR` → `AnalyzeQueryJit` → `TryCompileQueryJit` →
`ExecuteQueryJit` → `MaterializeQjitTempToLingoDB` (QjitTable → Arrow).
Falls back to MLIR execution (IRFrontend → PartialQueryOptimizer → LinGo-DB JIT)
when query-JIT rejects or fails.

#### 5. Config wiring (implemented)

- No-split path (`aqp_middleware.cpp`): `ConfigureQueryJit*` virtuals
- Split path (`ir_query_splitter.cpp`): same virtuals, engine-agnostic
- `--plan-optimizer=duckdb`: no-split uses DuckDB helper for IR generation
- StoragePlan: loads from `--storage-cache` or DuckDB helper

#### 6. Benchmark scripts (implemented)

- `measure/run_aqp.sh`, `hyperfine_aqp.sh`, `measure_breakdown_time_aqp.sh`:
  lingodb supports `jit_level=query/expr` (not forced to `none`)
- `measure/correctness_test_job_lingodb.sh`, `correctness_test_dsb_lingodb.sh`:
  all split x compile-mode x cache-mode configs
- DSB CSV fallback: uses `csv/` if `lingo_db_csv/` doesn't exist

### Remaining Open Questions

- **Arrow→FlatTable direct conversion**: Would avoid `--storage-cache` /
  `--helper-db-path` requirement for query-JIT. Can be added later.
- **Spec-JIT for LinGo-DB**: Not wired. Can be added if needed.

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
